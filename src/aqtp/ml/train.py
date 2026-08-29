"""Walk-forward training and model comparison (REQ 16/20/21).

The trainer never reports a single aggregate backtest as evidence (REQ 20). It
produces per-fold metrics, a consistency measure, and an explicit PASSED/FAILED
verdict against out-of-sample criteria (REQ 68).

It also never touches the test window during model selection: within each fold the
family is chosen on *validation* performance and then scored once on test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..configuration.schema import ModelConfig
from ..core.errors import ModelError
from ..core.ids import new_run_id
from ..core.logging import get_logger
from ..core.types import Regime
from .dataset import Dataset, Split, assert_split_ordering, walk_forward_splits
from .metrics import (
    ClassificationMetrics,
    WalkForwardReport,
    compare_models,
    evaluate_classification,
)
from .models import CalibratedModel, ModelEnsemble, build_model

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Everything produced by one training run, versioned for the registry."""

    run_id: str
    dataset_description: dict
    per_family: dict[str, WalkForwardReport] = field(default_factory=dict)
    best_family: str = ""
    final_model: CalibratedModel | None = None
    final_metrics: ClassificationMetrics | None = None
    verdict: str = ""
    passed: bool = False
    trained_at: datetime | None = None
    duration_seconds: float = 0.0
    config_snapshot: dict = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def comparison_table(self) -> pd.DataFrame:
        summaries = {}
        for family, report in self.per_family.items():
            summary = report.summary()
            if summary:
                summaries[family] = summary
        if not summaries:
            return pd.DataFrame()
        return pd.DataFrame(summaries).T.sort_values("mean_expected_value", ascending=False)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "best_family": self.best_family,
            "passed": self.passed,
            "verdict": self.verdict,
            "trained_at": str(self.trained_at),
            "duration_seconds": round(self.duration_seconds, 1),
            "dataset": self.dataset_description,
        }


class WalkForwardTrainer:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def run(
        self,
        dataset: Dataset,
        *,
        families: Sequence[str] | None = None,
        cost_r: float = 0.05,
        threshold: float = 0.55,
    ) -> TrainingResult:
        """Train and evaluate every family across walk-forward folds."""
        started = time.perf_counter()
        result = TrainingResult(
            run_id=new_run_id(),
            dataset_description=dataset.describe(),
            trained_at=datetime.now(),
            config_snapshot=self.config.model_dump(),
        )

        if dataset.is_empty:
            result.verdict = "FAILED: dataset is empty"
            return result
        if len(dataset) < self.config.min_training_samples:
            result.verdict = (
                f"FAILED: {len(dataset)} samples is below the configured minimum "
                f"{self.config.min_training_samples}"
            )
            return result

        splits = walk_forward_splits(
            dataset,
            train_days=self.config.walk_forward_train_days,
            validation_days=self.config.walk_forward_validation_days,
            test_days=self.config.walk_forward_test_days,
            step_days=self.config.walk_forward_step_days,
            embargo=timedelta(minutes=self.config.embargo_minutes),
        )
        if not splits:
            result.verdict = (
                "FAILED: not enough history for even one walk-forward fold "
                f"(need ~{self.config.walk_forward_train_days + self.config.walk_forward_validation_days + self.config.walk_forward_test_days} days)"
            )
            return result

        family_names = list(families or self.config.families)
        logger.info(
            "walk-forward: %d folds x %d families on %d samples",
            len(splits), len(family_names), len(dataset),
        )

        for family in family_names:
            report = WalkForwardReport()
            for split in splits:
                try:
                    assert_split_ordering(
                        split, min_embargo=timedelta(minutes=self.config.embargo_minutes)
                    )
                except Exception as exc:
                    logger.error("fold %d rejected: %s", split.fold, exc)
                    continue
                metrics = self._train_fold(family, split, cost_r=cost_r, threshold=threshold)
                if metrics is not None:
                    report.add(metrics, split.describe())
            if report.folds:
                result.per_family[family] = report
                summary = report.summary()
                logger.info(
                    "%-22s folds=%d mean_ev=%+.4fR profitable=%.0f%% brier=%.4f auc=%.3f",
                    family,
                    summary["folds"],
                    summary["mean_expected_value"],
                    summary["profitable_fold_fraction"] * 100,
                    summary["mean_brier"],
                    summary["mean_roc_auc"],
                )

        if not result.per_family:
            result.verdict = "FAILED: no family completed any fold"
            return result

        # Selection is on walk-forward *out-of-sample* performance across folds,
        # which is the only honest basis available.
        result.best_family = max(
            result.per_family,
            key=lambda f: result.per_family[f].summary().get("mean_expected_value", -np.inf),
        )
        best_report = result.per_family[result.best_family]
        result.passed, result.verdict = best_report.verdict()

        # Fit the deployable model on the most recent fold's train+validation data.
        final_split = splits[-1]
        try:
            result.final_model = self._fit_model(result.best_family, final_split)
            result.final_metrics = self._score(
                result.final_model, final_split.test, cost_r=cost_r, threshold=threshold
            )
            importance = result.final_model.feature_importance()
            if not importance.empty:
                result.feature_importance = importance.head(25).round(5).to_dict()
        except ModelError as exc:
            result.verdict = f"FAILED: could not fit final model: {exc}"
            result.passed = False

        result.duration_seconds = time.perf_counter() - started
        logger.info("training run %s: %s", result.run_id, result.verdict)
        return result

    # ------------------------------------------------------------------ #
    def _train_fold(
        self, family: str, split: Split, *, cost_r: float, threshold: float
    ) -> ClassificationMetrics | None:
        if split.train.is_empty or split.test.is_empty:
            return None
        try:
            model = self._fit_model(family, split)
        except ModelError as exc:
            logger.debug("fold %d family %s skipped: %s", split.fold, family, exc)
            return None
        return self._score(model, split.test, cost_r=cost_r, threshold=threshold)

    def _fit_model(self, family: str, split: Split) -> CalibratedModel:
        estimator = build_model(family, self.config.random_seed)
        model = CalibratedModel(estimator, method=self.config.calibration, family=family)
        model.fit(
            split.train.features,
            split.train.labels,
            X_validation=split.validation.features if not split.validation.is_empty else None,
            y_validation=split.validation.labels if not split.validation.is_empty else None,
        )
        return model

    @staticmethod
    def _score(
        model: CalibratedModel, test: Dataset, *, cost_r: float, threshold: float
    ) -> ClassificationMetrics:
        probabilities = model.predict_proba(test.features)
        realized = (
            test.metadata["realized_r"].to_numpy()
            if "realized_r" in test.metadata.columns
            else None
        )
        return evaluate_classification(
            test.labels.to_numpy(),
            probabilities,
            threshold=threshold,
            realized_r=realized,
            cost_r=cost_r,
        )

    # ------------------------------------------------------------------ #
    def build_ensemble(
        self, dataset: Dataset, result: TrainingResult, *, top_n: int = 3
    ) -> ModelEnsemble:
        """Combine the best families into a weighted ensemble (REQ 17)."""
        ranked = sorted(
            result.per_family.items(),
            key=lambda kv: kv[1].summary().get("mean_expected_value", -np.inf),
            reverse=True,
        )[:top_n]
        if not ranked:
            raise ModelError("no trained families to ensemble")

        splits = walk_forward_splits(
            dataset,
            train_days=self.config.walk_forward_train_days,
            validation_days=self.config.walk_forward_validation_days,
            test_days=self.config.walk_forward_test_days,
            step_days=self.config.walk_forward_step_days,
            embargo=timedelta(minutes=self.config.embargo_minutes),
        )
        if not splits:
            raise ModelError("cannot build ensemble without walk-forward folds")

        final_split = splits[-1]
        ensemble = ModelEnsemble()
        scores: dict[str, float] = {}
        for family, report in ranked:
            try:
                model = self._fit_model(family, final_split)
            except ModelError as exc:
                logger.warning("excluding %s from ensemble: %s", family, exc)
                continue
            ensemble.add(family, model)
            scores[family] = report.summary().get("mean_expected_value", 0.0)

        if not ensemble.models:
            raise ModelError("ensemble construction produced no usable models")
        ensemble.set_weights_from_scores(scores)
        logger.info("ensemble weights: %s", ensemble.describe())
        return ensemble


# --------------------------------------------------------------------------- #
def train_regime_specific(
    trainer: WalkForwardTrainer,
    dataset: Dataset,
    regime_labels: pd.Series,
    *,
    min_samples: int = 300,
) -> dict[Regime, TrainingResult]:
    """Train per-regime models, retaining only those that demonstrate value (REQ 21).

    REQ 21 warns against creating regime-specific models merely because it sounds
    sophisticated. So each regime model is compared against the pooled model on the
    same folds, and a regime model is returned only if it *beats* the pooled one out
    of sample. Everything else is discarded.
    """
    results: dict[Regime, TrainingResult] = {}
    pooled = trainer.run(dataset)
    pooled_ev = (
        pooled.per_family.get(pooled.best_family, WalkForwardReport())
        .summary()
        .get("mean_expected_value", -np.inf)
        if pooled.best_family
        else -np.inf
    )

    aligned = regime_labels.reindex(dataset.features.index)
    for regime in aligned.dropna().unique():
        mask = (aligned == regime).to_numpy()
        if mask.sum() < min_samples:
            logger.info("regime %s: %d samples is too few, skipping", regime, int(mask.sum()))
            continue
        subset = Dataset(
            features=dataset.features[mask],
            labels=dataset.labels[mask],
            metadata=dataset.metadata[mask] if not dataset.metadata.empty else dataset.metadata,
            feature_names=list(dataset.feature_names),
            label_spec=dataset.label_spec,
            symbol=dataset.symbol,
            timeframe=dataset.timeframe,
        )
        result = trainer.run(subset)
        if not result.best_family:
            continue
        regime_ev = result.per_family[result.best_family].summary().get("mean_expected_value", -np.inf)
        if regime_ev > pooled_ev:
            try:
                key = Regime(regime)
            except ValueError:
                continue
            results[key] = result
            logger.info(
                "regime %s model retained: %+.4fR vs pooled %+.4fR", regime, regime_ev, pooled_ev
            )
        else:
            logger.info(
                "regime %s model discarded: %+.4fR does not beat pooled %+.4fR",
                regime, regime_ev, pooled_ev,
            )
    return results
