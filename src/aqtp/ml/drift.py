"""Model drift detection (REQ 52).

Monitors whether the live world still resembles the world the model was trained in.
Four kinds of drift are tracked, because they fail differently:

  feature drift     — inputs have moved (PSI against the training distribution)
  prediction drift  — the model's output distribution has shifted
  calibration drift — probabilities no longer match observed frequencies
  performance drift — realized expectancy has decayed

Population Stability Index is used for distributional drift because it is
interpretable against widely-used bands (>0.25 = significant shift) and does not
require holding the full training sample in memory — only its bin edges.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Mapping

import numpy as np
import pandas as pd

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftReport:
    checked_at: datetime
    feature_psi: dict[str, float] = field(default_factory=dict)
    max_feature_psi: float = 0.0
    drifted_features: list[str] = field(default_factory=list)
    prediction_psi: float = 0.0
    calibration_error: float = float("nan")
    recent_expectancy_r: float = float("nan")
    baseline_expectancy_r: float = float("nan")
    samples: int = 0
    severity: str = "none"      # none | mild | significant | severe
    actions: list[str] = field(default_factory=list)

    @property
    def should_reduce(self) -> bool:
        return self.severity in ("significant", "severe")

    @property
    def should_pause(self) -> bool:
        return self.severity == "severe"

    def describe(self) -> str:
        if self.severity == "none":
            return "no material drift detected"
        return (
            f"{self.severity} drift: max feature PSI {self.max_feature_psi:.3f} "
            f"({len(self.drifted_features)} features), prediction PSI {self.prediction_psi:.3f}"
        )


def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """PSI between two samples using quantile bins from the baseline.

    Bins are derived from the baseline so the measure answers "where does current
    data fall relative to what we trained on", which is the question that matters.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    current = current[np.isfinite(current)]
    if len(baseline) < 20 or len(current) < 20:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(baseline, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0  # a near-constant feature cannot meaningfully drift
    edges[0], edges[-1] = -np.inf, np.inf

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing keeps an empty bin from producing an infinite PSI.
    baseline_pct = (baseline_counts + 1) / (baseline_counts.sum() + len(baseline_counts))
    current_pct = (current_counts + 1) / (current_counts.sum() + len(current_counts))
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


class DriftDetector:
    def __init__(
        self,
        *,
        psi_threshold: float = 0.25,
        window: int = 500,
        min_samples: int = 100,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.window = window
        self.min_samples = min_samples
        self._baseline_features: pd.DataFrame | None = None
        self._baseline_predictions: np.ndarray | None = None
        self._baseline_expectancy: float = float("nan")
        self._live_features: Deque[pd.Series] = deque(maxlen=window)
        self._live_predictions: Deque[float] = deque(maxlen=window)
        self._live_outcomes: Deque[tuple[float, float]] = deque(maxlen=window)  # (prob, label)
        self._live_r: Deque[float] = deque(maxlen=window)

    # ------------------------------------------------------------------ #
    def set_baseline(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray | None = None,
        *,
        expectancy_r: float = float("nan"),
    ) -> None:
        """Record the training-time distribution this model was fit on."""
        self._baseline_features = features.copy()
        self._baseline_predictions = (
            np.asarray(predictions, dtype=float) if predictions is not None else None
        )
        self._baseline_expectancy = expectancy_r
        logger.info(
            "drift baseline set: %d rows, %d features", len(features), features.shape[1]
        )

    def observe(
        self,
        features: pd.Series,
        prediction: float,
        *,
        realized_label: float | None = None,
        realized_r: float | None = None,
    ) -> None:
        """Record one live observation; outcomes arrive later when the trade closes."""
        self._live_features.append(features)
        self._live_predictions.append(float(prediction))
        if realized_label is not None:
            self._live_outcomes.append((float(prediction), float(realized_label)))
        if realized_r is not None:
            self._live_r.append(float(realized_r))

    def record_outcome(self, prediction: float, label: float, realized_r: float) -> None:
        self._live_outcomes.append((float(prediction), float(label)))
        self._live_r.append(float(realized_r))

    # ------------------------------------------------------------------ #
    def check(self, *, now: datetime | None = None) -> DriftReport:
        now = now or datetime.now()
        report = DriftReport(checked_at=now, samples=len(self._live_features))

        if self._baseline_features is None or len(self._live_features) < self.min_samples:
            report.severity = "none"
            report.actions.append(
                f"insufficient live samples ({len(self._live_features)}/{self.min_samples})"
            )
            return report

        live = pd.DataFrame(list(self._live_features))
        common = [c for c in self._baseline_features.columns if c in live.columns]

        for column in common:
            psi = population_stability_index(
                self._baseline_features[column].to_numpy(), live[column].to_numpy()
            )
            report.feature_psi[column] = round(psi, 4)
            if psi > self.psi_threshold:
                report.drifted_features.append(column)
        if report.feature_psi:
            report.max_feature_psi = max(report.feature_psi.values())

        if self._baseline_predictions is not None and len(self._live_predictions) >= self.min_samples:
            report.prediction_psi = population_stability_index(
                self._baseline_predictions, np.array(self._live_predictions)
            )

        if len(self._live_outcomes) >= self.min_samples:
            probabilities = np.array([p for p, _ in self._live_outcomes])
            labels = np.array([label for _, label in self._live_outcomes])
            from .metrics import expected_calibration_error

            report.calibration_error = expected_calibration_error(labels, probabilities)

        if len(self._live_r) >= max(20, self.min_samples // 4):
            report.recent_expectancy_r = float(np.mean(self._live_r))
            report.baseline_expectancy_r = self._baseline_expectancy

        report.severity, report.actions = self._assess(report)
        if report.severity != "none":
            logger.warning("model drift: %s", report.describe())
        return report

    def _assess(self, report: DriftReport) -> tuple[str, list[str]]:
        actions: list[str] = []
        score = 0

        drifted_fraction = (
            len(report.drifted_features) / len(report.feature_psi) if report.feature_psi else 0.0
        )
        if report.max_feature_psi > self.psi_threshold * 2:
            score += 2
            actions.append(
                f"feature PSI {report.max_feature_psi:.2f} is far beyond the "
                f"{self.psi_threshold} threshold"
            )
        elif report.max_feature_psi > self.psi_threshold:
            score += 1
            actions.append(f"feature PSI {report.max_feature_psi:.2f} exceeds threshold")
        if drifted_fraction > 0.3:
            score += 1
            actions.append(f"{drifted_fraction:.0%} of features have drifted")

        if report.prediction_psi > self.psi_threshold:
            score += 1
            actions.append(f"prediction distribution PSI {report.prediction_psi:.2f}")

        if np.isfinite(report.calibration_error) and report.calibration_error > 0.15:
            score += 2
            actions.append(
                f"calibration error {report.calibration_error:.2f} — probabilities are no "
                "longer trustworthy for sizing"
            )

        if np.isfinite(report.recent_expectancy_r) and report.recent_expectancy_r < 0:
            score += 2
            actions.append(
                f"recent expectancy {report.recent_expectancy_r:+.3f}R has turned negative"
            )

        if score >= 4:
            return "severe", actions + ["pause the model and retrain"]
        if score >= 2:
            return "significant", actions + ["reduce exposure and schedule retraining"]
        if score >= 1:
            return "mild", actions + ["monitor closely"]
        return "none", actions

    def reset_live(self) -> None:
        self._live_features.clear()
        self._live_predictions.clear()
        self._live_outcomes.clear()
        self._live_r.clear()
