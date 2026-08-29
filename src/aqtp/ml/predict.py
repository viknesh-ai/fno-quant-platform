"""PredictionEngine — serving layer for the ML models (REQ 14/17).

REQ 49 is explicit that the production trading loop must run without any Anthropic
API access. Nothing in this module (or anywhere in the execution path) calls an LLM;
predictions come from locally-loaded scikit-learn/LightGBM artifacts.

REQ 54's distinction is honoured here: **probability is not confidence**. The engine
returns both — the calibrated probability, and a separate confidence that accounts
for model disagreement, feature freshness and distance from the training regime. A
0.75 probability from models that disagree wildly is not the same trade as a 0.75
from models that agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

from ..configuration.schema import PredictionConfig
from ..core.errors import ModelError
from ..core.logging import get_logger
from ..core.types import Direction, Regime
from ..features.engine import FEATURE_VERSION
from .drift import DriftDetector
from .models import CalibratedModel, ModelEnsemble

logger = get_logger(__name__)


@dataclass
class Prediction:
    """One probabilistic forecast, with uncertainty attached (REQ 17)."""

    direction: Direction
    probability: float           # calibrated P(target before stop) for `direction`
    confidence: float            # separate from probability (REQ 54)
    uncertainty: float           # model disagreement
    expected_move_atr: float
    horizon_minutes: int
    model_id: str = ""
    feature_version: str = FEATURE_VERSION
    timestamp: datetime | None = None
    per_model: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    usable: bool = True

    @property
    def edge(self) -> float:
        """Distance from a coin flip. This, not the raw probability, is the signal."""
        return self.probability - 0.5

    def meets(self, minimum_probability: float) -> bool:
        return self.usable and self.probability >= minimum_probability

    def describe(self) -> str:
        return (
            f"{self.direction.value} p={self.probability:.3f} conf={self.confidence:.3f} "
            f"±{self.uncertainty:.3f} over {self.horizon_minutes}m"
        )


def unusable_prediction(reason: str, *, timestamp: datetime | None = None) -> Prediction:
    return Prediction(
        direction=Direction.FLAT,
        probability=0.5,
        confidence=0.0,
        uncertainty=1.0,
        expected_move_atr=0.0,
        horizon_minutes=0,
        timestamp=timestamp,
        reasons=[reason],
        usable=False,
    )


class PredictionEngine:
    """Serves directional predictions from registered models.

    Long and short are served by *separate* models trained on separate labels
    (`tb_long_*` / `tb_short_*`), because "target before stop for a long" and the
    same for a short are different questions — not complements. Treating one as
    1 minus the other is a modelling error that shows up as systematically
    overconfident shorts.
    """

    def __init__(
        self,
        config: PredictionConfig,
        *,
        drift_detector: DriftDetector | None = None,
    ) -> None:
        self.config = config
        self.drift = drift_detector or DriftDetector()
        self._long: ModelEnsemble | CalibratedModel | None = None
        self._short: ModelEnsemble | CalibratedModel | None = None
        self._volatility_model: CalibratedModel | None = None
        self._model_ids: dict[str, str] = {}
        self._horizon_minutes: int = config.horizons_minutes[0] if config.horizons_minutes else 30
        self._degraded_reason: str = ""

    # ------------------------------------------------------------------ #
    def load(
        self,
        *,
        long_model: ModelEnsemble | CalibratedModel | None = None,
        short_model: ModelEnsemble | CalibratedModel | None = None,
        horizon_minutes: int | None = None,
        model_ids: Mapping[str, str] | None = None,
    ) -> None:
        self._long = long_model
        self._short = short_model
        if horizon_minutes is not None:
            self._horizon_minutes = horizon_minutes
        if model_ids:
            self._model_ids.update(model_ids)
        logger.info(
            "prediction engine loaded: long=%s short=%s horizon=%dm",
            bool(long_model), bool(short_model), self._horizon_minutes,
        )

    @property
    def is_ready(self) -> bool:
        return self._long is not None or self._short is not None

    def degrade(self, reason: str) -> None:
        """Mark the engine as not-to-be-trusted (used by drift response)."""
        self._degraded_reason = reason
        logger.warning("prediction engine degraded: %s", reason)

    def restore(self) -> None:
        self._degraded_reason = ""

    # ------------------------------------------------------------------ #
    def predict(
        self,
        features: pd.Series,
        *,
        direction: Direction | None = None,
        regime: Regime | None = None,
        regime_confidence: float = 1.0,
        timestamp: datetime | None = None,
        feature_age_seconds: float = 0.0,
    ) -> Prediction:
        """Predict for a given direction, or pick the better of long/short."""
        if self._degraded_reason:
            return unusable_prediction(
                f"prediction engine degraded: {self._degraded_reason}", timestamp=timestamp
            )
        if not self.is_ready:
            return unusable_prediction("no model loaded", timestamp=timestamp)
        if feature_age_seconds > self.config.max_prediction_age_seconds:
            return unusable_prediction(
                f"features are {feature_age_seconds:.0f}s old, beyond the "
                f"{self.config.max_prediction_age_seconds:.0f}s limit",
                timestamp=timestamp,
            )

        frame = pd.DataFrame([features])
        if frame.isna().any().any():
            missing = frame.columns[frame.isna().any()].tolist()
            return unusable_prediction(
                f"{len(missing)} features are NaN (e.g. {missing[:3]}); refusing to impute at "
                "serving time",
                timestamp=timestamp,
            )

        candidates: list[Prediction] = []
        for candidate_direction, model in (
            (Direction.LONG, self._long),
            (Direction.SHORT, self._short),
        ):
            if model is None:
                continue
            if direction is not None and candidate_direction is not direction:
                continue
            prediction = self._predict_one(
                model, frame, candidate_direction, regime_confidence, timestamp
            )
            if prediction is not None:
                candidates.append(prediction)

        if not candidates:
            return unusable_prediction("no model available for the requested direction", timestamp=timestamp)

        best = max(candidates, key=lambda p: p.probability)

        # If both sides claim an edge, neither is trustworthy — they are modelling
        # mutually exclusive outcomes.
        if len(candidates) == 2:
            other = min(candidates, key=lambda p: p.probability)
            if other.probability > 0.5 and best.probability > 0.5:
                margin = best.probability - other.probability
                if margin < 0.08:
                    return unusable_prediction(
                        f"long and short models both indicate an edge "
                        f"({candidates[0].probability:.2f} / {candidates[1].probability:.2f}); "
                        "the signal is not directional",
                        timestamp=timestamp,
                    )
                best.reasons.append(
                    f"opposing model reads {other.probability:.2f}; margin {margin:.2f}"
                )

        self.drift.observe(features, best.probability)
        return best

    def _predict_one(
        self,
        model: ModelEnsemble | CalibratedModel,
        frame: pd.DataFrame,
        direction: Direction,
        regime_confidence: float,
        timestamp: datetime | None,
    ) -> Prediction | None:
        try:
            if isinstance(model, ModelEnsemble):
                probabilities, uncertainties = model.predict_with_uncertainty(frame)
                probability = float(probabilities[0])
                uncertainty = float(uncertainties[0])
                per_model = {
                    name: float(member.predict_proba(frame)[0])
                    for name, member in model.models.items()
                    if model.weights.get(name, 0.0) > 0
                }
            else:
                probability = float(model.predict_proba(frame)[0])
                # A single model reports no disagreement, but that is not the same
                # as certainty — a floor keeps downstream sizing honest.
                uncertainty = 0.10
                per_model = {model.family: probability}
        except ModelError as exc:
            logger.error("prediction failed for %s: %s", direction.value, exc)
            return None

        confidence = self._confidence(probability, uncertainty, regime_confidence)
        return Prediction(
            direction=direction,
            probability=probability,
            confidence=confidence,
            uncertainty=uncertainty,
            expected_move_atr=self.config.target_atr_multiple,
            horizon_minutes=self._horizon_minutes,
            model_id=self._model_ids.get(direction.value, ""),
            timestamp=timestamp,
            per_model=per_model,
            reasons=[
                f"calibrated probability {probability:.3f} that target precedes stop "
                f"within {self._horizon_minutes} minutes"
            ],
        )

    @staticmethod
    def _confidence(probability: float, uncertainty: float, regime_confidence: float) -> float:
        """Confidence is not probability (REQ 54).

        It combines how far the probability is from a coin flip, how much the
        models agree, and how sure we are about the regime the prediction was made
        in. A strong probability produced amid model disagreement and an unclear
        regime is downgraded — which is the whole point of tracking them separately.
        """
        edge_component = float(np.clip(abs(probability - 0.5) * 2.0, 0.0, 1.0))
        agreement_component = float(np.clip(1.0 - uncertainty * 3.0, 0.0, 1.0))
        regime_component = float(np.clip(regime_confidence, 0.0, 1.0))
        return float(
            np.clip(
                0.45 * edge_component + 0.35 * agreement_component + 0.20 * regime_component,
                0.0,
                1.0,
            )
        )

    # ------------------------------------------------------------------ #
    def check_drift(self) -> None:
        """Run the drift check and degrade the engine if the model has decayed."""
        report = self.drift.check()
        if report.should_pause:
            self.degrade(report.describe())
        elif report.should_reduce:
            logger.warning("model drift significant; exposure should be reduced: %s", report.describe())
        elif self._degraded_reason and report.severity == "none":
            self.restore()

    def health(self) -> dict[str, object]:
        return {
            "ready": self.is_ready,
            "degraded": bool(self._degraded_reason),
            "degraded_reason": self._degraded_reason,
            "horizon_minutes": self._horizon_minutes,
            "model_ids": dict(self._model_ids),
            "feature_version": FEATURE_VERSION,
        }
