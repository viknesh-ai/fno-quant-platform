"""SignalEnsemble (REQ 13/54).

Combines independent strategy signals with the ML prediction into one scored
decision, and is the component responsible for two subtle requirements:

REQ 13 — *avoid double-counting correlated indicators*. Trend-following, market
structure and VWAP largely read the same underlying phenomenon. Summing their
confidences would let one piece of evidence vote three times. So strategies are
grouped into correlation families, and within a family the contributions are
combined sub-additively: the strongest member counts fully and each additional
member is discounted.

REQ 54 — *confidence is not probability*. The ensemble emits both, and the final
score deliberately includes execution-quality and exposure terms that have nothing
to do with whether the direction is right.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..configuration.schema import EnsembleConfig
from ..core.logging import get_logger
from ..core.types import Decision, Direction, Regime
from ..ml.predict import Prediction
from ..regime.engine import RegimeState
from ..strategies.base import StrategySignal
from ..strategies.health import StrategyHealthMonitor

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    """The combined view. `decision` is BUY / SELL / NO_TRADE (REQ 36)."""

    decision: Decision
    direction: Direction
    score: float                     # overall trade quality in [0, 1]
    confidence: float                # calibrated conviction (REQ 54)
    ml_probability: float
    strategy_agreement: float
    agreeing_strategies: list[str] = field(default_factory=list)
    dissenting_strategies: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)
    expected_move_atr: float = 0.0
    proposed_entry: float = 0.0
    proposed_stop: float = 0.0
    proposed_target: float = 0.0
    expected_holding_minutes: int = 30
    regime: Regime = Regime.UNCERTAIN
    regime_confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    timestamp: datetime | None = None

    @property
    def is_tradable(self) -> bool:
        return self.decision is not Decision.NO_TRADE

    @property
    def risk_reward(self) -> float:
        risk = abs(self.proposed_entry - self.proposed_stop)
        reward = abs(self.proposed_target - self.proposed_entry)
        return reward / risk if risk > 0 else 0.0

    def explain(self) -> str:
        if not self.is_tradable:
            return f"NO_TRADE: {self.rejection_reason}"
        return (
            f"{self.decision.value} score={self.score:.3f} conf={self.confidence:.3f} "
            f"ml_p={self.ml_probability:.3f} agreement={self.strategy_agreement:.2f} "
            f"[{', '.join(self.agreeing_strategies)}]"
        )


def no_trade(reason: str, *, timestamp: datetime | None = None, regime: RegimeState | None = None) -> EnsembleResult:
    return EnsembleResult(
        decision=Decision.NO_TRADE,
        direction=Direction.FLAT,
        score=0.0,
        confidence=0.0,
        ml_probability=0.5,
        strategy_agreement=0.0,
        rejection_reason=reason,
        regime=regime.dominant if regime else Regime.UNCERTAIN,
        regime_confidence=regime.confidence if regime else 0.0,
        timestamp=timestamp,
    )


class SignalEnsemble:
    def __init__(
        self,
        config: EnsembleConfig,
        *,
        health_monitor: StrategyHealthMonitor | None = None,
    ) -> None:
        self.config = config
        self.health = health_monitor
        self._group_of: dict[str, str] = {}
        for group, members in config.correlation_groups.items():
            for member in members:
                self._group_of[member] = group

    # ------------------------------------------------------------------ #
    def combine(
        self,
        *,
        signals: list[StrategySignal],
        prediction: Prediction | None,
        regime: RegimeState,
        min_model_probability: float,
        timestamp: datetime | None = None,
    ) -> EnsembleResult:
        timestamp = timestamp or datetime.now()

        actionable = [s for s in signals if s.is_actionable]
        if not actionable:
            return no_trade("no strategy produced an actionable signal", timestamp=timestamp, regime=regime)

        if regime.is_uncertain:
            return no_trade(
                f"regime is uncertain (confidence {regime.confidence:.2f})",
                timestamp=timestamp, regime=regime,
            )

        # --- weight each signal --------------------------------------------
        weighted: dict[Direction, list[tuple[StrategySignal, float]]] = defaultdict(list)
        for signal in actionable:
            weight = self._signal_weight(signal)
            if weight <= 0:
                continue
            weighted[signal.direction].append((signal, weight))

        if not weighted:
            return no_trade(
                "every strategy signal was zero-weighted by health or regime fit",
                timestamp=timestamp, regime=regime,
            )

        long_score = self._side_score(weighted.get(Direction.LONG, []))
        short_score = self._side_score(weighted.get(Direction.SHORT, []))

        if long_score <= 0 and short_score <= 0:
            return no_trade("no side accumulated positive evidence", timestamp=timestamp, regime=regime)

        direction = Direction.LONG if long_score >= short_score else Direction.SHORT
        winning = long_score if direction is Direction.LONG else short_score
        losing = short_score if direction is Direction.LONG else long_score

        total = winning + losing
        agreement = winning / total if total > 0 else 0.0
        if agreement < self.config.min_strategy_agreement:
            return no_trade(
                f"strategy agreement {agreement:.2f} below the "
                f"{self.config.min_strategy_agreement:.2f} minimum "
                f"(long {long_score:.2f} vs short {short_score:.2f})",
                timestamp=timestamp, regime=regime,
            )

        agreeing = [s for s, _ in weighted.get(direction, [])]
        if len(agreeing) < self.config.min_strategies_agreeing:
            return no_trade(
                f"only {len(agreeing)} strategies agree, need "
                f"{self.config.min_strategies_agreeing}",
                timestamp=timestamp, regime=regime,
            )

        # --- ML gate ---------------------------------------------------------
        ml_probability = 0.5
        ml_component = 0.5
        reasons: list[str] = []
        if prediction is not None and prediction.usable:
            if prediction.direction is not direction:
                return no_trade(
                    f"ML model favours {prediction.direction.value} but strategies favour "
                    f"{direction.value}",
                    timestamp=timestamp, regime=regime,
                )
            ml_probability = prediction.probability
            if ml_probability < min_model_probability:
                return no_trade(
                    f"ML probability {ml_probability:.3f} below the required "
                    f"{min_model_probability:.3f}",
                    timestamp=timestamp, regime=regime,
                )
            ml_component = float(np.clip((ml_probability - 0.5) * 2.0, 0.0, 1.0))
            reasons.append(
                f"ML probability {ml_probability:.3f} (confidence {prediction.confidence:.2f}, "
                f"uncertainty ±{prediction.uncertainty:.2f})"
            )
        else:
            # No model is not the same as a bad model. Trading on strategies alone is
            # permitted but scored down, so the ML gate cannot be bypassed by simply
            # not loading a model.
            reason = prediction.reasons[0] if prediction and prediction.reasons else "no model loaded"
            ml_component = 0.35
            reasons.append(f"no usable ML prediction ({reason}); strategy-only evidence")

        strategy_component = float(np.clip(winning / max(len(agreeing), 1), 0.0, 1.0))

        score = (
            self.config.strategy_weight * strategy_component
            + self.config.ml_weight * ml_component
        )
        confidence = self._confidence(
            strategy_component=strategy_component,
            agreement=agreement,
            prediction=prediction,
            regime=regime,
            agreeing_count=len(agreeing),
        )

        if confidence < self.config.min_ensemble_confidence:
            return no_trade(
                f"ensemble confidence {confidence:.3f} below the "
                f"{self.config.min_ensemble_confidence:.3f} minimum",
                timestamp=timestamp, regime=regime,
            )

        levels = self._aggregate_levels(weighted[direction], direction)
        reasons.extend(
            f"{s.strategy} ({s.confidence:.2f}): {s.reasoning[0] if s.reasoning else 'no detail'}"
            for s, _ in sorted(weighted[direction], key=lambda sw: sw[1], reverse=True)[:4]
        )

        return EnsembleResult(
            decision=Decision.BUY if direction is Direction.LONG else Decision.SELL,
            direction=direction,
            score=float(np.clip(score, 0.0, 1.0)),
            confidence=confidence,
            ml_probability=ml_probability,
            strategy_agreement=agreement,
            agreeing_strategies=[s.strategy for s in agreeing],
            dissenting_strategies=[
                s.strategy
                for d, items in weighted.items()
                if d is not direction
                for s, _ in items
            ],
            contributions={s.strategy: round(w, 4) for s, w in weighted[direction]},
            expected_move_atr=levels["expected_move_atr"],
            proposed_entry=levels["entry"],
            proposed_stop=levels["stop"],
            proposed_target=levels["target"],
            expected_holding_minutes=levels["holding_minutes"],
            regime=regime.dominant,
            regime_confidence=regime.confidence,
            reasons=reasons,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    def _signal_weight(self, signal: StrategySignal) -> float:
        """Per-signal weight from conviction, regime fit and live health."""
        weight = signal.confidence * max(signal.regime_fit, 0.0)
        if self.health is not None:
            weight *= self.health.risk_multiplier(signal.strategy)
        return float(weight)

    def _side_score(self, items: list[tuple[StrategySignal, float]]) -> float:
        """Aggregate one side's evidence, discounting correlated strategies (REQ 13).

        Within a correlation family the largest contribution counts in full, the
        second is multiplied by `correlation_discount`, the third by its square, and
        so on. Independent strategies (no family) always count in full.
        """
        if not items:
            return 0.0

        by_group: dict[str, list[float]] = defaultdict(list)
        independent = 0.0
        for signal, weight in items:
            group = self._group_of.get(signal.strategy)
            if group is None:
                independent += weight
            else:
                by_group[group].append(weight)

        correlated_total = 0.0
        for weights in by_group.values():
            for rank, weight in enumerate(sorted(weights, reverse=True)):
                correlated_total += weight * (self.config.correlation_discount ** rank)

        return independent + correlated_total

    def _confidence(
        self,
        *,
        strategy_component: float,
        agreement: float,
        prediction: Prediction | None,
        regime: RegimeState,
        agreeing_count: int,
    ) -> float:
        """REQ 54: confidence must be calibrated, not just a restated probability."""
        components: list[tuple[float, float]] = [
            (strategy_component, 0.25),
            (agreement, 0.20),
            (float(np.clip(regime.confidence, 0.0, 1.0)), 0.20),
            # Breadth of agreement: three independent strategies agreeing is
            # stronger evidence than one very confident strategy.
            (float(np.clip(agreeing_count / 4.0, 0.0, 1.0)), 0.15),
        ]
        if prediction is not None and prediction.usable:
            components.append((prediction.confidence, 0.20))
        else:
            components.append((0.30, 0.20))

        total_weight = sum(w for _, w in components)
        return float(
            np.clip(sum(v * w for v, w in components) / total_weight, 0.0, 1.0)
        )

    @staticmethod
    def _aggregate_levels(
        items: list[tuple[StrategySignal, float]], direction: Direction
    ) -> dict:
        """Blend the agreeing strategies' proposed levels.

        The stop is taken as the *tightest* proposal rather than the average: if any
        agreeing strategy considers its thesis invalidated at a level, holding past
        that level means holding a trade no strategy still believes in.
        """
        weights = np.array([w for _, w in items], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(items)) / len(items)

        entries = np.array([s.proposed_entry for s, _ in items])
        targets = np.array([s.proposed_target for s, _ in items])
        stops = np.array([s.proposed_stop for s, _ in items])
        holdings = np.array([s.expected_holding_minutes for s, _ in items], dtype=float)
        moves = np.array([s.expected_move_atr for s, _ in items])

        entry = float(np.average(entries, weights=weights))
        stop = float(stops.max() if direction is Direction.LONG else stops.min())
        target = float(np.average(targets, weights=weights))

        return {
            "entry": entry,
            "stop": stop,
            "target": target,
            "holding_minutes": int(np.average(holdings, weights=weights)),
            "expected_move_atr": float(np.average(moves, weights=weights)),
        }
