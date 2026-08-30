"""Strategy framework (REQ 12/13).

Hard architectural rule (REQ 67.23): **a strategy may never place an order.** A
strategy receives a read-only context and returns a `StrategySignal` describing what
it believes. Everything downstream — contract selection, sizing, risk, execution —
happens in other layers. The `StrategyContext` is deliberately given no broker
handle, so this rule is enforced by construction rather than by discipline.

Every signal must carry reasoning (REQ 48) and a proposed invalidation level, so
that "why did we enter" and "why did we exit" are always answerable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Direction, Instrument, Regime, Timeframe
from ..features.options_features import OptionChainFeatures
from ..regime.engine import RegimeState, regime_fit

if TYPE_CHECKING:
    from ..analysis.engine import AnalysisReport

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategyContext:
    """Read-only view handed to a strategy. No broker, no order API, by design."""

    underlying: str
    instrument: Instrument
    timestamp: datetime
    last_price: float
    features: pd.Series                       # entry-timeframe features
    features_by_timeframe: Mapping[Timeframe, pd.Series]
    frames: Mapping[Timeframe, pd.DataFrame]  # closed bars only
    regime: RegimeState
    entry_timeframe: Timeframe
    setup_timeframe: Timeframe
    regime_timeframe: Timeframe
    atr: float
    option_features: OptionChainFeatures | None = None
    analysis: "AnalysisReport | None" = None
    days_to_expiry: int | None = None
    session_context: Mapping[str, float] = field(default_factory=dict)

    def feature(self, name: str, default: float = float("nan")) -> float:
        """Fetch an entry-timeframe feature by base name."""
        return _get(self.features, name, default, self.entry_timeframe)

    def feature_at(self, timeframe: Timeframe, name: str, default: float = float("nan")) -> float:
        series = self.features_by_timeframe.get(timeframe)
        if series is None:
            return default
        return _get(series, name, default, timeframe)

    def frame(self, timeframe: Timeframe) -> pd.DataFrame:
        return self.frames.get(timeframe, pd.DataFrame())


def _get(series: pd.Series, name: str, default: float, timeframe: Timeframe) -> float:
    """Resolve a feature whether or not it carries a timeframe prefix."""
    for key in (name, f"{timeframe.value}_{name}"):
        if key in series.index:
            value = series[key]
            try:
                out = float(value)
            except (TypeError, ValueError):
                return default
            return default if np.isnan(out) else out
    return default


@dataclass
class StrategySignal:
    """What a strategy believes. Not an order — an opinion with evidence."""

    strategy: str
    underlying: str
    direction: Direction
    confidence: float                 # [0, 1], the strategy's own conviction
    expected_move_atr: float          # expected favourable move, in ATR units
    proposed_entry: float
    proposed_stop: float              # invalidation level on the underlying
    proposed_target: float
    expected_holding_minutes: int
    regime_fit: float
    reasoning: list[str] = field(default_factory=list)
    evidence: dict[str, float] = field(default_factory=dict)
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        self.regime_fit = float(np.clip(self.regime_fit, 0.0, 1.0))

    @property
    def is_actionable(self) -> bool:
        return self.direction is not Direction.FLAT and self.confidence > 0.0

    @property
    def risk_distance(self) -> float:
        return abs(self.proposed_entry - self.proposed_stop)

    @property
    def reward_distance(self) -> float:
        return abs(self.proposed_target - self.proposed_entry)

    @property
    def risk_reward(self) -> float:
        return self.reward_distance / self.risk_distance if self.risk_distance > 0 else 0.0

    def explain(self) -> str:
        return f"{self.strategy}: {self.direction.value} @{self.confidence:.2f} — " + "; ".join(
            self.reasoning
        )


def no_signal(strategy: str, underlying: str, reason: str, *, timestamp: datetime | None = None) -> StrategySignal:
    """An explicit 'nothing here' — recorded, not silently dropped (REQ 36)."""
    return StrategySignal(
        strategy=strategy,
        underlying=underlying,
        direction=Direction.FLAT,
        confidence=0.0,
        expected_move_atr=0.0,
        proposed_entry=0.0,
        proposed_stop=0.0,
        proposed_target=0.0,
        expected_holding_minutes=0,
        regime_fit=0.0,
        reasoning=[reason],
        timestamp=timestamp,
    )


class Strategy(ABC):
    """Base class for all strategies."""

    name: str = "abstract"
    # Timeframes this strategy needs; the orchestrator ensures they are computed.
    required_timeframes: tuple[str, ...] = ()
    # Minimum closed bars before the strategy will produce anything.
    min_bars: int = 60

    def __init__(self, params: Mapping[str, Any] | None = None, weight: float = 1.0) -> None:
        self.params = dict(self.default_params())
        if params:
            unknown = set(params) - set(self.params)
            if unknown:
                # Silently ignoring a misspelled parameter would mean the operator
                # thinks they tuned something they didn't.
                raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
            self.params.update(params)
        self.weight = weight

    @staticmethod
    def default_params() -> dict[str, Any]:
        return {}

    @abstractmethod
    def generate(self, context: StrategyContext) -> StrategySignal:
        """Return a signal (possibly FLAT). Must never raise on bad data — return
        a FLAT signal with the reason instead."""

    # -- helpers available to subclasses ---------------------------------- #
    def fit(self, regime: RegimeState) -> float:
        return regime_fit(self.name, regime)

    def _levels(
        self,
        context: StrategyContext,
        direction: Direction,
        *,
        stop_atr: float,
        target_atr: float,
    ) -> tuple[float, float, float]:
        """Entry/stop/target derived from ATR, so levels scale with volatility
        rather than being fixed rupee amounts (REQ 67.7)."""
        entry = context.last_price
        atr = max(context.atr, entry * 0.0005)  # floor: never a zero-width stop
        if direction is Direction.LONG:
            return entry, entry - stop_atr * atr, entry + target_atr * atr
        return entry, entry + stop_atr * atr, entry - target_atr * atr

    def _ready(self, context: StrategyContext) -> str | None:
        """Common precondition check. Returns a reason string if not ready."""
        frame = context.frame(context.entry_timeframe)
        if len(frame) < self.min_bars:
            return f"insufficient history ({len(frame)} of {self.min_bars} bars)"
        if not np.isfinite(context.atr) or context.atr <= 0:
            return "ATR unavailable"
        if context.regime.is_uncertain:
            return "regime uncertain"
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, weight={self.weight})"


def combine_confidence(*components: tuple[float, float]) -> float:
    """Weighted mean of (value, weight) pairs, clipped to [0, 1].

    Used instead of multiplying confidences, because multiplication makes any single
    weak component veto the whole signal — which produces a strategy that almost
    never fires rather than one that expresses degrees of belief.
    """
    total_weight = sum(w for _, w in components)
    if total_weight <= 0:
        return 0.0
    return float(np.clip(sum(v * w for v, w in components) / total_weight, 0.0, 1.0))
