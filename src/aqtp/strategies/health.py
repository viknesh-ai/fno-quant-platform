"""Strategy health monitoring and loss-streak protection (REQ 27/28/53).

A strategy's live allocation is governed by its *measured* recent expectancy, not by
opinion. Health transitions are one-directional per evaluation and hysteretic: a
strategy that recovers must clear a higher bar than the one that demoted it, so a
single lucky trade cannot flip it straight back to ACTIVE.

REQ 53 warning respected: performance is tracked and used to *throttle*, but
parameters are never re-optimized from small recent samples.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Iterable, Mapping

import numpy as np

from ..configuration.schema import StrategyHealthConfig
from ..core.logging import get_logger
from ..core.types import Regime, StrategyState

logger = get_logger(__name__)


@dataclass(frozen=True)
class TradeOutcome:
    """One closed trade, reduced to what health measurement needs."""

    strategy: str
    underlying: str
    timestamp: datetime
    r_multiple: float          # P&L in units of the risk taken
    pnl: float
    regime: Regime
    won: bool
    holding_minutes: float = 0.0


@dataclass
class StrategyHealth:
    strategy: str
    state: StrategyState = StrategyState.ACTIVE
    trades: int = 0
    wins: int = 0
    losses: int = 0
    expectancy_r: float = 0.0
    win_rate: float = 0.0
    average_r: float = 0.0
    max_drawdown_r: float = 0.0
    consecutive_losses: int = 0
    total_pnl: float = 0.0
    paused_until: datetime | None = None
    last_evaluated: datetime | None = None
    by_regime: dict[str, float] = field(default_factory=dict)
    by_underlying: dict[str, float] = field(default_factory=dict)
    by_hour: dict[int, float] = field(default_factory=dict)
    note: str = ""

    @property
    def risk_multiplier(self) -> float:
        """What fraction of normal risk this strategy is allowed right now."""
        return {
            StrategyState.ACTIVE: 1.0,
            StrategyState.REDUCED: 0.5,
            StrategyState.PAUSED: 0.0,
            StrategyState.DISABLED: 0.0,
        }[self.state]

    @property
    def can_trade(self) -> bool:
        return self.state in (StrategyState.ACTIVE, StrategyState.REDUCED)


class StrategyHealthMonitor:
    def __init__(self, config: StrategyHealthConfig) -> None:
        self.config = config
        self._outcomes: dict[str, Deque[TradeOutcome]] = defaultdict(
            lambda: deque(maxlen=max(config.lookback_trades * 3, 50))
        )
        self._health: dict[str, StrategyHealth] = {}
        self._manual_overrides: dict[str, StrategyState] = {}

    # ------------------------------------------------------------------ #
    def record(self, outcome: TradeOutcome) -> StrategyHealth:
        self._outcomes[outcome.strategy].append(outcome)
        return self.evaluate(outcome.strategy, now=outcome.timestamp)

    def health(self, strategy: str) -> StrategyHealth:
        return self._health.setdefault(strategy, StrategyHealth(strategy=strategy))

    def all_health(self) -> dict[str, StrategyHealth]:
        return dict(self._health)

    def risk_multiplier(self, strategy: str) -> float:
        return self.health(strategy).risk_multiplier

    def can_trade(self, strategy: str, *, now: datetime | None = None) -> tuple[bool, str]:
        health = self.health(strategy)
        if health.paused_until and now and now < health.paused_until:
            return False, f"{strategy} paused until {health.paused_until:%H:%M}"
        if health.paused_until and now and now >= health.paused_until:
            # Pause expired: return to REDUCED, not straight to ACTIVE. The strategy
            # must earn its way back with real trades.
            health.paused_until = None
            health.state = StrategyState.REDUCED
            health.note = "pause expired; resumed at reduced size"
            logger.info("strategy %s resumed at reduced size after pause", strategy)
        if not health.can_trade:
            return False, f"{strategy} is {health.state.value}"
        return True, ""

    # ------------------------------------------------------------------ #
    def evaluate(self, strategy: str, *, now: datetime | None = None) -> StrategyHealth:
        """Recompute health metrics and apply state transitions."""
        outcomes = list(self._outcomes[strategy])[-self.config.lookback_trades :]
        health = self.health(strategy)
        health.last_evaluated = now

        if not outcomes:
            return health

        r_values = np.array([o.r_multiple for o in outcomes], dtype=float)
        health.trades = len(outcomes)
        health.wins = int(sum(1 for o in outcomes if o.won))
        health.losses = health.trades - health.wins
        health.win_rate = health.wins / health.trades
        health.average_r = float(r_values.mean())
        health.expectancy_r = float(r_values.mean())
        health.total_pnl = float(sum(o.pnl for o in outcomes))

        equity = np.cumsum(r_values)
        peak = np.maximum.accumulate(equity)
        health.max_drawdown_r = float((peak - equity).max()) if len(equity) else 0.0

        streak = 0
        for outcome in reversed(outcomes):
            if outcome.won:
                break
            streak += 1
        health.consecutive_losses = streak

        by_regime: dict[str, list[float]] = defaultdict(list)
        by_underlying: dict[str, list[float]] = defaultdict(list)
        by_hour: dict[int, list[float]] = defaultdict(list)
        for outcome in outcomes:
            by_regime[outcome.regime.value].append(outcome.r_multiple)
            by_underlying[outcome.underlying].append(outcome.r_multiple)
            by_hour[outcome.timestamp.hour].append(outcome.r_multiple)
        health.by_regime = {k: float(np.mean(v)) for k, v in by_regime.items()}
        health.by_underlying = {k: float(np.mean(v)) for k, v in by_underlying.items()}
        health.by_hour = {k: float(np.mean(v)) for k, v in by_hour.items()}

        self._transition(health, now)
        return health

    def _transition(self, health: StrategyHealth, now: datetime | None) -> None:
        """Apply state transitions with hysteresis."""
        if health.strategy in self._manual_overrides:
            health.state = self._manual_overrides[health.strategy]
            health.note = "manual override"
            return
        if health.state is StrategyState.DISABLED:
            return  # only a manual reset re-enables a disabled strategy

        config = self.config
        if health.trades < config.min_trades_for_judgement:
            health.note = (
                f"{health.trades}/{config.min_trades_for_judgement} trades — too few to judge"
            )
            return

        previous = health.state
        expectancy = health.expectancy_r

        if expectancy <= config.disable_below_expectancy_r:
            health.state = StrategyState.DISABLED
            health.note = (
                f"expectancy {expectancy:+.2f}R at or below the disable threshold "
                f"{config.disable_below_expectancy_r:+.2f}R"
            )
        elif expectancy <= config.pause_below_expectancy_r:
            health.state = StrategyState.PAUSED
            if now is not None:
                health.paused_until = now + timedelta(minutes=config.pause_duration_minutes)
            health.note = (
                f"expectancy {expectancy:+.2f}R at or below the pause threshold "
                f"{config.pause_below_expectancy_r:+.2f}R"
            )
        elif expectancy <= config.reduce_below_expectancy_r:
            health.state = StrategyState.REDUCED
            health.note = f"expectancy {expectancy:+.2f}R below the reduce threshold"
        else:
            # Hysteresis: recovering to ACTIVE requires clearing the reduce
            # threshold by a margin, not merely touching it.
            recovery_bar = abs(config.reduce_below_expectancy_r) * 0.5
            if previous is StrategyState.ACTIVE or expectancy >= recovery_bar:
                health.state = StrategyState.ACTIVE
                health.paused_until = None
                health.note = f"expectancy {expectancy:+.2f}R"
            else:
                health.note = (
                    f"expectancy {expectancy:+.2f}R positive but below the "
                    f"{recovery_bar:+.2f}R recovery bar; staying {previous.value}"
                )

        if health.state is not previous:
            logger.warning(
                "strategy %s: %s -> %s (%s)",
                health.strategy, previous.value, health.state.value, health.note,
            )

    # ------------------------------------------------------------------ #
    def regime_performance(self, strategy: str, regime: Regime) -> float | None:
        """Average R for a strategy in a given regime, if there is enough data."""
        outcomes = [o for o in self._outcomes[strategy] if o.regime is regime]
        if len(outcomes) < 5:
            return None
        return float(np.mean([o.r_multiple for o in outcomes]))

    def allocation_weights(self, strategies: Iterable[str]) -> dict[str, float]:
        """Opportunity allocation across strategies (REQ 53).

        Weights are shrunk toward equal allocation, deliberately. Allocating in
        proportion to a 30-trade sample is overfitting to noise; the shrinkage makes
        the allocation responsive without being credulous.
        """
        names = list(strategies)
        if not names:
            return {}
        equal = 1.0 / len(names)
        raw: dict[str, float] = {}
        for name in names:
            health = self.health(name)
            if health.trades < self.config.min_trades_for_judgement:
                raw[name] = equal
            else:
                # Map expectancy through a bounded transform so no strategy can
                # dominate on the strength of a short hot streak.
                score = float(np.clip(0.5 + health.expectancy_r, 0.1, 2.0))
                raw[name] = equal * score
        total = sum(raw.values())
        normalized = {k: v / total for k, v in raw.items()}
        shrinkage = 0.5
        return {k: shrinkage * equal + (1 - shrinkage) * v for k, v in normalized.items()}

    def override(self, strategy: str, state: StrategyState | None) -> None:
        """Manual operator control (used by the CLI kill switches)."""
        if state is None:
            self._manual_overrides.pop(strategy, None)
            health = self.health(strategy)
            health.state = StrategyState.ACTIVE
            health.note = "manual override cleared"
        else:
            self._manual_overrides[strategy] = state
            self.health(strategy).state = state
            logger.warning("strategy %s manually set to %s", strategy, state.value)

    def reset(self, strategy: str | None = None) -> None:
        if strategy is None:
            self._outcomes.clear()
            self._health.clear()
        else:
            self._outcomes.pop(strategy, None)
            self._health.pop(strategy, None)

    def summary(self) -> list[dict[str, object]]:
        return [
            {
                "strategy": h.strategy,
                "state": h.state.value,
                "trades": h.trades,
                "win_rate": round(h.win_rate, 3),
                "expectancy_r": round(h.expectancy_r, 3),
                "max_dd_r": round(h.max_drawdown_r, 2),
                "consecutive_losses": h.consecutive_losses,
                "note": h.note,
            }
            for h in sorted(self._health.values(), key=lambda x: x.strategy)
        ]
