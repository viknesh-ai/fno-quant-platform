"""Position management and the dynamic exit engine (REQ 34/35).

REQ 35 forbids a single fixed exit rule. Every managed position is re-evaluated on
each cycle against twelve independent exit conditions, and the first one that fires
supplies the recorded reason (REQ 35's final clause).

The evaluation order is deliberate: hard stops and emergencies are checked before
discretionary reasons, so a position is never held through its stop because a
softer rule happened to be evaluated first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import numpy as np

from ..configuration.schema import AppConfig
from ..core.clock import minutes_until_close, to_ist
from ..core.logging import get_logger
from ..core.types import (
    Direction,
    ExitReason,
    Greeks,
    Instrument,
    ManagementAction,
    Quote,
    Regime,
)
from ..ml.predict import Prediction
from ..regime.engine import RegimeState

logger = get_logger(__name__)


@dataclass
class ManagedPosition:
    """A position the platform is actively managing, with its original thesis."""

    position_id: str
    decision_id: str
    instrument: Instrument
    underlying: str
    strategy: str
    direction: Direction
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    initial_stop_price: float
    entry_regime: Regime
    entry_ml_probability: float = float("nan")
    expected_holding_minutes: int = 60
    underlying_entry_price: float = 0.0

    # live state
    last_price: float = 0.0
    unrealized_pnl: float = 0.0
    max_favourable_price: float = 0.0
    max_adverse_price: float = 0.0
    partial_exits: int = 0
    stop_moved_to_breakeven: bool = False
    trailing_active: bool = False
    entry_costs: float = 0.0

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.initial_stop_price)

    @property
    def r_multiple(self) -> float:
        risk = self.risk_per_unit
        if risk <= 0:
            return 0.0
        move = (self.last_price - self.entry_price) * self.direction.sign
        return move / risk

    @property
    def mae_r(self) -> float:
        risk = self.risk_per_unit
        if risk <= 0 or self.max_adverse_price <= 0:
            return 0.0
        return abs(self.entry_price - self.max_adverse_price) / risk

    @property
    def mfe_r(self) -> float:
        risk = self.risk_per_unit
        if risk <= 0 or self.max_favourable_price <= 0:
            return 0.0
        return abs(self.max_favourable_price - self.entry_price) / risk

    def holding_minutes(self, now: datetime) -> float:
        return (now - self.entry_time).total_seconds() / 60

    def update(self, price: float) -> None:
        self.last_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity * self.direction.sign
        if self.direction is Direction.LONG:
            self.max_favourable_price = max(self.max_favourable_price or price, price)
            self.max_adverse_price = min(self.max_adverse_price or price, price)
        else:
            self.max_favourable_price = min(self.max_favourable_price or price, price)
            self.max_adverse_price = max(self.max_adverse_price or price, price)


@dataclass
class ManagementDecision:
    """What to do with a position right now."""

    action: ManagementAction
    reason: ExitReason | None = None
    detail: str = ""
    new_stop: float | None = None
    exit_quantity: int = 0
    urgency: str = "normal"          # normal | immediate

    @property
    def is_exit(self) -> bool:
        return self.action in (
            ManagementAction.EXIT, ManagementAction.EMERGENCY_EXIT, ManagementAction.PARTIAL_EXIT
        )

    def describe(self) -> str:
        parts = [self.action.value]
        if self.reason:
            parts.append(self.reason.value)
        if self.detail:
            parts.append(self.detail)
        return " — ".join(parts)


class PositionManager:
    """Evaluates open positions and produces management decisions."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._positions: dict[str, ManagedPosition] = {}

    # ------------------------------------------------------------------ #
    def track(self, position: ManagedPosition) -> None:
        self._positions[position.position_id] = position
        logger.info(
            "tracking position %s: %s %s x%d @ %.2f (stop %.2f, target %.2f)",
            position.position_id, position.direction.value,
            position.instrument.trading_symbol, position.quantity,
            position.entry_price, position.stop_price, position.target_price,
        )

    def untrack(self, position_id: str) -> ManagedPosition | None:
        return self._positions.pop(position_id, None)

    def get(self, position_id: str) -> ManagedPosition | None:
        return self._positions.get(position_id)

    def all(self) -> list[ManagedPosition]:
        return list(self._positions.values())

    def for_underlying(self, underlying: str) -> list[ManagedPosition]:
        return [p for p in self._positions.values() if p.underlying == underlying]

    @property
    def count(self) -> int:
        return len(self._positions)

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        position: ManagedPosition,
        *,
        quote: Quote,
        now: datetime,
        regime: RegimeState | None = None,
        prediction: Prediction | None = None,
        greeks: Greeks | None = None,
        strategy_still_valid: bool = True,
        emergency: bool = False,
        risk_limit_breached: bool = False,
        portfolio_risk_breached: bool = False,
    ) -> ManagementDecision:
        """Assess one position against every exit condition, hardest first."""
        price = quote.last_price
        position.update(price)
        is_long = position.direction is Direction.LONG

        # --- 1. emergency shutdown (REQ 44) -------------------------------
        if emergency:
            return ManagementDecision(
                ManagementAction.EMERGENCY_EXIT,
                ExitReason.EMERGENCY_SHUTDOWN,
                "emergency shutdown is active",
                urgency="immediate",
            )

        # --- 2. stop ------------------------------------------------------
        stop_hit = price <= position.stop_price if is_long else price >= position.stop_price
        if stop_hit:
            return ManagementDecision(
                ManagementAction.EXIT,
                ExitReason.STOP_REACHED,
                f"price {price:.2f} reached the stop at {position.stop_price:.2f} "
                f"({position.r_multiple:+.2f}R)",
                urgency="immediate",
            )

        # --- 3. target ----------------------------------------------------
        target_hit = price >= position.target_price if is_long else price <= position.target_price
        if target_hit:
            return ManagementDecision(
                ManagementAction.EXIT,
                ExitReason.TARGET_REACHED,
                f"price {price:.2f} reached the target at {position.target_price:.2f} "
                f"({position.r_multiple:+.2f}R)",
            )

        # --- 4. risk limits ------------------------------------------------
        if risk_limit_breached:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.RISK_LIMIT,
                "a portfolio risk limit was breached", urgency="immediate",
            )
        if portfolio_risk_breached:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.PORTFOLIO_RISK,
                "aggregate portfolio risk exceeded its limit", urgency="immediate",
            )

        # --- 5. session end / square-off ------------------------------------
        moment = to_ist(now)
        square_off = self.config.session.square_off_time
        if moment.time() >= square_off:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.END_OF_SESSION,
                f"square-off time {square_off:%H:%M} reached", urgency="immediate",
            )

        # --- 6. expiry ------------------------------------------------------
        expiry = position.instrument.expiry_date
        if expiry is not None:
            if moment.date() > expiry:
                return ManagementDecision(
                    ManagementAction.EXIT, ExitReason.END_OF_SESSION,
                    "contract has expired", urgency="immediate",
                )
            if moment.date() == expiry and minutes_until_close(moment) <= 30:
                return ManagementDecision(
                    ManagementAction.EXIT, ExitReason.END_OF_SESSION,
                    "expiry day with under 30 minutes remaining; holding risks assignment "
                    "or a worthless expiry",
                    urgency="immediate",
                )

        # --- 7. liquidity deterioration ----------------------------------------
        spread_pct = quote.spread_pct
        if spread_pct is not None and spread_pct > self.config.risk.max_spread_pct * 2:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.LIQUIDITY_DETERIORATED,
                f"spread widened to {spread_pct:.2%}, twice the {self.config.risk.max_spread_pct:.2%} "
                "limit — exiting before it widens further",
                urgency="immediate",
            )

        # --- 8. regime change ---------------------------------------------------
        if regime is not None and self._regime_invalidated(position, regime):
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.REGIME_CHANGED,
                f"regime moved from {position.entry_regime.value} to {regime.dominant.value}, "
                "invalidating the entry thesis",
            )

        # --- 9. prediction invalidated --------------------------------------------
        if prediction is not None and prediction.usable:
            if prediction.direction is not position.direction and prediction.probability > 0.60:
                return ManagementDecision(
                    ManagementAction.EXIT, ExitReason.PREDICTION_INVALIDATED,
                    f"model now favours {prediction.direction.value} at "
                    f"{prediction.probability:.1%}, against the open position",
                )

        # --- 10. strategy invalidated ------------------------------------------------
        if not strategy_still_valid:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.STRATEGY_INVALIDATED,
                f"{position.strategy} no longer holds its entry thesis",
            )

        # --- 11. volatility change -------------------------------------------------
        if greeks is not None and position.instrument.is_option:
            decision = self._greeks_check(position, greeks, now)
            if decision is not None:
                return decision

        # --- 12. time limit ---------------------------------------------------------
        held = position.holding_minutes(now)
        if position.expected_holding_minutes and held >= position.expected_holding_minutes * 1.5:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.TIME_LIMIT,
                f"held {held:.0f} minutes, 1.5x the {position.expected_holding_minutes} minute "
                f"expectation, at {position.r_multiple:+.2f}R",
            )

        # --- otherwise: manage the stop ----------------------------------------------
        return self._manage_stop(position, price, now)

    # ------------------------------------------------------------------ #
    def _regime_invalidated(self, position: ManagedPosition, regime: RegimeState) -> bool:
        """A long entered in TRENDING_UP is invalidated by TRENDING_DOWN, not by
        every regime change — an uncertain regime is a reason to stop *entering*,
        not necessarily to abandon a working position."""
        opposing = {
            Regime.TRENDING_UP: Regime.TRENDING_DOWN,
            Regime.TRENDING_DOWN: Regime.TRENDING_UP,
        }
        entry = position.entry_regime
        if entry in opposing and regime.dominant is opposing[entry]:
            return regime.confidence >= 0.6
        if position.direction is Direction.LONG and regime.dominant is Regime.TRENDING_DOWN:
            return regime.confidence >= 0.7
        if position.direction is Direction.SHORT and regime.dominant is Regime.TRENDING_UP:
            return regime.confidence >= 0.7
        return False

    def _greeks_check(
        self, position: ManagedPosition, greeks: Greeks, now: datetime
    ) -> ManagementDecision | None:
        """Option-specific exit conditions (REQ 30/34)."""
        premium = position.last_price
        if premium <= 0:
            return None

        # Theta burning through the remaining premium faster than the position can
        # plausibly recover it.
        daily_decay_ratio = abs(greeks.theta) / premium
        if daily_decay_ratio > 0.25 and position.r_multiple < 0.5:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.VOLATILITY_CHANGED,
                f"theta is consuming {daily_decay_ratio:.0%} of premium per day while the "
                f"position sits at {position.r_multiple:+.2f}R",
            )

        # Delta collapse: the option has stopped tracking the underlying, so the
        # directional thesis can no longer be expressed through it.
        if abs(greeks.delta) < 0.10:
            return ManagementDecision(
                ManagementAction.EXIT, ExitReason.STRATEGY_INVALIDATED,
                f"delta collapsed to {greeks.delta:.3f}; the contract no longer tracks the "
                "underlying",
            )
        return None

    def _manage_stop(
        self, position: ManagedPosition, price: float, now: datetime
    ) -> ManagementDecision:
        """Move to breakeven, then trail. Stops only ever move in our favour."""
        r = position.r_multiple
        is_long = position.direction is Direction.LONG
        risk = position.risk_per_unit

        # Breakeven at +1R: removes the loss case while leaving the upside open.
        if not position.stop_moved_to_breakeven and r >= 1.0:
            breakeven = position.entry_price
            improves = breakeven > position.stop_price if is_long else breakeven < position.stop_price
            if improves:
                position.stop_moved_to_breakeven = True
                return ManagementDecision(
                    ManagementAction.MOVE_STOP,
                    detail=f"reached {r:+.2f}R; moving the stop to breakeven {breakeven:.2f}",
                    new_stop=breakeven,
                )

        # Trail at +1.5R, keeping 1R of give-back.
        if r >= 1.5:
            trail_distance = risk * 1.0
            candidate = (
                position.max_favourable_price - trail_distance
                if is_long
                else position.max_favourable_price + trail_distance
            )
            improves = candidate > position.stop_price if is_long else candidate < position.stop_price
            if improves:
                position.trailing_active = True
                return ManagementDecision(
                    ManagementAction.TRAIL_STOP,
                    detail=f"at {r:+.2f}R; trailing the stop to {candidate:.2f}",
                    new_stop=candidate,
                )

        # Scale out half at +2R once, locking in a profit while leaving a runner.
        if r >= 2.0 and position.partial_exits == 0 and position.quantity >= position.instrument.lot_size * 2:
            lots = position.quantity // position.instrument.lot_size
            exit_quantity = (lots // 2) * position.instrument.lot_size
            if exit_quantity > 0:
                return ManagementDecision(
                    ManagementAction.PARTIAL_EXIT,
                    detail=f"at {r:+.2f}R; taking partial profit on {exit_quantity} units",
                    exit_quantity=exit_quantity,
                )

        return ManagementDecision(
            ManagementAction.HOLD,
            detail=f"holding at {r:+.2f}R after {position.holding_minutes(now):.0f} minutes",
        )

    # ------------------------------------------------------------------ #
    def apply(self, position: ManagedPosition, decision: ManagementDecision) -> None:
        """Apply a non-exit decision to the tracked position."""
        if decision.new_stop is not None:
            old = position.stop_price
            position.stop_price = decision.new_stop
            logger.info(
                "position %s stop moved %.2f -> %.2f (%s)",
                position.position_id, old, decision.new_stop, decision.detail,
            )
        if decision.action is ManagementAction.PARTIAL_EXIT and decision.exit_quantity > 0:
            position.partial_exits += 1
            position.quantity -= decision.exit_quantity

    def total_unrealized(self) -> float:
        return sum(p.unrealized_pnl for p in self._positions.values())

    def summary(self) -> list[dict]:
        return [
            {
                "position_id": p.position_id,
                "instrument": p.instrument.trading_symbol,
                "strategy": p.strategy,
                "direction": p.direction.value,
                "quantity": p.quantity,
                "entry": round(p.entry_price, 2),
                "last": round(p.last_price, 2),
                "stop": round(p.stop_price, 2),
                "target": round(p.target_price, 2),
                "unrealized": round(p.unrealized_pnl, 2),
                "r": round(p.r_multiple, 2),
            }
            for p in self._positions.values()
        ]
