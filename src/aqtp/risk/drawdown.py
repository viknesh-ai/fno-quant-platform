"""Drawdown protection and loss-streak control (REQ 26/27).

Four levels, as REQ 26 requires. The important behavioural details:

  * Levels are evaluated against **peak equity**, so a drawdown is measured from the
    high-water mark rather than from the day's open.
  * The ladder is **ratcheting within a session**: once level 3 is reached, a small
    bounce does not immediately restore full risk. Recovery requires clearing back
    above the *next lower* threshold with a margin. Without this, equity oscillating
    around a boundary would flip trading on and off every cycle.
  * REQ 26 explicitly forbids increasing risk after winning streaks, so there is no
    upward adjustment anywhere in this module — only reduction and restoration to
    normal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Deque

import numpy as np

from ..configuration.schema import RiskConfig
from ..core.logging import get_logger
from ..core.types import DrawdownLevel

logger = get_logger(__name__)


@dataclass
class DrawdownState:
    level: DrawdownLevel = DrawdownLevel.NORMAL
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0
    day_start_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_loss_pct: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    cooldown_until: datetime | None = None
    last_loss_at: datetime | None = None
    session_date: date | None = None
    reason: str = ""

    @property
    def risk_multiplier(self) -> float:
        """How much of the configured per-trade risk is permitted right now."""
        return {
            DrawdownLevel.NORMAL: 1.0,
            DrawdownLevel.REDUCED: 0.5,
            DrawdownLevel.PAUSED: 0.0,
            DrawdownLevel.SHUTDOWN: 0.0,
        }[self.level]

    @property
    def can_open_new(self) -> bool:
        return self.level in (DrawdownLevel.NORMAL, DrawdownLevel.REDUCED)

    @property
    def is_shutdown(self) -> bool:
        return self.level is DrawdownLevel.SHUTDOWN

    def describe(self) -> str:
        return (
            f"L{self.level.value} {self.level.name}: drawdown {self.drawdown_pct:.2%} from peak "
            f"₹{self.peak_equity:,.0f}, daily P&L ₹{self.daily_pnl:+,.0f} "
            f"({self.daily_loss_pct:+.2%})"
        )


class DrawdownProtection:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self.state = DrawdownState()
        self._recent_outcomes: Deque[bool] = deque(maxlen=50)

    # ------------------------------------------------------------------ #
    def start_session(self, equity: float, *, session_date: date) -> None:
        """Reset intraday counters. Peak equity persists across sessions — the
        portfolio drawdown limit is not supposed to reset overnight."""
        self.state.day_start_equity = equity
        self.state.current_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.state.daily_pnl = 0.0
        self.state.daily_loss_pct = 0.0
        self.state.trades_today = 0
        self.state.session_date = session_date
        self.state.cooldown_until = None
        # A SHUTDOWN persists until an operator clears it; it must not evaporate
        # because the clock rolled over.
        if self.state.level is not DrawdownLevel.SHUTDOWN:
            self.state.level = DrawdownLevel.NORMAL
            self.state.reason = "new session"
        logger.info(
            "session started at equity %s (peak %s)",
            f"₹{equity:,.0f}", f"₹{self.state.peak_equity:,.0f}",
        )

    def update_equity(self, equity: float, *, now: datetime | None = None) -> DrawdownState:
        """Recompute the drawdown level from current equity."""
        state = self.state
        state.current_equity = equity
        state.peak_equity = max(state.peak_equity, equity)
        state.drawdown_pct = (
            (state.peak_equity - equity) / state.peak_equity if state.peak_equity > 0 else 0.0
        )
        if state.day_start_equity > 0:
            state.daily_pnl = equity - state.day_start_equity
            state.daily_loss_pct = state.daily_pnl / state.day_start_equity

        self._apply_ladder(now)
        return state

    def _apply_ladder(self, now: datetime | None) -> None:
        config = self.config
        state = self.state
        if state.level is DrawdownLevel.SHUTDOWN:
            return  # only an explicit reset leaves shutdown

        drawdown = state.drawdown_pct
        daily_loss = -state.daily_loss_pct  # positive when losing

        target = DrawdownLevel.NORMAL
        reason = "within normal operating range"

        if drawdown >= config.drawdown_level4_pct or drawdown >= config.max_portfolio_drawdown_pct:
            target = DrawdownLevel.SHUTDOWN
            reason = (
                f"drawdown {drawdown:.2%} reached the level-4 threshold "
                f"{config.drawdown_level4_pct:.2%}"
            )
        elif drawdown >= config.drawdown_level3_pct:
            target = DrawdownLevel.PAUSED
            reason = f"drawdown {drawdown:.2%} reached the level-3 threshold"
        elif drawdown >= config.drawdown_level2_pct:
            target = DrawdownLevel.REDUCED
            reason = f"drawdown {drawdown:.2%} reached the level-2 threshold"

        # The daily loss limit can independently force a pause.
        if daily_loss >= config.max_daily_loss_pct:
            if target.value < DrawdownLevel.PAUSED.value:
                target = DrawdownLevel.PAUSED
                reason = (
                    f"daily loss {daily_loss:.2%} reached the "
                    f"{config.max_daily_loss_pct:.2%} limit"
                )

        # --- ratchet with hysteresis ---------------------------------------
        if target.value > state.level.value:
            self._set_level(target, reason)
        elif target.value < state.level.value:
            # Require clearing back below the threshold that triggered the current
            # level, with a 20% buffer, before easing.
            thresholds = {
                DrawdownLevel.REDUCED: config.drawdown_level2_pct,
                DrawdownLevel.PAUSED: config.drawdown_level3_pct,
            }
            trigger = thresholds.get(state.level)
            if trigger is not None and drawdown < trigger * 0.8:
                self._set_level(
                    target, f"drawdown recovered to {drawdown:.2%}, below {trigger * 0.8:.2%}"
                )

    def _set_level(self, level: DrawdownLevel, reason: str) -> None:
        if level is self.state.level:
            return
        previous = self.state.level
        self.state.level = level
        self.state.reason = reason
        log = logger.error if level is DrawdownLevel.SHUTDOWN else logger.warning
        log("drawdown level %s -> %s: %s", previous.name, level.name, reason)

    # ------------------------------------------------------------------ #
    def record_trade(self, *, won: bool, now: datetime) -> None:
        """Update streak counters and start a cooldown after a loss (REQ 27)."""
        state = self.state
        state.trades_today += 1
        self._recent_outcomes.append(won)

        if won:
            state.consecutive_losses = 0
            return

        state.consecutive_losses += 1
        state.last_loss_at = now
        if self.config.cooldown_minutes_after_loss > 0:
            state.cooldown_until = now + timedelta(minutes=self.config.cooldown_minutes_after_loss)

        if state.consecutive_losses >= self.config.max_consecutive_losses:
            self._set_level(
                DrawdownLevel.PAUSED,
                f"{state.consecutive_losses} consecutive losses reached the "
                f"{self.config.max_consecutive_losses} limit",
            )

    def in_cooldown(self, now: datetime) -> tuple[bool, str]:
        until = self.state.cooldown_until
        if until is None:
            return False, ""
        if now < until:
            remaining = (until - now).total_seconds() / 60
            return True, f"cooling down for another {remaining:.0f} minutes after a loss"
        self.state.cooldown_until = None
        return False, ""

    def trades_remaining(self) -> int:
        return max(0, self.config.max_trades_per_day - self.state.trades_today)

    # ------------------------------------------------------------------ #
    def can_trade(self, now: datetime) -> tuple[bool, str]:
        """The single question the RiskEngine asks this module."""
        state = self.state
        if state.level is DrawdownLevel.SHUTDOWN:
            return False, f"EMERGENCY SHUTDOWN: {state.reason}"
        if state.level is DrawdownLevel.PAUSED:
            return False, f"trading paused: {state.reason}"

        cooling, cooldown_reason = self.in_cooldown(now)
        if cooling:
            return False, cooldown_reason

        if state.trades_today >= self.config.max_trades_per_day:
            return False, (
                f"daily trade limit reached ({state.trades_today}/"
                f"{self.config.max_trades_per_day})"
            )
        return True, ""

    def emergency_shutdown(self, reason: str) -> None:
        self._set_level(DrawdownLevel.SHUTDOWN, f"manual shutdown: {reason}")

    def reset_shutdown(self, *, operator_note: str = "") -> None:
        """Clear a shutdown. Deliberately explicit — never automatic."""
        if self.state.level is not DrawdownLevel.SHUTDOWN:
            return
        self.state.level = DrawdownLevel.NORMAL
        self.state.consecutive_losses = 0
        self.state.cooldown_until = None
        self.state.reason = f"shutdown cleared by operator: {operator_note}"
        logger.warning("emergency shutdown cleared: %s", operator_note)

    def recent_win_rate(self) -> float:
        if not self._recent_outcomes:
            return float("nan")
        return float(np.mean(self._recent_outcomes))
