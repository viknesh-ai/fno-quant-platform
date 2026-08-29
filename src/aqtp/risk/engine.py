"""RiskEngine — the final authority before execution (REQ 25).

Every proposed trade passes through `RiskEngine.evaluate()`. There is no other path
to the ExecutionEngine, and the ExecutionEngine refuses any order not carrying a
`RiskApproval` issued here (REQ 67.22).

Design points:

  * **Every control produces a named result**, pass or fail. The full checklist is
    attached to the decision so the journal can record which control rejected a
    trade and by how much — not merely that "risk said no".
  * **All checks run** even after the first failure. Knowing that a trade breached
    three limits rather than one is what tells an operator their configuration is
    wrong, rather than the market being unusual.
  * **Limits are checked against the projected portfolio**, i.e. the state that
    would exist after the fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from ..configuration.schema import AppConfig
from ..core.logging import get_logger
from ..core.types import (
    Decision,
    Direction,
    DrawdownLevel,
    Greeks,
    Instrument,
    MarginInfo,
    Quote,
    TransactionType,
)
from ..events.risk import EventRiskEngine, EventWindow
from ..signals.expected_value import ExpectedValueResult
from ..strategies.health import StrategyHealthMonitor
from .costs import TransactionCostModel
from .drawdown import DrawdownProtection
from .killswitch import KillScope, KillSwitchManager
from .portfolio import PortfolioManager, PortfolioState
from .sizing import PositionSizer, SizingResult

logger = get_logger(__name__)


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class RiskCheck:
    name: str
    status: CheckStatus
    detail: str = ""
    value: float | None = None
    limit: float | None = None

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAIL

    def describe(self) -> str:
        if self.value is not None and self.limit is not None:
            return f"[{self.status.value}] {self.name}: {self.detail} ({self.value:.4g} vs {self.limit:.4g})"
        return f"[{self.status.value}] {self.name}: {self.detail}"


@dataclass
class RiskApproval:
    """The token the ExecutionEngine requires. Cannot be constructed elsewhere."""

    approved: bool
    decision: Decision
    checks: list[RiskCheck] = field(default_factory=list)
    sizing: SizingResult | None = None
    risk_multiplier: float = 1.0
    rejection_reasons: list[str] = field(default_factory=list)
    portfolio_before: PortfolioState | None = None
    portfolio_after: PortfolioState | None = None
    expected_value: ExpectedValueResult | None = None
    event_window: EventWindow | None = None
    approved_at: datetime | None = None
    approval_id: str = ""

    @property
    def quantity(self) -> int:
        return self.sizing.quantity if self.sizing else 0

    def failed_checks(self) -> list[RiskCheck]:
        return [c for c in self.checks if c.failed]

    def explain(self) -> str:
        if self.approved:
            return (
                f"APPROVED {self.decision.value} qty={self.quantity} "
                f"({len(self.checks)} checks passed, risk x{self.risk_multiplier:.2f})"
            )
        return "NO_TRADE: " + "; ".join(self.rejection_reasons)

    def checklist(self) -> list[str]:
        return [c.describe() for c in self.checks]


def _rejected(reasons: list[str], checks: list[RiskCheck], **kwargs) -> RiskApproval:
    return RiskApproval(
        approved=False,
        decision=Decision.NO_TRADE,
        checks=checks,
        rejection_reasons=reasons,
        **kwargs,
    )


class RiskEngine:
    def __init__(
        self,
        config: AppConfig,
        *,
        portfolio: PortfolioManager,
        drawdown: DrawdownProtection,
        kill_switches: KillSwitchManager,
        sizer: PositionSizer,
        cost_model: TransactionCostModel,
        event_risk: EventRiskEngine,
        health: StrategyHealthMonitor | None = None,
    ) -> None:
        self.config = config
        self.risk = config.risk
        self.portfolio = portfolio
        self.drawdown = drawdown
        self.kill_switches = kill_switches
        self.sizer = sizer
        self.costs = cost_model
        self.event_risk = event_risk
        self.health = health
        self._approval_counter = 0

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        *,
        decision: Decision,
        instrument: Instrument,
        underlying: str,
        strategy: str,
        entry_price: float,
        stop_price: float,
        target_price: float,
        portfolio_state: PortfolioState,
        margin: MarginInfo | None,
        quote: Quote | None,
        expected_value: ExpectedValueResult | None,
        greeks: Greeks | None = None,
        underlying_price: float | None = None,
        now: datetime,
        margin_per_lot: float | None = None,
    ) -> RiskApproval:
        """Run every risk control. Returns an approval or a fully-reasoned rejection."""
        checks: list[RiskCheck] = []
        failures: list[str] = []

        def record(
            name: str, ok: bool, detail: str, *, value: float | None = None, limit: float | None = None
        ) -> None:
            checks.append(
                RiskCheck(
                    name=name,
                    status=CheckStatus.PASS if ok else CheckStatus.FAIL,
                    detail=detail,
                    value=value,
                    limit=limit,
                )
            )
            if not ok:
                failures.append(f"{name}: {detail}")

        def skip(name: str, detail: str) -> None:
            checks.append(RiskCheck(name=name, status=CheckStatus.SKIP, detail=detail))

        if decision is Decision.NO_TRADE:
            return _rejected(["upstream decision was NO_TRADE"], checks)

        # --- 1. kill switches ------------------------------------------------
        allowed, reason = self.kill_switches.can_trade(
            strategy=strategy, instrument=instrument.trading_symbol, underlying=underlying
        )
        record("kill_switch", allowed, reason or "no kill switch engaged")

        # --- 2. drawdown ladder / daily loss / streak / cooldown --------------
        can_trade, drawdown_reason = self.drawdown.can_trade(now)
        record(
            "drawdown_protection",
            can_trade,
            drawdown_reason or f"level {self.drawdown.state.level.name}",
            value=self.drawdown.state.drawdown_pct,
            limit=self.risk.max_portfolio_drawdown_pct,
        )
        record(
            "daily_loss_limit",
            -self.drawdown.state.daily_loss_pct < self.risk.max_daily_loss_pct,
            f"daily P&L {self.drawdown.state.daily_loss_pct:+.2%}",
            value=-self.drawdown.state.daily_loss_pct,
            limit=self.risk.max_daily_loss_pct,
        )
        record(
            "consecutive_losses",
            self.drawdown.state.consecutive_losses < self.risk.max_consecutive_losses,
            f"{self.drawdown.state.consecutive_losses} consecutive losses",
            value=self.drawdown.state.consecutive_losses,
            limit=self.risk.max_consecutive_losses,
        )
        record(
            "trades_per_day",
            self.drawdown.state.trades_today < self.risk.max_trades_per_day,
            f"{self.drawdown.state.trades_today} trades today",
            value=self.drawdown.state.trades_today,
            limit=self.risk.max_trades_per_day,
        )

        # --- 3. strategy health ----------------------------------------------
        health_multiplier = 1.0
        if self.health is not None:
            healthy, health_reason = self.health.can_trade(strategy, now=now)
            record("strategy_health", healthy, health_reason or f"{strategy} is healthy")
            health_multiplier = self.health.risk_multiplier(strategy)
        else:
            skip("strategy_health", "no health monitor attached")

        # --- 4. event risk ---------------------------------------------------
        window = self.event_risk.assess(now, underlying=underlying)
        record(
            "event_risk",
            not window.blocks_new_positions,
            window.reason or "no active event window",
        )

        # --- 5. open position count -------------------------------------------
        record(
            "max_open_positions",
            portfolio_state.open_positions < self.risk.max_open_positions,
            f"{portfolio_state.open_positions} positions open",
            value=portfolio_state.open_positions,
            limit=self.risk.max_open_positions,
        )

        # --- 6. liquidity and execution quality --------------------------------
        if quote is not None:
            spread_pct = quote.spread_pct
            if spread_pct is None:
                record("max_spread", False, "no bid/ask available to measure spread")
            else:
                record(
                    "max_spread",
                    spread_pct <= self.risk.max_spread_pct,
                    f"spread {spread_pct:.3%}",
                    value=spread_pct,
                    limit=self.risk.max_spread_pct,
                )
            volume = quote.volume or 0.0
            record(
                "min_liquidity",
                volume >= self.risk.min_liquidity_volume,
                f"volume {volume:,.0f}",
                value=volume,
                limit=self.risk.min_liquidity_volume,
            )
            if instrument.is_derivative:
                open_interest = quote.open_interest or 0.0
                record(
                    "min_open_interest",
                    open_interest >= self.risk.min_open_interest,
                    f"open interest {open_interest:,.0f}",
                    value=open_interest,
                    limit=self.risk.min_open_interest,
                )
            estimated_slippage = self.costs.slippage_pct(quote, entry_price)
            record(
                "max_slippage",
                estimated_slippage <= self.risk.max_slippage_pct,
                f"estimated slippage {estimated_slippage:.3%}",
                value=estimated_slippage,
                limit=self.risk.max_slippage_pct,
            )
        else:
            record("max_spread", False, "no quote available for the contract")

        # --- 7. expected value --------------------------------------------------
        if expected_value is not None:
            record(
                "expected_value",
                expected_value.passes,
                expected_value.reason or expected_value.explain(),
                value=expected_value.expected_value_r,
                limit=self.config.expected_value.min_expected_value_r_multiple,
            )
            record(
                "risk_reward",
                expected_value.risk_reward >= self.config.expected_value.min_risk_reward,
                f"risk/reward {expected_value.risk_reward:.2f}",
                value=expected_value.risk_reward,
                limit=self.config.expected_value.min_risk_reward,
            )
        else:
            record("expected_value", False, "expected value was not computed")

        # --- 8. position sizing ---------------------------------------------------
        risk_multiplier = self._risk_multiplier(health_multiplier, window)
        equity = portfolio_state.equity
        available_margin = margin.available_margin if margin else equity

        sizing = self.sizer.size(
            instrument=instrument,
            entry_price=entry_price,
            stop_price=stop_price,
            equity=equity,
            available_margin=available_margin,
            margin_per_lot=margin_per_lot,
            risk_multiplier=risk_multiplier,
            current_exposure=portfolio_state.total_exposure,
            quote=quote,
        )
        record(
            "position_sizing",
            sizing.ok,
            sizing.rejection_reason or sizing.explain(),
            value=sizing.quantity,
        )

        if not sizing.ok:
            return _rejected(
                failures, checks, sizing=sizing, portfolio_before=portfolio_state,
                event_window=window, expected_value=expected_value,
            )

        # --- 9. risk per trade (verify what sizing actually produced) --------------
        actual_risk_pct = sizing.risk_amount / equity if equity > 0 else 1.0
        record(
            "max_risk_per_trade",
            actual_risk_pct <= self.risk.risk_per_trade_pct * 1.001,  # rounding tolerance
            f"trade risks {actual_risk_pct:.3%} of equity",
            value=actual_risk_pct,
            limit=self.risk.risk_per_trade_pct,
        )

        # --- 10. margin utilization -------------------------------------------------
        if margin is not None:
            projected_used = margin.net_margin_used + sizing.margin_required
            total_margin = margin.available_margin + margin.net_margin_used
            utilization = projected_used / total_margin if total_margin > 0 else 1.0
            record(
                "margin_utilization",
                utilization <= self.risk.max_margin_utilization_pct,
                f"margin utilization would be {utilization:.1%}",
                value=utilization,
                limit=self.risk.max_margin_utilization_pct,
            )
            record(
                "margin_available",
                sizing.margin_required <= margin.available_margin,
                f"requires ₹{sizing.margin_required:,.0f} of ₹{margin.available_margin:,.0f}",
                value=sizing.margin_required,
                limit=margin.available_margin,
            )
        else:
            skip("margin_utilization", "broker margin unavailable")

        # --- 11. portfolio concentration (projected) ---------------------------------
        direction = Direction.LONG if decision is Decision.BUY else Direction.SHORT
        signed_quantity = sizing.quantity * direction.sign
        projected = self.portfolio.projected_state(
            portfolio_state,
            instrument=instrument,
            quantity=signed_quantity,
            price=entry_price,
            greeks=greeks,
            underlying_price=underlying_price,
        )

        instrument_exposure = (
            projected.exposure_by_instrument.get(instrument.trading_symbol, 0.0) / equity
            if equity > 0 else 1.0
        )
        record(
            "max_exposure_per_instrument",
            instrument_exposure <= self.risk.max_exposure_per_instrument_pct,
            f"exposure to {instrument.trading_symbol} would be {instrument_exposure:.1%}",
            value=instrument_exposure,
            limit=self.risk.max_exposure_per_instrument_pct,
        )

        underlying_exposure = projected.underlying_exposure_pct(underlying)
        record(
            "max_exposure_per_underlying",
            underlying_exposure <= self.risk.max_exposure_per_underlying_pct,
            f"exposure to {underlying} would be {underlying_exposure:.1%}",
            value=underlying_exposure,
            limit=self.risk.max_exposure_per_underlying_pct,
        )

        correlated = self.portfolio.correlated_exposure(projected, underlying)
        correlated_pct = correlated / equity if equity > 0 else 1.0
        overlapping = self.portfolio.overlapping_positions(portfolio_state, underlying)
        record(
            "max_correlated_exposure",
            correlated_pct <= self.risk.max_correlated_exposure_pct,
            f"correlated exposure would be {correlated_pct:.1%}"
            + (f" (overlaps {', '.join(overlapping[:3])})" if overlapping else ""),
            value=correlated_pct,
            limit=self.risk.max_correlated_exposure_pct,
        )

        group = self.portfolio.group_for(underlying)
        sector_exposure = projected.group_exposure_pct(group)
        record(
            "max_sector_exposure",
            sector_exposure <= self.risk.max_sector_exposure_pct,
            f"{group} sector exposure would be {sector_exposure:.1%}",
            value=sector_exposure,
            limit=self.risk.max_sector_exposure_pct,
        )

        position_value_pct = (entry_price * sizing.quantity) / equity if equity > 0 else 1.0
        record(
            "max_position_size",
            position_value_pct <= self.risk.max_position_size_pct,
            f"position would be {position_value_pct:.1%} of equity",
            value=position_value_pct,
            limit=self.risk.max_position_size_pct,
        )

        # --- 12. aggregate option greeks (REQ 29/30) ------------------------------
        for name, projected_value, limit in (
            ("max_portfolio_delta", abs(projected.net_delta_per_lakh), self.risk.max_portfolio_delta_per_lakh),
            ("max_portfolio_gamma", abs(projected.net_gamma_per_lakh), self.risk.max_portfolio_gamma_per_lakh),
            ("max_portfolio_vega", abs(projected.net_vega_per_lakh), self.risk.max_portfolio_vega_per_lakh),
            ("max_portfolio_theta", abs(projected.net_theta_per_lakh), self.risk.max_portfolio_theta_per_lakh),
        ):
            record(
                name,
                projected_value <= limit,
                f"{name.replace('max_portfolio_', 'net ')} would be {projected_value:.2f} per lakh",
                value=projected_value,
                limit=limit,
            )

        # --- verdict --------------------------------------------------------------
        if failures:
            logger.info(
                "risk rejected %s %s: %d control(s) failed — %s",
                decision.value, instrument.trading_symbol, len(failures), failures[0],
            )
            return _rejected(
                failures, checks, sizing=sizing, risk_multiplier=risk_multiplier,
                portfolio_before=portfolio_state, portfolio_after=projected,
                expected_value=expected_value, event_window=window,
            )

        self._approval_counter += 1
        approval = RiskApproval(
            approved=True,
            decision=decision,
            checks=checks,
            sizing=sizing,
            risk_multiplier=risk_multiplier,
            portfolio_before=portfolio_state,
            portfolio_after=projected,
            expected_value=expected_value,
            event_window=window,
            approved_at=now,
            approval_id=f"RA{self._approval_counter:06d}",
        )
        logger.info(
            "risk approved %s %s qty=%d (%d checks)",
            decision.value, instrument.trading_symbol, sizing.quantity, len(checks),
        )
        return approval

    # ------------------------------------------------------------------ #
    def _risk_multiplier(self, health_multiplier: float, window: EventWindow) -> float:
        """Combine every throttle. Multiplicative, so throttles compound rather than
        one overriding another — two independent reasons to be cautious should make
        the system more cautious, not equally cautious."""
        multiplier = self.drawdown.state.risk_multiplier
        if self.drawdown.state.level is DrawdownLevel.REDUCED:
            multiplier = min(multiplier, self.risk.level2_risk_multiplier)
        multiplier *= max(0.0, health_multiplier)
        multiplier *= self.event_risk.risk_multiplier(window)
        return float(np.clip(multiplier, 0.0, 1.0))

    def on_trade_closed(self, *, won: bool, now: datetime) -> None:
        self.drawdown.record_trade(won=won, now=now)

    def on_equity_update(self, equity: float, *, now: datetime) -> None:
        state = self.drawdown.update_equity(equity, now=now)
        if state.is_shutdown and not self.kill_switches.is_shutdown:
            self.kill_switches.emergency_shutdown(
                f"drawdown protection reached level 4: {state.reason}", engaged_by="risk_engine"
            )

    def status(self) -> dict:
        return {
            "drawdown_level": self.drawdown.state.level.name,
            "drawdown_pct": round(self.drawdown.state.drawdown_pct, 4),
            "daily_pnl": round(self.drawdown.state.daily_pnl, 2),
            "consecutive_losses": self.drawdown.state.consecutive_losses,
            "trades_today": self.drawdown.state.trades_today,
            "trades_remaining": self.drawdown.trades_remaining(),
            "kill_switches": self.kill_switches.summary(),
            "risk_multiplier": self.drawdown.state.risk_multiplier,
        }
