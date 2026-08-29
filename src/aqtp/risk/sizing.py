"""Position sizing (REQ 24).

Size is derived, never chosen. The chain is:

    risk budget (equity x risk%)  ->  risk per unit (stop distance)
    ->  raw quantity  ->  rounded DOWN to lot size
    ->  capped by margin, exposure, freeze limit and premium outlay

Rounding is always *down*. Rounding up to "get closer to the risk budget" silently
exceeds the configured risk, which is the exact failure the budget exists to
prevent. If one lot already exceeds the budget, the correct answer is zero lots —
and the sizer says so rather than trading a smaller-than-legal quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..configuration.schema import CapitalConfig, RiskConfig
from ..core.logging import get_logger
from ..core.types import Instrument, MarginInfo, Quote

logger = get_logger(__name__)


@dataclass
class SizingResult:
    quantity: int
    lots: int
    lot_size: int
    risk_amount: float               # rupees genuinely at risk if the stop fills
    risk_per_unit: float
    capital_required: float
    margin_required: float
    capped_by: str = ""              # which constraint bound the size
    reasons: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected and self.quantity > 0

    def explain(self) -> str:
        if self.rejected:
            return f"size 0: {self.rejection_reason}"
        return (
            f"{self.lots} lot(s) = {self.quantity} units, risking ₹{self.risk_amount:,.0f} "
            f"(₹{self.risk_per_unit:.2f}/unit), capital ₹{self.capital_required:,.0f}"
            + (f", bound by {self.capped_by}" if self.capped_by else "")
        )


class PositionSizer:
    def __init__(self, capital_config: CapitalConfig, risk_config: RiskConfig) -> None:
        self.capital = capital_config
        self.risk = risk_config

    def size(
        self,
        *,
        instrument: Instrument,
        entry_price: float,
        stop_price: float,
        equity: float,
        available_margin: float,
        margin_per_lot: float | None = None,
        risk_multiplier: float = 1.0,
        current_exposure: float = 0.0,
        quote: Quote | None = None,
    ) -> SizingResult:
        """Calculate the position size for one candidate trade.

        `risk_multiplier` is how the drawdown ladder and strategy health throttle
        size without changing the configured risk percentage.
        """
        lot_size = max(1, instrument.lot_size)  # from the instrument master (REQ 67.12)
        risk_per_unit = abs(entry_price - stop_price)

        if risk_per_unit <= 0:
            return self._reject("stop distance is zero — risk per unit is undefined", lot_size)
        if entry_price <= 0:
            return self._reject("entry price is not positive", lot_size)
        if equity <= 0:
            return self._reject("account equity is not positive", lot_size)

        risk_budget = equity * self.risk.risk_per_trade_pct * max(0.0, risk_multiplier)
        if risk_budget <= 0:
            return self._reject(
                f"risk budget is zero (risk multiplier {risk_multiplier:.2f})", lot_size
            )

        reasons: list[str] = [
            f"risk budget ₹{risk_budget:,.0f} = equity ₹{equity:,.0f} x "
            f"{self.risk.risk_per_trade_pct:.2%}"
            + (f" x {risk_multiplier:.2f} throttle" if risk_multiplier != 1.0 else "")
        ]

        # --- raw size from the risk budget ---------------------------------
        raw_units = risk_budget / risk_per_unit
        raw_lots = int(np.floor(raw_units / lot_size))
        if raw_lots < 1:
            one_lot_risk = risk_per_unit * lot_size
            return self._reject(
                f"one lot ({lot_size} units) would risk ₹{one_lot_risk:,.0f}, exceeding the "
                f"₹{risk_budget:,.0f} budget; rounding up would breach configured risk",
                lot_size,
            )

        lots = raw_lots
        capped_by = "risk budget"

        # --- premium / notional outlay --------------------------------------
        cost_per_lot = entry_price * lot_size
        deployable = equity * self.capital.max_capital_deployment_pct
        max_position_value = equity * self.risk.max_position_size_pct
        remaining_deployable = max(0.0, deployable - current_exposure)

        for limit, label in (
            (max_position_value, f"max position size ({self.risk.max_position_size_pct:.0%} of equity)"),
            (remaining_deployable, f"remaining deployable capital"),
            (available_margin, "available margin"),
        ):
            allowed = int(np.floor(limit / cost_per_lot)) if cost_per_lot > 0 else 0
            if allowed < lots:
                lots, capped_by = allowed, label

        # --- margin requirement ---------------------------------------------
        if margin_per_lot is not None and margin_per_lot > 0:
            margin_ceiling = available_margin * self.risk.max_margin_utilization_pct
            allowed = int(np.floor(margin_ceiling / margin_per_lot))
            if allowed < lots:
                lots, capped_by = allowed, (
                    f"margin utilization cap ({self.risk.max_margin_utilization_pct:.0%})"
                )

        # --- exchange freeze quantity ----------------------------------------
        if instrument.freeze_quantity and instrument.freeze_quantity > 0:
            # Orders above the freeze limit are rejected or frozen by the exchange;
            # this is a hard venue constraint, not a preference.
            max_lots_by_freeze = int(np.floor(instrument.freeze_quantity / lot_size))
            if max_lots_by_freeze < lots:
                lots, capped_by = max_lots_by_freeze, (
                    f"exchange freeze quantity ({instrument.freeze_quantity})"
                )

        # --- liquidity ---------------------------------------------------------
        if quote is not None and quote.volume is not None and quote.volume > 0:
            # Do not attempt to take more than a small share of the day's volume;
            # beyond that the fill price is not the quoted price.
            max_units_by_volume = int(quote.volume * 0.02)
            allowed = int(np.floor(max_units_by_volume / lot_size))
            if allowed < lots:
                lots, capped_by = allowed, "traded volume participation limit (2%)"

        if lots < 1:
            return self._reject(
                f"size reduced to zero by {capped_by}; one lot costs ₹{cost_per_lot:,.0f} "
                f"against ₹{available_margin:,.0f} available",
                lot_size,
            )

        quantity = lots * lot_size
        actual_risk = risk_per_unit * quantity
        capital_required = entry_price * quantity
        margin_required = (margin_per_lot * lots) if margin_per_lot else capital_required

        if capped_by != "risk budget":
            reasons.append(f"size limited by {capped_by}")
        reasons.append(
            f"{lots} lot(s) x {lot_size} = {quantity} units risks ₹{actual_risk:,.0f} "
            f"({actual_risk / equity:.2%} of equity)"
        )

        return SizingResult(
            quantity=quantity,
            lots=lots,
            lot_size=lot_size,
            risk_amount=actual_risk,
            risk_per_unit=risk_per_unit,
            capital_required=capital_required,
            margin_required=margin_required,
            capped_by=capped_by if capped_by != "risk budget" else "",
            reasons=reasons,
        )

    @staticmethod
    def _reject(reason: str, lot_size: int) -> SizingResult:
        return SizingResult(
            quantity=0,
            lots=0,
            lot_size=lot_size,
            risk_amount=0.0,
            risk_per_unit=0.0,
            capital_required=0.0,
            margin_required=0.0,
            rejected=True,
            rejection_reason=reason,
        )


def effective_equity(
    config: CapitalConfig, margin: MarginInfo | None, *, fallback: float | None = None
) -> tuple[float, str]:
    """Reconcile configured capital against what the broker reports (REQ 23).

    The smaller of the two always wins. Configuring ₹10 lakh while the broker shows
    ₹2 lakh available must not size positions against ₹10 lakh.
    """
    configured = config.available_capital
    if not config.use_broker_equity or margin is None:
        return configured, "configured capital (broker equity not used)"

    broker_equity = margin.equity
    if broker_equity <= 0:
        return configured, "configured capital (broker reported no equity)"

    if broker_equity < configured * (1 - config.broker_equity_tolerance_pct):
        logger.warning(
            "broker equity %s is materially below configured %s; sizing against the broker figure",
            f"₹{broker_equity:,.0f}", f"₹{configured:,.0f}",
        )
        return broker_equity, (
            f"broker equity ₹{broker_equity:,.0f} (below configured ₹{configured:,.0f})"
        )
    return min(configured, broker_equity), (
        f"min(configured ₹{configured:,.0f}, broker ₹{broker_equity:,.0f})"
    )
