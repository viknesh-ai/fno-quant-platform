"""Transaction cost model (REQ 56).

REQ 56 forbids assuming zero transaction costs. For Indian F&O the costs are not a
rounding error — on a ₹200 option premium, brokerage plus STT plus exchange charges
plus GST plus a half-spread of slippage can easily exceed 2% of notional round trip,
which is larger than many intraday edges.

Every rate is a configured input, not a literal in the code, and the defaults follow
published NSE/SEBI/Groww schedules. They must be re-verified against a current
contract note before LIVE trading — rates change by circular, and a stale rate makes
the expected-value gate wrong in the dangerous direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..configuration.schema import CostConfig
from ..core.types import Instrument, Quote, TransactionType


@dataclass
class CostBreakdown:
    """Itemized costs so the trade journal can record exactly what was charged."""

    brokerage: float = 0.0
    exchange_transaction: float = 0.0
    stt: float = 0.0
    stamp_duty: float = 0.0
    sebi_charges: float = 0.0
    gst: float = 0.0
    slippage: float = 0.0

    @property
    def statutory(self) -> float:
        return self.exchange_transaction + self.stt + self.stamp_duty + self.sebi_charges + self.gst

    @property
    def total(self) -> float:
        return self.brokerage + self.statutory + self.slippage

    @property
    def total_excluding_slippage(self) -> float:
        return self.brokerage + self.statutory

    def to_dict(self) -> dict[str, float]:
        return {
            "brokerage": round(self.brokerage, 2),
            "exchange_transaction": round(self.exchange_transaction, 2),
            "stt": round(self.stt, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "gst": round(self.gst, 2),
            "slippage": round(self.slippage, 2),
            "total": round(self.total, 2),
        }

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            exchange_transaction=self.exchange_transaction + other.exchange_transaction,
            stt=self.stt + other.stt,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            sebi_charges=self.sebi_charges + other.sebi_charges,
            gst=self.gst + other.gst,
            slippage=self.slippage + other.slippage,
        )


class TransactionCostModel:
    def __init__(self, config: CostConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def leg_cost(
        self,
        *,
        instrument: Instrument,
        transaction_type: TransactionType,
        quantity: int,
        price: float,
        quote: Quote | None = None,
    ) -> CostBreakdown:
        """Cost of a single leg (one buy or one sell)."""
        config = self.config
        turnover = price * quantity
        is_option = instrument.is_option
        breakdown = CostBreakdown()

        # Brokerage: flat per order, capped as a percentage of turnover.
        breakdown.brokerage = min(
            config.brokerage_per_order, turnover * config.brokerage_pct_cap
        ) if config.brokerage_pct_cap > 0 else config.brokerage_per_order

        # Options exchange charges are levied on *premium* turnover, futures on
        # contract value. Both are `price * quantity` here, but the rates differ by
        # an order of magnitude, which is why they are separate config fields.
        breakdown.exchange_transaction = turnover * (
            config.exchange_txn_charge_pct_options if is_option else config.exchange_txn_charge_pct_futures
        )

        # STT applies on the SELL leg only for both futures and options.
        if transaction_type is TransactionType.SELL:
            breakdown.stt = turnover * (
                config.stt_pct_options_sell if is_option else config.stt_pct_futures_sell
            )

        # Stamp duty applies on the BUY leg only.
        if transaction_type is TransactionType.BUY:
            breakdown.stamp_duty = turnover * (
                config.stamp_duty_pct_buy_options if is_option else config.stamp_duty_pct_buy_futures
            )

        breakdown.sebi_charges = turnover * config.sebi_charges_pct
        # GST is charged on brokerage + exchange charges + SEBI charges, not on the
        # statutory levies themselves.
        breakdown.gst = (
            breakdown.brokerage + breakdown.exchange_transaction + breakdown.sebi_charges
        ) * config.gst_pct

        breakdown.slippage = self.slippage(
            instrument=instrument, quantity=quantity, price=price, quote=quote
        )
        return breakdown

    def round_trip(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        entry_price: float,
        exit_price: float | None = None,
        entry_side: TransactionType = TransactionType.BUY,
        quote: Quote | None = None,
    ) -> CostBreakdown:
        """Full cost of entering and exiting. This is what the EV gate must clear."""
        exit_price = exit_price if exit_price is not None else entry_price
        exit_side = (
            TransactionType.SELL if entry_side is TransactionType.BUY else TransactionType.BUY
        )
        entry = self.leg_cost(
            instrument=instrument,
            transaction_type=entry_side,
            quantity=quantity,
            price=entry_price,
            quote=quote,
        )
        exit_ = self.leg_cost(
            instrument=instrument,
            transaction_type=exit_side,
            quantity=quantity,
            price=exit_price,
            quote=quote,
        )
        return entry + exit_

    # ------------------------------------------------------------------ #
    def slippage(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        price: float,
        quote: Quote | None = None,
    ) -> float:
        """Estimated slippage in rupees for one leg.

        The spread-fraction model is preferred because it uses the *observed* book:
        a wide-spread illiquid option is correctly penalised more than a tight
        index option. It falls back to a fixed percentage only when no book exists.
        """
        config = self.config
        if config.slippage_model == "spread_fraction" and quote is not None:
            spread = quote.spread
            if spread is not None and spread > 0:
                return spread * config.slippage_spread_fraction * quantity
        return price * config.slippage_fixed_pct * quantity

    def slippage_pct(self, quote: Quote | None, price: float) -> float:
        if quote is not None:
            spread_pct = quote.spread_pct
            if spread_pct is not None:
                return spread_pct * self.config.slippage_spread_fraction
        return self.config.slippage_fixed_pct

    def cost_as_r_multiple(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        entry_price: float,
        risk_per_unit: float,
        quote: Quote | None = None,
    ) -> float:
        """Round-trip cost expressed in R — directly comparable to a signal's edge."""
        if risk_per_unit <= 0 or quantity <= 0:
            return float("inf")
        costs = self.round_trip(
            instrument=instrument, quantity=quantity, entry_price=entry_price, quote=quote
        )
        total_risk = risk_per_unit * quantity
        return costs.total / total_risk if total_risk > 0 else float("inf")

    def breakeven_move_per_unit(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        price: float,
        quote: Quote | None = None,
    ) -> float:
        """Price move per unit needed just to cover the round trip."""
        if quantity <= 0:
            return float("inf")
        costs = self.round_trip(
            instrument=instrument, quantity=quantity, entry_price=price, quote=quote
        )
        return costs.total / quantity
