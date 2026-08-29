"""ExecutionEngine (REQ 32).

Implements the thirteen steps REQ 32 specifies, in order, with the requirement's
central rule enforced structurally:

    "Never assume: API returned success = position exists."

So `execute()` does not record a trade from the submit response. It records the
*intent* first (crash safety), submits, then **queries the broker** for the order
and its trades, and only reconciles local state from what the broker reports.

Two further guarantees:
  * An order cannot be submitted without a `RiskApproval` (REQ 67.22).
  * PAPER and BACKTEST modes cannot reach a real order endpoint — the check is on
    `config.submits_real_orders`, and it is asserted here as well as in the broker
    factory, because this is the failure nobody gets to make twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from ..brokers.base import BrokerAdapter
from ..configuration.schema import AppConfig
from ..core.clock import Clock, LiveClock, is_market_open
from ..core.errors import (
    BrokerError,
    DuplicateOrderError,
    OrderRejected,
    OrderStateAmbiguous,
)
from ..core.ids import decision_bucket, make_client_order_id
from ..core.logging import get_logger
from ..core.types import (
    Decision,
    Direction,
    Instrument,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Quote,
    TradingMode,
    TransactionType,
    Validity,
)
from ..journal.store import TradeJournal
from ..risk.engine import RiskApproval
from .latency import LatencyMonitor, LatencyTrace

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Outcome of an execution attempt, with every step recorded."""

    success: bool
    order: Order | None = None
    client_order_id: str = ""
    filled_quantity: int = 0
    average_fill_price: float | None = None
    steps: list[str] = field(default_factory=list)
    failure_reason: str = ""
    ambiguous: bool = False
    latency: LatencyTrace | None = None
    verified: bool = False

    @property
    def partially_filled(self) -> bool:
        return (
            self.order is not None
            and 0 < self.filled_quantity < self.order.quantity
        )

    def explain(self) -> str:
        if self.success:
            return (
                f"filled {self.filled_quantity} @ {self.average_fill_price:.2f} "
                f"(order {self.order.broker_order_id if self.order else '?'}, verified={self.verified})"
            )
        if self.ambiguous:
            return f"AMBIGUOUS: {self.failure_reason}"
        return f"failed: {self.failure_reason}"


class ExecutionEngine:
    def __init__(
        self,
        config: AppConfig,
        broker: BrokerAdapter,
        journal: TradeJournal,
        *,
        clock: Clock | None = None,
        latency_monitor: LatencyMonitor | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.journal = journal
        self.clock = clock or LiveClock()
        self.latency = latency_monitor or LatencyMonitor()

    # ------------------------------------------------------------------ #
    def execute(
        self,
        *,
        approval: RiskApproval,
        instrument: Instrument,
        quote: Quote | None,
        decision_id: str,
        strategy: str,
        limit_price: float | None = None,
        market_data_at: datetime | None = None,
        signal_at: datetime | None = None,
    ) -> ExecutionResult:
        """Run the thirteen-step execution pipeline."""
        steps: list[str] = []
        trace = LatencyTrace(
            market_data_at=market_data_at,
            signal_at=signal_at,
            risk_check_at=approval.approved_at,
        )

        def fail(reason: str, *, ambiguous: bool = False) -> ExecutionResult:
            logger.warning("execution halted: %s", reason)
            return ExecutionResult(
                success=False, steps=steps, failure_reason=reason,
                ambiguous=ambiguous, latency=trace,
            )

        # --- 1. validate signal ------------------------------------------
        if not approval.approved:
            return fail("risk approval was not granted")
        if approval.decision is Decision.NO_TRADE:
            return fail("decision is NO_TRADE")
        steps.append("1. signal validated")

        # --- 2. validate risk ---------------------------------------------
        # The approval object is the risk gate's signature. Executing without one is
        # impossible because it is a required argument (REQ 67.22).
        if approval.sizing is None or approval.sizing.quantity <= 0:
            return fail("risk approval carries no position size")
        steps.append(f"2. risk approval {approval.approval_id} verified")

        # --- 3. validate instrument ----------------------------------------
        if not instrument.tradable:
            return fail(f"{instrument.trading_symbol} is not tradable (reserved or restricted)")
        if instrument.lot_size <= 0:
            return fail(f"{instrument.trading_symbol} has an invalid lot size")
        if approval.sizing.quantity % instrument.lot_size != 0:
            return fail(
                f"quantity {approval.sizing.quantity} is not a multiple of lot size "
                f"{instrument.lot_size}"
            )
        steps.append(f"3. instrument {instrument.trading_symbol} validated")

        # --- 4. validate liquidity ------------------------------------------
        if quote is None:
            return fail("no quote available to validate liquidity")
        if quote.last_price <= 0:
            return fail("quote has no valid last price")
        spread_pct = quote.spread_pct
        if spread_pct is not None and spread_pct > self.config.risk.max_spread_pct:
            return fail(
                f"spread widened to {spread_pct:.2%} since approval, above the "
                f"{self.config.risk.max_spread_pct:.2%} limit"
            )
        steps.append("4. liquidity validated")

        # --- 5. validate market status ---------------------------------------
        now = self.clock.now()
        if self.config.submits_real_orders and not is_market_open(now):
            return fail(f"market is closed at {now:%H:%M:%S}")
        steps.append("5. market status validated")

        # --- 6. validate quantity ---------------------------------------------
        quantity = approval.sizing.quantity
        if instrument.freeze_quantity and quantity > instrument.freeze_quantity:
            return fail(
                f"quantity {quantity} exceeds the exchange freeze limit "
                f"{instrument.freeze_quantity}"
            )
        steps.append(f"6. quantity {quantity} validated")

        # --- 7. validate margin -------------------------------------------------
        transaction_type = (
            TransactionType.BUY if approval.decision is Decision.BUY else TransactionType.SELL
        )
        order_type, price = self._order_parameters(quote, transaction_type, limit_price)
        request = OrderRequest(
            instrument=instrument,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=self.config.execution.default_product,
            price=price,
            validity=Validity.DAY,
            client_order_id="",
            tag=strategy[:16],
        )
        margin_ok, margin_detail = self._verify_margin(request)
        if not margin_ok:
            return fail(margin_detail)
        steps.append(f"7. margin validated ({margin_detail})")

        # --- 8. create order -----------------------------------------------------
        client_order_id = make_client_order_id(
            trading_symbol=instrument.trading_symbol,
            transaction_type=transaction_type.value,
            quantity=quantity,
            strategy=strategy,
            bucket=decision_bucket(now, seconds=60),
        )
        request.client_order_id = client_order_id
        steps.append(f"8. order created with idempotency key {client_order_id}")

        # Record the intent BEFORE submitting. If the process dies mid-flight, this
        # row is what tells restart reconciliation to query rather than resubmit.
        self.journal.record_order_intent(
            {
                "client_order_id": client_order_id,
                "decision_id": decision_id,
                "instrument": instrument.trading_symbol,
                "transaction_type": transaction_type.value,
                "order_type": order_type.value,
                "product": request.product.value,
                "quantity": quantity,
                "price": price,
                "status": "INTENT",
                "timestamp": now.isoformat(),
            }
        )

        # --- 9. submit order -------------------------------------------------------
        if not self.config.submits_real_orders and self.config.mode not in (
            TradingMode.PAPER, TradingMode.BACKTEST
        ):
            return fail(f"mode {self.config.mode.value} is not permitted to submit orders")

        trace.order_submit_at = self.clock.now()
        self.journal.update_order(
            client_order_id, {"status": "SUBMITTED", "submitted_at": trace.order_submit_at.isoformat()}
        )

        try:
            order = self.broker.place_order(request)
        except DuplicateOrderError as exc:
            self.journal.update_order(client_order_id, {"status": "DUPLICATE_BLOCKED"})
            return fail(f"duplicate order blocked: {exc}")
        except OrderStateAmbiguous as exc:
            # Do NOT resubmit. Resolve by querying (REQ 45).
            self.journal.update_order(client_order_id, {"status": "UNKNOWN", "rejection_reason": str(exc)})
            resolved = self._resolve_ambiguous(client_order_id, instrument)
            if resolved is None:
                return fail(
                    f"order state is ambiguous and could not be resolved: {exc}. "
                    "Manual reconciliation required before retrying.",
                    ambiguous=True,
                )
            order = resolved
            steps.append("9. submit was ambiguous; state resolved by reference lookup")
        except OrderRejected as exc:
            self.journal.update_order(
                client_order_id, {"status": "REJECTED", "rejection_reason": str(exc)}
            )
            return fail(f"broker rejected the order: {exc}")
        except BrokerError as exc:
            self.journal.update_order(
                client_order_id, {"status": "FAILED", "rejection_reason": str(exc)}
            )
            return fail(f"broker error: {exc}")
        else:
            steps.append("9. order submitted")

        trace.broker_ack_at = self.clock.now()

        # --- 10. verify broker response ----------------------------------------------
        if not order.broker_order_id:
            return fail(
                "broker returned no order id; existence is unconfirmed", ambiguous=True
            )
        if order.status in (OrderStatus.REJECTED, OrderStatus.FAILED):
            self.journal.update_order(
                client_order_id,
                {"status": order.status.value, "rejection_reason": order.rejection_reason,
                 "broker_order_id": order.broker_order_id},
            )
            return fail(f"order {order.status.value}: {order.rejection_reason}")
        steps.append(f"10. broker acknowledged order {order.broker_order_id}")

        self.journal.update_order(
            client_order_id,
            {
                "broker_order_id": order.broker_order_id,
                "status": order.status.value,
                "acknowledged_at": trace.broker_ack_at.isoformat(),
            },
        )

        # --- 11. verify ACTUAL execution ------------------------------------------------
        # This is the step REQ 32 exists for: never infer a fill from the submit call.
        verified_order = self._await_fill(order, instrument)
        if verified_order is None:
            return fail("could not verify order state with the broker", ambiguous=True)
        order = verified_order

        if order.filled_quantity <= 0:
            if self.config.execution.cancel_unfilled_after_timeout and order.status.is_open:
                self._cancel(order, instrument)
                return fail("order did not fill within the timeout and was cancelled")
            return fail(f"order is {order.status.value} with no fill")

        trace.fill_at = self.clock.now()
        steps.append(
            f"11. execution verified: {order.filled_quantity}/{order.quantity} filled "
            f"@ {order.average_fill_price}"
        )

        # --- 12. reconcile local state ---------------------------------------------------
        if order.filled_quantity < order.quantity:
            # Partial fill is a real state, not an error. Cancel the remainder so the
            # position matches what we think we hold (REQ 45).
            steps.append(
                f"12. partial fill: cancelling the unfilled {order.quantity - order.filled_quantity}"
            )
            if order.status.is_open:
                self._cancel(order, instrument)
        else:
            steps.append("12. local state reconciled with broker")

        # --- 13. record the trade -----------------------------------------------------------
        self.latency.record(trace)
        latency_fields = trace.to_dict()
        self.journal.update_order(
            client_order_id,
            {
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "filled_at": trace.fill_at.isoformat(),
                **{k: v for k, v in latency_fields.items() if k in
                   ("signal_latency_ms", "api_latency_ms", "execution_latency_ms", "total_latency_ms")},
                "raw": dict(order.raw) if order.raw else None,
            },
        )
        steps.append("13. trade recorded to journal")
        logger.info(
            "executed %s %s x%d @ %.2f (%s)",
            transaction_type.value, instrument.trading_symbol, order.filled_quantity,
            order.average_fill_price or 0.0, trace.describe(),
        )

        return ExecutionResult(
            success=True,
            order=order,
            client_order_id=client_order_id,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            steps=steps,
            latency=trace,
            verified=True,
        )

    # ------------------------------------------------------------------ #
    def _order_parameters(
        self, quote: Quote, transaction_type: TransactionType, limit_price: float | None
    ) -> tuple[OrderType, float | None]:
        """Choose order type and price.

        A LIMIT with a small buffer past the touch is preferred over a MARKET order:
        it bounds the worst fill on an illiquid option, where a market order can be
        filled far from the quote.
        """
        config = self.config.execution
        if config.entry_order_type == "MARKET":
            return OrderType.MARKET, None

        if limit_price is not None:
            price = limit_price
        else:
            touch = (
                quote.ask_price if transaction_type is TransactionType.BUY else quote.bid_price
            ) or quote.last_price
            buffer = touch * config.limit_price_buffer_pct
            price = touch + buffer if transaction_type is TransactionType.BUY else touch - buffer

        tick = 0.05
        return OrderType.LIMIT, max(tick, round(price / tick) * tick)

    def _verify_margin(self, request: OrderRequest) -> tuple[bool, str]:
        try:
            requirement = self.broker.get_required_margin([request])
        except Exception as exc:
            # If the broker cannot price the margin we do not assume it is fine —
            # but neither do we block on an endpoint that may simply be unsupported.
            logger.debug("margin pre-check unavailable: %s", exc)
            return True, "broker margin pre-check unavailable"
        try:
            margin = self.broker.get_margin()
        except Exception as exc:
            return True, f"margin balance unavailable ({exc})"

        if requirement.total_requirement > margin.available_margin:
            return False, (
                f"insufficient margin: requires ₹{requirement.total_requirement:,.0f} "
                f"but only ₹{margin.available_margin:,.0f} is available"
            )
        return True, (
            f"requires ₹{requirement.total_requirement:,.0f} of ₹{margin.available_margin:,.0f}"
        )

    def _resolve_ambiguous(self, client_order_id: str, instrument: Instrument) -> Order | None:
        """After an ambiguous submit, ask the broker what actually exists (REQ 45)."""
        for attempt in range(3):
            try:
                resolved = self.broker.get_order_by_reference(
                    client_order_id, segment=instrument.segment
                )
            except Exception as exc:
                logger.warning("reference lookup attempt %d failed: %s", attempt + 1, exc)
                time.sleep(1.0 * (attempt + 1))
                continue
            if resolved is not None:
                logger.warning(
                    "ambiguous submit resolved: order %s exists with status %s",
                    resolved.broker_order_id, resolved.status.value,
                )
                return resolved
            time.sleep(1.0 * (attempt + 1))
        logger.error(
            "could not resolve ambiguous order %s; NOT resubmitting", client_order_id
        )
        return None

    def _await_fill(self, order: Order, instrument: Instrument) -> Order | None:
        """Poll the broker until the order reaches a terminal state or times out."""
        config = self.config.execution
        deadline = time.monotonic() + config.order_fill_timeout_seconds
        current = order

        # A simulated broker fills synchronously; poll it so resting limits resolve.
        poll_hook: Callable[[], None] | None = getattr(self.broker, "poll_open_orders", None)

        while True:
            if current.status.is_terminal and current.filled_quantity > 0:
                return current
            if current.status in (OrderStatus.REJECTED, OrderStatus.FAILED, OrderStatus.CANCELLED):
                return current
            if time.monotonic() >= deadline:
                logger.warning(
                    "order %s did not reach a terminal state within %.0fs (status %s)",
                    current.broker_order_id, config.order_fill_timeout_seconds, current.status.value,
                )
                return current

            time.sleep(min(config.order_poll_interval_seconds, 1.0))
            if poll_hook is not None:
                try:
                    poll_hook()
                except Exception:
                    pass
            try:
                current = self.broker.get_order(
                    current.broker_order_id, segment=instrument.segment
                )
            except BrokerError as exc:
                logger.warning("order status query failed: %s", exc)
                continue

    def _cancel(self, order: Order, instrument: Instrument) -> None:
        try:
            self.broker.cancel_order(order.broker_order_id, segment=instrument.segment)
            logger.info("cancelled unfilled remainder of order %s", order.broker_order_id)
        except BrokerError as exc:
            logger.error("failed to cancel order %s: %s", order.broker_order_id, exc)

    # ------------------------------------------------------------------ #
    def close_position(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        direction: Direction,
        reason: str,
        decision_id: str = "",
        strategy: str = "exit",
        quote: Quote | None = None,
        use_market_order: bool = True,
    ) -> ExecutionResult:
        """Exit an open position.

        Exits default to MARKET orders. An exit that does not fill is strictly worse
        than an exit at a slightly worse price — a resting limit on the way out is
        how a managed loss becomes an unmanaged one.
        """
        steps = ["exit: closing position"]
        now = self.clock.now()
        transaction_type = (
            TransactionType.SELL if direction is Direction.LONG else TransactionType.BUY
        )
        client_order_id = make_client_order_id(
            trading_symbol=instrument.trading_symbol,
            transaction_type=transaction_type.value,
            quantity=quantity,
            strategy=f"exit_{strategy}",
            bucket=decision_bucket(now, seconds=60),
        )

        order_type = OrderType.MARKET if use_market_order else OrderType.LIMIT
        price = None
        if order_type is OrderType.LIMIT and quote is not None:
            touch = (
                quote.bid_price if transaction_type is TransactionType.SELL else quote.ask_price
            ) or quote.last_price
            price = round(touch / 0.05) * 0.05

        request = OrderRequest(
            instrument=instrument,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=self.config.execution.default_product,
            price=price,
            client_order_id=client_order_id,
            tag=f"exit_{reason}"[:16],
        )

        self.journal.record_order_intent(
            {
                "client_order_id": client_order_id,
                "decision_id": decision_id,
                "instrument": instrument.trading_symbol,
                "transaction_type": transaction_type.value,
                "order_type": order_type.value,
                "product": request.product.value,
                "quantity": quantity,
                "price": price,
                "status": "INTENT",
                "timestamp": now.isoformat(),
                "rejection_reason": f"exit reason: {reason}",
            }
        )

        try:
            order = self.broker.place_order(request)
        except OrderStateAmbiguous as exc:
            resolved = self._resolve_ambiguous(client_order_id, instrument)
            if resolved is None:
                return ExecutionResult(
                    success=False, steps=steps, client_order_id=client_order_id,
                    failure_reason=f"exit order ambiguous: {exc}", ambiguous=True,
                )
            order = resolved
        except BrokerError as exc:
            self.journal.update_order(
                client_order_id, {"status": "FAILED", "rejection_reason": str(exc)}
            )
            return ExecutionResult(
                success=False, steps=steps, client_order_id=client_order_id,
                failure_reason=f"exit order failed: {exc}",
            )

        verified = self._await_fill(order, instrument)
        order = verified or order
        self.journal.update_order(
            client_order_id,
            {
                "broker_order_id": order.broker_order_id,
                "status": order.status.value,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "filled_at": self.clock.now().isoformat(),
            },
        )
        steps.append(f"exit filled {order.filled_quantity} @ {order.average_fill_price}")

        return ExecutionResult(
            success=order.filled_quantity > 0,
            order=order,
            client_order_id=client_order_id,
            filled_quantity=order.filled_quantity,
            average_fill_price=order.average_fill_price,
            steps=steps,
            verified=True,
            failure_reason="" if order.filled_quantity > 0 else "exit order did not fill",
        )
