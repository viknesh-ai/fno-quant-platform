"""SimulatedBroker — execution simulator for PAPER and BACKTEST modes (REQ 41).

Real market data goes in; no order ever leaves the process. Fills are modelled
against the quote that was live at submission time, with configurable slippage,
latency and partial-fill behaviour, so paper results are comparable to live ones
rather than flattering.

Fault injection (`fail_next_*`, `ambiguous_next_submit`) exists so the simulation
tests in REQ 64 can drive the failure paths that are otherwise unreachable.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable

from ..core.clock import IST, Clock, LiveClock
from ..core.errors import OrderRejected, OrderStateAmbiguous, TransientBrokerError
from ..core.logging import get_logger
from ..core.types import (
    Candle,
    Holding,
    Instrument,
    MarginInfo,
    MarginRequirement,
    OptionChain,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Segment,
    Timeframe,
    Trade,
    TransactionType,
)
from .base import BrokerAdapter, Capability

logger = get_logger(__name__)


@dataclass
class SimulationSettings:
    slippage_spread_fraction: float = 0.5
    slippage_fixed_pct: float = 0.0005
    latency_ms: int = 250
    partial_fill_probability: float = 0.0
    reject_probability: float = 0.0
    seed: int = 7
    # A market order beyond this fraction of the book is only partially filled.
    depth_consumption_ratio: float = 0.5


@dataclass
class SimulatedFill:
    quantity: int
    price: float


class SimulatedBroker(BrokerAdapter):
    """In-process broker. Quotes are supplied by a callable so the same class works
    for live-data paper trading and for historical replay."""

    name = "simulated"

    def __init__(
        self,
        *,
        quote_source: Callable[[Instrument], Quote | None],
        starting_cash: float,
        settings: SimulationSettings | None = None,
        clock: Clock | None = None,
        instruments: list[Instrument] | None = None,
    ) -> None:
        self.quote_source = quote_source
        self.settings = settings or SimulationSettings()
        self.clock = clock or LiveClock()
        self._rng = random.Random(self.settings.seed)
        self._cash = starting_cash
        self._starting_cash = starting_cash
        self._orders: dict[str, Order] = {}
        self._orders_by_reference: dict[str, str] = {}
        self._trades: dict[str, list[Trade]] = {}
        self._positions: dict[str, Position] = {}
        self._realized_pnl = 0.0
        self._counter = itertools.count(1)
        self._instruments = instruments or []
        self._authenticated = False

        # Fault injection switches for simulation tests.
        self.fail_next_submit: Exception | None = None
        self.ambiguous_next_submit = False
        self.fail_next_query: Exception | None = None

    # ------------------------------------------------------------------ #
    def authenticate(self) -> None:
        self._authenticated = True

    def is_authenticated(self) -> bool:
        return self._authenticated

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.AUTHENTICATION,
                Capability.INSTRUMENTS,
                Capability.LTP,
                Capability.QUOTE,
                Capability.PLACE_ORDER,
                Capability.MODIFY_ORDER,
                Capability.CANCEL_ORDER,
                Capability.ORDER_STATUS,
                Capability.ORDER_LIST,
                Capability.TRADE_STATUS,
                Capability.POSITIONS,
                Capability.HOLDINGS,
                Capability.MARGIN,
            }
        )

    def fetch_instruments(self, *, force_refresh: bool = False) -> list[Instrument]:
        return list(self._instruments)

    def set_instruments(self, instruments: list[Instrument]) -> None:
        self._instruments = list(instruments)

    # ------------------------------------------------------------------ #
    # Market data passthrough
    # ------------------------------------------------------------------ #
    def get_ltp(self, instruments: list[Instrument]) -> dict[str, float]:
        out: dict[str, float] = {}
        for instrument in instruments:
            quote = self.quote_source(instrument)
            if quote is not None:
                out[instrument.trading_symbol] = quote.last_price
        return out

    def get_ohlc(self, instruments: list[Instrument]) -> dict[str, Candle]:
        out: dict[str, Candle] = {}
        for instrument in instruments:
            quote = self.quote_source(instrument)
            if quote is None:
                continue
            out[instrument.trading_symbol] = Candle(
                timestamp=quote.timestamp,
                open=quote.open or quote.last_price,
                high=quote.high or quote.last_price,
                low=quote.low or quote.last_price,
                close=quote.last_price,
                volume=quote.volume or 0.0,
            )
        return out

    def get_quote(self, instrument: Instrument) -> Quote:
        quote = self.quote_source(instrument)
        if quote is None:
            raise TransientBrokerError(f"no simulated quote for {instrument.trading_symbol}")
        return quote

    def get_option_chain(self, underlying: str, expiry: date, *, exchange: str = "NSE") -> OptionChain:
        raise NotImplementedError(
            "SimulatedBroker does not synthesize option chains; supply them through the "
            "MarketDataEngine's historical/live source"
        )

    def get_historical_candles(
        self, instrument: Instrument, timeframe: Timeframe, start: datetime, end: datetime
    ) -> list[Candle]:
        return []

    # ------------------------------------------------------------------ #
    # Order lifecycle
    # ------------------------------------------------------------------ #
    def place_order(self, request: OrderRequest) -> Order:
        if self.fail_next_submit is not None:
            error, self.fail_next_submit = self.fail_next_submit, None
            raise error
        if self.ambiguous_next_submit:
            self.ambiguous_next_submit = False
            # Mirror the real hazard: the order *is* accepted but the caller is told
            # the state is unknown. Only a reference lookup reveals the truth.
            order = self._accept(request)
            raise OrderStateAmbiguous(
                "simulated ambiguous submit", client_order_id=request.client_order_id
            )

        if request.client_order_id and request.client_order_id in self._orders_by_reference:
            existing = self._orders[self._orders_by_reference[request.client_order_id]]
            logger.warning(
                "duplicate client_order_id %s; returning existing order %s",
                request.client_order_id,
                existing.broker_order_id,
            )
            return existing

        if self._rng.random() < self.settings.reject_probability:
            raise OrderRejected("simulated random rejection", code="SIM_REJECT")

        return self._accept(request)

    def _accept(self, request: OrderRequest) -> Order:
        quote = self.quote_source(request.instrument)
        if quote is None:
            raise OrderRejected(
                f"no market data for {request.instrument.trading_symbol}", code="SIM_NO_DATA"
            )

        order_id = f"SIM{next(self._counter):08d}"
        now = self.clock.now()
        order = Order(
            broker_order_id=order_id,
            instrument=request.instrument,
            transaction_type=request.transaction_type,
            quantity=request.quantity,
            order_type=request.order_type,
            product=request.product,
            status=OrderStatus.ACKED,
            remaining_quantity=request.quantity,
            price=request.price,
            trigger_price=request.trigger_price,
            client_order_id=request.client_order_id,
            created_at=now,
            updated_at=now,
        )
        self._orders[order_id] = order
        if request.client_order_id:
            self._orders_by_reference[request.client_order_id] = order_id

        fill = self._simulate_fill(request, quote)
        if fill is None:
            # Resting limit order: stays open until price comes to it or it is cancelled.
            order.status = OrderStatus.ACKED
            return order

        self._apply_fill(order, fill, now)
        return order

    def _simulate_fill(self, request: OrderRequest, quote: Quote) -> SimulatedFill | None:
        """Decide whether and at what price this order fills right now."""
        is_buy = request.transaction_type is TransactionType.BUY
        touch = (quote.ask_price if is_buy else quote.bid_price) or quote.last_price
        if touch is None or touch <= 0:
            touch = quote.last_price

        if request.order_type is OrderType.LIMIT:
            assert request.price is not None
            # A buy limit fills only if the offer has come down to our price.
            if is_buy and touch > request.price:
                return None
            if not is_buy and touch < request.price:
                return None
            # Price improvement is capped at the limit price.
            fill_price = min(touch, request.price) if is_buy else max(touch, request.price)
        elif request.order_type in (OrderType.SL, OrderType.SL_M):
            assert request.trigger_price is not None
            triggered = (
                quote.last_price >= request.trigger_price
                if is_buy
                else quote.last_price <= request.trigger_price
            )
            if not triggered:
                return None
            fill_price = touch + self._slippage(quote, is_buy)
        else:  # MARKET
            fill_price = touch + self._slippage(quote, is_buy)

        quantity = request.quantity
        available = self._available_depth(quote, is_buy)
        if available is not None and available < quantity:
            quantity = max(request.instrument.lot_size, available - (available % max(1, request.instrument.lot_size)))
            quantity = min(quantity, request.quantity)
        elif self._rng.random() < self.settings.partial_fill_probability:
            lots = max(1, request.quantity // max(1, request.instrument.lot_size))
            if lots > 1:
                quantity = self._rng.randint(1, lots - 1) * request.instrument.lot_size

        fill_price = max(0.05, round(fill_price / request.instrument.tick_size) * request.instrument.tick_size)
        return SimulatedFill(quantity=quantity, price=round(fill_price, 2))

    def _slippage(self, quote: Quote, is_buy: bool) -> float:
        """Signed slippage, always against us."""
        spread = quote.spread
        if spread is not None and spread > 0:
            magnitude = spread * self.settings.slippage_spread_fraction
        else:
            magnitude = quote.last_price * self.settings.slippage_fixed_pct
        return magnitude if is_buy else -magnitude

    def _available_depth(self, quote: Quote, is_buy: bool) -> int | None:
        levels = quote.depth_sell if is_buy else quote.depth_buy
        if not levels:
            return None
        total = sum(level.quantity for level in levels)
        return int(total * self.settings.depth_consumption_ratio)

    def _apply_fill(self, order: Order, fill: SimulatedFill, now: datetime) -> None:
        order.filled_quantity += fill.quantity
        order.remaining_quantity = order.quantity - order.filled_quantity
        prior_value = (order.average_fill_price or 0.0) * (order.filled_quantity - fill.quantity)
        order.average_fill_price = (prior_value + fill.price * fill.quantity) / order.filled_quantity
        order.status = OrderStatus.EXECUTED if order.remaining_quantity == 0 else OrderStatus.ACKED
        order.updated_at = now

        self._trades.setdefault(order.broker_order_id, []).append(
            Trade(
                trade_id=f"T{order.broker_order_id}-{len(self._trades.get(order.broker_order_id, []))+1}",
                broker_order_id=order.broker_order_id,
                instrument=order.instrument,
                transaction_type=order.transaction_type,
                quantity=fill.quantity,
                price=fill.price,
                timestamp=now,
            )
        )
        self._update_position(order.instrument, order.transaction_type, fill, now)

    def _update_position(
        self, instrument: Instrument, side: TransactionType, fill: SimulatedFill, now: datetime
    ) -> None:
        key = instrument.trading_symbol
        signed = fill.quantity if side is TransactionType.BUY else -fill.quantity
        cash_delta = -signed * fill.price
        self._cash += cash_delta

        position = self._positions.get(key)
        if position is None:
            self._positions[key] = Position(
                instrument=instrument,
                quantity=signed,
                average_price=fill.price,
                last_price=fill.price,
                opened_at=now,
            )
            return

        if position.quantity == 0 or (position.quantity > 0) == (signed > 0):
            # Adding to the same side: weighted-average the entry.
            total = position.quantity + signed
            position.average_price = (
                position.average_price * position.quantity + fill.price * signed
            ) / total
            position.quantity = total
        else:
            # Reducing or flipping: realize P&L on the closed portion.
            closing = min(abs(signed), abs(position.quantity))
            direction = 1 if position.quantity > 0 else -1
            realized = (fill.price - position.average_price) * closing * direction
            position.realized_pnl += realized
            self._realized_pnl += realized
            remaining = position.quantity + signed
            if (remaining > 0) != (position.quantity > 0) and remaining != 0:
                position.average_price = fill.price  # flipped side
            position.quantity = remaining

        position.last_price = fill.price
        if position.quantity == 0:
            self._positions.pop(key, None)

    def poll_open_orders(self) -> list[Order]:
        """Re-evaluate resting orders against the current quote.

        The orchestrator calls this each loop; the backtester calls it on every bar.
        This is what lets a limit order fill later rather than only at submission.
        """
        filled: list[Order] = []
        now = self.clock.now()
        for order in list(self._orders.values()):
            if not order.status.is_open or order.remaining_quantity <= 0:
                continue
            quote = self.quote_source(order.instrument)
            if quote is None:
                continue
            request = OrderRequest(
                instrument=order.instrument,
                transaction_type=order.transaction_type,
                quantity=order.remaining_quantity,
                order_type=order.order_type,
                product=order.product,
                price=order.price,
                trigger_price=order.trigger_price,
                client_order_id=order.client_order_id or "POLLFILL0",
            )
            fill = self._simulate_fill(request, quote)
            if fill is not None:
                self._apply_fill(order, fill, now)
                filled.append(order)
        return filled

    def mark_to_market(self) -> None:
        for position in self._positions.values():
            quote = self.quote_source(position.instrument)
            if quote is not None:
                position.last_price = quote.last_price

    def modify_order(
        self,
        broker_order_id: str,
        *,
        segment: Segment,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        order_type: str | None = None,
    ) -> Order:
        order = self._orders.get(broker_order_id)
        if order is None:
            raise OrderRejected(f"unknown order {broker_order_id}")
        if order.status.is_terminal:
            raise OrderRejected(f"cannot modify order in state {order.status.value}")
        if quantity is not None:
            order.quantity = quantity
            order.remaining_quantity = quantity - order.filled_quantity
        if price is not None:
            order.price = price
        if trigger_price is not None:
            order.trigger_price = trigger_price
        if order_type is not None:
            order.order_type = OrderType(order_type.upper())
        order.updated_at = self.clock.now()
        return order

    def cancel_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        order = self._orders.get(broker_order_id)
        if order is None:
            raise OrderRejected(f"unknown order {broker_order_id}")
        if order.status.is_terminal:
            return order
        order.status = OrderStatus.CANCELLED
        order.remaining_quantity = 0
        order.updated_at = self.clock.now()
        return order

    def get_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        if self.fail_next_query is not None:
            error, self.fail_next_query = self.fail_next_query, None
            raise error
        order = self._orders.get(broker_order_id)
        if order is None:
            raise OrderRejected(f"unknown order {broker_order_id}", code="404")
        return order

    def get_order_by_reference(self, client_order_id: str, *, segment: Segment) -> Order | None:
        order_id = self._orders_by_reference.get(client_order_id)
        return self._orders.get(order_id) if order_id else None

    def list_orders(self) -> list[Order]:
        return list(self._orders.values())

    def get_trades(self, broker_order_id: str, *, segment: Segment) -> list[Trade]:
        return list(self._trades.get(broker_order_id, []))

    # ------------------------------------------------------------------ #
    def get_positions(self) -> list[Position]:
        self.mark_to_market()
        return list(self._positions.values())

    def get_holdings(self) -> list[Holding]:
        return []

    def get_margin(self) -> MarginInfo:
        self.mark_to_market()
        used = sum(abs(p.quantity) * p.average_price for p in self._positions.values())
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return MarginInfo(
            clear_cash=max(0.0, self._starting_cash + self._realized_pnl + unrealized - used),
            net_margin_used=used,
            raw={"simulated": True, "realized_pnl": self._realized_pnl},
        )

    def get_required_margin(self, requests: list[OrderRequest]) -> MarginRequirement:
        """Premium-outlay approximation. Deliberately crude and labelled as such:
        SPAN/exposure for short options cannot be reproduced without the exchange
        risk arrays, so short-option strategies must size against the real broker."""
        total = 0.0
        for request in requests:
            quote = self.quote_source(request.instrument)
            price = request.price or (quote.last_price if quote else 0.0)
            total += price * request.quantity
        return MarginRequirement(total_requirement=total, option_buy_premium=total)

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def equity(self) -> float:
        self.mark_to_market()
        return (
            self._starting_cash
            + self._realized_pnl
            + sum(p.unrealized_pnl for p in self._positions.values())
        )

    def reset(self) -> None:
        self._cash = self._starting_cash
        self._orders.clear()
        self._orders_by_reference.clear()
        self._trades.clear()
        self._positions.clear()
        self._realized_pnl = 0.0
        self._counter = itertools.count(1)
