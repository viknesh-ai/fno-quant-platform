"""Core domain vocabulary shared by every layer.

Nothing in this module may import a broker, a strategy or a model. These types are
the contract between layers; keeping them dependency-free is what allows the
Market-Data -> Feature -> Regime -> Strategy -> ML -> Risk -> Execution pipeline to
stay decoupled (REQ 2).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class TradingMode(str, enum.Enum):
    """REQ 41/42/43 — modes are isolated and PAPER is the default everywhere."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    DEMO = "DEMO"
    LIVE = "LIVE"


class Exchange(str, enum.Enum):
    NSE = "NSE"
    BSE = "BSE"


class Segment(str, enum.Enum):
    CASH = "CASH"
    FNO = "FNO"


class InstrumentType(str, enum.Enum):
    EQ = "EQ"
    IDX = "IDX"
    FUT = "FUT"
    CE = "CE"
    PE = "PE"


class OptionType(str, enum.Enum):
    CE = "CE"
    PE = "PE"


class Product(str, enum.Enum):
    CNC = "CNC"
    MIS = "MIS"
    NRML = "NRML"


class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL_M"


class TransactionType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class Validity(str, enum.Enum):
    DAY = "DAY"


class OrderStatus(str, enum.Enum):
    """Mirrors the broker's documented order-status vocabulary plus a local UNKNOWN.

    UNKNOWN is used when the broker response is ambiguous — the ExecutionEngine must
    resolve it by querying, never by assuming (REQ 32/45).
    """

    NEW = "NEW"
    ACKED = "ACKED"
    TRIGGER_PENDING = "TRIGGER_PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    EXECUTED = "EXECUTED"
    DELIVERY_AWAITED = "DELIVERY_AWAITED"
    CANCELLED = "CANCELLED"
    CANCELLATION_REQUESTED = "CANCELLATION_REQUESTED"
    MODIFICATION_REQUESTED = "MODIFICATION_REQUESTED"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES

    @property
    def is_open(self) -> bool:
        return self in _OPEN_STATUSES


_TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.EXECUTED,
        OrderStatus.COMPLETED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.FAILED,
    }
)
_OPEN_STATUSES = frozenset(
    {
        OrderStatus.NEW,
        OrderStatus.ACKED,
        OrderStatus.APPROVED,
        OrderStatus.TRIGGER_PENDING,
        OrderStatus.MODIFICATION_REQUESTED,
        OrderStatus.CANCELLATION_REQUESTED,
    }
)


class Direction(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        return {Direction.LONG: 1, Direction.SHORT: -1, Direction.FLAT: 0}[self]


class Decision(str, enum.Enum):
    """REQ 36 — NO_TRADE is an explicit, first-class outcome."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO_TRADE"


class Regime(str, enum.Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_COMPRESSION = "VOLATILITY_COMPRESSION"
    EXPIRY_DRIVEN = "EXPIRY_DRIVEN"
    EVENT_DRIVEN = "EVENT_DRIVEN"
    UNCERTAIN = "UNCERTAIN"


class Timeframe(str, enum.Enum):
    """REQ 9 — no timeframe is hard-coded; strategies declare what they need."""

    TICK = "tick"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M10 = "10m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def minutes(self) -> int:
        return _TF_MINUTES[self]

    @classmethod
    def from_minutes(cls, minutes: int) -> "Timeframe":
        for tf, m in _TF_MINUTES.items():
            if m == minutes:
                return tf
        raise ValueError(f"no timeframe for {minutes} minutes")


_TF_MINUTES: Mapping[Timeframe, int] = {
    Timeframe.TICK: 0,
    Timeframe.M1: 1,
    Timeframe.M3: 3,
    Timeframe.M5: 5,
    Timeframe.M10: 10,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 375,  # one NSE equity session
}


class StrategyState(str, enum.Enum):
    """REQ 28 — strategies are throttled by measured health, not by opinion."""

    ACTIVE = "ACTIVE"
    REDUCED = "REDUCED"
    PAUSED = "PAUSED"
    DISABLED = "DISABLED"


class DrawdownLevel(int, enum.Enum):
    """REQ 26."""

    NORMAL = 1
    REDUCED = 2
    PAUSED = 3
    SHUTDOWN = 4


class ExitReason(str, enum.Enum):
    """REQ 35 — every exit carries a recorded reason."""

    TARGET_REACHED = "TARGET_REACHED"
    STOP_REACHED = "STOP_REACHED"
    PREDICTION_INVALIDATED = "PREDICTION_INVALIDATED"
    REGIME_CHANGED = "REGIME_CHANGED"
    STRATEGY_INVALIDATED = "STRATEGY_INVALIDATED"
    VOLATILITY_CHANGED = "VOLATILITY_CHANGED"
    TIME_LIMIT = "TIME_LIMIT"
    LIQUIDITY_DETERIORATED = "LIQUIDITY_DETERIORATED"
    RISK_LIMIT = "RISK_LIMIT"
    PORTFOLIO_RISK = "PORTFOLIO_RISK"
    END_OF_SESSION = "END_OF_SESSION"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"
    EVENT_RISK = "EVENT_RISK"


class ManagementAction(str, enum.Enum):
    """REQ 34."""

    HOLD = "HOLD"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    MOVE_STOP = "MOVE_STOP"
    TRAIL_STOP = "TRAIL_STOP"
    EXIT = "EXIT"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable contract, normalized away from any broker's field names (REQ 3)."""

    trading_symbol: str
    exchange: Exchange
    segment: Segment
    instrument_type: InstrumentType
    name: str = ""
    underlying_symbol: str = ""
    exchange_token: str = ""
    groww_symbol: str = ""
    isin: str = ""
    series: str = ""
    lot_size: int = 1
    tick_size: float = 0.05
    freeze_quantity: int = 0
    expiry_date: date | None = None
    strike_price: float | None = None
    buy_allowed: bool = True
    sell_allowed: bool = True
    is_reserved: bool = False

    @property
    def is_option(self) -> bool:
        return self.instrument_type in (InstrumentType.CE, InstrumentType.PE)

    @property
    def is_future(self) -> bool:
        return self.instrument_type is InstrumentType.FUT

    @property
    def is_derivative(self) -> bool:
        return self.is_option or self.is_future

    @property
    def option_type(self) -> OptionType | None:
        if self.instrument_type is InstrumentType.CE:
            return OptionType.CE
        if self.instrument_type is InstrumentType.PE:
            return OptionType.PE
        return None

    @property
    def tradable(self) -> bool:
        return self.buy_allowed and self.sell_allowed and not self.is_reserved

    def days_to_expiry(self, as_of: date) -> int | None:
        if self.expiry_date is None:
            return None
        return (self.expiry_date - as_of).days

    @property
    def key(self) -> str:
        return f"{self.exchange.value}:{self.segment.value}:{self.trading_symbol}"


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    open_interest: float = 0.0


@dataclass(frozen=True, slots=True)
class DepthLevel:
    price: float
    quantity: int


@dataclass(slots=True)
class Quote:
    """Normalized snapshot. `received_at` is stamped locally so staleness is
    measurable even when the venue timestamp is missing or wrong (REQ 8)."""

    instrument: Instrument
    last_price: float
    timestamp: datetime
    received_at: datetime
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    previous_close: float | None = None
    volume: float | None = None
    average_price: float | None = None
    bid_price: float | None = None
    bid_quantity: int | None = None
    ask_price: float | None = None
    ask_quantity: int | None = None
    total_buy_quantity: int | None = None
    total_sell_quantity: int | None = None
    open_interest: float | None = None
    previous_open_interest: float | None = None
    oi_day_change: float | None = None
    implied_volatility: float | None = None
    upper_circuit: float | None = None
    lower_circuit: float | None = None
    depth_buy: tuple[DepthLevel, ...] = ()
    depth_sell: tuple[DepthLevel, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def spread(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        if self.bid_price <= 0 or self.ask_price <= 0:
            return None
        return self.ask_price - self.bid_price

    @property
    def mid(self) -> float | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        if self.bid_price <= 0 or self.ask_price <= 0:
            return None
        return (self.bid_price + self.ask_price) / 2.0

    @property
    def spread_pct(self) -> float | None:
        """Spread as a fraction of mid. This is the liquidity metric the option
        selector and risk engine gate on, so it is defined once, here."""
        s, m = self.spread, self.mid
        if s is None or m is None or m <= 0:
            return None
        return s / m

    @property
    def oi_change(self) -> float | None:
        if self.oi_day_change is not None:
            return self.oi_day_change
        if self.open_interest is not None and self.previous_open_interest is not None:
            return self.open_interest - self.previous_open_interest
        return None


@dataclass(frozen=True, slots=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float


@dataclass(slots=True)
class OptionContract:
    """One strike/type row of an option chain with its market state and greeks."""

    instrument: Instrument
    strike: float
    option_type: OptionType
    expiry: date
    quote: Quote | None = None
    greeks: Greeks | None = None

    @property
    def last_price(self) -> float | None:
        return self.quote.last_price if self.quote else None

    def intrinsic_value(self, underlying: float) -> float:
        if self.option_type is OptionType.CE:
            return max(0.0, underlying - self.strike)
        return max(0.0, self.strike - underlying)

    def time_value(self, underlying: float) -> float | None:
        if self.last_price is None:
            return None
        return self.last_price - self.intrinsic_value(underlying)

    def moneyness(self, underlying: float) -> float:
        """Signed log-moneyness in the direction that favours the option.

        Positive = in the money. Using log keeps the measure comparable across
        instruments priced at 200 and at 50,000.
        """
        import math

        if underlying <= 0 or self.strike <= 0:
            return 0.0
        raw = math.log(underlying / self.strike)
        return raw if self.option_type is OptionType.CE else -raw


@dataclass(slots=True)
class OptionChain:
    underlying_symbol: str
    expiry: date
    underlying_ltp: float
    contracts: list[OptionContract] = field(default_factory=list)
    fetched_at: datetime | None = None

    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type is OptionType.CE]

    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.option_type is OptionType.PE]

    def strikes(self) -> list[float]:
        return sorted({c.strike for c in self.contracts})

    def atm_strike(self) -> float | None:
        strikes = self.strikes()
        if not strikes:
            return None
        return min(strikes, key=lambda s: abs(s - self.underlying_ltp))


# --------------------------------------------------------------------------- #
# Orders, fills, positions
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class OrderRequest:
    """What we intend to send. `client_order_id` is the idempotency key — the broker
    layer refuses to send two requests carrying the same one (REQ 45)."""

    instrument: Instrument
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType
    product: Product
    price: float | None = None
    trigger_price: float | None = None
    validity: Validity = Validity.DAY
    client_order_id: str = ""
    tag: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("order quantity must be positive")
        if self.order_type in (OrderType.LIMIT, OrderType.SL) and self.price is None:
            raise ValueError(f"{self.order_type.value} order requires a price")
        if self.order_type in (OrderType.SL, OrderType.SL_M) and self.trigger_price is None:
            raise ValueError(f"{self.order_type.value} order requires a trigger price")


@dataclass(slots=True)
class Order:
    """Broker-acknowledged order state."""

    broker_order_id: str
    instrument: Instrument
    transaction_type: TransactionType
    quantity: int
    order_type: OrderType
    product: Product
    status: OrderStatus
    filled_quantity: int = 0
    remaining_quantity: int = 0
    average_fill_price: float | None = None
    price: float | None = None
    trigger_price: float | None = None
    client_order_id: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    rejection_reason: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status.is_terminal


@dataclass(slots=True)
class Trade:
    trade_id: str
    broker_order_id: str
    instrument: Instrument
    transaction_type: TransactionType
    quantity: int
    price: float
    timestamp: datetime


@dataclass(slots=True)
class Position:
    instrument: Instrument
    quantity: int  # signed: positive long, negative short
    average_price: float
    last_price: float = 0.0
    realized_pnl: float = 0.0
    product: Product = Product.NRML
    opened_at: datetime | None = None

    @property
    def direction(self) -> Direction:
        if self.quantity > 0:
            return Direction.LONG
        if self.quantity < 0:
            return Direction.SHORT
        return Direction.FLAT

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.last_price


@dataclass(slots=True)
class Holding:
    instrument: Instrument
    quantity: int
    average_price: float


@dataclass(slots=True)
class MarginInfo:
    """REQ 23 — account equity comes from the broker, never from a constant."""

    clear_cash: float = 0.0
    net_margin_used: float = 0.0
    collateral_available: float = 0.0
    collateral_used: float = 0.0
    adhoc_margin: float = 0.0
    fno_span_margin: float = 0.0
    fno_exposure_margin: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def available_margin(self) -> float:
        return max(0.0, self.clear_cash + self.collateral_available)

    @property
    def equity(self) -> float:
        return self.clear_cash + self.collateral_available + self.net_margin_used


@dataclass(slots=True)
class MarginRequirement:
    total_requirement: float
    span_required: float = 0.0
    exposure_required: float = 0.0
    option_buy_premium: float = 0.0
    brokerage_and_charges: float = 0.0
