"""BrokerAdapter — the boundary between the platform and any broker (REQ 3).

Everything above this interface speaks in `aqtp.core.types`. No Groww field name,
URL, enum spelling or error code may leak past an implementation of this class.

Capability negotiation: not every broker exposes every endpoint. Rather than having
callers guess, each adapter declares what it supports via `capabilities()`, and
unsupported operations raise `NotSupportedError` instead of silently returning
fabricated data.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import date, datetime

from ..core.errors import BrokerError
from ..core.types import (
    Candle,
    Holding,
    Instrument,
    MarginInfo,
    MarginRequirement,
    OptionChain,
    Order,
    OrderRequest,
    Position,
    Quote,
    Segment,
    Timeframe,
    Trade,
)


class NotSupportedError(BrokerError):
    """The broker does not document this capability. Never fake it (REQ 3/5)."""

    def __init__(self, capability: str) -> None:
        super().__init__(f"broker does not support capability: {capability}", retryable=False)
        self.capability = capability


class Capability(str, enum.Enum):
    AUTHENTICATION = "authentication"
    INSTRUMENTS = "instruments"
    LTP = "ltp"
    OHLC = "ohlc"
    QUOTE = "quote"
    MARKET_DEPTH = "market_depth"
    OPTION_CHAIN = "option_chain"
    GREEKS = "greeks"
    PLACE_ORDER = "place_order"
    MODIFY_ORDER = "modify_order"
    CANCEL_ORDER = "cancel_order"
    ORDER_STATUS = "order_status"
    ORDER_LIST = "order_list"
    TRADE_STATUS = "trade_status"
    POSITIONS = "positions"
    HOLDINGS = "holdings"
    MARGIN = "margin"
    ORDER_MARGIN = "order_margin"
    HISTORICAL = "historical"


class BrokerAdapter(ABC):
    """Abstract broker. Implementations must be safe to call concurrently."""

    name: str = "abstract"

    # -- lifecycle ---------------------------------------------------------- #
    @abstractmethod
    def authenticate(self) -> None:
        """Establish or refresh a session. Idempotent."""

    @abstractmethod
    def is_authenticated(self) -> bool: ...

    def close(self) -> None:
        """Release sockets/sessions. Default is a no-op."""

    # -- capability negotiation --------------------------------------------- #
    @abstractmethod
    def capabilities(self) -> frozenset[Capability]: ...

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities()

    def require(self, capability: Capability) -> None:
        if not self.supports(capability):
            raise NotSupportedError(capability.value)

    # -- reference data ----------------------------------------------------- #
    @abstractmethod
    def fetch_instruments(self, *, force_refresh: bool = False) -> list[Instrument]:
        """Full tradable instrument master (REQ 4)."""

    # -- market data -------------------------------------------------------- #
    @abstractmethod
    def get_ltp(self, instruments: list[Instrument]) -> dict[str, float]:
        """Last traded price keyed by `Instrument.trading_symbol`."""

    @abstractmethod
    def get_ohlc(self, instruments: list[Instrument]) -> dict[str, Candle]: ...

    @abstractmethod
    def get_quote(self, instrument: Instrument) -> Quote:
        """Full snapshot including depth, OI and IV where the venue provides them."""

    def get_option_chain(self, underlying: str, expiry: date, *, exchange: str = "NSE") -> OptionChain:
        raise NotSupportedError(Capability.OPTION_CHAIN.value)

    def get_greeks(self, instrument: Instrument) -> object:
        raise NotSupportedError(Capability.GREEKS.value)

    def get_historical_candles(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        raise NotSupportedError(Capability.HISTORICAL.value)

    # -- orders ------------------------------------------------------------- #
    @abstractmethod
    def place_order(self, request: OrderRequest) -> Order:
        """Submit an order.

        Contract for implementations:
          * Must honour `request.client_order_id` as an idempotency key.
          * On an ambiguous failure (timeout after send, unparsable response) MUST
            raise `OrderStateAmbiguous` rather than reporting failure — the caller
            resolves it by querying, never by resubmitting (REQ 45).
        """

    @abstractmethod
    def modify_order(
        self,
        broker_order_id: str,
        *,
        segment: Segment,
        quantity: int | None = None,
        price: float | None = None,
        trigger_price: float | None = None,
        order_type: str | None = None,
    ) -> Order: ...

    @abstractmethod
    def cancel_order(self, broker_order_id: str, *, segment: Segment) -> Order: ...

    @abstractmethod
    def get_order(self, broker_order_id: str, *, segment: Segment) -> Order: ...

    def get_order_by_reference(self, client_order_id: str, *, segment: Segment) -> Order | None:
        """Look up an order by our own idempotency key.

        This is the primitive that makes ambiguous submits recoverable: after a
        timeout we ask "does an order with my reference exist?" instead of resending.
        Returning None means "definitively not present".
        """
        raise NotSupportedError("order_status_by_reference")

    @abstractmethod
    def list_orders(self) -> list[Order]: ...

    @abstractmethod
    def get_trades(self, broker_order_id: str, *, segment: Segment) -> list[Trade]: ...

    # -- portfolio ---------------------------------------------------------- #
    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_holdings(self) -> list[Holding]: ...

    @abstractmethod
    def get_margin(self) -> MarginInfo: ...

    def get_required_margin(self, requests: list[OrderRequest]) -> MarginRequirement:
        raise NotSupportedError(Capability.ORDER_MARGIN.value)

    # -- health ------------------------------------------------------------- #
    def health_check(self) -> bool:
        """Cheap liveness probe used by the monitoring layer."""
        try:
            self.get_margin()
            return True
        except Exception:
            return False


class FutureBrokerAdapter(BrokerAdapter):
    """Placeholder for the next broker integration (REQ 3).

    It exists so the abstraction is exercised by more than one implementation and
    so adding a broker is a matter of filling in methods, not restructuring the
    application. Every method raises rather than pretending to work.
    """

    name = "future_broker"

    def authenticate(self) -> None:
        raise NotSupportedError("authentication")

    def is_authenticated(self) -> bool:
        return False

    def capabilities(self) -> frozenset[Capability]:
        return frozenset()

    def fetch_instruments(self, *, force_refresh: bool = False) -> list[Instrument]:
        raise NotSupportedError(Capability.INSTRUMENTS.value)

    def get_ltp(self, instruments: list[Instrument]) -> dict[str, float]:
        raise NotSupportedError(Capability.LTP.value)

    def get_ohlc(self, instruments: list[Instrument]) -> dict[str, Candle]:
        raise NotSupportedError(Capability.OHLC.value)

    def get_quote(self, instrument: Instrument) -> Quote:
        raise NotSupportedError(Capability.QUOTE.value)

    def place_order(self, request: OrderRequest) -> Order:
        raise NotSupportedError(Capability.PLACE_ORDER.value)

    def modify_order(self, broker_order_id: str, **kwargs: object) -> Order:  # type: ignore[override]
        raise NotSupportedError(Capability.MODIFY_ORDER.value)

    def cancel_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        raise NotSupportedError(Capability.CANCEL_ORDER.value)

    def get_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        raise NotSupportedError(Capability.ORDER_STATUS.value)

    def list_orders(self) -> list[Order]:
        raise NotSupportedError(Capability.ORDER_LIST.value)

    def get_trades(self, broker_order_id: str, *, segment: Segment) -> list[Trade]:
        raise NotSupportedError(Capability.TRADE_STATUS.value)

    def get_positions(self) -> list[Position]:
        raise NotSupportedError(Capability.POSITIONS.value)

    def get_holdings(self) -> list[Holding]:
        raise NotSupportedError(Capability.HOLDINGS.value)

    def get_margin(self) -> MarginInfo:
        raise NotSupportedError(Capability.MARGIN.value)
