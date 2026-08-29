"""Groww Trading API adapter (REQ 4).

Every endpoint below is taken from the official Groww Trading API documentation.
No endpoint is invented. Where Groww documents no equivalent for something the
platform would like (e.g. a streaming feed contract, commodity segment), the
adapter raises `NotSupportedError` rather than guessing a URL.

Endpoint reference used:
  POST /v1/token/api/access                                     auth
  POST /v1/order/create | /modify | /cancel                     orders
  GET  /v1/order/detail/{id} | /status/{id}                     orders
  GET  /v1/order/status/reference/{reference_id}                orders (idempotency)
  GET  /v1/order/list                                           orders
  GET  /v1/order/trades/{id}                                    trades
  GET  /v1/live-data/quote | /ltp | /ohlc                       live data
  GET  /v1/option-chain/exchange/{ex}/underlying/{u}            option chain
  GET  /v1/live-data/greeks/...                                 greeks
  GET  /v1/positions/user | /positions/trading-symbol           portfolio
  GET  /v1/holdings/user                                        portfolio
  GET  /v1/margins/detail/user                                  margin
  POST /v1/margins/detail/orders                                margin
  GET  /v1/historical/candle/range                              historical
  GET  https://growwapi-assets.groww.in/instruments/instrument.csv   instruments
"""

from __future__ import annotations

import csv
import hashlib
import io
import time as _time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from ..core.clock import IST, to_ist
from ..core.errors import AuthenticationError, BrokerError, OrderStateAmbiguous
from ..core.logging import get_logger
from ..core.types import (
    Candle,
    DepthLevel,
    Exchange,
    Greeks,
    Holding,
    Instrument,
    InstrumentType,
    MarginInfo,
    MarginRequirement,
    OptionChain,
    OptionContract,
    OptionType,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    Segment,
    Timeframe,
    Trade,
    TransactionType,
)
from .base import BrokerAdapter, Capability, NotSupportedError
from .rest import IdempotencyGuard, RateCategory, RateLimiter, RestClient

logger = get_logger(__name__)

INSTRUMENTS_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"

# Documented per-category rate limits (requests/second, requests/minute).
_DEFAULT_LIMITS = {
    RateCategory.AUTH: (5, 30),
    RateCategory.ORDERS: (10, 250),
    RateCategory.LIVE_DATA: (10, 300),
    RateCategory.NON_TRADING: (20, 500),
}

# Groww documents interval_in_minutes; these are the values it accepts.
_TIMEFRAME_MINUTES = {
    Timeframe.M1: 1,
    Timeframe.M3: 3,
    Timeframe.M5: 5,
    Timeframe.M10: 10,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
}

# Documented history limits per interval: (max span per request, max lookback).
_HISTORY_LIMITS = {
    1: (timedelta(days=7), timedelta(days=90)),
    5: (timedelta(days=15), timedelta(days=90)),
    1440: (timedelta(days=1080), timedelta(days=36500)),
}


def _parse_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value or value.lower() in ("nan", "none", "null"):
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "NaN"):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y"):
        return True
    if text in ("false", "0", "no", "n"):
        return False
    return default


class GrowwAdapter(BrokerAdapter):
    """Concrete adapter for the Groww Trading API."""

    name = "groww"

    def __init__(
        self,
        *,
        access_token: str = "",
        api_key: str = "",
        api_secret: str = "",
        totp_secret: str = "",
        auth_mode: str = "token",
        base_url: str = "https://api.groww.in",
        instruments_url: str = INSTRUMENTS_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        api_version: str = "1.0",
        cache_dir: str | Path = "data/cache",
        session: requests.Session | None = None,
        rate_limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self._access_token = access_token
        self._api_key = api_key
        self._api_secret = api_secret
        self._totp_secret = totp_secret
        self._auth_mode = auth_mode
        self._token_expiry: datetime | None = None
        self.instruments_url = instruments_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        limits = dict(_DEFAULT_LIMITS)
        if rate_limits:
            limits.update(rate_limits)  # type: ignore[arg-type]
        limiters = {cat: RateLimiter(per_second=ps, per_minute=pm) for cat, (ps, pm) in limits.items()}

        self.client = RestClient(
            base_url,
            timeout=timeout,
            max_retries=max_retries,
            backoff=backoff,
            limiters=limiters,
            session=session,
            api_version=api_version,
        )
        self.idempotency = IdempotencyGuard()
        self._instruments_cache: list[Instrument] | None = None
        self._instrument_by_symbol: dict[str, Instrument] = {}

    # ------------------------------------------------------------------ #
    # Authentication
    # ------------------------------------------------------------------ #
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.AUTHENTICATION,
                Capability.INSTRUMENTS,
                Capability.LTP,
                Capability.OHLC,
                Capability.QUOTE,
                Capability.MARKET_DEPTH,
                Capability.OPTION_CHAIN,
                Capability.GREEKS,
                Capability.PLACE_ORDER,
                Capability.MODIFY_ORDER,
                Capability.CANCEL_ORDER,
                Capability.ORDER_STATUS,
                Capability.ORDER_LIST,
                Capability.TRADE_STATUS,
                Capability.POSITIONS,
                Capability.HOLDINGS,
                Capability.MARGIN,
                Capability.ORDER_MARGIN,
                Capability.HISTORICAL,
            }
        )

    def authenticate(self) -> None:
        """Obtain a session token according to the configured flow."""
        if self._auth_mode == "token":
            if not self._access_token:
                raise AuthenticationError("no GROWW_ACCESS_TOKEN provided")
            self.client.set_token(self._access_token)
            # Documented behaviour: daily tokens expire at 06:00 IST.
            self._token_expiry = self._next_0600_ist()
            return

        if self._auth_mode == "approval":
            payload = self._approval_payload()
        elif self._auth_mode == "totp":
            payload = {"key_type": "totp", "totp": self._current_totp()}
        else:
            raise AuthenticationError(f"unknown auth mode {self._auth_mode!r}")

        result = self.client.request(
            "POST",
            "/v1/token/api/access",
            category=RateCategory.AUTH,
            json_body=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        token = result.get("token") or result.get("access_token")
        if not token:
            raise AuthenticationError("token endpoint returned no token")
        self.client.set_token(token)
        self._access_token = token
        self._token_expiry = self._next_0600_ist()
        logger.info("groww authentication succeeded via %s flow", self._auth_mode)

    def _approval_payload(self) -> dict[str, Any]:
        """Checksum = SHA256(api_secret + epoch_seconds), per the documentation."""
        epoch = str(int(_time.time()))
        checksum = hashlib.sha256(f"{self._api_secret}{epoch}".encode()).hexdigest()
        return {"key_type": "approval", "checksum": checksum, "timestamp": epoch}

    def _current_totp(self) -> str:
        try:
            import pyotp  # optional dependency, only needed for the TOTP flow
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AuthenticationError(
                "auth_mode=totp requires the 'pyotp' package (pip install pyotp)"
            ) from exc
        return pyotp.TOTP(self._totp_secret).now()

    @staticmethod
    def _next_0600_ist() -> datetime:
        now = datetime.now(IST)
        expiry = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now >= expiry:
            expiry += timedelta(days=1)
        return expiry

    def is_authenticated(self) -> bool:
        if not self.client.has_token:
            return False
        if self._token_expiry and datetime.now(IST) >= self._token_expiry:
            return False
        return True

    def _ensure_auth(self) -> None:
        if not self.is_authenticated():
            self.authenticate()

    def close(self) -> None:
        self.client.close()

    # ------------------------------------------------------------------ #
    # Instruments (REQ 4/6)
    # ------------------------------------------------------------------ #
    def fetch_instruments(self, *, force_refresh: bool = False) -> list[Instrument]:
        cache_file = self.cache_dir / "instruments.csv"
        stale = True
        if cache_file.exists():
            age = _time.time() - cache_file.stat().st_mtime
            stale = age > 6 * 3600
        if force_refresh or stale or not cache_file.exists():
            try:
                response = requests.get(self.instruments_url, timeout=60)
                response.raise_for_status()
                cache_file.write_bytes(response.content)
                logger.info("downloaded instrument master (%d bytes)", len(response.content))
            except requests.RequestException as exc:
                if not cache_file.exists():
                    raise BrokerError(f"could not download instrument master: {exc}") from exc
                logger.warning("instrument master download failed, using cached copy: %s", exc)

        text = cache_file.read_text(errors="replace")
        instruments = self.parse_instruments_csv(text)
        self._instruments_cache = instruments
        self._instrument_by_symbol = {i.trading_symbol: i for i in instruments}
        return instruments

    @staticmethod
    def parse_instruments_csv(text: str) -> list[Instrument]:
        """Parse the documented instrument CSV into normalized Instruments.

        Unknown instrument types are skipped rather than coerced — an unrecognized
        product must not silently become tradable (REQ 5).
        """
        reader = csv.DictReader(io.StringIO(text))
        out: list[Instrument] = []
        for row in reader:
            raw_type = (row.get("instrument_type") or "").strip().upper()
            try:
                instrument_type = InstrumentType(raw_type)
            except ValueError:
                continue
            try:
                exchange = Exchange((row.get("exchange") or "NSE").strip().upper())
                segment = Segment((row.get("segment") or "CASH").strip().upper())
            except ValueError:
                continue

            strike = _to_float(row.get("strike_price"), 0.0)
            out.append(
                Instrument(
                    trading_symbol=(row.get("trading_symbol") or "").strip(),
                    exchange=exchange,
                    segment=segment,
                    instrument_type=instrument_type,
                    name=(row.get("name") or "").strip(),
                    underlying_symbol=(row.get("underlying_symbol") or "").strip(),
                    exchange_token=(row.get("exchange_token") or "").strip(),
                    groww_symbol=(row.get("groww_symbol") or "").strip(),
                    isin=(row.get("isin") or "").strip(),
                    series=(row.get("series") or "").strip(),
                    lot_size=max(1, _to_int(row.get("lot_size"), 1)),
                    tick_size=_to_float(row.get("tick_size"), 0.05) or 0.05,
                    freeze_quantity=_to_int(row.get("freeze_quantity"), 0),
                    expiry_date=_parse_date(row.get("expiry_date") or ""),
                    strike_price=strike if strike > 0 else None,
                    buy_allowed=_to_bool(row.get("buy_allowed"), True),
                    sell_allowed=_to_bool(row.get("sell_allowed"), True),
                    is_reserved=_to_bool(row.get("is_reserved"), False),
                )
            )
        return [i for i in out if i.trading_symbol]

    def lookup(self, trading_symbol: str) -> Instrument | None:
        if not self._instrument_by_symbol:
            self.fetch_instruments()
        return self._instrument_by_symbol.get(trading_symbol)

    # ------------------------------------------------------------------ #
    # Market data (REQ 7)
    # ------------------------------------------------------------------ #
    def _exchange_symbols(self, instruments: Iterable[Instrument]) -> list[str]:
        """Groww's batch endpoints key instruments as ``EXCHANGE_SYMBOL``."""
        return [f"{i.exchange.value}_{i.trading_symbol}" for i in instruments]

    def get_ltp(self, instruments: list[Instrument]) -> dict[str, float]:
        if not instruments:
            return {}
        self._ensure_auth()
        out: dict[str, float] = {}
        # Documented cap: 50 instruments per request.
        for chunk in _chunks(instruments, 50):
            segment = chunk[0].segment.value
            payload = self.client.request(
                "GET",
                "/v1/live-data/ltp",
                category=RateCategory.LIVE_DATA,
                params={
                    "segment": segment,
                    "exchange_symbols": ",".join(self._exchange_symbols(chunk)),
                },
            )
            for key, value in (payload or {}).items():
                out[_strip_exchange_prefix(key)] = _to_float(value)
        return out

    def get_ohlc(self, instruments: list[Instrument]) -> dict[str, Candle]:
        if not instruments:
            return {}
        self._ensure_auth()
        now = datetime.now(IST)
        out: dict[str, Candle] = {}
        for chunk in _chunks(instruments, 50):
            segment = chunk[0].segment.value
            payload = self.client.request(
                "GET",
                "/v1/live-data/ohlc",
                category=RateCategory.LIVE_DATA,
                params={
                    "segment": segment,
                    "exchange_symbols": ",".join(self._exchange_symbols(chunk)),
                },
            )
            for key, value in (payload or {}).items():
                if not isinstance(value, dict):
                    continue
                out[_strip_exchange_prefix(key)] = Candle(
                    timestamp=now,
                    open=_to_float(value.get("open")),
                    high=_to_float(value.get("high")),
                    low=_to_float(value.get("low")),
                    close=_to_float(value.get("close")),
                )
        return out

    def get_quote(self, instrument: Instrument) -> Quote:
        self._ensure_auth()
        payload = self.client.request(
            "GET",
            "/v1/live-data/quote",
            category=RateCategory.LIVE_DATA,
            params={
                "exchange": instrument.exchange.value,
                "segment": instrument.segment.value,
                "trading_symbol": instrument.trading_symbol,
            },
        )
        return self._parse_quote(instrument, payload or {})

    @staticmethod
    def _parse_quote(instrument: Instrument, payload: Mapping[str, Any]) -> Quote:
        received_at = datetime.now(IST)
        ohlc = payload.get("ohlc") or {}
        depth = payload.get("depth") or {}

        def levels(side: str) -> tuple[DepthLevel, ...]:
            rows = depth.get(side) or []
            return tuple(
                DepthLevel(price=_to_float(r.get("price")), quantity=_to_int(r.get("quantity")))
                for r in rows
                if isinstance(r, dict)
            )

        last_trade_time = payload.get("last_trade_time")
        if isinstance(last_trade_time, (int, float)) and last_trade_time > 0:
            # Groww reports epoch milliseconds for this field.
            seconds = last_trade_time / 1000 if last_trade_time > 1e11 else last_trade_time
            timestamp = datetime.fromtimestamp(seconds, IST)
        else:
            timestamp = received_at

        bid = _to_float(payload.get("bid_price"), 0.0)
        ask = _to_float(payload.get("offer_price"), 0.0)
        buy_levels, sell_levels = levels("buy"), levels("sell")
        if bid <= 0 and buy_levels:
            bid = buy_levels[0].price
        if ask <= 0 and sell_levels:
            ask = sell_levels[0].price

        return Quote(
            instrument=instrument,
            last_price=_to_float(payload.get("last_price")),
            timestamp=timestamp,
            received_at=received_at,
            open=_to_float(ohlc.get("open")) or None,
            high=_to_float(ohlc.get("high")) or None,
            low=_to_float(ohlc.get("low")) or None,
            close=_to_float(ohlc.get("close")) or None,
            previous_close=_to_float(ohlc.get("close")) or None,
            volume=_to_float(payload.get("volume")) or None,
            average_price=_to_float(payload.get("average_price")) or None,
            bid_price=bid or None,
            bid_quantity=_to_int(payload.get("bid_quantity")) or None,
            ask_price=ask or None,
            ask_quantity=_to_int(payload.get("offer_quantity")) or None,
            total_buy_quantity=_to_int(payload.get("total_buy_quantity")) or None,
            total_sell_quantity=_to_int(payload.get("total_sell_quantity")) or None,
            open_interest=_to_float(payload.get("open_interest")) or None,
            previous_open_interest=_to_float(payload.get("previous_open_interest")) or None,
            oi_day_change=_to_float(payload.get("oi_day_change")) or None,
            implied_volatility=_to_float(payload.get("implied_volatility")) or None,
            upper_circuit=_to_float(payload.get("upper_circuit_limit")) or None,
            lower_circuit=_to_float(payload.get("lower_circuit_limit")) or None,
            depth_buy=buy_levels,
            depth_sell=sell_levels,
            raw=dict(payload),
        )

    def get_option_chain(self, underlying: str, expiry: date, *, exchange: str = "NSE") -> OptionChain:
        self._ensure_auth()
        payload = self.client.request(
            "GET",
            f"/v1/option-chain/exchange/{exchange}/underlying/{underlying}",
            category=RateCategory.LIVE_DATA,
            params={"expiry_date": expiry.isoformat()},
        )
        return self._parse_option_chain(underlying, expiry, exchange, payload or {})

    def _parse_option_chain(
        self, underlying: str, expiry: date, exchange: str, payload: Mapping[str, Any]
    ) -> OptionChain:
        underlying_ltp = _to_float(payload.get("underlying_ltp"))
        received_at = datetime.now(IST)
        contracts: list[OptionContract] = []

        strikes = payload.get("strikes") or {}
        # The documented shape is {strike: {"CE": {...}, "PE": {...}}}; accept a
        # list-of-rows shape too so a documented change does not crash the loop.
        rows: list[tuple[float, str, Mapping[str, Any]]] = []
        if isinstance(strikes, Mapping):
            for strike_key, sides in strikes.items():
                if not isinstance(sides, Mapping):
                    continue
                for side, data in sides.items():
                    if isinstance(data, Mapping):
                        rows.append((_to_float(strike_key), str(side).upper(), data))
        elif isinstance(strikes, list):
            for entry in strikes:
                if not isinstance(entry, Mapping):
                    continue
                strike = _to_float(entry.get("strike_price") or entry.get("strike"))
                for side in ("CE", "PE"):
                    data = entry.get(side.lower()) or entry.get(side)
                    if isinstance(data, Mapping):
                        rows.append((strike, side, data))

        for strike, side, data in rows:
            if side not in ("CE", "PE") or strike <= 0:
                continue
            option_type = OptionType(side)
            symbol = str(data.get("trading_symbol") or "").strip()
            instrument = self._instrument_by_symbol.get(symbol) or Instrument(
                trading_symbol=symbol,
                exchange=Exchange(exchange),
                segment=Segment.FNO,
                instrument_type=InstrumentType(side),
                underlying_symbol=underlying,
                expiry_date=expiry,
                strike_price=strike,
                lot_size=_to_int(data.get("lot_size"), 1) or 1,
            )
            greeks_raw = data.get("greeks") or {}
            greeks = None
            if isinstance(greeks_raw, Mapping) and greeks_raw:
                greeks = Greeks(
                    delta=_to_float(greeks_raw.get("delta")),
                    gamma=_to_float(greeks_raw.get("gamma")),
                    theta=_to_float(greeks_raw.get("theta")),
                    vega=_to_float(greeks_raw.get("vega")),
                    rho=_to_float(greeks_raw.get("rho")),
                    iv=_to_float(greeks_raw.get("iv") or greeks_raw.get("implied_volatility")),
                )
            quote = Quote(
                instrument=instrument,
                last_price=_to_float(data.get("ltp")),
                timestamp=received_at,
                received_at=received_at,
                bid_price=_to_float(data.get("bid_price")) or None,
                ask_price=_to_float(data.get("offer_price") or data.get("ask_price")) or None,
                volume=_to_float(data.get("volume")) or None,
                open_interest=_to_float(data.get("open_interest")) or None,
                previous_open_interest=_to_float(data.get("previous_open_interest")) or None,
                implied_volatility=greeks.iv if greeks else None,
                raw=dict(data),
            )
            contracts.append(
                OptionContract(
                    instrument=instrument,
                    strike=strike,
                    option_type=option_type,
                    expiry=expiry,
                    quote=quote,
                    greeks=greeks,
                )
            )

        return OptionChain(
            underlying_symbol=underlying,
            expiry=expiry,
            underlying_ltp=underlying_ltp,
            contracts=contracts,
            fetched_at=received_at,
        )

    def get_greeks(self, instrument: Instrument) -> Greeks:
        if not instrument.is_option or instrument.expiry_date is None:
            raise NotSupportedError("greeks are only defined for option contracts")
        self._ensure_auth()
        payload = self.client.request(
            "GET",
            f"/v1/live-data/greeks/exchange/{instrument.exchange.value}"
            f"/underlying/{instrument.underlying_symbol}"
            f"/trading_symbol/{instrument.trading_symbol}"
            f"/expiry/{instrument.expiry_date.isoformat()}",
            category=RateCategory.LIVE_DATA,
        )
        payload = payload or {}
        return Greeks(
            delta=_to_float(payload.get("delta")),
            gamma=_to_float(payload.get("gamma")),
            theta=_to_float(payload.get("theta")),
            vega=_to_float(payload.get("vega")),
            rho=_to_float(payload.get("rho")),
            iv=_to_float(payload.get("iv") or payload.get("implied_volatility")),
        )

    def get_historical_candles(
        self,
        instrument: Instrument,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        minutes = _TIMEFRAME_MINUTES.get(timeframe)
        if minutes is None:
            raise NotSupportedError(f"historical candles for timeframe {timeframe.value}")
        self._ensure_auth()

        max_span, max_lookback = _closest_history_limit(minutes)
        earliest = to_ist(end) - max_lookback
        cursor = max(to_ist(start), earliest)
        end = to_ist(end)
        candles: list[Candle] = []

        # Chunk the request to respect the documented per-request span limits.
        while cursor < end:
            window_end = min(cursor + max_span, end)
            payload = self.client.request(
                "GET",
                "/v1/historical/candle/range",
                category=RateCategory.NON_TRADING,
                params={
                    "exchange": instrument.exchange.value,
                    "segment": instrument.segment.value,
                    "trading_symbol": instrument.trading_symbol,
                    "start_time": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": window_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "interval_in_minutes": minutes,
                },
            )
            candles.extend(self._parse_candles(payload or {}))
            cursor = window_end

        # De-duplicate on timestamp; overlapping chunk boundaries are expected.
        seen: set[datetime] = set()
        unique: list[Candle] = []
        for candle in sorted(candles, key=lambda c: c.timestamp):
            if candle.timestamp not in seen:
                seen.add(candle.timestamp)
                unique.append(candle)
        return unique

    @staticmethod
    def _parse_candles(payload: Mapping[str, Any]) -> list[Candle]:
        rows = payload.get("candles") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            return []
        out: list[Candle] = []
        for row in rows:
            # Documented as [timestamp, open, high, low, close, volume].
            if isinstance(row, (list, tuple)) and len(row) >= 5:
                ts_raw = row[0]
                seconds = float(ts_raw) / 1000 if float(ts_raw) > 1e11 else float(ts_raw)
                out.append(
                    Candle(
                        timestamp=datetime.fromtimestamp(seconds, IST),
                        open=_to_float(row[1]),
                        high=_to_float(row[2]),
                        low=_to_float(row[3]),
                        close=_to_float(row[4]),
                        volume=_to_float(row[5]) if len(row) > 5 else 0.0,
                    )
                )
            elif isinstance(row, Mapping):
                ts_raw = row.get("timestamp") or row.get("time")
                if ts_raw is None:
                    continue
                seconds = float(ts_raw) / 1000 if float(ts_raw) > 1e11 else float(ts_raw)
                out.append(
                    Candle(
                        timestamp=datetime.fromtimestamp(seconds, IST),
                        open=_to_float(row.get("open")),
                        high=_to_float(row.get("high")),
                        low=_to_float(row.get("low")),
                        close=_to_float(row.get("close")),
                        volume=_to_float(row.get("volume")),
                    )
                )
        return out

    # ------------------------------------------------------------------ #
    # Orders (REQ 31/32/45)
    # ------------------------------------------------------------------ #
    def place_order(self, request: OrderRequest) -> Order:
        self._ensure_auth()
        if not request.client_order_id:
            raise ValueError("client_order_id (idempotency key) is required")

        # Reserve first: if this key was used before, we refuse without a network call.
        self.idempotency.reserve(request.client_order_id)

        body: dict[str, Any] = {
            "trading_symbol": request.instrument.trading_symbol,
            "quantity": request.quantity,
            "exchange": request.instrument.exchange.value,
            "segment": request.instrument.segment.value,
            "product": request.product.value,
            "order_type": request.order_type.value,
            "transaction_type": request.transaction_type.value,
            "validity": request.validity.value,
            "order_reference_id": request.client_order_id,
        }
        if request.price is not None:
            body["price"] = round(request.price, 2)
        if request.trigger_price is not None:
            body["trigger_price"] = round(request.trigger_price, 2)

        try:
            payload = self.client.request(
                "POST",
                "/v1/order/create",
                category=RateCategory.ORDERS,
                json_body=body,
                idempotent=False,  # never retried — see rest.RestClient
            )
        except OrderStateAmbiguous as exc:
            # We do NOT release the idempotency reservation: the order may exist.
            logger.error(
                "order submission ambiguous for %s; resolving by reference lookup",
                request.client_order_id,
            )
            resolved = self.get_order_by_reference(
                request.client_order_id, segment=request.instrument.segment
            )
            if resolved is not None:
                return resolved
            raise OrderStateAmbiguous(
                f"{exc}. Reference lookup found no order, but existence is still not "
                "guaranteed; operator confirmation required before resubmitting.",
                client_order_id=request.client_order_id,
            ) from exc
        except Exception:
            # A definitive rejection means nothing reached the book; free the key
            # so a corrected order can reuse it.
            self.idempotency.release(request.client_order_id)
            raise

        return self._parse_order(payload or {}, fallback_instrument=request.instrument)

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
        self._ensure_auth()
        body: dict[str, Any] = {"groww_order_id": broker_order_id, "segment": segment.value}
        if quantity is not None:
            body["quantity"] = quantity
        if price is not None:
            body["price"] = round(price, 2)
        if trigger_price is not None:
            body["trigger_price"] = round(trigger_price, 2)
        if order_type is not None:
            body["order_type"] = order_type
        payload = self.client.request(
            "POST",
            "/v1/order/modify",
            category=RateCategory.ORDERS,
            json_body=body,
            idempotent=False,
        )
        return self._parse_order(payload or {})

    def cancel_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        self._ensure_auth()
        payload = self.client.request(
            "POST",
            "/v1/order/cancel",
            category=RateCategory.ORDERS,
            json_body={"groww_order_id": broker_order_id, "segment": segment.value},
            idempotent=False,
        )
        return self._parse_order(payload or {})

    def get_order(self, broker_order_id: str, *, segment: Segment) -> Order:
        self._ensure_auth()
        payload = self.client.request(
            "GET",
            f"/v1/order/detail/{broker_order_id}",
            category=RateCategory.NON_TRADING,
            params={"segment": segment.value},
        )
        return self._parse_order(payload or {})

    def get_order_by_reference(self, client_order_id: str, *, segment: Segment) -> Order | None:
        """Resolve our idempotency key to a broker order, if one exists (REQ 45)."""
        self._ensure_auth()
        try:
            payload = self.client.request(
                "GET",
                f"/v1/order/status/reference/{client_order_id}",
                category=RateCategory.NON_TRADING,
                params={"segment": segment.value},
            )
        except BrokerError as exc:
            # A 404-style failure is a definitive "no such order"; anything else
            # leaves the question open and must propagate.
            if str(getattr(exc, "code", "")) in ("404", "GA004") or "not found" in str(exc).lower():
                return None
            raise
        if not payload:
            return None
        return self._parse_order(payload)

    def list_orders(self) -> list[Order]:
        self._ensure_auth()
        payload = self.client.request(
            "GET", "/v1/order/list", category=RateCategory.NON_TRADING
        )
        rows = _rows_of(payload, "order_list", "orders")
        return [self._parse_order(r) for r in rows]

    def get_trades(self, broker_order_id: str, *, segment: Segment) -> list[Trade]:
        self._ensure_auth()
        payload = self.client.request(
            "GET",
            f"/v1/order/trades/{broker_order_id}",
            category=RateCategory.NON_TRADING,
            params={"segment": segment.value},
        )
        rows = _rows_of(payload, "trade_list", "trades")
        out: list[Trade] = []
        for row in rows:
            symbol = str(row.get("trading_symbol", ""))
            instrument = self._instrument_by_symbol.get(symbol) or _unknown_instrument(symbol, segment)
            ts = row.get("trade_date_time") or row.get("timestamp")
            out.append(
                Trade(
                    trade_id=str(row.get("trade_id") or row.get("exchange_trade_id") or ""),
                    broker_order_id=broker_order_id,
                    instrument=instrument,
                    transaction_type=TransactionType(
                        str(row.get("transaction_type", "BUY")).upper()
                    ),
                    quantity=_to_int(row.get("quantity")),
                    price=_to_float(row.get("price")),
                    timestamp=_parse_timestamp(ts),
                )
            )
        return out

    def _parse_order(
        self, payload: Mapping[str, Any], *, fallback_instrument: Instrument | None = None
    ) -> Order:
        symbol = str(payload.get("trading_symbol") or "")
        instrument = (
            self._instrument_by_symbol.get(symbol)
            or fallback_instrument
            or _unknown_instrument(symbol, Segment(str(payload.get("segment", "FNO")).upper()))
        )
        raw_status = str(payload.get("order_status") or payload.get("status") or "").upper()
        try:
            status = OrderStatus(raw_status)
        except ValueError:
            status = OrderStatus.UNKNOWN

        quantity = _to_int(payload.get("quantity"))
        filled = _to_int(payload.get("filled_quantity"))
        remaining = _to_int(payload.get("remaining_quantity"), max(0, quantity - filled))

        return Order(
            broker_order_id=str(payload.get("groww_order_id") or payload.get("order_id") or ""),
            instrument=instrument,
            transaction_type=TransactionType(str(payload.get("transaction_type", "BUY")).upper()),
            quantity=quantity,
            order_type=_safe_order_type(payload.get("order_type")),
            product=_safe_product(payload.get("product")),
            status=status,
            filled_quantity=filled,
            remaining_quantity=remaining,
            average_fill_price=_to_float(payload.get("average_fill_price")) or None,
            price=_to_float(payload.get("price")) or None,
            trigger_price=_to_float(payload.get("trigger_price")) or None,
            client_order_id=str(payload.get("order_reference_id") or ""),
            created_at=_parse_timestamp(payload.get("created_at") or payload.get("order_time")),
            updated_at=_parse_timestamp(payload.get("updated_at")),
            rejection_reason=str(payload.get("remark") or payload.get("rejection_reason") or ""),
            raw=dict(payload),
        )

    # ------------------------------------------------------------------ #
    # Portfolio & margin (REQ 23)
    # ------------------------------------------------------------------ #
    def get_positions(self) -> list[Position]:
        self._ensure_auth()
        payload = self.client.request(
            "GET", "/v1/positions/user", category=RateCategory.NON_TRADING
        )
        rows = _rows_of(payload, "positions", "position_list")
        out: list[Position] = []
        for row in rows:
            symbol = str(row.get("trading_symbol", ""))
            segment = Segment(str(row.get("segment", "FNO")).upper())
            instrument = self._instrument_by_symbol.get(symbol) or _unknown_instrument(symbol, segment)

            # Groww reports credit/debit legs separately (intraday + carry forward).
            credit_qty = _to_int(row.get("credit_quantity")) + _to_int(row.get("cf_credit_quantity"))
            debit_qty = _to_int(row.get("debit_quantity")) + _to_int(row.get("cf_debit_quantity"))
            net_qty = credit_qty - debit_qty
            if net_qty == 0:
                continue

            credit_price = _to_float(row.get("credit_price")) + _to_float(row.get("cf_credit_price"))
            debit_price = _to_float(row.get("debit_price")) + _to_float(row.get("cf_debit_price"))
            # Average entry price of the *open* side.
            if net_qty > 0:
                avg = credit_price / credit_qty if credit_qty else 0.0
            else:
                avg = debit_price / debit_qty if debit_qty else 0.0

            out.append(
                Position(
                    instrument=instrument,
                    quantity=net_qty,
                    average_price=avg,
                    realized_pnl=_to_float(row.get("net_carry_forward_price"))
                    if "net_carry_forward_price" in row
                    else 0.0,
                    product=_safe_product(row.get("product")),
                )
            )
        return out

    def get_holdings(self) -> list[Holding]:
        self._ensure_auth()
        payload = self.client.request(
            "GET", "/v1/holdings/user", category=RateCategory.NON_TRADING
        )
        rows = _rows_of(payload, "holdings", "holding_list")
        out: list[Holding] = []
        for row in rows:
            symbol = str(row.get("trading_symbol", ""))
            instrument = self._instrument_by_symbol.get(symbol) or _unknown_instrument(
                symbol, Segment.CASH
            )
            out.append(
                Holding(
                    instrument=instrument,
                    quantity=_to_int(row.get("quantity")),
                    average_price=_to_float(row.get("average_price")),
                )
            )
        return out

    def get_margin(self) -> MarginInfo:
        self._ensure_auth()
        payload = self.client.request(
            "GET", "/v1/margins/detail/user", category=RateCategory.NON_TRADING
        ) or {}
        fno = payload.get("fno_margin_details") or {}
        return MarginInfo(
            clear_cash=_to_float(payload.get("clear_cash")),
            net_margin_used=_to_float(payload.get("net_margin_used")),
            collateral_available=_to_float(payload.get("collateral_available")),
            collateral_used=_to_float(payload.get("collateral_used")),
            adhoc_margin=_to_float(payload.get("adhoc_margin")),
            fno_span_margin=_to_float(fno.get("span_margin_required")),
            fno_exposure_margin=_to_float(fno.get("exposure_margin_required")),
            raw=dict(payload),
        )

    def get_required_margin(self, requests_: list[OrderRequest]) -> MarginRequirement:
        if not requests_:
            return MarginRequirement(total_requirement=0.0)
        self._ensure_auth()
        segment = requests_[0].instrument.segment
        body = [
            {
                "trading_symbol": r.instrument.trading_symbol,
                "quantity": r.quantity,
                "exchange": r.instrument.exchange.value,
                "segment": r.instrument.segment.value,
                "product": r.product.value,
                "order_type": r.order_type.value,
                "transaction_type": r.transaction_type.value,
                **({"price": round(r.price, 2)} if r.price is not None else {}),
            }
            for r in requests_
        ]
        payload = self.client.request(
            "POST",
            "/v1/margins/detail/orders",
            category=RateCategory.NON_TRADING,
            params={"segment": segment.value},
            json_body=body,  # type: ignore[arg-type]
        ) or {}
        return MarginRequirement(
            total_requirement=_to_float(payload.get("total_requirement")),
            span_required=_to_float(payload.get("span_required")),
            exposure_required=_to_float(payload.get("exposure_required")),
            option_buy_premium=_to_float(payload.get("option_buy_premium")),
            brokerage_and_charges=_to_float(payload.get("brokerage_and_charges")),
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _chunks(items: list[Instrument], size: int) -> Iterable[list[Instrument]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _strip_exchange_prefix(key: str) -> str:
    """``NSE_NIFTY24AUG24000CE`` -> ``NIFTY24AUG24000CE``."""
    for prefix in ("NSE_", "BSE_"):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _rows_of(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """Pull a list of rows out of a payload that may be a list or a wrapper dict."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, Mapping):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _unknown_instrument(symbol: str, segment: Segment) -> Instrument:
    """Minimal placeholder for a symbol absent from the local master.

    Deliberately marked non-tradable: reconciliation can see the position, but the
    strategy layer must not act on an instrument whose lot size we do not know.
    """
    return Instrument(
        trading_symbol=symbol,
        exchange=Exchange.NSE,
        segment=segment,
        instrument_type=InstrumentType.EQ if segment is Segment.CASH else InstrumentType.FUT,
        name=symbol,
        buy_allowed=False,
        sell_allowed=False,
    )


def _safe_order_type(value: Any) -> OrderType:
    try:
        return OrderType(str(value).upper())
    except (ValueError, AttributeError):
        return OrderType.MARKET


def _safe_product(value: Any) -> Product:
    try:
        return Product(str(value).upper())
    except (ValueError, AttributeError):
        return Product.NRML


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        return datetime.fromtimestamp(seconds, IST)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%d-%m-%Y %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt).replace(tzinfo=IST)
        except ValueError:
            continue
    return None


def _closest_history_limit(minutes: int) -> tuple[timedelta, timedelta]:
    """Pick the documented span/lookback limits for the nearest documented interval."""
    if minutes in _HISTORY_LIMITS:
        return _HISTORY_LIMITS[minutes]
    if minutes >= 1440:
        return _HISTORY_LIMITS[1440]
    if minutes <= 1:
        return _HISTORY_LIMITS[1]
    return _HISTORY_LIMITS[5]
