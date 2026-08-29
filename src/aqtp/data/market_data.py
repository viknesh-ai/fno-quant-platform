"""MarketDataEngine — the normalized market-data layer (REQ 7/8).

Responsibilities:
  * fetch quotes / chains / candles through the BrokerAdapter,
  * validate everything through the DataQualityGate before it is stored,
  * maintain multi-timeframe candle stores per instrument,
  * expose a `MarketSnapshot` — the single consistent view a decision is made from.

Nothing downstream ever calls the broker directly. If data is unusable the snapshot
says so, and the orchestrator turns that into NO_TRADE with a recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from ..brokers.base import BrokerAdapter, Capability, NotSupportedError
from ..configuration.schema import AppConfig
from ..core.clock import Clock, LiveClock, minutes_since_open, minutes_until_close, to_ist
from ..core.errors import BrokerError
from ..core.logging import get_logger
from ..core.types import (
    Candle,
    Greeks,
    Instrument,
    OptionChain,
    Quote,
    Timeframe,
)
from .candles import MultiTimeframeStore, session_vwap
from .quality import DataQualityGate, QualityVerdict

logger = get_logger(__name__)


@dataclass
class MarketContext:
    """REQ 7 'Market Context' block — session-level state for one underlying."""

    previous_close: float | None = None
    previous_high: float | None = None
    previous_low: float | None = None
    open_price: float | None = None
    session_high: float | None = None
    session_low: float | None = None
    gap: float | None = None
    gap_pct: float | None = None
    intraday_range: float | None = None
    intraday_range_pct: float | None = None
    minutes_since_open: float = 0.0
    minutes_until_close: float = 0.0
    vwap: float | None = None


@dataclass
class InstrumentSnapshot:
    """Everything known about one instrument at one instant, already validated."""

    instrument: Instrument
    quote: Quote
    verdict: QualityVerdict
    store: MultiTimeframeStore | None = None
    context: MarketContext = field(default_factory=MarketContext)
    greeks: Greeks | None = None

    @property
    def usable(self) -> bool:
        return self.verdict.ok

    @property
    def last_price(self) -> float:
        return self.quote.last_price


@dataclass
class MarketSnapshot:
    """A consistent view across the universe at one decision point."""

    timestamp: datetime
    instruments: dict[str, InstrumentSnapshot] = field(default_factory=dict)
    chains: dict[str, OptionChain] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)

    def get(self, symbol: str) -> InstrumentSnapshot | None:
        return self.instruments.get(symbol)

    def usable_symbols(self) -> list[str]:
        return [s for s, snap in self.instruments.items() if snap.usable]

    def chain_key(self, underlying: str, expiry: date) -> str:
        return f"{underlying}:{expiry.isoformat()}"

    def chain(self, underlying: str, expiry: date) -> OptionChain | None:
        return self.chains.get(self.chain_key(underlying, expiry))


class MarketDataEngine:
    def __init__(
        self,
        broker: BrokerAdapter,
        config: AppConfig,
        *,
        clock: Clock | None = None,
        quality_gate: DataQualityGate | None = None,
    ) -> None:
        self.broker = broker
        self.config = config
        self.clock = clock or LiveClock()
        self.quality = quality_gate or DataQualityGate(config.data_quality)
        self._stores: dict[str, MultiTimeframeStore] = {}
        self._contexts: dict[str, MarketContext] = {}
        self._last_quotes: dict[str, Quote] = {}
        self._chain_cache: dict[str, tuple[datetime, OptionChain]] = {}
        self._base_timeframe = Timeframe(min(
            (Timeframe(tf) for tf in config.timeframes.enabled_timeframes),
            key=lambda tf: tf.minutes if tf.minutes > 0 else 10**6,
        ).value)
        self._enabled = tuple(
            sorted(
                (Timeframe(tf) for tf in config.timeframes.enabled_timeframes),
                key=lambda tf: tf.minutes,
            )
        )

    # ------------------------------------------------------------------ #
    def store_for(self, instrument: Instrument) -> MultiTimeframeStore:
        key = instrument.trading_symbol
        if key not in self._stores:
            self._stores[key] = MultiTimeframeStore(
                base_timeframe=self._base_timeframe,
                timeframes=self._enabled,
                max_bars=max(self.config.timeframes.history_bars * 4, 500),
            )
        return self._stores[key]

    def warm_up(self, instruments: Iterable[Instrument], *, days: int = 10) -> dict[str, int]:
        """Load historical candles so features have depth on the first cycle.

        Returns bars loaded per symbol. Failures are recorded, not raised: one
        illiquid symbol must not prevent the session from starting.
        """
        loaded: dict[str, int] = {}
        end = self.clock.now()
        start = end - timedelta(days=days)
        for instrument in instruments:
            try:
                candles = self.broker.get_historical_candles(
                    instrument, self._base_timeframe, start, end
                )
            except NotSupportedError:
                logger.warning("broker cannot supply history; features will warm up live")
                return loaded
            except BrokerError as exc:
                logger.warning("history fetch failed for %s: %s", instrument.trading_symbol, exc)
                self.quality.record_failure(instrument.trading_symbol)
                continue
            if candles:
                store = self.store_for(instrument)
                store.ingest_candles(candles, self._base_timeframe)
                store.refresh_all()
                loaded[instrument.trading_symbol] = len(candles)
        return loaded

    # ------------------------------------------------------------------ #
    def fetch_quotes(self, instruments: list[Instrument]) -> dict[str, Quote]:
        """Fetch full quotes one instrument at a time (the documented quote endpoint
        is single-instrument). Symbols that are failing repeatedly are skipped so a
        broken instrument cannot consume the whole rate-limit budget."""
        out: dict[str, Quote] = {}
        for instrument in instruments:
            symbol = instrument.trading_symbol
            if self.quality.is_circuit_broken(symbol):
                continue
            try:
                out[symbol] = self.broker.get_quote(instrument)
            except BrokerError as exc:
                logger.warning("quote failed for %s: %s", symbol, exc)
                self.quality.record_failure(symbol)
        return out

    def ingest_quote(self, quote: Quote) -> QualityVerdict:
        """Validate a quote and, if valid, fold it into the candle store."""
        now = self.clock.now()
        verdict = self.quality.check_quote(quote, now=now)
        symbol = quote.instrument.trading_symbol
        if not verdict.ok:
            logger.debug("rejected quote for %s: %s", symbol, verdict.reason)
            return verdict

        self._last_quotes[symbol] = quote
        store = self.store_for(quote.instrument)
        store.ingest_tick(
            quote.timestamp,
            quote.last_price,
            volume=0.0,  # cumulative volume is handled in _update_context
            open_interest=quote.open_interest or 0.0,
        )
        self._update_context(quote, now)
        return verdict

    def _update_context(self, quote: Quote, now: datetime) -> None:
        symbol = quote.instrument.trading_symbol
        context = self._contexts.setdefault(symbol, MarketContext())

        if context.previous_close is None and quote.previous_close:
            context.previous_close = quote.previous_close
        if context.open_price is None and quote.open:
            context.open_price = quote.open

        price = quote.last_price
        context.session_high = max(context.session_high or price, quote.high or price, price)
        context.session_low = min(context.session_low or price, quote.low or price, price)

        if context.previous_close and context.open_price:
            context.gap = context.open_price - context.previous_close
            context.gap_pct = context.gap / context.previous_close
        if context.session_high is not None and context.session_low is not None:
            context.intraday_range = context.session_high - context.session_low
            if price > 0:
                context.intraday_range_pct = context.intraday_range / price

        context.minutes_since_open = minutes_since_open(now)
        context.minutes_until_close = minutes_until_close(now)

        store = self._stores.get(symbol)
        if store is not None:
            frame = store.closed(self._base_timeframe, now)
            if not frame.empty:
                vwap = session_vwap(frame)
                if not vwap.empty:
                    context.vwap = float(vwap.iloc[-1])

    # ------------------------------------------------------------------ #
    def fetch_option_chain(
        self, underlying: str, expiry: date, *, exchange: str = "NSE", max_age_seconds: float = 5.0
    ) -> OptionChain | None:
        """Fetch (and briefly cache) an option chain, validated before return."""
        key = f"{underlying}:{expiry.isoformat()}"
        now = self.clock.now()
        cached = self._chain_cache.get(key)
        if cached and (now - cached[0]).total_seconds() < max_age_seconds:
            return cached[1]

        if not self.broker.supports(Capability.OPTION_CHAIN):
            return None
        try:
            chain = self.broker.get_option_chain(underlying, expiry, exchange=exchange)
        except BrokerError as exc:
            logger.warning("option chain failed for %s %s: %s", underlying, expiry, exc)
            self.quality.record_failure(key)
            return None

        verdict = self.quality.check_option_chain(chain, now=now)
        if not verdict.ok:
            logger.warning("option chain rejected for %s %s: %s", underlying, expiry, verdict.reason)
            return None

        # Drop individual contracts whose own quote or greeks are unusable, so the
        # selector never scores a contract on bad numbers.
        good: list = []
        for contract in chain.contracts:
            if contract.quote is None:
                continue
            if not self.quality.check_quote(contract.quote, now=now).ok:
                continue
            if contract.greeks is not None and not self.quality.check_greeks(contract.greeks).ok:
                contract.greeks = None
            good.append(contract)
        chain.contracts = good

        self._chain_cache[key] = (now, chain)
        return chain

    # ------------------------------------------------------------------ #
    def build_snapshot(
        self,
        instruments: list[Instrument],
        *,
        chains_for: dict[str, date] | None = None,
    ) -> MarketSnapshot:
        """Fetch, validate and assemble a decision-ready view of the market."""
        now = self.clock.now()
        snapshot = MarketSnapshot(timestamp=now)

        for symbol, quote in self.fetch_quotes(instruments).items():
            verdict = self.ingest_quote(quote)
            if not verdict.ok:
                snapshot.rejected[symbol] = verdict.reason
                continue
            snapshot.instruments[symbol] = InstrumentSnapshot(
                instrument=quote.instrument,
                quote=quote,
                verdict=verdict,
                store=self._stores.get(symbol),
                context=self._contexts.get(symbol, MarketContext()),
            )

        for symbol in instruments:
            name = symbol.trading_symbol
            if name not in snapshot.instruments and name not in snapshot.rejected:
                snapshot.rejected[name] = "no quote returned"

        for underlying, expiry in (chains_for or {}).items():
            chain = self.fetch_option_chain(underlying, expiry)
            if chain is not None:
                snapshot.chains[snapshot.chain_key(underlying, expiry)] = chain

        return snapshot

    # ------------------------------------------------------------------ #
    def frame(self, symbol: str, timeframe: Timeframe) -> pd.DataFrame:
        """Closed bars for a symbol/timeframe. The only frame feature code sees."""
        store = self._stores.get(symbol)
        if store is None:
            return pd.DataFrame()
        return store.closed(timeframe, self.clock.now())

    def last_quote(self, symbol: str) -> Quote | None:
        return self._last_quotes.get(symbol)

    def context(self, symbol: str) -> MarketContext:
        return self._contexts.get(symbol, MarketContext())

    def seed_from_candles(self, instrument: Instrument, candles: list[Candle]) -> None:
        """Used by the backtester to push historical bars through the same store."""
        store = self.store_for(instrument)
        store.ingest_candles(candles, self._base_timeframe)
        store.refresh_all()

    def reset_session(self) -> None:
        """Clear intraday context at the start of a new session."""
        self._contexts.clear()
        self._chain_cache.clear()
        self.quality.reset()

    def health(self) -> dict[str, object]:
        return {
            "tracked_symbols": len(self._stores),
            "failing_symbols": self.quality.summary(),
            "circuit_broken": self.quality.broken_symbols(),
            "cached_chains": len(self._chain_cache),
        }
