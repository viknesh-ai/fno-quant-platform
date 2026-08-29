"""Instrument discovery and universe construction (REQ 4/5/6).

The universe is *discovered* from the broker's instrument master on a schedule. No
symbol list, lot size or expiry is hard-coded anywhere. Expiries, new listings,
delistings, lot-size revisions and trading-status changes are picked up simply by
refreshing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ..configuration.schema import UniverseConfig
from ..core.clock import to_ist
from ..core.logging import get_logger
from ..core.types import Instrument, InstrumentType, Segment

logger = get_logger(__name__)


@dataclass(frozen=True)
class UnderlyingUniverse:
    """Everything tradable for one underlying, grouped for fast lookup."""

    symbol: str
    is_index: bool
    spot: Instrument | None
    futures: tuple[Instrument, ...]
    options: tuple[Instrument, ...]

    def expiries(self) -> list[date]:
        dates = {i.expiry_date for i in self.futures + self.options if i.expiry_date}
        return sorted(dates)

    def option_expiries(self) -> list[date]:
        return sorted({i.expiry_date for i in self.options if i.expiry_date})

    def future_expiries(self) -> list[date]:
        return sorted({i.expiry_date for i in self.futures if i.expiry_date})

    def options_for_expiry(self, expiry: date) -> list[Instrument]:
        return [i for i in self.options if i.expiry_date == expiry]

    def strikes_for_expiry(self, expiry: date) -> list[float]:
        return sorted({i.strike_price for i in self.options if i.expiry_date == expiry and i.strike_price})

    def future_for_expiry(self, expiry: date) -> Instrument | None:
        for instrument in self.futures:
            if instrument.expiry_date == expiry:
                return instrument
        return None

    def nearest_future(self, as_of: date) -> Instrument | None:
        candidates = [f for f in self.futures if f.expiry_date and f.expiry_date >= as_of]
        return min(candidates, key=lambda f: f.expiry_date) if candidates else None  # type: ignore[arg-type]

    @property
    def lot_size(self) -> int:
        """Lot size is read from the master, never assumed (REQ 67.12)."""
        for instrument in self.futures + self.options:
            if instrument.lot_size > 0:
                return instrument.lot_size
        return 1


@dataclass
class InstrumentUniverse:
    """The discovered, filtered tradable set for this session."""

    underlyings: dict[str, UnderlyingUniverse] = field(default_factory=dict)
    refreshed_at: datetime | None = None
    total_instruments: int = 0

    def get(self, symbol: str) -> UnderlyingUniverse | None:
        return self.underlyings.get(symbol)

    def symbols(self) -> list[str]:
        return sorted(self.underlyings)

    def index_symbols(self) -> list[str]:
        return sorted(s for s, u in self.underlyings.items() if u.is_index)

    def stock_symbols(self) -> list[str]:
        return sorted(s for s, u in self.underlyings.items() if not u.is_index)

    def all_instruments(self) -> list[Instrument]:
        out: list[Instrument] = []
        for underlying in self.underlyings.values():
            if underlying.spot:
                out.append(underlying.spot)
            out.extend(underlying.futures)
            out.extend(underlying.options)
        return out

    def find(self, trading_symbol: str) -> Instrument | None:
        for instrument in self.all_instruments():
            if instrument.trading_symbol == trading_symbol:
                return instrument
        return None

    def __len__(self) -> int:
        return len(self.underlyings)


class InstrumentDiscovery:
    """Builds and refreshes the universe from a broker's instrument master."""

    def __init__(self, config: UniverseConfig) -> None:
        self.config = config
        self._universe = InstrumentUniverse()
        self._last_refresh_monotonic: float | None = None

    @property
    def universe(self) -> InstrumentUniverse:
        return self._universe

    def needs_refresh(self) -> bool:
        if self._last_refresh_monotonic is None:
            return True
        elapsed_minutes = (time.monotonic() - self._last_refresh_monotonic) / 60.0
        return elapsed_minutes >= self.config.refresh_interval_minutes

    def refresh(
        self, instruments: list[Instrument], *, as_of: datetime | None = None
    ) -> InstrumentUniverse:
        """Rebuild the universe from a fresh instrument master."""
        as_of = to_ist(as_of) if as_of else datetime.now()
        today = as_of.date()

        eligible = [i for i in instruments if self._is_eligible(i, today)]
        by_underlying: dict[str, list[Instrument]] = defaultdict(list)
        spots: dict[str, Instrument] = {}

        for instrument in eligible:
            if instrument.segment is Segment.FNO and instrument.is_derivative:
                key = instrument.underlying_symbol or instrument.name
                if key:
                    by_underlying[key].append(instrument)
            elif instrument.instrument_type in (InstrumentType.IDX, InstrumentType.EQ):
                # Spot/index reference used for underlying prices and moneyness.
                spots.setdefault(instrument.trading_symbol, instrument)

        index_set = {s.upper() for s in self.config.index_underlyings}
        underlyings: dict[str, UnderlyingUniverse] = {}

        for symbol, group in by_underlying.items():
            upper = symbol.upper()
            is_index = upper in index_set
            if not is_index and not self.config.include_stock_fno:
                continue
            if not is_index and not self._stock_allowed(upper):
                continue
            if is_index and upper not in index_set:
                continue

            futures = tuple(
                sorted(
                    (i for i in group if i.is_future),
                    key=lambda i: (i.expiry_date or date.max),
                )
            )
            options = tuple(
                sorted(
                    (i for i in group if i.is_option),
                    key=lambda i: (i.expiry_date or date.max, i.strike_price or 0.0),
                )
            )
            if not self.config.include_futures:
                futures = ()
            if not self.config.include_options:
                options = ()
            if not futures and not options:
                continue

            underlyings[symbol] = UnderlyingUniverse(
                symbol=symbol,
                is_index=is_index,
                spot=spots.get(symbol),
                futures=futures,
                options=options,
            )

        underlyings = self._apply_size_cap(underlyings, index_set)

        self._universe = InstrumentUniverse(
            underlyings=underlyings,
            refreshed_at=as_of,
            total_instruments=sum(
                len(u.futures) + len(u.options) + (1 if u.spot else 0) for u in underlyings.values()
            ),
        )
        self._last_refresh_monotonic = time.monotonic()
        logger.info(
            "universe refreshed: %d underlyings (%d index, %d stock), %d contracts",
            len(underlyings),
            len(self._universe.index_symbols()),
            len(self._universe.stock_symbols()),
            self._universe.total_instruments,
        )
        return self._universe

    def _is_eligible(self, instrument: Instrument, today: date) -> bool:
        """Filters that apply to every contract, regardless of underlying."""
        if not instrument.trading_symbol:
            return False
        if instrument.is_reserved:
            return False  # suspended / not available for trading
        if instrument.is_derivative:
            if not instrument.tradable:
                return False
            if instrument.expiry_date is None:
                return False
            days = (instrument.expiry_date - today).days
            # Expired contracts drop out automatically on the next refresh.
            if days < self.config.min_days_to_expiry:
                return False
            if days > self.config.max_days_to_expiry:
                return False
            if instrument.lot_size <= 0:
                return False
            if instrument.is_option and not instrument.strike_price:
                return False
        return True

    def _stock_allowed(self, symbol: str) -> bool:
        allow = {s.upper() for s in self.config.stock_underlyings_allowlist}
        block = {s.upper() for s in self.config.stock_underlyings_blocklist}
        if symbol in block:
            return False
        if allow and symbol not in allow:
            return False
        return True

    def _apply_size_cap(
        self, underlyings: dict[str, UnderlyingUniverse], index_set: set[str]
    ) -> dict[str, UnderlyingUniverse]:
        """Cap universe size, keeping every configured index first.

        Indices are never dropped: they are the ones the user explicitly named, and
        silently discarding them would change the strategy mix without saying so.
        """
        if len(underlyings) <= self.config.max_universe_size:
            return underlyings

        indices = {s: u for s, u in underlyings.items() if s.upper() in index_set}
        stocks = {s: u for s, u in underlyings.items() if s.upper() not in index_set}
        room = max(0, self.config.max_universe_size - len(indices))

        # Rank stocks by contract breadth — a proxy for liquidity available from the
        # master alone. Real liquidity ranking happens in the OpportunityScanner
        # once quotes exist.
        ranked = sorted(stocks.items(), key=lambda kv: len(kv[1].options) + len(kv[1].futures), reverse=True)
        kept = dict(indices)
        kept.update(dict(ranked[:room]))
        logger.info(
            "universe capped at %d (kept %d indices + %d stocks of %d)",
            self.config.max_universe_size,
            len(indices),
            min(room, len(ranked)),
            len(stocks),
        )
        return kept

    # ------------------------------------------------------------------ #
    def tradable_expiries(
        self, underlying: str, *, as_of: date, kind: str = "options"
    ) -> list[date]:
        """The next N expiries the config permits, nearest first (REQ 6)."""
        group = self._universe.get(underlying)
        if group is None:
            return []
        expiries = group.option_expiries() if kind == "options" else group.future_expiries()
        upcoming = [e for e in expiries if e >= as_of]
        return upcoming[: self.config.max_expiries_ahead]

    def detect_changes(self, previous: InstrumentUniverse) -> dict[str, list[str]]:
        """Diff two universes so the operator sees what the market changed.

        REQ 4 requires the system to account for new contracts, expiries and
        lot-size changes; surfacing them explicitly is how that becomes visible
        rather than implicit.
        """
        changes: dict[str, list[str]] = {
            "added_underlyings": [],
            "removed_underlyings": [],
            "lot_size_changed": [],
            "new_expiries": [],
            "expired": [],
        }
        old_symbols, new_symbols = set(previous.underlyings), set(self._universe.underlyings)
        changes["added_underlyings"] = sorted(new_symbols - old_symbols)
        changes["removed_underlyings"] = sorted(old_symbols - new_symbols)

        for symbol in sorted(old_symbols & new_symbols):
            old, new = previous.underlyings[symbol], self._universe.underlyings[symbol]
            if old.lot_size != new.lot_size:
                changes["lot_size_changed"].append(f"{symbol}: {old.lot_size} -> {new.lot_size}")
            old_expiries, new_expiries = set(old.expiries()), set(new.expiries())
            for expiry in sorted(new_expiries - old_expiries):
                changes["new_expiries"].append(f"{symbol}@{expiry}")
            for expiry in sorted(old_expiries - new_expiries):
                changes["expired"].append(f"{symbol}@{expiry}")
        return {k: v for k, v in changes.items() if v}


def next_weekly_expiry(expiries: list[date], as_of: date) -> date | None:
    upcoming = sorted(e for e in expiries if e >= as_of)
    return upcoming[0] if upcoming else None


def monthly_expiry(expiries: list[date], as_of: date) -> date | None:
    """The last expiry within the current expiry month — that is the monthly series.

    Derived from the actual expiry list rather than from a rule about "last Thursday",
    because the exchange has changed that rule and will again.
    """
    upcoming = sorted(e for e in expiries if e >= as_of)
    if not upcoming:
        return None
    by_month: dict[tuple[int, int], list[date]] = defaultdict(list)
    for expiry in upcoming:
        by_month[(expiry.year, expiry.month)].append(expiry)
    first_month = min(by_month)
    return max(by_month[first_month])


def is_expiry_week(expiry: date, as_of: date) -> bool:
    return 0 <= (expiry - as_of).days <= 7
