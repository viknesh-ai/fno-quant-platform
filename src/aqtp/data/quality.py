"""Data quality gate (REQ 8).

Invalid data must never reach the strategy engine. Everything entering the platform
passes through `DataQualityGate.check_quote`, which returns a verdict rather than
raising, so the orchestrator can record *why* a symbol produced NO_TRADE instead of
just seeing an exception.

The gate is stateful on purpose: staleness and duplicate detection need memory of
what the previous tick looked like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterable

from ..configuration.schema import DataQualityConfig
from ..core.logging import get_logger
from ..core.types import Greeks, OptionChain, Quote

logger = get_logger(__name__)


class QualityIssue(str, Enum):
    STALE_PRICE = "stale_price"
    MISSING_PRICE = "missing_price"
    TIMESTAMP_ANOMALY = "timestamp_anomaly"
    DUPLICATE_TICK = "duplicate_tick"
    IMPOSSIBLE_VALUE = "impossible_value"
    ABNORMAL_SPREAD = "abnormal_spread"
    CROSSED_BOOK = "crossed_book"
    PRICE_JUMP = "price_jump"
    MISSING_DEPTH = "missing_depth"
    OUTSIDE_CIRCUIT = "outside_circuit"
    MISSING_CHAIN_CONTRACTS = "missing_chain_contracts"
    STALE_GREEKS = "stale_greeks"
    IMPOSSIBLE_GREEKS = "impossible_greeks"
    BROKER_FAILURE = "broker_failure"


@dataclass(frozen=True)
class QualityVerdict:
    ok: bool
    issues: tuple[QualityIssue, ...] = ()
    detail: str = ""

    @property
    def reason(self) -> str:
        if self.ok:
            return ""
        names = ", ".join(i.value for i in self.issues)
        return f"{names}{': ' + self.detail if self.detail else ''}"


GOOD = QualityVerdict(ok=True)


@dataclass
class _SymbolState:
    last_price: float | None = None
    last_timestamp: datetime | None = None
    last_received: datetime | None = None
    consecutive_failures: int = 0
    identical_count: int = 0


@dataclass
class DataQualityGate:
    config: DataQualityConfig
    _state: dict[str, _SymbolState] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def check_quote(self, quote: Quote, *, now: datetime | None = None) -> QualityVerdict:
        now = now or quote.received_at
        symbol = quote.instrument.trading_symbol
        state = self._state.setdefault(symbol, _SymbolState())
        issues: list[QualityIssue] = []
        details: list[str] = []

        # --- price sanity -------------------------------------------------
        price = quote.last_price
        if price is None or price != price:  # None or NaN
            issues.append(QualityIssue.MISSING_PRICE)
            details.append("last_price is missing")
        elif price < self.config.min_price:
            issues.append(QualityIssue.IMPOSSIBLE_VALUE)
            details.append(f"last_price {price} below minimum {self.config.min_price}")
        elif price <= 0:
            issues.append(QualityIssue.IMPOSSIBLE_VALUE)
            details.append(f"non-positive price {price}")

        for name, value in (
            ("high", quote.high),
            ("low", quote.low),
            ("open", quote.open),
        ):
            if value is not None and value < 0:
                issues.append(QualityIssue.IMPOSSIBLE_VALUE)
                details.append(f"negative {name} {value}")

        if quote.high is not None and quote.low is not None and quote.high < quote.low:
            issues.append(QualityIssue.IMPOSSIBLE_VALUE)
            details.append(f"high {quote.high} < low {quote.low}")

        if quote.volume is not None and quote.volume < 0:
            issues.append(QualityIssue.IMPOSSIBLE_VALUE)
            details.append("negative volume")

        if quote.open_interest is not None and quote.open_interest < 0:
            issues.append(QualityIssue.IMPOSSIBLE_VALUE)
            details.append("negative open interest")

        # --- circuit limits -----------------------------------------------
        if price and quote.upper_circuit and price > quote.upper_circuit * 1.001:
            issues.append(QualityIssue.OUTSIDE_CIRCUIT)
            details.append(f"price {price} above upper circuit {quote.upper_circuit}")
        if price and quote.lower_circuit and 0 < quote.lower_circuit and price < quote.lower_circuit * 0.999:
            issues.append(QualityIssue.OUTSIDE_CIRCUIT)
            details.append(f"price {price} below lower circuit {quote.lower_circuit}")

        # --- timestamp ----------------------------------------------------
        age = (now - quote.timestamp).total_seconds()
        if age < -2.0:
            # A venue timestamp meaningfully in the future means clocks disagree;
            # trusting it would make every staleness check meaningless.
            issues.append(QualityIssue.TIMESTAMP_ANOMALY)
            details.append(f"timestamp {age:.1f}s in the future")
        elif age > self.config.max_quote_age_seconds:
            issues.append(QualityIssue.STALE_PRICE)
            details.append(f"quote age {age:.1f}s exceeds {self.config.max_quote_age_seconds}s")

        if state.last_timestamp is not None and quote.timestamp < state.last_timestamp:
            issues.append(QualityIssue.TIMESTAMP_ANOMALY)
            details.append("timestamp moved backwards")

        # --- duplicates ---------------------------------------------------
        if (
            state.last_timestamp is not None
            and quote.timestamp == state.last_timestamp
            and state.last_price == price
        ):
            state.identical_count += 1
            # One repeat is normal (an untraded instrument). Many repeats while the
            # clock advances means the feed is frozen, which is not tradable.
            if state.identical_count >= 3:
                issues.append(QualityIssue.DUPLICATE_TICK)
                details.append(f"identical tick repeated {state.identical_count}x")
        else:
            state.identical_count = 0

        # --- spread -------------------------------------------------------
        bid, ask = quote.bid_price, quote.ask_price
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            if bid > ask:
                issues.append(QualityIssue.CROSSED_BOOK)
                details.append(f"crossed book bid {bid} > ask {ask}")
            else:
                spread_pct = quote.spread_pct
                if spread_pct is not None and spread_pct > self.config.max_spread_pct:
                    issues.append(QualityIssue.ABNORMAL_SPREAD)
                    details.append(
                        f"spread {spread_pct:.2%} exceeds {self.config.max_spread_pct:.2%}"
                    )
        elif self.config.require_depth:
            issues.append(QualityIssue.MISSING_DEPTH)
            details.append("bid/ask unavailable")

        # --- jump ---------------------------------------------------------
        if state.last_price and price and state.last_price > 0:
            change = abs(price - state.last_price) / state.last_price
            if change > self.config.max_price_jump_pct:
                issues.append(QualityIssue.PRICE_JUMP)
                details.append(
                    f"price moved {change:.1%} from {state.last_price} to {price} in one update"
                )

        verdict = (
            GOOD
            if not issues
            else QualityVerdict(ok=False, issues=tuple(issues), detail="; ".join(details))
        )

        # Update memory only from structurally sane quotes, so one corrupt tick does
        # not poison the jump baseline for every subsequent check.
        if price and price > 0 and QualityIssue.IMPOSSIBLE_VALUE not in issues:
            state.last_price = price
            state.last_timestamp = quote.timestamp
            state.last_received = now

        if verdict.ok:
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1

        return verdict

    # ------------------------------------------------------------------ #
    def check_greeks(
        self, greeks: Greeks | None, *, age_seconds: float | None = None
    ) -> QualityVerdict:
        if greeks is None:
            return QualityVerdict(ok=False, issues=(QualityIssue.STALE_GREEKS,), detail="no greeks")
        if age_seconds is not None and age_seconds > self.config.max_greeks_age_seconds:
            return QualityVerdict(
                ok=False,
                issues=(QualityIssue.STALE_GREEKS,),
                detail=f"greeks age {age_seconds:.0f}s exceeds {self.config.max_greeks_age_seconds}s",
            )
        problems: list[str] = []
        if not (-1.0001 <= greeks.delta <= 1.0001):
            problems.append(f"delta {greeks.delta} outside [-1, 1]")
        if greeks.gamma < 0:
            problems.append(f"negative gamma {greeks.gamma}")
        if greeks.vega < 0:
            problems.append(f"negative vega {greeks.vega}")
        if greeks.iv <= 0 or greeks.iv > 5.0:
            problems.append(f"implausible IV {greeks.iv}")
        if problems:
            return QualityVerdict(
                ok=False, issues=(QualityIssue.IMPOSSIBLE_GREEKS,), detail="; ".join(problems)
            )
        return GOOD

    def check_option_chain(
        self, chain: OptionChain, *, min_strikes: int = 5, now: datetime | None = None
    ) -> QualityVerdict:
        issues: list[QualityIssue] = []
        details: list[str] = []

        if chain.underlying_ltp <= 0:
            issues.append(QualityIssue.MISSING_PRICE)
            details.append("underlying LTP missing from chain")

        strikes = chain.strikes()
        if len(strikes) < min_strikes:
            issues.append(QualityIssue.MISSING_CHAIN_CONTRACTS)
            details.append(f"only {len(strikes)} strikes available, need {min_strikes}")

        # Both sides must be present around the money; a one-sided chain usually
        # means a partial response rather than a real market.
        calls = {c.strike for c in chain.calls()}
        puts = {c.strike for c in chain.puts()}
        atm = chain.atm_strike()
        if atm is not None:
            near = sorted(strikes, key=lambda s: abs(s - atm))[:5]
            missing = [s for s in near if s not in calls or s not in puts]
            if missing:
                issues.append(QualityIssue.MISSING_CHAIN_CONTRACTS)
                details.append(f"near-ATM strikes missing a side: {missing}")

        if chain.fetched_at is not None and now is not None:
            age = (now - chain.fetched_at).total_seconds()
            if age > self.config.max_greeks_age_seconds:
                issues.append(QualityIssue.STALE_GREEKS)
                details.append(f"chain age {age:.0f}s")

        if issues:
            return QualityVerdict(ok=False, issues=tuple(issues), detail="; ".join(details))
        return GOOD

    # ------------------------------------------------------------------ #
    def record_failure(self, symbol: str) -> None:
        """Called by the market-data engine when the broker call itself failed."""
        self._state.setdefault(symbol, _SymbolState()).consecutive_failures += 1

    def is_circuit_broken(self, symbol: str) -> bool:
        """True when a symbol has failed enough times to be dropped this cycle."""
        state = self._state.get(symbol)
        return bool(state and state.consecutive_failures >= self.config.max_consecutive_failures)

    def broken_symbols(self) -> list[str]:
        return [s for s in self._state if self.is_circuit_broken(s)]

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._state.clear()
        else:
            self._state.pop(symbol, None)

    def summary(self) -> dict[str, int]:
        return {s: st.consecutive_failures for s, st in self._state.items() if st.consecutive_failures}


def all_ok(verdicts: Iterable[QualityVerdict]) -> bool:
    return all(v.ok for v in verdicts)
