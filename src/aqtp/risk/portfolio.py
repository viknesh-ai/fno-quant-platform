"""Portfolio-level intelligence (REQ 29/30).

REQ 29 is that trades must not be evaluated independently. This module holds the
aggregate view — exposure, correlation, underlying/sector overlap and net option
greeks — that the RiskEngine consults before approving anything.

The greek aggregation is the part that most distinguishes an options portfolio from
an equity one: four separate 30-delta long calls on correlated underlyings are, in
aggregate, one large directional bet with concentrated gamma and theta. Netting them
is the only way to see that.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ..core.clock import time_to_expiry_years
from ..core.logging import get_logger
from ..core.types import (
    Direction,
    Greeks,
    Instrument,
    OptionType,
    Position,
    Quote,
)
from ..options.pricing import compute_greeks, implied_volatility

logger = get_logger(__name__)

# Correlation groups for Indian F&O. Index members overlap heavily by construction;
# sector groupings are coarse but sufficient for a concentration check. This is a
# starting map that the operator is expected to extend, not a claim of completeness.
DEFAULT_CORRELATION_GROUPS: dict[str, list[str]] = {
    "broad_index": ["NIFTY", "NIFTYNXT50", "MIDCPNIFTY"],
    "banking": [
        "BANKNIFTY", "FINNIFTY", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK",
        "SBIN", "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "AUBANK", "IDFCFIRSTB",
        "PNB", "BANKBARODA", "CANBK",
    ],
    "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT"],
    "auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT", "ASHOKLEY", "TVSMOTOR"],
    "energy": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "NTPC", "POWERGRID", "COALINDIA", "TATAPOWER"],
    "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA", "LUPIN", "TORNTPHARM", "ALKEM"],
    "metals": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL", "JINDALSTEL", "NATIONALUM", "NMDC"],
    "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "TATACONSUM"],
    "financials_nonbank": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "ICICIPRULI", "CHOLAFIN", "MUTHOOTFIN"],
}


@dataclass
class PositionGreeks:
    """Greeks for one position, scaled by quantity and sign."""

    symbol: str
    underlying: str
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    notional: float = 0.0
    quantity: int = 0
    expiry: date | None = None


@dataclass
class PortfolioState:
    """Aggregate portfolio view at one instant."""

    equity: float = 0.0
    positions: list[Position] = field(default_factory=list)
    total_exposure: float = 0.0
    exposure_by_instrument: dict[str, float] = field(default_factory=dict)
    exposure_by_underlying: dict[str, float] = field(default_factory=dict)
    exposure_by_group: dict[str, float] = field(default_factory=dict)
    directional_exposure: float = 0.0     # signed, in rupees
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    per_position_greeks: list[PositionGreeks] = field(default_factory=list)
    expiry_concentration: dict[str, float] = field(default_factory=dict)
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    open_positions: int = 0
    timestamp: datetime | None = None

    @property
    def exposure_pct(self) -> float:
        return self.total_exposure / self.equity if self.equity > 0 else 0.0

    @property
    def net_delta_per_lakh(self) -> float:
        return self.net_delta / (self.equity / 100_000) if self.equity > 0 else 0.0

    @property
    def net_gamma_per_lakh(self) -> float:
        return self.net_gamma / (self.equity / 100_000) if self.equity > 0 else 0.0

    @property
    def net_vega_per_lakh(self) -> float:
        return self.net_vega / (self.equity / 100_000) if self.equity > 0 else 0.0

    @property
    def net_theta_per_lakh(self) -> float:
        return self.net_theta / (self.equity / 100_000) if self.equity > 0 else 0.0

    def underlying_exposure_pct(self, underlying: str) -> float:
        if self.equity <= 0:
            return 0.0
        return self.exposure_by_underlying.get(underlying, 0.0) / self.equity

    def group_exposure_pct(self, group: str) -> float:
        if self.equity <= 0:
            return 0.0
        return self.exposure_by_group.get(group, 0.0) / self.equity

    def describe(self) -> dict[str, float]:
        return {
            "equity": round(self.equity, 2),
            "open_positions": self.open_positions,
            "exposure_pct": round(self.exposure_pct, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "net_delta_per_lakh": round(self.net_delta_per_lakh, 2),
            "net_gamma_per_lakh": round(self.net_gamma_per_lakh, 4),
            "net_theta_per_lakh": round(self.net_theta_per_lakh, 2),
            "net_vega_per_lakh": round(self.net_vega_per_lakh, 2),
        }


class PortfolioManager:
    def __init__(
        self, *, correlation_groups: dict[str, list[str]] | None = None
    ) -> None:
        groups = correlation_groups or DEFAULT_CORRELATION_GROUPS
        self.groups = groups
        self._group_of: dict[str, str] = {}
        for group, members in groups.items():
            for member in members:
                # First mapping wins; index names appear in more than one group and
                # should be attributed to their most specific one.
                self._group_of.setdefault(member.upper(), group)

    def group_for(self, underlying: str) -> str:
        return self._group_of.get(underlying.upper(), "unclassified")

    def are_correlated(self, a: str, b: str) -> bool:
        if a.upper() == b.upper():
            return True
        group_a, group_b = self.group_for(a), self.group_for(b)
        if group_a == "unclassified" or group_b == "unclassified":
            return False
        return group_a == group_b

    # ------------------------------------------------------------------ #
    def build_state(
        self,
        *,
        positions: list[Position],
        equity: float,
        quotes: dict[str, Quote] | None = None,
        underlying_prices: dict[str, float] | None = None,
        greeks_by_symbol: dict[str, Greeks] | None = None,
        now: datetime | None = None,
        realized_pnl: float = 0.0,
    ) -> PortfolioState:
        """Aggregate current positions into a portfolio view."""
        now = now or datetime.now()
        quotes = quotes or {}
        underlying_prices = underlying_prices or {}
        greeks_by_symbol = greeks_by_symbol or {}

        state = PortfolioState(
            equity=equity, positions=positions, timestamp=now, realized_pnl=realized_pnl
        )
        state.open_positions = len([p for p in positions if p.quantity != 0])

        for position in positions:
            if position.quantity == 0:
                continue
            instrument = position.instrument
            symbol = instrument.trading_symbol
            underlying = instrument.underlying_symbol or symbol

            quote = quotes.get(symbol)
            price = quote.last_price if quote else position.last_price or position.average_price
            notional = abs(position.quantity) * price

            state.total_exposure += notional
            state.exposure_by_instrument[symbol] = (
                state.exposure_by_instrument.get(symbol, 0.0) + notional
            )
            state.exposure_by_underlying[underlying] = (
                state.exposure_by_underlying.get(underlying, 0.0) + notional
            )
            group = self.group_for(underlying)
            state.exposure_by_group[group] = state.exposure_by_group.get(group, 0.0) + notional
            state.unrealized_pnl += position.unrealized_pnl

            position_greeks = self._position_greeks(
                position=position,
                price=price,
                underlying_price=underlying_prices.get(underlying),
                supplied_greeks=greeks_by_symbol.get(symbol),
                now=now,
            )
            state.per_position_greeks.append(position_greeks)
            state.net_delta += position_greeks.delta
            state.net_gamma += position_greeks.gamma
            state.net_theta += position_greeks.theta
            state.net_vega += position_greeks.vega
            state.directional_exposure += position_greeks.delta * (
                underlying_prices.get(underlying, price)
            )

            if instrument.expiry_date:
                key = instrument.expiry_date.isoformat()
                state.expiry_concentration[key] = state.expiry_concentration.get(key, 0.0) + notional

        return state

    def _position_greeks(
        self,
        *,
        position: Position,
        price: float,
        underlying_price: float | None,
        supplied_greeks: Greeks | None,
        now: datetime,
    ) -> PositionGreeks:
        """Scale per-unit greeks by signed quantity.

        For a future, delta is 1 per unit by definition and the other greeks are
        zero — treating a future as having no delta would badly understate the
        portfolio's directional exposure.
        """
        instrument = position.instrument
        result = PositionGreeks(
            symbol=instrument.trading_symbol,
            underlying=instrument.underlying_symbol or instrument.trading_symbol,
            quantity=position.quantity,
            notional=abs(position.quantity) * price,
            expiry=instrument.expiry_date,
        )

        if instrument.is_future:
            result.delta = float(position.quantity)
            return result
        if not instrument.is_option:
            result.delta = float(position.quantity)
            return result

        greeks = supplied_greeks
        if greeks is None and underlying_price and instrument.strike_price and instrument.expiry_date:
            option_type = instrument.option_type or OptionType.CE
            tte = time_to_expiry_years(now, instrument.expiry_date)
            iv = implied_volatility(price, underlying_price, instrument.strike_price, tte, option_type)
            if iv is not None:
                greeks = compute_greeks(
                    underlying_price, instrument.strike_price, tte, iv, option_type
                )

        if greeks is None:
            # Without greeks the position's risk is unknown. Recording zeros would
            # make the portfolio look safer than it is, so the delta is approximated
            # conservatively at 0.5 per unit and flagged.
            logger.debug("no greeks for %s; using a 0.5 delta approximation", instrument.trading_symbol)
            result.delta = 0.5 * position.quantity
            return result

        quantity = float(position.quantity)
        result.delta = greeks.delta * quantity
        result.gamma = greeks.gamma * quantity
        result.theta = greeks.theta * quantity
        result.vega = greeks.vega * quantity
        return result

    # ------------------------------------------------------------------ #
    def projected_state(
        self,
        current: PortfolioState,
        *,
        instrument: Instrument,
        quantity: int,
        price: float,
        greeks: Greeks | None = None,
        underlying_price: float | None = None,
    ) -> PortfolioState:
        """What the portfolio would look like *after* a candidate trade.

        Checking limits against the projected state rather than the current one is
        the difference between "we are within limits" and "we will still be within
        limits after this fills" — only the latter prevents a breach.
        """
        hypothetical = Position(
            instrument=instrument,
            quantity=quantity,
            average_price=price,
            last_price=price,
        )
        return self.build_state(
            positions=list(current.positions) + [hypothetical],
            equity=current.equity,
            underlying_prices=(
                {**({} if underlying_price is None else
                    {instrument.underlying_symbol or instrument.trading_symbol: underlying_price})}
            ),
            greeks_by_symbol={instrument.trading_symbol: greeks} if greeks else {},
            now=current.timestamp,
            realized_pnl=current.realized_pnl,
        )

    def correlated_exposure(self, state: PortfolioState, underlying: str) -> float:
        """Total exposure to everything correlated with `underlying`, including itself."""
        total = 0.0
        for symbol, exposure in state.exposure_by_underlying.items():
            if self.are_correlated(symbol, underlying):
                total += exposure
        return total

    def overlapping_positions(self, state: PortfolioState, underlying: str) -> list[str]:
        return [
            g.symbol
            for g in state.per_position_greeks
            if self.are_correlated(g.underlying, underlying)
        ]
