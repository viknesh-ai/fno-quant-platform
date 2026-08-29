"""Option-chain derived features (REQ 10 'Options' block).

These describe *positioning* — where open interest sits, how it moved, what the
skew looks like — not a directional prediction. REQ 12.8 is explicit that OI alone
does not predict direction, so this module deliberately produces measurements and
leaves interpretation to the strategy and the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ..core.logging import get_logger
from ..core.types import OptionChain, OptionContract, OptionType

logger = get_logger(__name__)


@dataclass
class OptionChainFeatures:
    """Measured state of one option chain at one instant."""

    underlying: str
    expiry: date
    underlying_ltp: float

    total_call_oi: float = 0.0
    total_put_oi: float = 0.0
    pcr_oi: float = 0.0
    total_call_volume: float = 0.0
    total_put_volume: float = 0.0
    pcr_volume: float = 0.0

    call_oi_change: float = 0.0
    put_oi_change: float = 0.0
    net_oi_change: float = 0.0

    max_call_oi_strike: float | None = None
    max_put_oi_strike: float | None = None
    max_call_oi_change_strike: float | None = None
    max_put_oi_change_strike: float | None = None

    atm_strike: float | None = None
    atm_call_iv: float | None = None
    atm_put_iv: float | None = None
    atm_iv: float | None = None
    iv_skew: float | None = None          # put IV minus call IV at equal distance
    iv_term_slope: float | None = None

    atm_straddle_price: float | None = None
    expected_move_pct: float | None = None

    net_delta: float = 0.0
    net_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0

    gamma_concentration_strike: float | None = None
    support_strike: float | None = None    # heaviest put OI below spot
    resistance_strike: float | None = None  # heaviest call OI above spot

    contracts_analyzed: int = 0

    def to_dict(self) -> dict[str, float]:
        """Flat numeric view for the ML feature vector."""
        out: dict[str, float] = {}
        for key, value in self.__dict__.items():
            if key in ("underlying", "expiry"):
                continue
            if isinstance(value, (int, float)) and value is not None:
                out[f"opt_{key}"] = float(value)
        if self.atm_strike and self.underlying_ltp:
            out["opt_atm_offset_pct"] = (self.underlying_ltp - self.atm_strike) / self.underlying_ltp
        if self.support_strike and self.underlying_ltp:
            out["opt_support_dist_pct"] = (self.underlying_ltp - self.support_strike) / self.underlying_ltp
        if self.resistance_strike and self.underlying_ltp:
            out["opt_resistance_dist_pct"] = (
                self.resistance_strike - self.underlying_ltp
            ) / self.underlying_ltp
        return out


def _oi(contract: OptionContract) -> float:
    return float(contract.quote.open_interest or 0.0) if contract.quote else 0.0


def _oi_change(contract: OptionContract) -> float:
    if contract.quote is None:
        return 0.0
    return float(contract.quote.oi_change or 0.0)


def _volume(contract: OptionContract) -> float:
    return float(contract.quote.volume or 0.0) if contract.quote else 0.0


def compute_chain_features(chain: OptionChain) -> OptionChainFeatures:
    """Reduce a chain to measured aggregates."""
    features = OptionChainFeatures(
        underlying=chain.underlying_symbol,
        expiry=chain.expiry,
        underlying_ltp=chain.underlying_ltp,
    )
    calls, puts = chain.calls(), chain.puts()
    features.contracts_analyzed = len(calls) + len(puts)
    if not calls and not puts:
        return features

    features.total_call_oi = sum(_oi(c) for c in calls)
    features.total_put_oi = sum(_oi(p) for p in puts)
    features.total_call_volume = sum(_volume(c) for c in calls)
    features.total_put_volume = sum(_volume(p) for p in puts)
    features.call_oi_change = sum(_oi_change(c) for c in calls)
    features.put_oi_change = sum(_oi_change(p) for p in puts)
    features.net_oi_change = features.put_oi_change - features.call_oi_change

    # PCR is undefined with no call OI; 0.0 marks "not measurable" rather than
    # silently reporting an extreme value.
    features.pcr_oi = (
        features.total_put_oi / features.total_call_oi if features.total_call_oi > 0 else 0.0
    )
    features.pcr_volume = (
        features.total_put_volume / features.total_call_volume
        if features.total_call_volume > 0
        else 0.0
    )

    if calls:
        heaviest = max(calls, key=_oi)
        features.max_call_oi_strike = heaviest.strike if _oi(heaviest) > 0 else None
        builder = max(calls, key=_oi_change)
        features.max_call_oi_change_strike = builder.strike if _oi_change(builder) > 0 else None
    if puts:
        heaviest = max(puts, key=_oi)
        features.max_put_oi_strike = heaviest.strike if _oi(heaviest) > 0 else None
        builder = max(puts, key=_oi_change)
        features.max_put_oi_change_strike = builder.strike if _oi_change(builder) > 0 else None

    spot = chain.underlying_ltp
    calls_above = [c for c in calls if c.strike > spot and _oi(c) > 0]
    puts_below = [p for p in puts if p.strike < spot and _oi(p) > 0]
    if calls_above:
        features.resistance_strike = max(calls_above, key=_oi).strike
    if puts_below:
        features.support_strike = max(puts_below, key=_oi).strike

    atm = chain.atm_strike()
    features.atm_strike = atm
    if atm is not None:
        atm_call = next((c for c in calls if c.strike == atm), None)
        atm_put = next((p for p in puts if p.strike == atm), None)
        if atm_call and atm_call.greeks:
            features.atm_call_iv = atm_call.greeks.iv
        if atm_put and atm_put.greeks:
            features.atm_put_iv = atm_put.greeks.iv
        ivs = [v for v in (features.atm_call_iv, features.atm_put_iv) if v]
        features.atm_iv = float(np.mean(ivs)) if ivs else None

        if atm_call and atm_put and atm_call.last_price and atm_put.last_price:
            straddle = atm_call.last_price + atm_put.last_price
            features.atm_straddle_price = straddle
            if spot > 0:
                # The straddle is the market's own estimate of the move to expiry.
                features.expected_move_pct = straddle / spot

        features.iv_skew = _compute_skew(calls, puts, atm)

    for contract in calls + puts:
        if contract.greeks is None:
            continue
        oi = _oi(contract)
        sign = 1.0 if contract.option_type is OptionType.CE else -1.0
        features.net_delta += contract.greeks.delta * oi
        features.net_gamma += contract.greeks.gamma * oi * sign
        features.total_vega += contract.greeks.vega * oi
        features.total_theta += contract.greeks.theta * oi

    with_gamma = [c for c in calls + puts if c.greeks and _oi(c) > 0]
    if with_gamma:
        features.gamma_concentration_strike = max(
            with_gamma, key=lambda c: c.greeks.gamma * _oi(c)  # type: ignore[union-attr]
        ).strike

    return features


def _compute_skew(
    calls: list[OptionContract], puts: list[OptionContract], atm: float
) -> float | None:
    """Put IV minus call IV at roughly equal distance from ATM.

    Comparing IVs at matched distance (rather than matched strike) is what makes
    this a skew measure rather than a restatement of moneyness.
    """
    strikes = sorted({c.strike for c in calls} & {p.strike for p in puts})
    if len(strikes) < 3:
        return None
    below = [s for s in strikes if s < atm]
    above = [s for s in strikes if s > atm]
    if not below or not above:
        return None

    otm_put_strike = max(below, key=lambda s: s) if len(below) == 1 else sorted(below)[-2]
    otm_call_strike = min(above) if len(above) == 1 else sorted(above)[1]

    put = next((p for p in puts if p.strike == otm_put_strike and p.greeks), None)
    call = next((c for c in calls if c.strike == otm_call_strike and c.greeks), None)
    if put is None or call is None:
        return None
    return put.greeks.iv - call.greeks.iv  # type: ignore[union-attr]


@dataclass
class IVHistory:
    """Rolling ATM IV history, used for IV percentile (REQ 10).

    Percentile is computed against this instrument's own history only. Comparing a
    stock's IV to an index's would be meaningless, and comparing against a global
    pool would leak information across instruments.
    """

    max_points: int = 500
    _by_underlying: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)

    def record(self, underlying: str, moment: datetime, iv: float) -> None:
        if iv <= 0:
            return
        series = self._by_underlying.setdefault(underlying, [])
        series.append((moment, iv))
        if len(series) > self.max_points:
            del series[: len(series) - self.max_points]

    def percentile(self, underlying: str, iv: float, *, min_points: int = 30) -> float | None:
        """Rank of `iv` within this underlying's own history, or None if unknown.

        A degenerate history (every observation identical) carries no information
        about whether volatility is expensive. Returning 1.0 there would read as
        "IV is at its maximum" and cause the option selector to reject every
        contract, so the honest answer is None — not measurable.
        """
        series = self._by_underlying.get(underlying)
        if not series or len(series) < min_points:
            return None
        values = np.array([v for _, v in series])
        if float(values.std()) < 1e-9:
            return None
        return float((values <= iv).mean())

    def change(self, underlying: str, iv: float, *, lookback: int = 20) -> float | None:
        series = self._by_underlying.get(underlying)
        if not series or len(series) < lookback:
            return None
        past = series[-lookback][1]
        return (iv - past) / past if past > 0 else None

    def points(self, underlying: str) -> int:
        return len(self._by_underlying.get(underlying, []))
