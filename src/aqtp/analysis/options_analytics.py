"""Dealer-positioning analytics over the option chain.

`features/options_features.py` measures the chain: PCR, OI, skew, straddle price.
This module asks the harder question — what those numbers do to the people who
have to hedge them. Dealer gamma changes whether a market mean-reverts or trends;
charm and vanna change what happens into expiry and into a volatility shift; max
pain and OI walls mark where hedging flow concentrates.

None of this is a directional forecast, and the module never returns one. It
returns a positioning map with a signed bias and the reasoning behind it, which
the confluence layer weighs against everything else.

Sign convention: exposures are stated from the **dealer's** point of view, on the
standard assumption that customers are net long options at strikes with heavy
call OI above spot and heavy put OI below it. That assumption is stated rather
than hidden because it is the one thing here that can be wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ..core.logging import get_logger
from ..core.types import Greeks, OptionChain, OptionContract, OptionType
from ..options.pricing import DEFAULT_RISK_FREE_RATE, compute_greeks

logger = get_logger(__name__)

MINUTES_PER_YEAR = 365.0 * 24 * 60


@dataclass
class StrikeExposure:
    """Per-strike dealer exposure, in rupees of underlying per unit move."""

    strike: float
    call_oi: float = 0.0
    put_oi: float = 0.0
    gamma_exposure: float = 0.0
    delta_exposure: float = 0.0
    vanna_exposure: float = 0.0
    charm_exposure: float = 0.0
    net_oi_change: float = 0.0


@dataclass
class OptionsAnalyticsReport:
    """Dealer positioning and volatility structure for one underlying."""

    underlying: str = ""
    expiry: date | None = None
    spot: float = 0.0
    days_to_expiry: float = 0.0
    lot_size: int = 1
    strikes_analyzed: int = 0

    # --- gamma ---------------------------------------------------------------
    total_gamma_exposure: float = 0.0     # rupees of dealer hedging per 1% move
    zero_gamma_level: float | None = None  # the gamma flip point
    gamma_regime: str = "unknown"          # positive | negative | unknown
    largest_gamma_strike: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None

    # --- other exposures -----------------------------------------------------
    total_delta_exposure: float = 0.0
    total_vanna_exposure: float = 0.0
    total_charm_exposure: float = 0.0

    # --- pinning -------------------------------------------------------------
    max_pain: float | None = None
    pin_risk: float = 0.0                  # [0, 1]

    # --- volatility structure ------------------------------------------------
    atm_iv: float | None = None
    iv_rank: float | None = None
    risk_reversal_25d: float | None = None  # 25Δ put IV − 25Δ call IV
    butterfly_25d: float | None = None
    skew_slope: float | None = None         # IV per unit log-moneyness
    term_structure_slope: float | None = None
    expected_move_pct: float | None = None
    expected_move_points: float | None = None

    # --- flow ----------------------------------------------------------------
    pcr_oi: float = 0.0
    pcr_volume: float = 0.0
    oi_buildup: str = "none"                # long_buildup | short_buildup | ...
    call_oi_change: float = 0.0
    put_oi_change: float = 0.0
    aggregate_iv_change: float | None = None

    exposures: list[StrikeExposure] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def positioning_bias(self) -> float:
        """Signed [-1, 1] directional read from positioning alone.

        Built from four independent reads that are combined rather than
        cherry-picked: put/call OI, the direction of OI change, skew, and where
        spot sits between the put and call walls.
        """
        votes: list[float] = []

        if self.pcr_oi > 0:
            # A high PCR means put writers dominate — support beneath. Mapped so
            # 1.0 is neutral, saturating either side of it.
            votes.append(float(np.clip((self.pcr_oi - 1.0) / 0.5, -1.0, 1.0)))

        net_change = self.put_oi_change - self.call_oi_change
        total_change = abs(self.put_oi_change) + abs(self.call_oi_change)
        if total_change > 0:
            votes.append(float(np.clip(net_change / total_change, -1.0, 1.0)))

        if self.risk_reversal_25d is not None:
            # Puts bid over calls is demand for downside protection.
            votes.append(float(np.clip(-self.risk_reversal_25d / 0.05, -1.0, 1.0)))

        if self.call_wall and self.put_wall and self.call_wall > self.put_wall and self.spot > 0:
            span = self.call_wall - self.put_wall
            if span > 0:
                # Position inside the walls: near the put wall is support, near the
                # call wall is resistance.
                relative = (self.spot - self.put_wall) / span
                votes.append(float(np.clip((0.5 - relative) * 2.0, -1.0, 1.0)))

        return float(np.mean(votes)) if votes else 0.0

    @property
    def volatility_bias(self) -> float:
        """Signed [-1, 1]: +1 means options are cheap relative to their own
        history (favour buying premium), -1 means rich (favour selling it)."""
        if self.iv_rank is None:
            return 0.0
        return float(np.clip((0.5 - self.iv_rank) * 2.0, -1.0, 1.0))

    @property
    def suppresses_movement(self) -> bool:
        """Positive dealer gamma means hedging flow leans against every move.

        Breakouts fail in this regime and ranges hold; it is the single most
        useful thing the chain tells a directional strategy.
        """
        return self.gamma_regime == "positive"

    @property
    def amplifies_movement(self) -> bool:
        """Negative dealer gamma: hedging flow chases price and moves extend."""
        return self.gamma_regime == "negative"

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "opt_positioning_bias": self.positioning_bias,
            "opt_volatility_bias": self.volatility_bias,
            "opt_gamma_exposure": self.total_gamma_exposure,
            "opt_delta_exposure": self.total_delta_exposure,
            "opt_vanna_exposure": self.total_vanna_exposure,
            "opt_charm_exposure": self.total_charm_exposure,
            "opt_pin_risk": self.pin_risk,
            "opt_positive_gamma": 1.0 if self.suppresses_movement else 0.0,
            "opt_negative_gamma": 1.0 if self.amplifies_movement else 0.0,
        }
        for name in (
            "atm_iv", "iv_rank", "risk_reversal_25d", "butterfly_25d", "skew_slope",
            "term_structure_slope", "expected_move_pct", "aggregate_iv_change",
        ):
            value = getattr(self, name)
            if value is not None:
                out[f"opt_{name}"] = float(value)
        if self.spot > 0:
            for name in ("zero_gamma_level", "max_pain", "call_wall", "put_wall"):
                value = getattr(self, name)
                if value:
                    out[f"opt_{name}_dist_pct"] = (value - self.spot) / self.spot
        return out

    def explain(self) -> list[str]:
        return list(self.reasons)


# --------------------------------------------------------------------------- #
def _years_to_expiry(expiry: date, now: datetime) -> float:
    """Calendar years to expiry, floored so that expiry-day maths stays finite."""
    expiry_moment = datetime.combine(expiry, datetime.min.time()).replace(
        hour=15, minute=30, tzinfo=now.tzinfo
    )
    minutes = max((expiry_moment - now).total_seconds() / 60.0, 5.0)
    return minutes / MINUTES_PER_YEAR


def _greeks_for(contract: OptionContract, spot: float, years: float) -> Greeks | None:
    """Use broker greeks when present, otherwise reprice from the quoted IV."""
    if contract.greeks is not None and contract.greeks.gamma:
        return contract.greeks
    iv = None
    if contract.quote is not None and contract.quote.implied_volatility:
        iv = float(contract.quote.implied_volatility)
        if iv > 1.5:            # some feeds quote IV in percent
            iv /= 100.0
    if contract.greeks is not None and contract.greeks.iv:
        iv = iv or float(contract.greeks.iv)
    if not iv or iv <= 0 or spot <= 0 or contract.strike <= 0:
        return None
    try:
        return compute_greeks(spot, contract.strike, years, iv, contract.option_type)
    except Exception:  # pragma: no cover — pricing is defensive already
        return None


def _vanna_charm(spot: float, strike: float, years: float, iv: float) -> tuple[float, float]:
    """Vanna (∂delta/∂vol) and charm (∂delta/∂time), per unit of underlying.

    Both matter for the same reason: they say how a dealer's hedge changes when
    nothing about price changes. Charm drives the systematic unwind into expiry
    that makes expiry-day drift a real, measurable thing.

    Neither takes an option type: put delta is call delta minus one, so both
    partial derivatives are identical for a call and a put at the same strike.
    """
    if spot <= 0 or strike <= 0 or years <= 0 or iv <= 0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (DEFAULT_RISK_FREE_RATE + 0.5 * iv * iv) * years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vanna = -pdf * d2 / iv
    charm = -pdf * (2.0 * DEFAULT_RISK_FREE_RATE * years - d2 * iv * sqrt_t) / (2.0 * years * iv * sqrt_t)
    return float(vanna), float(charm / 365.0)   # charm per day


def _max_pain(chain: OptionChain) -> float | None:
    """The strike at which the total value of expiring options is smallest.

    Not a prediction — a measurement of where option writers collectively have
    the least to pay out, which is where price is most often pulled on expiry
    day and is close to meaningless on any other day.
    """
    strikes = chain.strikes()
    if len(strikes) < 3:
        return None
    best_strike, best_pain = None, math.inf
    for candidate in strikes:
        pain = 0.0
        for contract in chain.contracts:
            oi = float(contract.quote.open_interest or 0.0) if contract.quote else 0.0
            if oi <= 0:
                continue
            if contract.option_type is OptionType.CE:
                pain += max(0.0, candidate - contract.strike) * oi
            else:
                pain += max(0.0, contract.strike - candidate) * oi
        if pain < best_pain:
            best_strike, best_pain = candidate, pain
    return float(best_strike) if best_strike is not None else None


def _interpolate_iv(
    contracts: list[OptionContract], spot: float, years: float, target_delta: float
) -> float | None:
    """IV at a target absolute delta, interpolated across the smile."""
    points: list[tuple[float, float]] = []
    for contract in contracts:
        greeks = _greeks_for(contract, spot, years)
        if greeks is None or not greeks.iv:
            continue
        points.append((abs(greeks.delta), float(greeks.iv)))
    if len(points) < 2:
        return None
    points.sort(key=lambda item: item[0])
    deltas = np.array([p[0] for p in points])
    ivs = np.array([p[1] for p in points])
    if target_delta < deltas[0] or target_delta > deltas[-1]:
        return None
    return float(np.interp(target_delta, deltas, ivs))


# --------------------------------------------------------------------------- #
def analyze_chain(
    chain: OptionChain,
    *,
    now: datetime,
    lot_size: int = 1,
    iv_rank: float | None = None,
    next_expiry_chain: OptionChain | None = None,
    previous_atm_iv: float | None = None,
) -> OptionsAnalyticsReport:
    """Full dealer-positioning analysis of one option chain."""
    report = OptionsAnalyticsReport(
        underlying=chain.underlying_symbol,
        expiry=chain.expiry,
        spot=float(chain.underlying_ltp or 0.0),
        lot_size=max(1, lot_size),
        iv_rank=iv_rank,
    )
    if report.spot <= 0 or not chain.contracts:
        report.reasons.append("option chain is empty or has no underlying price")
        return report

    years = _years_to_expiry(chain.expiry, now)
    report.days_to_expiry = years * 365.0
    spot = report.spot

    by_strike: dict[float, StrikeExposure] = {}
    call_oi_total = put_oi_total = 0.0
    call_volume_total = put_volume_total = 0.0

    for contract in chain.contracts:
        quote = contract.quote
        oi = float(quote.open_interest or 0.0) if quote else 0.0
        oi_change = float(quote.oi_change or 0.0) if quote else 0.0
        volume = float(quote.volume or 0.0) if quote else 0.0
        exposure = by_strike.setdefault(contract.strike, StrikeExposure(strike=contract.strike))

        if contract.option_type is OptionType.CE:
            call_oi_total += oi
            call_volume_total += volume
            exposure.call_oi = oi
            report.call_oi_change += oi_change
        else:
            put_oi_total += oi
            put_volume_total += volume
            exposure.put_oi = oi
            report.put_oi_change += oi_change
        exposure.net_oi_change += oi_change

        greeks = _greeks_for(contract, spot, years)
        if greeks is None or oi <= 0:
            continue

        contracts_outstanding = oi * report.lot_size
        # Dealer sign: short the calls customers buy, short the puts they buy.
        # Gamma exposure is quoted per 1% move in spot, the unit desks use.
        sign = 1.0 if contract.option_type is OptionType.CE else -1.0
        gamma_rupees = greeks.gamma * contracts_outstanding * spot * spot * 0.01
        exposure.gamma_exposure += sign * gamma_rupees
        exposure.delta_exposure += greeks.delta * contracts_outstanding * spot

        iv = float(greeks.iv or 0.0)
        if iv > 0:
            vanna, charm = _vanna_charm(spot, contract.strike, years, iv)
            exposure.vanna_exposure += sign * vanna * contracts_outstanding * spot
            exposure.charm_exposure += sign * charm * contracts_outstanding * spot

    report.exposures = sorted(by_strike.values(), key=lambda e: e.strike)
    report.strikes_analyzed = len(report.exposures)
    report.total_gamma_exposure = float(sum(e.gamma_exposure for e in report.exposures))
    report.total_delta_exposure = float(sum(e.delta_exposure for e in report.exposures))
    report.total_vanna_exposure = float(sum(e.vanna_exposure for e in report.exposures))
    report.total_charm_exposure = float(sum(e.charm_exposure for e in report.exposures))

    report.pcr_oi = float(put_oi_total / call_oi_total) if call_oi_total > 0 else 0.0
    report.pcr_volume = float(put_volume_total / call_volume_total) if call_volume_total > 0 else 0.0

    # --- gamma regime and the flip level -----------------------------------
    if report.exposures:
        report.largest_gamma_strike = max(
            report.exposures, key=lambda e: abs(e.gamma_exposure)
        ).strike
        report.gamma_regime = (
            "positive" if report.total_gamma_exposure > 0
            else "negative" if report.total_gamma_exposure < 0
            else "unknown"
        )
        # Cumulative gamma by strike crosses zero at the flip point: below it
        # dealers are short gamma and amplify moves, above it they dampen them.
        cumulative = 0.0
        previous_strike, previous_cumulative = None, 0.0
        for exposure in report.exposures:
            cumulative += exposure.gamma_exposure
            if previous_strike is not None and previous_cumulative * cumulative < 0:
                span = cumulative - previous_cumulative
                if abs(span) > 1e-9:
                    fraction = -previous_cumulative / span
                    report.zero_gamma_level = float(
                        previous_strike + fraction * (exposure.strike - previous_strike)
                    )
                    break
            previous_strike, previous_cumulative = exposure.strike, cumulative

    calls_above = [e for e in report.exposures if e.strike > spot and e.call_oi > 0]
    puts_below = [e for e in report.exposures if e.strike < spot and e.put_oi > 0]
    if calls_above:
        report.call_wall = max(calls_above, key=lambda e: e.call_oi).strike
    if puts_below:
        report.put_wall = max(puts_below, key=lambda e: e.put_oi).strike

    # --- pinning ------------------------------------------------------------
    report.max_pain = _max_pain(chain)
    if report.max_pain and spot > 0 and report.days_to_expiry <= 2.0:
        distance = abs(spot - report.max_pain) / spot
        proximity = float(np.clip(1.0 - distance / 0.01, 0.0, 1.0))
        urgency = float(np.clip(1.0 - report.days_to_expiry / 2.0, 0.0, 1.0))
        report.pin_risk = proximity * urgency

    # --- volatility structure -----------------------------------------------
    atm_strike = chain.atm_strike()
    if atm_strike is not None:
        atm_ivs = []
        for contract in chain.contracts:
            if contract.strike != atm_strike:
                continue
            greeks = _greeks_for(contract, spot, years)
            if greeks and greeks.iv:
                atm_ivs.append(float(greeks.iv))
        if atm_ivs:
            report.atm_iv = float(np.mean(atm_ivs))

    call_iv_25 = _interpolate_iv(chain.calls(), spot, years, 0.25)
    put_iv_25 = _interpolate_iv(chain.puts(), spot, years, 0.25)
    if call_iv_25 is not None and put_iv_25 is not None:
        report.risk_reversal_25d = float(put_iv_25 - call_iv_25)
        if report.atm_iv:
            report.butterfly_25d = float((call_iv_25 + put_iv_25) / 2.0 - report.atm_iv)

    smile: list[tuple[float, float]] = []
    for contract in chain.contracts:
        greeks = _greeks_for(contract, spot, years)
        if greeks is None or not greeks.iv or contract.strike <= 0:
            continue
        smile.append((math.log(contract.strike / spot), float(greeks.iv)))
    if len(smile) >= 4:
        xs = np.array([p[0] for p in smile])
        ys = np.array([p[1] for p in smile])
        if xs.std() > 1e-9:
            slope, _ = np.polyfit(xs, ys, 1)
            report.skew_slope = float(slope)

    if next_expiry_chain is not None:
        far_years = _years_to_expiry(next_expiry_chain.expiry, now)
        far_report_iv = None
        far_atm = next_expiry_chain.atm_strike()
        if far_atm is not None:
            ivs = []
            for contract in next_expiry_chain.contracts:
                if contract.strike != far_atm:
                    continue
                greeks = _greeks_for(contract, next_expiry_chain.underlying_ltp, far_years)
                if greeks and greeks.iv:
                    ivs.append(float(greeks.iv))
            if ivs:
                far_report_iv = float(np.mean(ivs))
        if far_report_iv is not None and report.atm_iv and far_years > years:
            report.term_structure_slope = float(
                (far_report_iv - report.atm_iv) / max(far_years - years, 1e-6) / 365.0
            )

    if report.atm_iv and years > 0:
        report.expected_move_pct = float(report.atm_iv * math.sqrt(years))
        report.expected_move_points = report.expected_move_pct * spot

    if previous_atm_iv and report.atm_iv:
        report.aggregate_iv_change = float(report.atm_iv - previous_atm_iv)

    # --- OI buildup ---------------------------------------------------------
    report.oi_buildup = _classify_buildup(report, chain)

    _narrate(report)
    return report


def _classify_buildup(report: OptionsAnalyticsReport, chain: OptionChain) -> str:
    """Classify the day's OI change into the four standard buildups.

    Price up + OI up  = long buildup      (fresh longs, bullish)
    Price down + OI up = short buildup    (fresh shorts, bearish)
    Price up + OI down = short covering   (bearish positions closing, bullish)
    Price down + OI down = long unwinding (longs closing, bearish)
    """
    net_oi_change = report.call_oi_change + report.put_oi_change
    if abs(net_oi_change) < 1e-9:
        return "none"

    price_change = 0.0
    reference = 0.0
    for contract in chain.contracts:
        quote = contract.quote
        if quote is None or not quote.previous_close or not quote.last_price:
            continue
        weight = float(quote.open_interest or 0.0)
        if weight <= 0:
            continue
        price_change += (quote.last_price - quote.previous_close) * weight
        reference += quote.previous_close * weight
    if reference <= 0:
        return "none"
    direction = price_change / reference

    if abs(direction) < 0.005:
        return "none"
    if direction > 0:
        return "long_buildup" if net_oi_change > 0 else "short_covering"
    return "short_buildup" if net_oi_change > 0 else "long_unwinding"


def _narrate(report: OptionsAnalyticsReport) -> None:
    spot = report.spot
    if report.gamma_regime == "positive":
        report.reasons.append(
            f"dealers are LONG gamma (₹{report.total_gamma_exposure:,.0f} per 1% move) — "
            "hedging flow dampens moves, ranges hold and breakouts tend to fail"
        )
    elif report.gamma_regime == "negative":
        report.reasons.append(
            f"dealers are SHORT gamma (₹{report.total_gamma_exposure:,.0f} per 1% move) — "
            "hedging flow chases price, so moves extend and stops get run"
        )
    if report.zero_gamma_level and spot > 0:
        report.reasons.append(
            f"gamma flip level {report.zero_gamma_level:,.0f} "
            f"({(report.zero_gamma_level - spot) / spot:+.2%} from spot)"
        )
    if report.call_wall:
        report.reasons.append(f"call wall (resistance) at {report.call_wall:,.0f}")
    if report.put_wall:
        report.reasons.append(f"put wall (support) at {report.put_wall:,.0f}")
    if report.max_pain and spot > 0:
        report.reasons.append(
            f"max pain {report.max_pain:,.0f} ({(report.max_pain - spot) / spot:+.2%})"
            + (f", pin risk {report.pin_risk:.2f}" if report.pin_risk > 0.1 else "")
        )
    if report.pcr_oi:
        report.reasons.append(f"PCR(OI) {report.pcr_oi:.2f}, PCR(volume) {report.pcr_volume:.2f}")
    if report.oi_buildup != "none":
        report.reasons.append(f"OI buildup: {report.oi_buildup.replace('_', ' ')}")
    if report.atm_iv is not None:
        line = f"ATM IV {report.atm_iv:.1%}"
        if report.iv_rank is not None:
            line += f" (IV rank {report.iv_rank:.0%})"
        if report.expected_move_pct:
            line += f", implied move to expiry ±{report.expected_move_pct:.2%}"
        report.reasons.append(line)
    if report.risk_reversal_25d is not None:
        direction = "puts bid over calls" if report.risk_reversal_25d > 0 else "calls bid over puts"
        report.reasons.append(f"25Δ risk reversal {report.risk_reversal_25d:+.2%} — {direction}")
    if report.term_structure_slope is not None:
        shape = "contango" if report.term_structure_slope > 0 else "backwardation"
        report.reasons.append(f"IV term structure in {shape}")
    if abs(report.total_charm_exposure) > 0 and report.days_to_expiry <= 3:
        report.reasons.append(
            f"charm exposure ₹{report.total_charm_exposure:,.0f}/day with "
            f"{report.days_to_expiry:.1f} days left — expect a systematic hedge unwind"
        )
