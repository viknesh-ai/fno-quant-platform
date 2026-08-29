"""Black-Scholes pricing, greeks and implied volatility.

The broker supplies greeks for most contracts, and those are preferred. This module
exists for two cases the broker cannot cover:

  * backtesting, where historical greeks are not available and must be reconstructed
    from the option price actually recorded at that time, and
  * validation, where a broker-supplied greek that disagrees badly with theory is a
    data-quality signal rather than something to trade on.

Greeks are returned in *practical* units (theta per day, vega per 1 vol point),
because that is how the risk limits are expressed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.types import Greeks, OptionType

# Indian risk-free proxy. Configurable because it moves; the sensitivity of the
# greeks we use to this input is small, but it should not be a hidden constant.
DEFAULT_RISK_FREE_RATE = 0.065

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(
    spot: float, strike: float, time_to_expiry: float, volatility: float, rate: float, dividend: float
) -> tuple[float, float]:
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or volatility <= 0:
        raise ValueError("d1/d2 require positive spot, strike, time and volatility")
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * time_to_expiry
    ) / sigma_sqrt_t
    return d1, d1 - sigma_sqrt_t


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    *,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend: float = 0.0,
) -> float:
    """European option price. At/after expiry this correctly returns intrinsic value."""
    if time_to_expiry <= 0 or volatility <= 0:
        if option_type is OptionType.CE:
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)

    d1, d2 = d1_d2(spot, strike, time_to_expiry, volatility, rate, dividend)
    discount = math.exp(-rate * time_to_expiry)
    carry = math.exp(-dividend * time_to_expiry)
    if option_type is OptionType.CE:
        return spot * carry * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * carry * _norm_cdf(-d1)


def compute_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    *,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend: float = 0.0,
) -> Greeks:
    """Greeks in the units the risk engine uses.

    delta : per 1 unit of underlying
    gamma : per 1 unit of underlying, per unit
    theta : per calendar DAY (not per year) — this is what a trader reasons about
    vega  : per 1 volatility POINT (a move from 20% to 21%), not per 1.0 of vol
    rho   : per 1 percentage point of rate
    """
    if time_to_expiry <= 0 or volatility <= 0:
        # At expiry: delta is a step function, every other greek collapses.
        if option_type is OptionType.CE:
            delta = 1.0 if spot > strike else 0.0
        else:
            delta = -1.0 if spot < strike else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0, rho=0.0, iv=volatility)

    d1, d2 = d1_d2(spot, strike, time_to_expiry, volatility, rate, dividend)
    sqrt_t = math.sqrt(time_to_expiry)
    pdf_d1 = _norm_pdf(d1)
    discount = math.exp(-rate * time_to_expiry)
    carry = math.exp(-dividend * time_to_expiry)

    gamma = carry * pdf_d1 / (spot * volatility * sqrt_t)
    vega_per_unit = spot * carry * pdf_d1 * sqrt_t

    if option_type is OptionType.CE:
        delta = carry * _norm_cdf(d1)
        theta_per_year = (
            -spot * carry * pdf_d1 * volatility / (2 * sqrt_t)
            - rate * strike * discount * _norm_cdf(d2)
            + dividend * spot * carry * _norm_cdf(d1)
        )
        rho_per_unit = strike * time_to_expiry * discount * _norm_cdf(d2)
    else:
        delta = -carry * _norm_cdf(-d1)
        theta_per_year = (
            -spot * carry * pdf_d1 * volatility / (2 * sqrt_t)
            + rate * strike * discount * _norm_cdf(-d2)
            - dividend * spot * carry * _norm_cdf(-d1)
        )
        rho_per_unit = -strike * time_to_expiry * discount * _norm_cdf(-d2)

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta=theta_per_year / 365.0,   # per day
        vega=vega_per_unit / 100.0,     # per vol point
        rho=rho_per_unit / 100.0,       # per rate point
        iv=volatility,
    )


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_type: OptionType,
    *,
    rate: float = DEFAULT_RISK_FREE_RATE,
    dividend: float = 0.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float | None:
    """Solve for implied volatility. Returns None when no solution exists.

    Newton-Raphson is used first because it converges in a handful of iterations for
    liquid options; it falls back to bisection when vega is near zero (deep ITM/OTM),
    where Newton is unstable. Returning None rather than a fabricated number matters:
    REQ 18 forbids inventing data, and a bogus IV would poison the option selector.
    """
    if price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return None

    intrinsic = (
        max(0.0, spot - strike) if option_type is OptionType.CE else max(0.0, strike - spot)
    )
    # A price below intrinsic is an arbitrage or bad data, not a volatility.
    if price < intrinsic - 1e-6:
        return None
    upper_bound = spot if option_type is OptionType.CE else strike
    if price >= upper_bound:
        return None

    sigma = 0.25
    for _ in range(max_iterations):
        try:
            theoretical = black_scholes_price(
                spot, strike, time_to_expiry, sigma, option_type, rate=rate, dividend=dividend
            )
            d1, _ = d1_d2(spot, strike, time_to_expiry, sigma, rate, dividend)
            vega = spot * math.exp(-dividend * time_to_expiry) * _norm_pdf(d1) * math.sqrt(time_to_expiry)
        except (ValueError, OverflowError):
            break

        difference = theoretical - price
        if abs(difference) < tolerance:
            return sigma
        if vega < 1e-8:
            break  # Newton is unreliable here; hand over to bisection
        sigma -= difference / vega
        if sigma <= 0.0 or sigma > 5.0:
            break

    low, high = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        theoretical = black_scholes_price(
            spot, strike, time_to_expiry, mid, option_type, rate=rate, dividend=dividend
        )
        if abs(theoretical - price) < tolerance:
            return mid
        if theoretical > price:
            high = mid
        else:
            low = mid
        if high - low < 1e-8:
            break
    solution = 0.5 * (low + high)
    return solution if 1e-3 < solution < 4.99 else None


@dataclass(frozen=True)
class PricedOption:
    price: float
    greeks: Greeks


def price_and_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: OptionType,
    *,
    rate: float = DEFAULT_RISK_FREE_RATE,
) -> PricedOption:
    return PricedOption(
        price=black_scholes_price(spot, strike, time_to_expiry, volatility, option_type, rate=rate),
        greeks=compute_greeks(spot, strike, time_to_expiry, volatility, option_type, rate=rate),
    )


def expected_option_move(
    greeks: Greeks, underlying_move: float, *, days_held: float = 0.0, iv_change: float = 0.0
) -> float:
    """Second-order estimate of an option's price change.

    delta + 0.5*gamma*move^2 captures the convexity that makes a directional view
    profitable for an option buyer; theta and vega then subtract what holding it
    costs. Ignoring the theta term is the most common way an option trade that was
    "right on direction" still loses money — so it is included here, not later.
    """
    delta_term = greeks.delta * underlying_move
    gamma_term = 0.5 * greeks.gamma * underlying_move ** 2
    theta_term = greeks.theta * days_held
    vega_term = greeks.vega * (iv_change * 100.0)
    return delta_term + gamma_term + theta_term + vega_term


def breakeven_move(
    premium: float, greeks: Greeks, *, days_held: float, option_type: OptionType
) -> float | None:
    """Underlying move required just to cover time decay over the holding period.

    This is the number that decides whether a directional option trade is worth
    taking at all: if the expected move is smaller than this, the trade is a
    guaranteed loser regardless of direction.
    """
    if abs(greeks.delta) < 1e-6:
        return None
    decay = abs(greeks.theta * days_held)
    return decay / abs(greeks.delta)
