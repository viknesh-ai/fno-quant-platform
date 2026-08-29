"""Dynamic contract selection (REQ 6, REQ 30).

The requirement is emphatic that the system must not permanently trade "nearest
expiry ATM CE" or any fixed strike. So this module builds a *candidate set* across
strikes and expiries, scores every candidate on the sixteen dimensions REQ 6 lists,
and returns the highest risk-adjusted one — along with the full scorecard, so the
decision report can answer "why this strike" and "why this expiry" (REQ 48).

REQ 30's point is enforced here too: a high-probability directional view does not
imply a good option trade. A candidate can be rejected outright for a wide spread,
punishing theta, expensive IV, thin liquidity or insufficient expected movement,
regardless of how confident the signal was.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np

from ..configuration.schema import OptionSelectionConfig
from ..core.clock import time_to_expiry_years, to_ist
from ..core.logging import get_logger
from ..core.types import (
    Direction,
    Greeks,
    Instrument,
    OptionChain,
    OptionContract,
    OptionType,
)
from .pricing import breakeven_move, compute_greeks, expected_option_move, implied_volatility

logger = get_logger(__name__)


@dataclass
class ContractScore:
    """Full scorecard for one candidate contract."""

    contract: OptionContract
    total: float
    components: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    rejected: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return self.contract.instrument.trading_symbol

    def explain(self) -> str:
        if self.rejected:
            return f"{self.symbol}: REJECTED — {'; '.join(self.rejection_reasons)}"
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self.components.items()))
        return f"{self.symbol}: score {self.total:.3f} ({parts})"


@dataclass
class SelectionResult:
    selected: ContractScore | None
    candidates: list[ContractScore]
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.selected is not None

    def rejected_candidates(self) -> list[ContractScore]:
        return [c for c in self.candidates if c.rejected]

    def explain(self) -> str:
        if self.selected is None:
            return f"no contract selected: {self.reason}"
        lines = [f"selected {self.selected.symbol} — {self.selected.explain()}"]
        runners = [c for c in self.candidates if not c.rejected and c is not self.selected][:3]
        for candidate in runners:
            lines.append(f"  runner-up: {candidate.explain()}")
        for candidate in self.rejected_candidates()[:3]:
            lines.append(f"  {candidate.explain()}")
        return "\n".join(lines)


class OptionSelector:
    def __init__(self, config: OptionSelectionConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def select(
        self,
        *,
        chains: Sequence[OptionChain],
        direction: Direction,
        underlying_price: float,
        expected_move: float,
        atr: float,
        now: datetime,
        holding_minutes: int,
        capital_available: float,
        lot_size: int,
        iv_percentile: float | None = None,
    ) -> SelectionResult:
        """Choose the best contract to express `direction` across the given chains.

        `expected_move` is the absolute expected favourable move in the *underlying*,
        which is what makes the risk/reward and breakeven comparisons meaningful.
        """
        if direction is Direction.FLAT:
            return SelectionResult(None, [], "no direction to express")
        if not chains:
            return SelectionResult(None, [], "no option chains supplied")

        option_type = OptionType.CE if direction is Direction.LONG else OptionType.PE
        candidates = self._gather_candidates(chains, option_type, underlying_price)
        if not candidates:
            return SelectionResult(None, [], "no candidate contracts near the money")

        scored = [
            self._score(
                contract=contract,
                underlying_price=underlying_price,
                expected_move=expected_move,
                atr=atr,
                now=now,
                holding_minutes=holding_minutes,
                capital_available=capital_available,
                lot_size=lot_size,
                iv_percentile=iv_percentile,
            )
            for contract in candidates
        ]
        viable = [s for s in scored if not s.rejected]
        if not viable:
            top_reasons = "; ".join(
                sorted({r for s in scored for r in s.rejection_reasons})
            )
            return SelectionResult(None, scored, f"all {len(scored)} candidates rejected: {top_reasons}")

        viable.sort(key=lambda s: s.total, reverse=True)
        ordered = viable + [s for s in scored if s.rejected]
        return SelectionResult(viable[0], ordered)

    # ------------------------------------------------------------------ #
    def _gather_candidates(
        self, chains: Sequence[OptionChain], option_type: OptionType, underlying_price: float
    ) -> list[OptionContract]:
        """Candidates across the configured strike offsets and expiries (REQ 6)."""
        out: list[OptionContract] = []
        for chain in list(chains)[: self.config.consider_expiries]:
            strikes = chain.strikes()
            if not strikes:
                continue
            atm = min(strikes, key=lambda s: abs(s - underlying_price))
            atm_index = strikes.index(atm)
            for offset in self.config.strike_offsets:
                index = atm_index + offset
                if not (0 <= index < len(strikes)):
                    continue
                strike = strikes[index]
                contract = next(
                    (
                        c
                        for c in chain.contracts
                        if c.strike == strike and c.option_type is option_type
                    ),
                    None,
                )
                if contract is not None:
                    out.append(contract)
        return out

    # ------------------------------------------------------------------ #
    def _score(
        self,
        *,
        contract: OptionContract,
        underlying_price: float,
        expected_move: float,
        atr: float,
        now: datetime,
        holding_minutes: int,
        capital_available: float,
        lot_size: int,
        iv_percentile: float | None,
    ) -> ContractScore:
        config = self.config
        quote = contract.quote
        rejections: list[str] = []
        metrics: dict[str, float] = {}

        if quote is None or not quote.last_price or quote.last_price <= 0:
            return ContractScore(contract, 0.0, rejected=True, rejection_reasons=["no valid quote"])

        premium = quote.last_price
        metrics["premium"] = premium
        metrics["strike"] = contract.strike
        metrics["days_to_expiry"] = float((contract.expiry - to_ist(now).date()).days)

        # --- greeks: broker-supplied, or reconstructed ---------------------
        greeks = contract.greeks
        time_to_expiry = time_to_expiry_years(now, contract.expiry)
        if greeks is None:
            iv = implied_volatility(
                premium, underlying_price, contract.strike, time_to_expiry, contract.option_type
            )
            if iv is None:
                return ContractScore(
                    contract, 0.0, rejected=True,
                    rejection_reasons=["greeks unavailable and IV could not be solved"],
                )
            greeks = compute_greeks(
                underlying_price, contract.strike, time_to_expiry, iv, contract.option_type
            )
        metrics.update(
            delta=greeks.delta, gamma=greeks.gamma, theta=greeks.theta,
            vega=greeks.vega, iv=greeks.iv,
        )

        # --- hard filters (REQ 30) ------------------------------------------
        abs_delta = abs(greeks.delta)
        if abs_delta < config.min_delta:
            rejections.append(
                f"delta {abs_delta:.2f} below {config.min_delta} — too little participation "
                "in the underlying move"
            )
        elif abs_delta > config.max_delta:
            rejections.append(
                f"delta {abs_delta:.2f} above {config.max_delta} — capital-inefficient, "
                "close to owning the underlying"
            )

        spread_pct = quote.spread_pct
        if spread_pct is None:
            # No book means we cannot bound execution cost. For an option that is a
            # rejection, not an assumption of zero spread.
            rejections.append("no bid/ask available to measure spread")
            spread_pct = float("nan")
        elif spread_pct > config.max_spread_pct:
            rejections.append(
                f"spread {spread_pct:.2%} exceeds {config.max_spread_pct:.2%}"
            )
        metrics["spread_pct"] = spread_pct

        volume = float(quote.volume or 0.0)
        open_interest = float(quote.open_interest or 0.0)
        metrics["volume"] = volume
        metrics["open_interest"] = open_interest
        if volume < config.min_volume:
            rejections.append(f"volume {volume:.0f} below {config.min_volume}")
        if open_interest < config.min_open_interest:
            rejections.append(f"open interest {open_interest:.0f} below {config.min_open_interest}")

        days_held = max(holding_minutes / (60.0 * 6.25), 1.0 / 6.25)
        theta_ratio = abs(greeks.theta * days_held) / premium if premium > 0 else 1.0
        metrics["theta_to_premium"] = theta_ratio
        if theta_ratio > config.max_theta_to_premium_ratio:
            rejections.append(
                f"theta would burn {theta_ratio:.1%} of premium over the expected hold, "
                f"above the {config.max_theta_to_premium_ratio:.1%} limit"
            )

        if iv_percentile is not None:
            metrics["iv_percentile"] = iv_percentile
            if iv_percentile > config.max_iv_percentile:
                rejections.append(
                    f"IV percentile {iv_percentile:.0%} above {config.max_iv_percentile:.0%} — "
                    "paying for volatility that is already elevated"
                )

        # --- expected value of the option leg --------------------------------
        signed_move = expected_move if contract.option_type is OptionType.CE else -expected_move
        expected_pnl = expected_option_move(greeks, signed_move, days_held=days_held)
        metrics["expected_option_move"] = expected_pnl
        move_to_premium = expected_pnl / premium if premium > 0 else 0.0
        metrics["expected_move_to_premium"] = move_to_premium

        if move_to_premium < 0:
            rejections.append(
                "expected move net of decay is negative — the option loses money even if "
                "the directional view is correct"
            )
        elif move_to_premium < (config.min_expected_move_to_premium - 1.0):
            rejections.append(
                f"expected return {move_to_premium:.2f}x premium below the "
                f"{config.min_expected_move_to_premium - 1.0:.2f}x minimum"
            )

        breakeven = breakeven_move(premium, greeks, days_held=days_held, option_type=contract.option_type)
        if breakeven is not None:
            metrics["breakeven_move"] = breakeven
            if abs(expected_move) < breakeven:
                rejections.append(
                    f"expected underlying move {abs(expected_move):.2f} is below the "
                    f"{breakeven:.2f} needed just to cover decay"
                )

        cost_per_lot = premium * lot_size
        metrics["cost_per_lot"] = cost_per_lot
        if cost_per_lot > capital_available:
            rejections.append(
                f"one lot costs {cost_per_lot:,.0f} but only {capital_available:,.0f} is available"
            )

        if rejections:
            return ContractScore(
                contract, 0.0, metrics=metrics, rejected=True, rejection_reasons=rejections
            )

        # --- component scores, all in [0, 1] ---------------------------------
        components: dict[str, float] = {}

        liquidity_volume = float(np.clip(np.log1p(volume) / np.log1p(config.min_volume * 20), 0.0, 1.0))
        liquidity_oi = float(np.clip(np.log1p(open_interest) / np.log1p(config.min_open_interest * 20), 0.0, 1.0))
        components["liquidity"] = 0.5 * liquidity_volume + 0.5 * liquidity_oi

        components["spread"] = float(np.clip(1.0 - spread_pct / config.max_spread_pct, 0.0, 1.0))

        # Greek quality: reward delta near the middle of the permitted band (best
        # balance of participation and cost) and penalise heavy theta.
        band_centre = 0.5 * (config.min_delta + config.max_delta)
        band_half = max(0.5 * (config.max_delta - config.min_delta), 1e-6)
        delta_quality = float(np.clip(1.0 - abs(abs_delta - band_centre) / band_half, 0.0, 1.0))
        theta_quality = float(np.clip(1.0 - theta_ratio / config.max_theta_to_premium_ratio, 0.0, 1.0))
        gamma_quality = float(np.clip(greeks.gamma * underlying_price / max(abs_delta, 1e-6) / 5.0, 0.0, 1.0))
        components["greeks"] = 0.45 * delta_quality + 0.40 * theta_quality + 0.15 * gamma_quality

        moneyness = contract.moneyness(underlying_price)
        # Slightly OTM (moneyness a little negative) is usually the best convexity
        # per rupee; deep ITM and far OTM both score down.
        components["moneyness"] = float(np.clip(1.0 - abs(moneyness + 0.005) / 0.05, 0.0, 1.0))

        affordability = float(np.clip(1.0 - cost_per_lot / max(capital_available, 1.0), 0.0, 1.0))
        components["cost"] = affordability

        components["risk_reward"] = float(np.clip(move_to_premium / 2.0, 0.0, 1.0))

        total = sum(config.weights.get(name, 0.0) * value for name, value in components.items())
        return ContractScore(contract, float(total), components=components, metrics=metrics)


def contracts_for_expiry(chain: OptionChain, option_type: OptionType) -> list[OptionContract]:
    return [c for c in chain.contracts if c.option_type is option_type]
