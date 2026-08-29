"""Expiry strategy (REQ 12.9).

The requirement is not "trade expiry" — it is that the system must be able to
*decide* TRADE_EXPIRY or AVOID_EXPIRY from measured conditions. This strategy
therefore begins with an explicit tradability assessment, exposed as
`assess_expiry()`, and only then looks for a direction.

Expiry day is characterised by gamma dominating delta and by theta bleeding
premium fast. Both cut against option buyers, so the bar for taking a trade is
raised, not lowered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.types import Direction, Regime
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class ExpiryVerdict(str, Enum):
    TRADE_EXPIRY = "TRADE_EXPIRY"
    AVOID_EXPIRY = "AVOID_EXPIRY"
    NOT_EXPIRY = "NOT_EXPIRY"


@dataclass(frozen=True)
class ExpiryAssessment:
    verdict: ExpiryVerdict
    reasons: tuple[str, ...]
    score: float
    days_to_expiry: int | None


class ExpiryStrategy(Strategy):
    name = "expiry"
    min_bars = 40

    @staticmethod
    def default_params() -> dict:
        return {
            "expiry_days_threshold": 1,
            "min_underlying_move_atr": 0.8,
            "max_theta_to_premium": 0.10,
            "min_iv": 0.08,
            "max_iv": 0.90,
            "min_expected_move_pct": 0.002,
            "min_trend_strength": 20.0,
            "min_time_remaining_minutes": 60,
            "stop_atr": 0.8,
            "target_atr": 1.8,
            "holding_minutes": 20,
            "min_confidence": 0.45,
        }

    # ------------------------------------------------------------------ #
    def assess_expiry(self, context: StrategyContext) -> ExpiryAssessment:
        """Decide TRADE_EXPIRY vs AVOID_EXPIRY from measured conditions (REQ 12.9)."""
        p = self.params
        days = context.days_to_expiry
        if days is None or days > p["expiry_days_threshold"]:
            return ExpiryAssessment(ExpiryVerdict.NOT_EXPIRY, ("not an expiry session",), 0.0, days)

        reasons: list[str] = []
        blockers: list[str] = []
        score = 0.0

        minutes_left = context.session_context.get("minutes_until_close", np.nan)
        if np.isfinite(minutes_left) and minutes_left < p["min_time_remaining_minutes"]:
            # Too little time left for a directional thesis to play out, while theta
            # keeps working against a long option position.
            blockers.append(
                f"only {minutes_left:.0f} minutes remain, below the "
                f"{p['min_time_remaining_minutes']} minute minimum"
            )
        elif np.isfinite(minutes_left):
            score += 0.2
            reasons.append(f"{minutes_left:.0f} minutes of session remain")

        adx = context.feature("adx", np.nan)
        if np.isfinite(adx):
            if adx >= p["min_trend_strength"]:
                score += 0.3
                reasons.append(f"ADX {adx:.1f} indicates a directional expiry session")
            else:
                blockers.append(
                    f"ADX {adx:.1f} below {p['min_trend_strength']}: choppy expiry sessions "
                    "punish option buyers"
                )

        options = context.option_features
        if options is not None:
            if options.atm_iv is not None:
                if options.atm_iv < p["min_iv"]:
                    blockers.append(f"ATM IV {options.atm_iv:.2f} is too low to price a move")
                elif options.atm_iv > p["max_iv"]:
                    blockers.append(f"ATM IV {options.atm_iv:.2f} is too expensive to buy")
                else:
                    score += 0.25
                    reasons.append(f"ATM IV {options.atm_iv:.2f} is within a workable band")

            if options.expected_move_pct is not None:
                if options.expected_move_pct < p["min_expected_move_pct"]:
                    blockers.append(
                        f"straddle-implied move {options.expected_move_pct:.2%} is too small "
                        "to cover costs"
                    )
                else:
                    score += 0.25
                    reasons.append(
                        f"straddle implies a {options.expected_move_pct:.2%} move"
                    )
        else:
            blockers.append("no option chain available to assess expiry conditions")

        vol_expansion = context.feature("vol_expansion", np.nan)
        if np.isfinite(vol_expansion) and vol_expansion >= 1.2:
            score += 0.2
            reasons.append(f"volatility expanding into expiry ({vol_expansion:.2f}x)")

        if blockers:
            return ExpiryAssessment(
                ExpiryVerdict.AVOID_EXPIRY, tuple(blockers), float(np.clip(score, 0.0, 1.0)), days
            )
        return ExpiryAssessment(
            ExpiryVerdict.TRADE_EXPIRY, tuple(reasons), float(np.clip(score, 0.0, 1.0)), days
        )

    # ------------------------------------------------------------------ #
    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        assessment = self.assess_expiry(context)
        if assessment.verdict is ExpiryVerdict.NOT_EXPIRY:
            return no_signal(
                self.name, context.underlying, "not an expiry session", timestamp=context.timestamp
            )
        if assessment.verdict is ExpiryVerdict.AVOID_EXPIRY:
            return no_signal(
                self.name,
                context.underlying,
                "AVOID_EXPIRY: " + "; ".join(assessment.reasons),
                timestamp=context.timestamp,
            )

        p = self.params
        di_spread = context.feature("di_spread", np.nan)
        macd_hist = context.feature("macd_hist", np.nan)
        return_5 = context.feature("return_5", np.nan)
        vwap_position = context.feature("above_vwap", np.nan)

        votes = [
            np.sign(di_spread) if np.isfinite(di_spread) and abs(di_spread) > 5 else 0.0,
            np.sign(macd_hist) if np.isfinite(macd_hist) and abs(macd_hist) > 0.02 else 0.0,
            np.sign(return_5) if np.isfinite(return_5) and abs(return_5) > 0.0008 else 0.0,
            (1.0 if vwap_position > 0.5 else -1.0) if np.isfinite(vwap_position) else 0.0,
        ]
        active = [v for v in votes if v != 0]
        if len(active) < 3 or abs(sum(active)) < len(active):
            # Expiry sessions are unforgiving: require near-unanimous agreement.
            return no_signal(
                self.name,
                context.underlying,
                "expiry conditions are tradable but direction is not unanimous",
                timestamp=context.timestamp,
            )

        direction = Direction.LONG if sum(active) > 0 else Direction.SHORT
        reasoning = ["TRADE_EXPIRY: " + "; ".join(assessment.reasons)]
        reasoning.append(f"all {len(active)} directional inputs agree on {direction.value.lower()}")

        move_atr = abs(return_5) / (context.atr / context.last_price) if np.isfinite(return_5) and context.atr > 0 and context.last_price > 0 else 0.0
        move_score = float(np.clip(move_atr / p["min_underlying_move_atr"], 0.0, 1.0))
        fit = self.fit(context.regime)
        expiry_probability = context.regime.probability(Regime.EXPIRY_DRIVEN)

        confidence = combine_confidence(
            (assessment.score, 0.30),
            (1.0, 0.20),  # unanimity, already enforced above
            (move_score, 0.15),
            (float(np.clip(expiry_probability * 1.5, 0.0, 1.0)), 0.15),
            (fit, 0.20),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"expiry confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

        # Tighter stops and nearer targets on expiry: holding through adverse gamma
        # is what turns a small expiry loss into a total premium loss.
        entry, stop, target = self._levels(
            context, direction, stop_atr=p["stop_atr"], target_atr=p["target_atr"]
        )
        return StrategySignal(
            strategy=self.name,
            underlying=context.underlying,
            direction=direction,
            confidence=confidence,
            expected_move_atr=p["target_atr"],
            proposed_entry=entry,
            proposed_stop=stop,
            proposed_target=target,
            expected_holding_minutes=p["holding_minutes"],
            regime_fit=fit,
            reasoning=reasoning,
            evidence={
                "expiry_score": assessment.score,
                "days_to_expiry": float(assessment.days_to_expiry or 0),
                "direction_votes": float(sum(active)),
            },
            timestamp=context.timestamp,
        )
