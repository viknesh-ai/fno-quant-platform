"""Options flow / positioning strategy (REQ 12.8).

REQ 12.8 states plainly: do not assume OI alone predicts direction. This strategy
respects that in a specific way — raw OI *level* is never used as a directional
input. What is used is the **change** in OI interpreted jointly with the price
change, which is the only reading that distinguishes the four positioning states:

    price up   + OI up   -> long build-up      (bullish)
    price down + OI up   -> short build-up     (bearish)
    price up   + OI down -> short covering     (weak bullish, fades)
    price down + OI down -> long unwinding     (weak bearish, fades)

Even then the strategy requires corroboration from price action, and it refuses to
act on positioning alone.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class OptionsFlowStrategy(Strategy):
    name = "options_flow"
    min_bars = 40

    @staticmethod
    def default_params() -> dict:
        return {
            "min_oi_change_ratio": 0.03,
            "min_price_change": 0.0008,
            "pcr_bullish": 1.2,
            "pcr_bearish": 0.7,
            "min_contracts": 10,
            "require_price_confirmation": True,
            "stop_atr": 1.2,
            "target_atr": 2.2,
            "holding_minutes": 45,
            "min_confidence": 0.40,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        options = context.option_features
        if options is None:
            return no_signal(
                self.name, context.underlying, "no option chain data for this underlying",
                timestamp=context.timestamp,
            )
        p = self.params
        if options.contracts_analyzed < p["min_contracts"]:
            return no_signal(
                self.name,
                context.underlying,
                f"only {options.contracts_analyzed} contracts available, need {p['min_contracts']}",
                timestamp=context.timestamp,
            )

        price_change = context.feature("return_5", np.nan)
        if not np.isfinite(price_change):
            price_change = context.feature("return_1", 0.0)

        total_oi = options.total_call_oi + options.total_put_oi
        if total_oi <= 0:
            return no_signal(
                self.name, context.underlying, "open interest unavailable",
                timestamp=context.timestamp,
            )

        call_change_ratio = options.call_oi_change / total_oi
        put_change_ratio = options.put_oi_change / total_oi
        net_change_ratio = abs(call_change_ratio) + abs(put_change_ratio)

        if net_change_ratio < p["min_oi_change_ratio"]:
            return no_signal(
                self.name,
                context.underlying,
                f"OI change {net_change_ratio:.3f} of total OI is below the "
                f"{p['min_oi_change_ratio']} threshold — no meaningful repositioning",
                timestamp=context.timestamp,
            )

        # --- classify the positioning state --------------------------------
        # Call writing (call OI up) is bearish pressure; put writing is bullish.
        # This is the joint price/OI reading REQ 12.8 requires.
        state, direction, base_score = self._classify(
            price_change, call_change_ratio, put_change_ratio, p
        )
        if direction is Direction.FLAT:
            return no_signal(
                self.name,
                context.underlying,
                f"positioning state '{state}' does not imply a direction",
                timestamp=context.timestamp,
            )

        reasoning = [f"option positioning reads as {state.replace('_', ' ')}"]

        if p["require_price_confirmation"]:
            if not np.isfinite(price_change) or abs(price_change) < p["min_price_change"]:
                return no_signal(
                    self.name,
                    context.underlying,
                    "price confirmation required but underlying is not moving meaningfully",
                    timestamp=context.timestamp,
                )
            if np.sign(price_change) != direction.sign and state in (
                "long_buildup", "short_buildup"
            ):
                return no_signal(
                    self.name,
                    context.underlying,
                    "underlying price move contradicts the positioning read",
                    timestamp=context.timestamp,
                )
            reasoning.append(f"underlying return {price_change:+.3%} confirms")

        # PCR is corroboration only — extreme PCR is as often a contrarian signal
        # as a confirming one, so it never drives direction by itself.
        pcr_score = 0.5
        if options.pcr_oi > 0:
            if direction is Direction.LONG and options.pcr_oi >= p["pcr_bullish"]:
                pcr_score = 0.75
                reasoning.append(f"PCR {options.pcr_oi:.2f} shows put-heavy positioning")
            elif direction is Direction.SHORT and options.pcr_oi <= p["pcr_bearish"]:
                pcr_score = 0.75
                reasoning.append(f"PCR {options.pcr_oi:.2f} shows call-heavy positioning")

        # Proximity to the OI wall matters: entering long straight into the heaviest
        # call strike is entering into supply.
        wall_score = 0.6
        spot = options.underlying_ltp or context.last_price
        if direction is Direction.LONG and options.resistance_strike and spot > 0:
            headroom = (options.resistance_strike - spot) / spot
            wall_score = float(np.clip(headroom / 0.01, 0.0, 1.0))
            reasoning.append(f"call OI wall at {options.resistance_strike:.0f} ({headroom:+.2%} away)")
        elif direction is Direction.SHORT and options.support_strike and spot > 0:
            headroom = (spot - options.support_strike) / spot
            wall_score = float(np.clip(headroom / 0.01, 0.0, 1.0))
            reasoning.append(f"put OI wall at {options.support_strike:.0f} ({headroom:+.2%} away)")

        magnitude_score = float(np.clip(net_change_ratio / (p["min_oi_change_ratio"] * 4), 0.0, 1.0))
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (base_score, 0.30),
            (magnitude_score, 0.20),
            (pcr_score, 0.15),
            (wall_score, 0.20),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"options flow confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

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
                "call_oi_change_ratio": call_change_ratio,
                "put_oi_change_ratio": put_change_ratio,
                "pcr_oi": options.pcr_oi,
                "price_change": price_change,
            },
            timestamp=context.timestamp,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _classify(
        price_change: float, call_change: float, put_change: float, params: dict
    ) -> tuple[str, Direction, float]:
        """Joint price/OI classification. Returns (state, direction, base score)."""
        threshold = params["min_oi_change_ratio"] / 2
        price_up = np.isfinite(price_change) and price_change > 0
        calls_building = call_change > threshold
        puts_building = put_change > threshold
        calls_unwinding = call_change < -threshold
        puts_unwinding = put_change < -threshold

        # Put writing with price rising: the strongest bullish positioning read.
        if puts_building and price_up and not calls_building:
            return "long_buildup", Direction.LONG, 0.85
        # Call writing with price falling: the strongest bearish read.
        if calls_building and not price_up and not puts_building:
            return "short_buildup", Direction.SHORT, 0.85
        # Unwinding is a weaker, faster-fading signal — scored lower on purpose.
        if calls_unwinding and price_up:
            return "short_covering", Direction.LONG, 0.55
        if puts_unwinding and not price_up:
            return "long_unwinding", Direction.SHORT, 0.55
        if puts_building and calls_building:
            return "two_sided_writing", Direction.FLAT, 0.0
        return "indeterminate", Direction.FLAT, 0.0
