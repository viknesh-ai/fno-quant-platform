"""Mean-reversion strategy (REQ 12.4).

REQ 12.4 mandates that this strategy disable itself during strong directional
trends. That is enforced twice, deliberately:

  * the regime-fit table gives mean_reversion a fit of 0.0 in trending regimes, and
  * `generate()` performs its own hard trend check and returns FLAT.

The redundancy is intentional — a strategy that fades a real trend is the fastest
way to lose money in this system, so a single misconfigured table entry should not
be able to enable it.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction, Regime
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    min_bars = 60

    @staticmethod
    def default_params() -> dict:
        return {
            "min_vwap_deviation_atr": 1.5,
            "min_zscore": 1.8,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "max_adx": 22.0,
            "max_trend_slope": 0.0005,
            "min_trend_probability_to_block": 0.45,
            "stop_atr": 1.0,
            "target_atr": 1.5,
            "holding_minutes": 25,
            "min_confidence": 0.40,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params

        # --- mandatory trend lockout (REQ 12.4) ---------------------------
        trend_probability = context.regime.probability(Regime.TRENDING_UP) + context.regime.probability(
            Regime.TRENDING_DOWN
        )
        if trend_probability >= p["min_trend_probability_to_block"]:
            return no_signal(
                self.name,
                context.underlying,
                f"disabled: trending regime probability {trend_probability:.2f} "
                f">= {p['min_trend_probability_to_block']}",
                timestamp=context.timestamp,
            )
        if context.regime.dominant is Regime.BREAKOUT:
            return no_signal(
                self.name, context.underlying, "disabled: breakout regime",
                timestamp=context.timestamp,
            )

        adx = context.feature("adx")
        if np.isfinite(adx) and adx > p["max_adx"]:
            return no_signal(
                self.name,
                context.underlying,
                f"disabled: ADX {adx:.1f} above {p['max_adx']} indicates a directional market",
                timestamp=context.timestamp,
            )

        slope = context.feature("trend_slope_20")
        if np.isfinite(slope) and abs(slope) > p["max_trend_slope"]:
            return no_signal(
                self.name,
                context.underlying,
                f"disabled: trend slope {slope:+.5f} exceeds {p['max_trend_slope']}",
                timestamp=context.timestamp,
            )

        # --- deviation ----------------------------------------------------
        vwap_dev = context.feature("vwap_dist_atr")
        zscore = context.feature("zscore_close")
        rsi = context.feature("rsi")

        if not np.isfinite(vwap_dev) and not np.isfinite(zscore):
            return no_signal(
                self.name, context.underlying, "no deviation measure available",
                timestamp=context.timestamp,
            )

        stretched_up = (np.isfinite(vwap_dev) and vwap_dev >= p["min_vwap_deviation_atr"]) or (
            np.isfinite(zscore) and zscore >= p["min_zscore"]
        )
        stretched_down = (np.isfinite(vwap_dev) and vwap_dev <= -p["min_vwap_deviation_atr"]) or (
            np.isfinite(zscore) and zscore <= -p["min_zscore"]
        )

        if not stretched_up and not stretched_down:
            return no_signal(
                self.name,
                context.underlying,
                f"price not stretched (VWAP dev {vwap_dev:.2f} ATR, z {zscore:.2f})",
                timestamp=context.timestamp,
            )

        # Fade the stretch: extended above the mean means look for a short.
        direction = Direction.SHORT if stretched_up else Direction.LONG
        reasoning = [
            f"price is {abs(vwap_dev):.2f} ATR {'above' if stretched_up else 'below'} VWAP "
            f"in a non-trending regime",
        ]

        # RSI confirmation makes the fade more credible but is not required — a
        # stretch can be extreme before RSI registers it.
        rsi_score = 0.5
        if np.isfinite(rsi):
            if direction is Direction.SHORT and rsi >= p["rsi_overbought"]:
                rsi_score = 0.9
                reasoning.append(f"RSI {rsi:.1f} is overbought")
            elif direction is Direction.LONG and rsi <= p["rsi_oversold"]:
                rsi_score = 0.9
                reasoning.append(f"RSI {rsi:.1f} is oversold")
            elif (direction is Direction.SHORT and rsi < 50) or (
                direction is Direction.LONG and rsi > 50
            ):
                # RSI already pointing the other way undercuts the fade.
                rsi_score = 0.2
                reasoning.append(f"RSI {rsi:.1f} does not support the reversion")

        deviation_score = float(
            np.clip(abs(vwap_dev) / (p["min_vwap_deviation_atr"] * 2), 0.0, 1.0)
        ) if np.isfinite(vwap_dev) else 0.4
        zscore_score = float(np.clip(abs(zscore) / (p["min_zscore"] * 2), 0.0, 1.0)) if np.isfinite(zscore) else 0.4
        range_score = float(np.clip(context.regime.probability(Regime.RANGE) * 1.5, 0.0, 1.0))
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (deviation_score, 0.30),
            (zscore_score, 0.20),
            (rsi_score, 0.20),
            (range_score, 0.15),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"reversion confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

        # Target the mean (VWAP) rather than a fixed multiple: that is the thesis.
        entry = context.last_price
        atr = context.atr
        vwap = context.feature("vwap", entry)
        if direction is Direction.SHORT:
            stop = entry + p["stop_atr"] * atr
            target = max(vwap, entry - p["target_atr"] * atr) if np.isfinite(vwap) else entry - p["target_atr"] * atr
        else:
            stop = entry - p["stop_atr"] * atr
            target = min(vwap, entry + p["target_atr"] * atr) if np.isfinite(vwap) else entry + p["target_atr"] * atr

        return StrategySignal(
            strategy=self.name,
            underlying=context.underlying,
            direction=direction,
            confidence=confidence,
            expected_move_atr=abs(target - entry) / atr if atr > 0 else 0.0,
            proposed_entry=entry,
            proposed_stop=stop,
            proposed_target=target,
            expected_holding_minutes=p["holding_minutes"],
            regime_fit=fit,
            reasoning=reasoning,
            evidence={"vwap_dist_atr": vwap_dev, "zscore_close": zscore, "rsi": rsi, "adx": adx},
            timestamp=context.timestamp,
        )
