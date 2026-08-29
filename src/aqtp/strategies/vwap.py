"""VWAP strategy (REQ 12.7).

Handles the four distinct VWAP behaviours the requirement names — trend, reclaim,
rejection and mean reversion — and selects between them using the VWAP slope. VWAP
is a trend anchor when it is sloping and a magnet when it is flat, so the same
distance from VWAP means opposite things in the two cases. Deciding which regime
VWAP itself is in, before deciding direction, is the core of this strategy.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class VWAPStrategy(Strategy):
    name = "vwap"
    min_bars = 60

    @staticmethod
    def default_params() -> dict:
        return {
            "flat_slope_threshold": 0.00015,
            "min_distance_atr": 0.4,
            "max_trend_distance_atr": 1.8,
            "reversion_distance_atr": 1.6,
            "reclaim_lookback": 5,
            "stop_atr": 1.0,
            "target_atr": 2.0,
            "holding_minutes": 35,
            "min_confidence": 0.38,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        vwap = context.feature("vwap")
        distance_atr = context.feature("vwap_dist_atr")
        slope = context.feature("vwap_slope")
        above = context.feature("above_vwap", np.nan)

        if not np.isfinite(vwap) or not np.isfinite(distance_atr):
            return no_signal(
                self.name, context.underlying, "VWAP unavailable for this instrument",
                timestamp=context.timestamp,
            )

        frame = context.frame(context.entry_timeframe)
        lookback = int(p["reclaim_lookback"])
        recent_close = frame["close"].iloc[-lookback:] if len(frame) >= lookback else frame["close"]

        vwap_is_trending = np.isfinite(slope) and abs(slope) > p["flat_slope_threshold"]
        mode: str
        direction: Direction
        reasoning: list[str] = []

        if vwap_is_trending:
            # --- trend / reclaim / rejection --------------------------------
            trend_direction = Direction.LONG if slope > 0 else Direction.SHORT
            price_above = above > 0.5 if np.isfinite(above) else distance_atr > 0

            if (trend_direction is Direction.LONG) != price_above:
                # Price on the wrong side of a sloping VWAP: this is a rejection
                # setup against the VWAP trend, which we do not take.
                return no_signal(
                    self.name,
                    context.underlying,
                    f"price is {'below' if not price_above else 'above'} a "
                    f"{'rising' if slope > 0 else 'falling'} VWAP — no aligned setup",
                    timestamp=context.timestamp,
                )

            if abs(distance_atr) > p["max_trend_distance_atr"]:
                return no_signal(
                    self.name,
                    context.underlying,
                    f"price extended {abs(distance_atr):.2f} ATR from VWAP; entry would be chasing",
                    timestamp=context.timestamp,
                )

            direction = trend_direction
            # A reclaim is a stronger entry than simply riding above VWAP.
            crossed = (
                (recent_close < vwap).any() if direction is Direction.LONG else (recent_close > vwap).any()
            )
            if crossed and abs(distance_atr) <= p["min_distance_atr"] * 2:
                mode = "reclaim"
                reasoning.append(
                    f"price reclaimed VWAP within the last {lookback} bars with VWAP sloping "
                    f"{'up' if slope > 0 else 'down'}"
                )
            else:
                mode = "trend"
                reasoning.append(
                    f"price holding {'above' if direction is Direction.LONG else 'below'} a "
                    f"{'rising' if slope > 0 else 'falling'} VWAP ({abs(distance_atr):.2f} ATR away)"
                )
        else:
            # --- mean reversion to a flat VWAP -------------------------------
            if abs(distance_atr) < p["reversion_distance_atr"]:
                return no_signal(
                    self.name,
                    context.underlying,
                    f"VWAP is flat and price is only {abs(distance_atr):.2f} ATR away; "
                    f"need {p['reversion_distance_atr']}",
                    timestamp=context.timestamp,
                )
            mode = "reversion"
            direction = Direction.SHORT if distance_atr > 0 else Direction.LONG
            reasoning.append(
                f"VWAP is flat (slope {slope:+.5f}) and price is {abs(distance_atr):.2f} ATR "
                f"{'above' if distance_atr > 0 else 'below'} it"
            )

        mode_score = {"reclaim": 0.85, "trend": 0.70, "reversion": 0.65}[mode]

        slope_score = (
            float(np.clip(abs(slope) / (p["flat_slope_threshold"] * 6), 0.0, 1.0))
            if np.isfinite(slope)
            else 0.4
        )
        if mode == "reversion":
            # For reversion, a *flatter* VWAP is better evidence.
            slope_score = 1.0 - slope_score

        distance_score = (
            float(np.clip(abs(distance_atr) / p["reversion_distance_atr"], 0.0, 1.0))
            if mode == "reversion"
            else float(np.clip(1.0 - abs(distance_atr) / p["max_trend_distance_atr"], 0.0, 1.0))
        )

        rel_volume = context.feature("rel_volume_20", np.nan)
        volume_score = float(np.clip(rel_volume / 2.0, 0.0, 1.0)) if np.isfinite(rel_volume) else 0.5
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (mode_score, 0.30),
            (slope_score, 0.20),
            (distance_score, 0.20),
            (volume_score, 0.10),
            (fit, 0.20),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"VWAP confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

        entry = context.last_price
        atr = context.atr
        if mode == "reversion":
            # Target VWAP itself — that is the thesis of the trade.
            if direction is Direction.SHORT:
                stop, target = entry + p["stop_atr"] * atr, vwap
            else:
                stop, target = entry - p["stop_atr"] * atr, vwap
        else:
            # VWAP is the invalidation for a trend/reclaim entry.
            buffer = 0.3 * atr
            if direction is Direction.LONG:
                stop = min(vwap - buffer, entry - p["stop_atr"] * atr)
                target = entry + p["target_atr"] * atr
            else:
                stop = max(vwap + buffer, entry + p["stop_atr"] * atr)
                target = entry - p["target_atr"] * atr

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
            evidence={
                "vwap": vwap,
                "vwap_dist_atr": distance_atr,
                "vwap_slope": slope,
                "mode": mode_score,
            },
            timestamp=context.timestamp,
        )
