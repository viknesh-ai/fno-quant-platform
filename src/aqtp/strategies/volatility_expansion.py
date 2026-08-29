"""Volatility-expansion strategy (REQ 12.5).

Trades the transition from compression to expansion. Compression alone is not a
signal — it is a *setup*. The trade is taken only when expansion has actually begun
and a direction has resolved, because a squeeze can break either way and guessing
which is not an edge.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class VolatilityExpansionStrategy(Strategy):
    name = "volatility_expansion"
    min_bars = 80

    @staticmethod
    def default_params() -> dict:
        return {
            "compression_lookback": 10,
            "max_compression_percentile": 0.30,
            "min_expansion_ratio": 1.20,
            "min_range_expansion": 1.3,
            "min_volume_ratio": 1.2,
            "min_directional_strength": 0.3,
            "stop_atr": 1.1,
            "target_atr": 2.5,
            "holding_minutes": 40,
            "min_confidence": 0.40,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        frame = context.frame(context.entry_timeframe)
        vol_expansion = context.feature("vol_expansion")
        bb_percentile = context.feature("bb_width_percentile")
        squeeze = context.feature("is_squeeze", 0.0)

        if not np.isfinite(vol_expansion):
            return no_signal(
                self.name, context.underlying, "volatility expansion measure unavailable",
                timestamp=context.timestamp,
            )

        # --- was there a compression to expand out of? ---------------------
        lookback = int(p["compression_lookback"])
        was_compressed = False
        if "bb_width_percentile" in frame.columns:
            recent = frame["bb_width_percentile"].iloc[-lookback:]
            was_compressed = bool((recent <= p["max_compression_percentile"]).any())
        else:
            # bb_width_percentile lives in the feature frame, not the OHLC frame;
            # fall back to the current reading plus the squeeze flag.
            was_compressed = squeeze > 0.5 or (
                np.isfinite(bb_percentile) and bb_percentile <= p["max_compression_percentile"]
            )

        if vol_expansion < p["min_expansion_ratio"]:
            state = "still compressed" if squeeze > 0.5 else "not expanding"
            return no_signal(
                self.name,
                context.underlying,
                f"volatility {state} (expansion ratio {vol_expansion:.2f} < {p['min_expansion_ratio']})",
                timestamp=context.timestamp,
            )

        # --- direction resolution -----------------------------------------
        # Expansion tells us *that* something is happening; these tell us which way.
        di_spread = context.feature("di_spread", 0.0)
        macd_hist = context.feature("macd_hist", 0.0)
        return_5 = context.feature("return_5", 0.0)
        candle_body = context.feature("candle_body", 0.0)

        votes = [
            np.sign(di_spread) if np.isfinite(di_spread) and abs(di_spread) > 3 else 0.0,
            np.sign(macd_hist) if np.isfinite(macd_hist) and abs(macd_hist) > 0.02 else 0.0,
            np.sign(return_5) if np.isfinite(return_5) and abs(return_5) > 0.0005 else 0.0,
            np.sign(candle_body) if np.isfinite(candle_body) and abs(candle_body) > 0.3 else 0.0,
        ]
        net = sum(votes)
        strength = abs(net) / len([v for v in votes if v != 0]) if any(votes) else 0.0

        if strength < p["min_directional_strength"] or net == 0:
            return no_signal(
                self.name,
                context.underlying,
                f"volatility expanding but direction unresolved (agreement {strength:.2f})",
                timestamp=context.timestamp,
            )

        direction = Direction.LONG if net > 0 else Direction.SHORT
        reasoning = [
            f"volatility expanding at {vol_expansion:.2f}x its baseline",
            f"directional agreement {strength:.2f} favours {direction.value.lower()}",
        ]
        if was_compressed:
            reasoning.append("expansion follows a measured compression phase")

        rel_volume = context.feature("rel_volume_20", np.nan)
        volume_score = 0.5
        if np.isfinite(rel_volume):
            volume_score = float(np.clip(rel_volume / (p["min_volume_ratio"] * 2), 0.0, 1.0))
            if rel_volume >= p["min_volume_ratio"]:
                reasoning.append(f"volume {rel_volume:.2f}x supports the expansion")

        expansion_score = float(np.clip(vol_expansion / (p["min_expansion_ratio"] * 1.8), 0.0, 1.0))
        compression_bonus = 0.9 if was_compressed else 0.35
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (expansion_score, 0.30),
            (strength, 0.25),
            (compression_bonus, 0.20),
            (volume_score, 0.10),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"expansion confidence {confidence:.2f} below {p['min_confidence']}",
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
                "vol_expansion": vol_expansion,
                "bb_width_percentile": bb_percentile,
                "direction_agreement": strength,
                "was_compressed": float(was_compressed),
            },
            timestamp=context.timestamp,
        )
