"""Momentum strategy (REQ 12.2).

Momentum here means *acceleration* rather than level: RSI at 70 tells you where
price has been, while rising MACD histogram plus positive price acceleration tells
you the move is still being fed. The strategy therefore weights rate-of-change
terms above absolute oscillator levels, and treats an extreme oscillator with
decelerating momentum as a reason to stand aside.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class MomentumStrategy(Strategy):
    name = "momentum"
    min_bars = 60

    @staticmethod
    def default_params() -> dict:
        return {
            "rsi_bull": 55.0,
            "rsi_bear": 45.0,
            "rsi_exhausted_high": 82.0,
            "rsi_exhausted_low": 18.0,
            "min_macd_hist": 0.05,
            "min_roc": 0.0008,
            "min_volume_ratio": 1.0,
            "stop_atr": 1.0,
            "target_atr": 2.0,
            "holding_minutes": 30,
            "min_confidence": 0.35,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        rsi = context.feature("rsi")
        macd_hist = context.feature("macd_hist")
        roc10 = context.feature("roc_10")
        acceleration = context.feature("acceleration_5")
        volume_accel = context.feature("volume_acceleration", 0.0)
        rel_volume = context.feature("rel_volume_20", 1.0)

        if not np.isfinite(rsi) or not np.isfinite(macd_hist):
            return no_signal(
                self.name, context.underlying, "momentum features unavailable",
                timestamp=context.timestamp,
            )

        bullish = rsi >= p["rsi_bull"] and macd_hist >= p["min_macd_hist"]
        bearish = rsi <= p["rsi_bear"] and macd_hist <= -p["min_macd_hist"]
        if not bullish and not bearish:
            return no_signal(
                self.name,
                context.underlying,
                f"RSI {rsi:.1f} / MACD hist {macd_hist:+.3f} do not agree on momentum",
                timestamp=context.timestamp,
            )

        direction = Direction.LONG if bullish else Direction.SHORT

        # Exhaustion filter: an extreme oscillator with momentum already fading is
        # where late entries get trapped.
        if direction is Direction.LONG and rsi >= p["rsi_exhausted_high"] and macd_hist <= 0:
            return no_signal(
                self.name, context.underlying,
                f"RSI {rsi:.1f} exhausted while MACD histogram is rolling over",
                timestamp=context.timestamp,
            )
        if direction is Direction.SHORT and rsi <= p["rsi_exhausted_low"] and macd_hist >= 0:
            return no_signal(
                self.name, context.underlying,
                f"RSI {rsi:.1f} exhausted while MACD histogram is turning up",
                timestamp=context.timestamp,
            )

        if np.isfinite(roc10) and abs(roc10) < p["min_roc"]:
            return no_signal(
                self.name, context.underlying,
                f"rate of change {roc10:+.4f} below minimum {p['min_roc']}",
                timestamp=context.timestamp,
            )

        reasoning = [
            f"RSI {rsi:.1f} and MACD histogram {macd_hist:+.3f} both favour "
            f"{direction.value.lower()}",
        ]

        rsi_score = float(np.clip(abs(rsi - 50.0) / 25.0, 0.0, 1.0))
        macd_score = float(np.clip(abs(macd_hist) / 0.5, 0.0, 1.0))
        roc_score = float(np.clip(abs(roc10) / (p["min_roc"] * 5), 0.0, 1.0)) if np.isfinite(roc10) else 0.4

        accel_score = 0.5
        if np.isfinite(acceleration):
            aligned = np.sign(acceleration) == direction.sign
            accel_score = 0.85 if aligned else 0.2
            reasoning.append(
                "price acceleration confirms" if aligned else "price acceleration is fading"
            )

        volume_score = 0.5
        if np.isfinite(rel_volume):
            volume_score = float(np.clip(rel_volume / max(p["min_volume_ratio"], 0.1) / 2.0, 0.0, 1.0))
            if rel_volume >= p["min_volume_ratio"] * 1.5:
                reasoning.append(f"relative volume {rel_volume:.2f}x supports the move")
        if np.isfinite(volume_accel) and volume_accel > 0:
            volume_score = min(1.0, volume_score + 0.15)

        fit = self.fit(context.regime)
        confidence = combine_confidence(
            (rsi_score, 0.20),
            (macd_score, 0.25),
            (roc_score, 0.15),
            (accel_score, 0.20),
            (volume_score, 0.10),
            (fit, 0.10),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"momentum confidence {confidence:.2f} below {p['min_confidence']}",
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
                "rsi": rsi,
                "macd_hist": macd_hist,
                "roc_10": roc10,
                "acceleration_5": acceleration,
                "rel_volume_20": rel_volume,
            },
            timestamp=context.timestamp,
        )
