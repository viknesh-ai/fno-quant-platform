"""Trend-following strategy (REQ 12.1).

Works symmetrically long and short. Trend is established on the *setup* timeframe
and confirmed on the entry timeframe, so a fast wiggle cannot by itself create a
position against the higher-timeframe structure.
"""

from __future__ import annotations

import numpy as np

from ..core.types import Direction
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    min_bars = 60

    @staticmethod
    def default_params() -> dict:
        return {
            "adx_min": 22.0,
            "adx_strong": 32.0,
            "min_ema_spread_atr": 0.15,
            "min_slope": 0.0003,
            "stop_atr": 1.2,
            "target_atr": 2.4,
            "holding_minutes": 45,
            "require_vwap_agreement": True,
            "min_confidence": 0.35,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        adx = context.feature("adx")
        di_spread = context.feature("di_spread")
        ema_spread = context.feature("ema_spread_atr")
        slope20 = context.feature("trend_slope_20")
        setup_slope = context.feature_at(context.setup_timeframe, "trend_slope_20", np.nan)
        above_vwap = context.feature("above_vwap", 0.5)
        higher_high = context.feature("higher_high", 0.0)
        higher_low = context.feature("higher_low", 0.0)
        lower_high = context.feature("lower_high", 0.0)
        lower_low = context.feature("lower_low", 0.0)

        if not np.isfinite(adx) or adx < p["adx_min"]:
            return no_signal(
                self.name,
                context.underlying,
                f"ADX {adx:.1f} below trend threshold {p['adx_min']}",
                timestamp=context.timestamp,
            )

        # Direction requires agreement between DI, EMA structure and slope. Any two
        # can agree by chance; requiring a consistent sign across all three is what
        # makes this a trend filter rather than a momentum echo.
        votes = []
        if np.isfinite(di_spread):
            votes.append(np.sign(di_spread))
        if np.isfinite(ema_spread) and abs(ema_spread) >= p["min_ema_spread_atr"]:
            votes.append(np.sign(ema_spread))
        if np.isfinite(slope20) and abs(slope20) >= p["min_slope"]:
            votes.append(np.sign(slope20))

        if len(votes) < 2 or abs(sum(votes)) < len(votes):
            return no_signal(
                self.name,
                context.underlying,
                "trend components disagree on direction",
                timestamp=context.timestamp,
            )

        direction = Direction.LONG if votes[0] > 0 else Direction.SHORT
        reasoning: list[str] = [
            f"ADX {adx:.1f} indicates a trending market",
            f"DI spread {di_spread:+.1f} and EMA spread {ema_spread:+.2f} ATR agree on "
            f"{direction.value.lower()}",
        ]

        # Higher-timeframe agreement is a filter, not a tiebreaker: fighting the
        # setup timeframe is the most reliable way to lose in a trend system.
        if np.isfinite(setup_slope):
            if np.sign(setup_slope) != direction.sign and abs(setup_slope) > p["min_slope"]:
                return no_signal(
                    self.name,
                    context.underlying,
                    f"{context.setup_timeframe.value} slope opposes the entry-timeframe trend",
                    timestamp=context.timestamp,
                )
            reasoning.append(f"{context.setup_timeframe.value} slope confirms")

        if p["require_vwap_agreement"] and np.isfinite(above_vwap):
            wants_above = direction is Direction.LONG
            if (above_vwap > 0.5) != wants_above:
                return no_signal(
                    self.name,
                    context.underlying,
                    "price is on the wrong side of VWAP for this trend direction",
                    timestamp=context.timestamp,
                )
            reasoning.append("VWAP position agrees")

        structure_score = 0.5
        if direction is Direction.LONG and (higher_high > 0 or higher_low > 0):
            structure_score = 0.9
            reasoning.append("market structure shows higher highs/lows")
        elif direction is Direction.SHORT and (lower_high > 0 or lower_low > 0):
            structure_score = 0.9
            reasoning.append("market structure shows lower highs/lows")

        adx_score = float(np.clip((adx - p["adx_min"]) / (p["adx_strong"] - p["adx_min"]), 0.0, 1.0))
        spread_score = float(np.clip(abs(ema_spread) / 1.5, 0.0, 1.0)) if np.isfinite(ema_spread) else 0.4
        slope_score = float(np.clip(abs(slope20) / (p["min_slope"] * 4), 0.0, 1.0)) if np.isfinite(slope20) else 0.4
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (adx_score, 0.30),
            (spread_score, 0.20),
            (slope_score, 0.20),
            (structure_score, 0.15),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name,
                context.underlying,
                f"trend confidence {confidence:.2f} below {p['min_confidence']}",
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
                "adx": adx,
                "di_spread": di_spread,
                "ema_spread_atr": ema_spread,
                "trend_slope_20": slope20,
            },
            timestamp=context.timestamp,
        )
