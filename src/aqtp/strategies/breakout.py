"""Breakout strategy (REQ 12.3).

Detects opening-range, previous-day, consolidation and volatility breakouts, and
requires confirmation from volume, volatility and momentum before acting.

The confirmation requirement is the substance of this strategy. An unconfirmed
break of a level is the most common false signal in intraday F&O, so a break with
no volume expansion is explicitly reported as a *rejected* setup rather than a
low-confidence one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.clock import minutes_since_open
from ..core.types import Direction, Timeframe
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class BreakoutStrategy(Strategy):
    name = "breakout"
    min_bars = 60

    @staticmethod
    def default_params() -> dict:
        return {
            "opening_range_minutes": 15,
            "lookback_bars": 20,
            "min_range_position": 0.92,
            "min_volume_ratio": 1.4,
            "min_vol_expansion": 1.1,
            "require_volume_confirmation": True,
            "stop_atr": 1.0,
            "target_atr": 2.2,
            "holding_minutes": 40,
            "min_confidence": 0.40,
            "max_extension_atr": 2.0,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        frame = context.frame(context.entry_timeframe)
        price = context.last_price
        atr = context.atr

        levels = self._breakout_levels(context, frame)
        if not levels:
            return no_signal(
                self.name, context.underlying, "no breakout reference levels available",
                timestamp=context.timestamp,
            )

        broken: list[tuple[str, float, Direction]] = []
        for label, level in levels.items():
            if not np.isfinite(level) or level <= 0:
                continue
            if price > level and label.endswith("_high"):
                broken.append((label, level, Direction.LONG))
            elif price < level and label.endswith("_low"):
                broken.append((label, level, Direction.SHORT))

        if not broken:
            return no_signal(
                self.name, context.underlying, "price has not cleared any reference level",
                timestamp=context.timestamp,
            )

        directions = {d for _, _, d in broken}
        if len(directions) > 1:
            # Simultaneously above a high and below a low means the levels are stale
            # or the instrument gapped through both; neither is tradable.
            return no_signal(
                self.name, context.underlying, "conflicting breakout directions",
                timestamp=context.timestamp,
            )
        direction = directions.pop()

        # Nearest broken level is the invalidation point.
        label, level, _ = min(broken, key=lambda item: abs(price - item[1]))
        extension_atr = abs(price - level) / atr if atr > 0 else 0.0
        if extension_atr > p["max_extension_atr"]:
            return no_signal(
                self.name,
                context.underlying,
                f"price already extended {extension_atr:.1f} ATR beyond {label}; chasing rejected",
                timestamp=context.timestamp,
            )

        rel_volume = context.feature("rel_volume_20", np.nan)
        vol_expansion = context.feature("vol_expansion", np.nan)
        macd_hist = context.feature("macd_hist", np.nan)
        range_position = context.feature(f"range_position_{p['lookback_bars']}", np.nan)

        if p["require_volume_confirmation"]:
            if not np.isfinite(rel_volume):
                return no_signal(
                    self.name, context.underlying,
                    "volume confirmation required but volume data is unavailable",
                    timestamp=context.timestamp,
                )
            if rel_volume < p["min_volume_ratio"]:
                return no_signal(
                    self.name,
                    context.underlying,
                    f"breakout of {label} unconfirmed: relative volume {rel_volume:.2f}x "
                    f"below {p['min_volume_ratio']}x",
                    timestamp=context.timestamp,
                )

        reasoning = [
            f"price {price:.2f} cleared {label} at {level:.2f} "
            f"({extension_atr:.2f} ATR of extension)",
        ]
        if np.isfinite(rel_volume):
            reasoning.append(f"relative volume {rel_volume:.2f}x confirms participation")

        momentum_score = 0.5
        if np.isfinite(macd_hist):
            aligned = np.sign(macd_hist) == direction.sign
            momentum_score = 0.85 if aligned else 0.15
            if aligned:
                reasoning.append("momentum agrees with the breakout direction")
            else:
                reasoning.append("momentum does not confirm the breakout")

        volume_score = float(np.clip(rel_volume / (p["min_volume_ratio"] * 2), 0.0, 1.0)) if np.isfinite(rel_volume) else 0.4
        volatility_score = (
            float(np.clip(vol_expansion / (p["min_vol_expansion"] * 1.6), 0.0, 1.0))
            if np.isfinite(vol_expansion)
            else 0.4
        )
        if np.isfinite(vol_expansion) and vol_expansion >= p["min_vol_expansion"]:
            reasoning.append(f"volatility expanding ({vol_expansion:.2f}x)")

        position_score = 0.5
        if np.isfinite(range_position):
            position_score = range_position if direction is Direction.LONG else 1.0 - range_position

        # A fresh break is worth more than a stale one.
        freshness = float(np.clip(1.0 - extension_atr / p["max_extension_atr"], 0.0, 1.0))
        fit = self.fit(context.regime)

        confidence = combine_confidence(
            (volume_score, 0.25),
            (momentum_score, 0.20),
            (volatility_score, 0.15),
            (position_score, 0.10),
            (freshness, 0.15),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"breakout confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

        # The broken level is the natural invalidation — a return through it means
        # the breakout failed, which is more meaningful than a fixed ATR stop.
        entry = price
        buffer = 0.25 * atr
        if direction is Direction.LONG:
            stop = min(level - buffer, entry - p["stop_atr"] * atr)
            target = entry + p["target_atr"] * atr
        else:
            stop = max(level + buffer, entry + p["stop_atr"] * atr)
            target = entry - p["target_atr"] * atr

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
                "broken_level": level,
                "extension_atr": extension_atr,
                "rel_volume_20": rel_volume,
                "vol_expansion": vol_expansion,
            },
            timestamp=context.timestamp,
        )

    # ------------------------------------------------------------------ #
    def _breakout_levels(self, context: StrategyContext, frame: pd.DataFrame) -> dict[str, float]:
        """Assemble the reference levels a breakout can occur against.

        All levels come from *closed* bars strictly before the current one, so a
        level can never be defined by the very bar that breaks it.
        """
        p = self.params
        levels: dict[str, float] = {}
        if frame.empty:
            return levels

        lookback = int(p["lookback_bars"])
        if len(frame) > lookback:
            prior = frame.iloc[-(lookback + 1) : -1]
            levels["consolidation_high"] = float(prior["high"].max())
            levels["consolidation_low"] = float(prior["low"].min())

        # Opening range, built from bars within N minutes of the open.
        try:
            today = frame.index[-1].normalize()
            session = frame[frame.index >= today]
            if not session.empty:
                cutoff = int(p["opening_range_minutes"])
                opening = session[
                    [minutes_since_open(ts.to_pydatetime()) <= cutoff for ts in session.index]
                ]
                # Only usable once the opening range has actually closed.
                if not opening.empty and len(session) > len(opening):
                    levels["opening_range_high"] = float(opening["high"].max())
                    levels["opening_range_low"] = float(opening["low"].min())
        except (AttributeError, TypeError):
            pass

        # Previous session's extremes.
        daily = context.frame(Timeframe.D1)
        if len(daily) >= 2:
            previous = daily.iloc[-2]
            levels["previous_day_high"] = float(previous["high"])
            levels["previous_day_low"] = float(previous["low"])

        return levels
