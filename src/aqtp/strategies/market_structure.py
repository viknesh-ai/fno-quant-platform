"""Market-structure strategy (REQ 12.6).

REQ 12.6 is explicit: discretionary market-structure terminology must be converted
into measurable rules and must not be treated as truth. Accordingly, every concept
below has an operational definition stated in the code:

  swing high        : a bar whose high exceeds the `left` bars before and `right`
                      bars after it — only confirmed `right` bars later.
  break of structure: a *close* beyond the last confirmed pivot in the trend
                      direction (a wick through it is not a break).
  change of character: a break of structure against the prevailing pivot sequence.
  rejection         : a bar that traded through a level intrabar but closed back
                      inside it, with the wick exceeding a configured fraction of
                      the bar's range.
  failed breakout   : a break of structure that closes back through the level
                      within `failure_bars`.
  liquidity sweep   : a rejection at a prior swing extreme, i.e. price took out the
                      level, failed, and closed back — measured, not eyeballed.

None of these are claimed to be predictive on their own; they are inputs whose
value is decided by the backtest and the ML layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.types import Direction
from ..features import indicators as ind
from .base import Strategy, StrategyContext, StrategySignal, combine_confidence, no_signal


class MarketStructureStrategy(Strategy):
    name = "market_structure"
    min_bars = 80

    @staticmethod
    def default_params() -> dict:
        return {
            "swing_left": 3,
            "swing_right": 3,
            "failure_bars": 3,
            "min_rejection_wick": 0.5,
            "sweep_tolerance_atr": 0.15,
            "stop_atr": 1.0,
            "target_atr": 2.2,
            "holding_minutes": 45,
            "min_confidence": 0.40,
        }

    def generate(self, context: StrategyContext) -> StrategySignal:
        blocked = self._ready(context)
        if blocked:
            return no_signal(self.name, context.underlying, blocked, timestamp=context.timestamp)

        p = self.params
        frame = context.frame(context.entry_timeframe)
        if len(frame) < self.min_bars:
            return no_signal(
                self.name, context.underlying, "insufficient bars for structure analysis",
                timestamp=context.timestamp,
            )

        structure = self.analyze_structure(frame, p, context.atr)
        if structure["event"] is None:
            return no_signal(
                self.name,
                context.underlying,
                "no measurable structure event on the latest bars",
                timestamp=context.timestamp,
            )

        event: str = structure["event"]
        direction: Direction = structure["direction"]
        level: float = structure["level"]
        reasoning: list[str] = list(structure["reasoning"])

        # A failed breakout is traded *against* the break — that is its whole point.
        if event == "failed_breakout":
            direction = Direction.SHORT if direction is Direction.LONG else Direction.LONG
            reasoning.append("trading the failure, against the original break")

        event_score = {
            "break_of_structure": 0.85,
            "change_of_character": 0.75,
            "liquidity_sweep": 0.80,
            "failed_breakout": 0.70,
            "rejection": 0.60,
        }.get(event, 0.5)

        trend_alignment = 0.5
        di_spread = context.feature("di_spread", np.nan)
        if np.isfinite(di_spread):
            aligned = np.sign(di_spread) == direction.sign
            trend_alignment = 0.85 if aligned else 0.3
            reasoning.append(
                "directional index agrees" if aligned else "directional index disagrees"
            )

        volume_score = 0.5
        rel_volume = context.feature("rel_volume_20", np.nan)
        if np.isfinite(rel_volume):
            volume_score = float(np.clip(rel_volume / 2.5, 0.0, 1.0))

        fit = self.fit(context.regime)
        confidence = combine_confidence(
            (event_score, 0.35),
            (trend_alignment, 0.25),
            (float(structure["strength"]), 0.15),
            (volume_score, 0.10),
            (fit, 0.15),
        )
        if confidence < p["min_confidence"]:
            return no_signal(
                self.name, context.underlying,
                f"structure confidence {confidence:.2f} below {p['min_confidence']}",
                timestamp=context.timestamp,
            )

        entry = context.last_price
        atr = context.atr
        buffer = p["sweep_tolerance_atr"] * atr
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
                "structure_event": event_score,
                "level": level,
                "strength": float(structure["strength"]),
            },
            timestamp=context.timestamp,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def analyze_structure(frame: pd.DataFrame, params: dict, atr: float) -> dict:
        """Detect the most recent measurable structure event.

        Returns a dict with event name, direction, the level involved, a strength in
        [0, 1] and human-readable reasoning. Separated from `generate` so it can be
        unit-tested against hand-built price sequences.
        """
        high, low, close, open_ = frame["high"], frame["low"], frame["close"], frame["open"]
        left, right = int(params["swing_left"]), int(params["swing_right"])

        pivot_highs = ind.swing_highs(high, left, right)
        pivot_lows = ind.swing_lows(low, left, right)
        last_high = ind.last_swing_level(high, pivot_highs, right)
        last_low = ind.last_swing_level(low, pivot_lows, right)

        result: dict = {"event": None, "direction": Direction.FLAT, "level": 0.0,
                        "strength": 0.0, "reasoning": []}
        # Only bail out when NEITHER pivot type has formed. Requiring both would
        # miss a break of a swing high in a series that has not yet printed a
        # confirmed swing low — each check below guards its own level separately.
        if last_high.isna().all() and last_low.isna().all():
            return result

        current_close = float(close.iloc[-1])
        prior_close = float(close.iloc[-2])
        level_high = float(last_high.iloc[-1]) if np.isfinite(last_high.iloc[-1]) else np.nan
        level_low = float(last_low.iloc[-1]) if np.isfinite(last_low.iloc[-1]) else np.nan
        atr = max(atr, 1e-9)

        # --- failed breakout: broke recently, closed back through ----------
        failure_bars = int(params["failure_bars"])
        window = slice(-(failure_bars + 1), None)
        if np.isfinite(level_high):
            broke_up = (close.iloc[window] > level_high).any()
            if broke_up and current_close < level_high:
                result.update(
                    event="failed_breakout",
                    direction=Direction.LONG,  # the *original* break direction
                    level=level_high,
                    strength=float(np.clip((level_high - current_close) / atr, 0.0, 1.0)),
                    reasoning=[
                        f"price closed above the swing high {level_high:.2f} within the last "
                        f"{failure_bars} bars and has closed back below it"
                    ],
                )
                return result
        if np.isfinite(level_low):
            broke_down = (close.iloc[window] < level_low).any()
            if broke_down and current_close > level_low:
                result.update(
                    event="failed_breakout",
                    direction=Direction.SHORT,
                    level=level_low,
                    strength=float(np.clip((current_close - level_low) / atr, 0.0, 1.0)),
                    reasoning=[
                        f"price closed below the swing low {level_low:.2f} within the last "
                        f"{failure_bars} bars and has closed back above it"
                    ],
                )
                return result

        # --- break of structure: close beyond the last pivot ---------------
        if np.isfinite(level_high) and current_close > level_high >= prior_close:
            result.update(
                event="break_of_structure",
                direction=Direction.LONG,
                level=level_high,
                strength=float(np.clip((current_close - level_high) / atr, 0.0, 1.0)),
                reasoning=[f"close {current_close:.2f} broke the swing high {level_high:.2f}"],
            )
            return result
        if np.isfinite(level_low) and current_close < level_low <= prior_close:
            result.update(
                event="break_of_structure",
                direction=Direction.SHORT,
                level=level_low,
                strength=float(np.clip((level_low - current_close) / atr, 0.0, 1.0)),
                reasoning=[f"close {current_close:.2f} broke the swing low {level_low:.2f}"],
            )
            return result

        # --- liquidity sweep / rejection -----------------------------------
        bar_high, bar_low = float(high.iloc[-1]), float(low.iloc[-1])
        bar_open, bar_close = float(open_.iloc[-1]), current_close
        bar_range = max(bar_high - bar_low, 1e-9)
        upper_wick = (bar_high - max(bar_open, bar_close)) / bar_range
        lower_wick = (min(bar_open, bar_close) - bar_low) / bar_range
        tolerance = params["sweep_tolerance_atr"] * atr

        if (
            np.isfinite(level_high)
            and bar_high >= level_high - tolerance
            and bar_close < level_high
            and upper_wick >= params["min_rejection_wick"]
        ):
            result.update(
                event="liquidity_sweep",
                direction=Direction.SHORT,
                level=level_high,
                strength=float(np.clip(upper_wick, 0.0, 1.0)),
                reasoning=[
                    f"bar swept the swing high {level_high:.2f} and closed back below with a "
                    f"{upper_wick:.0%} upper wick"
                ],
            )
            return result

        if (
            np.isfinite(level_low)
            and bar_low <= level_low + tolerance
            and bar_close > level_low
            and lower_wick >= params["min_rejection_wick"]
        ):
            result.update(
                event="liquidity_sweep",
                direction=Direction.LONG,
                level=level_low,
                strength=float(np.clip(lower_wick, 0.0, 1.0)),
                reasoning=[
                    f"bar swept the swing low {level_low:.2f} and closed back above with a "
                    f"{lower_wick:.0%} lower wick"
                ],
            )
            return result

        return result
