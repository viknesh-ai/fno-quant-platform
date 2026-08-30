"""Volume-at-price structure.

Moving averages describe where price has been in time. A volume profile describes
where it has been in *value* — which prices the market actually agreed on, and
which it passed through in a hurry. Those are different questions, and the second
one is what tells you whether a level is likely to hold.

Built from OHLCV candles: each bar's volume is spread across its high–low range,
weighted toward the close, because the close is the only price in a bar we know
was transacted at the end of it. That approximation is standard for bar data and
is honest about being an approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VolumeProfile:
    """Where volume actually traded, and where price sits relative to it."""

    poc: float | None = None              # point of control — heaviest price
    value_area_high: float | None = None
    value_area_low: float | None = None
    value_area_pct: float = 0.70
    high_volume_nodes: list[float] = field(default_factory=list)
    low_volume_nodes: list[float] = field(default_factory=list)
    initial_balance_high: float | None = None
    initial_balance_low: float | None = None
    session_vwap: float | None = None
    vwap_upper_1: float | None = None
    vwap_lower_1: float | None = None
    vwap_upper_2: float | None = None
    vwap_lower_2: float | None = None
    total_volume: float = 0.0
    bins: int = 0
    price: float | None = None
    reasons: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def location(self) -> str:
        """Where price is relative to the value area — the single most useful
        line of a profile."""
        if self.price is None or self.value_area_high is None or self.value_area_low is None:
            return "unknown"
        if self.price > self.value_area_high:
            return "above_value"
        if self.price < self.value_area_low:
            return "below_value"
        return "inside_value"

    @property
    def distance_to_poc_pct(self) -> float | None:
        if self.price is None or self.poc is None or self.price <= 0:
            return None
        return (self.price - self.poc) / self.price

    @property
    def structure_score(self) -> float:
        """Signed [-1, 1] structural bias.

        Acceptance above value is bullish, rejection back inside is not. The
        magnitude scales with how far outside value price has been able to hold.
        """
        if self.price is None or self.value_area_high is None or self.value_area_low is None:
            return 0.0
        width = self.value_area_high - self.value_area_low
        if width <= 0:
            return 0.0
        if self.price > self.value_area_high:
            return float(np.clip((self.price - self.value_area_high) / width, 0.0, 1.0))
        if self.price < self.value_area_low:
            return float(-np.clip((self.value_area_low - self.price) / width, 0.0, 1.0))
        # Inside value: mild pull toward the POC, which is where the auction is
        # balanced. Fading the edges of value is a mean-reversion setup.
        if self.poc:
            return float(np.clip((self.poc - self.price) / width, -0.5, 0.5))
        return 0.0

    @property
    def nearest_hvn(self) -> float | None:
        if self.price is None or not self.high_volume_nodes:
            return None
        return min(self.high_volume_nodes, key=lambda level: abs(level - self.price))

    @property
    def nearest_lvn(self) -> float | None:
        """The nearest low-volume node.

        Price moves fast through LVNs because nobody has inventory there, which
        makes the far side of an LVN a realistic first target.
        """
        if self.price is None or not self.low_volume_nodes:
            return None
        return min(self.low_volume_nodes, key=lambda level: abs(level - self.price))

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"vp_structure_score": self.structure_score}
        if self.distance_to_poc_pct is not None:
            out["vp_distance_to_poc_pct"] = self.distance_to_poc_pct
        if self.price and self.value_area_high and self.value_area_low:
            out["vp_value_area_width_pct"] = (self.value_area_high - self.value_area_low) / self.price
        if self.price and self.session_vwap:
            out["vp_vwap_deviation_pct"] = (self.price - self.session_vwap) / self.price
        out["vp_above_value"] = 1.0 if self.location == "above_value" else 0.0
        out["vp_below_value"] = 1.0 if self.location == "below_value" else 0.0
        return out

    def explain(self) -> list[str]:
        return list(self.reasons)


# --------------------------------------------------------------------------- #
def build_volume_profile(
    frame: pd.DataFrame,
    *,
    bins: int = 60,
    value_area_pct: float = 0.70,
    initial_balance_bars: int = 12,
    price: float | None = None,
) -> VolumeProfile:
    """Build a volume profile from an OHLCV frame.

    `initial_balance_bars` counts bars, not minutes — with a 5-minute entry
    timeframe, 12 bars is the conventional first hour.
    """
    profile = VolumeProfile(value_area_pct=value_area_pct)
    if frame is None or frame.empty:
        profile.reasons.append("no candles available for a volume profile")
        return profile
    if not {"high", "low", "close", "volume"}.issubset(frame.columns):
        profile.reasons.append("candles lack the OHLCV columns a profile needs")
        return profile

    high = frame["high"].astype(float).to_numpy()
    low = frame["low"].astype(float).to_numpy()
    close = frame["close"].astype(float).to_numpy()
    volume = frame["volume"].astype(float).to_numpy()

    finite = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    high, low, close, volume = high[finite], low[finite], close[finite], np.nan_to_num(volume[finite])
    if high.size < 5:
        profile.reasons.append(f"only {high.size} usable bars — profile suppressed")
        return profile

    profile.price = float(price if price is not None else close[-1])
    top, bottom = float(high.max()), float(low.min())
    if not math.isfinite(top) or not math.isfinite(bottom) or top <= bottom:
        profile.reasons.append("degenerate price range — profile suppressed")
        return profile

    bins = max(10, min(bins, 200))
    edges = np.linspace(bottom, top, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    histogram = np.zeros(bins, dtype=float)

    if volume.sum() <= 0:
        # No volume (indices often report none): fall back to time-at-price, which
        # still identifies acceptance, just without the conviction weighting.
        volume = np.ones_like(volume)
        profile.reasons.append("no volume reported — profile built from time at price")

    for bar_high, bar_low, bar_close, bar_volume in zip(high, low, close, volume):
        if bar_volume <= 0:
            continue
        lo_index = int(np.searchsorted(edges, bar_low, side="right") - 1)
        hi_index = int(np.searchsorted(edges, bar_high, side="right") - 1)
        lo_index = max(0, min(bins - 1, lo_index))
        hi_index = max(0, min(bins - 1, hi_index))
        if hi_index < lo_index:
            lo_index, hi_index = hi_index, lo_index
        span = hi_index - lo_index + 1
        if span == 1:
            histogram[lo_index] += bar_volume
            continue
        # 60% spread uniformly across the bar's range, 40% concentrated at the
        # close — the one price we know cleared.
        histogram[lo_index : hi_index + 1] += (bar_volume * 0.6) / span
        close_index = int(np.searchsorted(edges, bar_close, side="right") - 1)
        histogram[max(0, min(bins - 1, close_index))] += bar_volume * 0.4

    profile.bins = bins
    profile.total_volume = float(histogram.sum())
    if profile.total_volume <= 0:
        profile.reasons.append("profile is empty")
        return profile

    poc_index = int(np.argmax(histogram))
    profile.poc = float(centres[poc_index])

    # Value area: expand from the POC, always taking the heavier neighbour, until
    # the requested share of volume is enclosed. This is the standard Steidlmayer
    # construction and it is not the same as a percentile band.
    target = profile.total_volume * value_area_pct
    accumulated = histogram[poc_index]
    lower, upper = poc_index, poc_index
    while accumulated < target and (lower > 0 or upper < bins - 1):
        below = histogram[lower - 1] if lower > 0 else -1.0
        above = histogram[upper + 1] if upper < bins - 1 else -1.0
        if above >= below:
            upper += 1
            accumulated += histogram[upper]
        else:
            lower -= 1
            accumulated += histogram[lower]
    profile.value_area_low = float(edges[lower])
    profile.value_area_high = float(edges[upper + 1])

    mean_volume = float(histogram.mean())
    for index, amount in enumerate(histogram):
        if amount >= mean_volume * 1.6:
            profile.high_volume_nodes.append(float(centres[index]))
        elif 0 < amount <= mean_volume * 0.35:
            profile.low_volume_nodes.append(float(centres[index]))

    if high.size >= initial_balance_bars:
        profile.initial_balance_high = float(high[:initial_balance_bars].max())
        profile.initial_balance_low = float(low[:initial_balance_bars].min())

    typical = (high + low + close) / 3.0
    cumulative_volume = float(volume.sum())
    if cumulative_volume > 0:
        vwap = float((typical * volume).sum() / cumulative_volume)
        profile.session_vwap = vwap
        variance = float((volume * (typical - vwap) ** 2).sum() / cumulative_volume)
        sigma = math.sqrt(max(variance, 0.0))
        profile.vwap_upper_1, profile.vwap_lower_1 = vwap + sigma, vwap - sigma
        profile.vwap_upper_2, profile.vwap_lower_2 = vwap + 2 * sigma, vwap - 2 * sigma

    # --- narrative ---------------------------------------------------------
    profile.reasons.append(
        f"POC {profile.poc:,.2f}, value area {profile.value_area_low:,.2f}–"
        f"{profile.value_area_high:,.2f} ({value_area_pct:.0%} of volume)"
    )
    profile.reasons.append(f"price is {profile.location.replace('_', ' ')}")
    if profile.session_vwap and profile.price:
        deviation = (profile.price - profile.session_vwap) / profile.session_vwap
        profile.reasons.append(f"VWAP {profile.session_vwap:,.2f} ({deviation:+.2%} away)")
    nearest_lvn = profile.nearest_lvn
    if nearest_lvn is not None and profile.price:
        profile.reasons.append(
            f"nearest low-volume node at {nearest_lvn:,.2f} — thin, price travels through it"
        )
    if profile.initial_balance_high and profile.price:
        if profile.price > profile.initial_balance_high:
            profile.reasons.append(
                f"trading above the initial balance high {profile.initial_balance_high:,.2f}"
            )
        elif profile.initial_balance_low and profile.price < profile.initial_balance_low:
            profile.reasons.append(
                f"trading below the initial balance low {profile.initial_balance_low:,.2f}"
            )
    return profile
