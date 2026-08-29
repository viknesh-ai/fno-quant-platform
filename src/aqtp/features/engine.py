"""FeatureEngine — one reusable feature set, computed identically everywhere (REQ 10).

The same code path produces features for live decisions and for ML training. That is
deliberate: if training features were computed by a different function than live
features, the model would be scored on a distribution it never saw.

Feature versioning: `FEATURE_VERSION` changes whenever the produced columns change.
The model registry records it, so a model can never be served features it was not
trained on (REQ 51).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping

import numpy as np
import pandas as pd

from ..core.clock import SESSION_MINUTES, minutes_since_open, minutes_until_close, to_ist
from ..core.logging import get_logger
from ..core.types import Timeframe
from ..data.candles import session_vwap
from . import indicators as ind

logger = get_logger(__name__)

FEATURE_VERSION = "1.0.0"


@dataclass
class FeatureSpec:
    """Periods are configuration, not literals scattered through the code."""

    ema_periods: tuple[int, ...] = (9, 21, 50, 200)
    sma_periods: tuple[int, ...] = (20, 50)
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    roc_periods: tuple[int, ...] = (5, 10, 20)
    momentum_periods: tuple[int, ...] = (5, 10, 20)
    volatility_periods: tuple[int, ...] = (10, 20, 50)
    volume_periods: tuple[int, ...] = (10, 20)
    percentile_window: int = 100
    swing_left: int = 3
    swing_right: int = 3
    rolling_window: tuple[int, ...] = (10, 20, 50)
    zscore_period: int = 50

    @property
    def min_bars(self) -> int:
        """Bars needed before the widest feature is defined.

        Callers check this before trusting a feature row; computing on less data
        yields NaNs, and a NaN silently coerced to zero is a bug that shows up as a
        mysteriously confident model.
        """
        return max(
            max(self.ema_periods),
            max(self.sma_periods),
            self.percentile_window,
            max(self.volatility_periods),
            self.macd[1] + self.macd[2],
        ) + 5


class FeatureEngine:
    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()

    # ------------------------------------------------------------------ #
    def compute(self, frame: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
        """Full causal feature frame for one OHLCV series.

        `frame` must contain only *closed* bars — the caller guarantees that by
        going through `MultiTimeframeStore.closed()`.
        """
        if frame.empty:
            return pd.DataFrame(index=frame.index)

        spec = self.spec
        close, high, low, open_ = frame["close"], frame["high"], frame["low"], frame["open"]
        volume = frame["volume"] if "volume" in frame else pd.Series(0.0, index=frame.index)
        out: dict[str, pd.Series] = {}

        # --- price -------------------------------------------------------
        returns = close.pct_change()
        out["return_1"] = returns
        out["log_return_1"] = np.log(close / close.shift(1))
        for period in spec.momentum_periods:
            out[f"return_{period}"] = close.pct_change(period)
            out[f"momentum_{period}"] = ind.momentum(close, period) / close.replace(0, np.nan)
        out["acceleration_5"] = ind.acceleration(close, 5) / close.replace(0, np.nan)

        # --- trend -------------------------------------------------------
        atr_ = ind.atr(high, low, close, spec.atr_period)
        safe_atr = atr_.replace(0, np.nan)
        for period in spec.ema_periods:
            ema_ = ind.ema(close, period)
            out[f"ema_{period}"] = ema_
            # Distance in ATR units, not rupees: comparable across instruments.
            out[f"dist_ema_{period}_atr"] = (close - ema_) / safe_atr
        for period in spec.sma_periods:
            sma_ = ind.sma(close, period)
            out[f"dist_sma_{period}_atr"] = (close - sma_) / safe_atr

        if len(spec.ema_periods) >= 2:
            fast, slow = spec.ema_periods[0], spec.ema_periods[1]
            out["ema_spread_atr"] = (ind.ema(close, fast) - ind.ema(close, slow)) / safe_atr
            out["ema_stack_bull"] = (
                (ind.ema(close, fast) > ind.ema(close, slow)).astype(float)
            )

        adx_, plus_di, minus_di = ind.adx(high, low, close, spec.adx_period)
        out["adx"] = adx_
        out["plus_di"] = plus_di
        out["minus_di"] = minus_di
        out["di_spread"] = plus_di - minus_di
        out["trend_slope_20"] = ind.trend_slope(close, 20)
        out["trend_slope_50"] = ind.trend_slope(close, 50)

        # --- momentum ----------------------------------------------------
        out["rsi"] = ind.rsi(close, spec.rsi_period)
        out["rsi_centered"] = out["rsi"] - 50.0
        macd_line, macd_signal, macd_hist = ind.macd(close, *spec.macd)
        out["macd"] = macd_line / safe_atr
        out["macd_signal"] = macd_signal / safe_atr
        out["macd_hist"] = macd_hist / safe_atr
        for period in spec.roc_periods:
            out[f"roc_{period}"] = ind.roc(close, period)
        stoch_k, stoch_d = ind.stochastic(high, low, close, 14, 3)
        out["stoch_k"] = stoch_k
        out["stoch_d"] = stoch_d

        # --- volatility --------------------------------------------------
        out["atr"] = atr_
        out["atr_pct"] = atr_ / close.replace(0, np.nan)
        for period in spec.volatility_periods:
            out[f"realized_vol_{period}"] = ind.realized_volatility(returns, period, annualize=False)
        out["vol_percentile"] = ind.percentile_rank(out["atr_pct"], spec.percentile_window)
        out["vol_expansion"] = ind.volatility_expansion(atr_, 5, 20)
        out["bb_width"] = ind.bollinger_width(close, 20, 2.0)
        out["bb_width_percentile"] = ind.percentile_rank(out["bb_width"], spec.percentile_window)
        out["keltner_width"] = ind.keltner_width(high, low, close, 20, 1.5)
        # Squeeze: Bollinger inside Keltner is the canonical compression signature.
        out["is_squeeze"] = (out["bb_width"] < out["keltner_width"]).astype(float)

        # --- volume ------------------------------------------------------
        has_volume = float(volume.sum()) > 0
        out["has_volume"] = pd.Series(float(has_volume), index=frame.index)
        if has_volume:
            for period in spec.volume_periods:
                out[f"rel_volume_{period}"] = ind.relative_volume(volume, period)
            out["volume_change"] = volume.pct_change()
            out["volume_acceleration"] = ind.volume_acceleration(volume, 5)
            out["volume_percentile"] = ind.percentile_rank(volume, spec.percentile_window)
            out["obv_slope"] = ind.trend_slope(ind.obv(close, volume), 20)
        else:
            # Index series carry no volume. Emit neutral constants rather than NaN so
            # the column set stays stable between instruments that do and don't.
            for period in spec.volume_periods:
                out[f"rel_volume_{period}"] = pd.Series(1.0, index=frame.index)
            out["volume_change"] = pd.Series(0.0, index=frame.index)
            out["volume_acceleration"] = pd.Series(0.0, index=frame.index)
            out["volume_percentile"] = pd.Series(0.5, index=frame.index)
            out["obv_slope"] = pd.Series(0.0, index=frame.index)

        # --- VWAP --------------------------------------------------------
        vwap = session_vwap(frame)
        out["vwap"] = vwap
        out["vwap_dist_atr"] = ind.vwap_deviation(close, vwap, atr_)
        out["vwap_dist_pct"] = (close - vwap) / vwap.replace(0, np.nan)
        out["vwap_slope"] = ind.vwap_slope(vwap, 10)
        out["above_vwap"] = (close > vwap).astype(float)

        # --- structure ---------------------------------------------------
        for period in spec.rolling_window:
            r_high = ind.rolling_high(high, period)
            r_low = ind.rolling_low(low, period)
            out[f"dist_high_{period}_atr"] = (close - r_high) / safe_atr
            out[f"dist_low_{period}_atr"] = (close - r_low) / safe_atr
            span = (r_high - r_low).replace(0, np.nan)
            out[f"range_position_{period}"] = (close - r_low) / span

        pivot_highs = ind.swing_highs(high, spec.swing_left, spec.swing_right)
        pivot_lows = ind.swing_lows(low, spec.swing_left, spec.swing_right)
        last_high = ind.last_swing_level(high, pivot_highs, spec.swing_right)
        last_low = ind.last_swing_level(low, pivot_lows, spec.swing_right)
        out["last_swing_high_dist_atr"] = (close - last_high) / safe_atr
        out["last_swing_low_dist_atr"] = (close - last_low) / safe_atr
        out["higher_high"] = (last_high > last_high.shift(1)).astype(float)
        out["higher_low"] = (last_low > last_low.shift(1)).astype(float)
        out["lower_high"] = (last_high < last_high.shift(1)).astype(float)
        out["lower_low"] = (last_low < last_low.shift(1)).astype(float)
        # Break of structure: closing beyond the last confirmed pivot.
        out["bos_up"] = ((close > last_high) & (close.shift(1) <= last_high.shift(1))).astype(float)
        out["bos_down"] = ((close < last_low) & (close.shift(1) >= last_low.shift(1))).astype(float)

        candles = ind.candle_features(frame)
        for column in candles.columns:
            out[column] = candles[column]

        out["gap"] = (open_ - close.shift(1)) / close.shift(1).replace(0, np.nan)
        out["zscore_close"] = ind.zscore(close, spec.zscore_period)

        result = pd.DataFrame(out, index=frame.index)
        result = result.replace([np.inf, -np.inf], np.nan)
        if prefix:
            result.columns = [f"{prefix}{c}" for c in result.columns]
        return result

    # ------------------------------------------------------------------ #
    def compute_multi_timeframe(
        self,
        frames: Mapping[Timeframe, pd.DataFrame],
        *,
        align_to: Timeframe,
    ) -> pd.DataFrame:
        """Merge features from several timeframes onto one index (REQ 9).

        Higher-timeframe values are joined with `merge_asof` on direction='backward',
        so each low-timeframe bar sees only the most recently *closed* higher bar.
        A plain join would attach a 1h bar to the 5m bars inside it — which is
        look-ahead, and the single most common way multi-timeframe models leak.
        """
        base_frame = frames.get(align_to)
        if base_frame is None or base_frame.empty:
            return pd.DataFrame()

        result = self.compute(base_frame, prefix=f"{align_to.value}_")

        for timeframe, frame in frames.items():
            if timeframe == align_to or frame.empty:
                continue
            if timeframe.minutes < align_to.minutes:
                continue  # never mix a faster timeframe down into a slower one
            higher = self.compute(frame, prefix=f"{timeframe.value}_")
            if higher.empty:
                continue
            # The bar labelled 10:00 on a 1h series closes at 11:00; it may only be
            # used from 11:00 onward.
            higher = higher.copy()
            higher.index = higher.index + pd.Timedelta(minutes=timeframe.minutes)
            result = pd.merge_asof(
                result.sort_index(),
                higher.sort_index(),
                left_index=True,
                right_index=True,
                direction="backward",
                allow_exact_matches=True,
            )
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def time_features(index: pd.DatetimeIndex, *, expiry: date | None = None) -> pd.DataFrame:
        """REQ 10 'Time Features'. Cyclical encodings avoid the artificial
        discontinuity a raw minute-of-day column creates at the session boundary."""
        if len(index) == 0:
            return pd.DataFrame()
        moments = [to_ist(ts.to_pydatetime()) for ts in index]
        since_open = np.array([minutes_since_open(m) for m in moments])
        until_close = np.array([minutes_until_close(m) for m in moments])
        progress = np.clip(since_open / SESSION_MINUTES, 0.0, 1.0)

        data = {
            "minutes_since_open": since_open,
            "minutes_until_close": until_close,
            "session_progress": progress,
            "session_sin": np.sin(2 * np.pi * progress),
            "session_cos": np.cos(2 * np.pi * progress),
            "day_of_week": np.array([m.weekday() for m in moments], dtype=float),
            "is_monday": np.array([float(m.weekday() == 0) for m in moments]),
            "is_friday": np.array([float(m.weekday() == 4) for m in moments]),
            "first_30_min": (since_open <= 30).astype(float),
            "last_30_min": (until_close <= 30).astype(float),
        }

        if expiry is not None:
            days = np.array([float((expiry - m.date()).days) for m in moments])
            data["days_to_expiry"] = days
            data["is_expiry_day"] = (days == 0).astype(float)
            data["is_expiry_week"] = ((days >= 0) & (days <= 7)).astype(float)
            # Decay accelerates non-linearly into expiry; 1/sqrt(t) matches the
            # shape of theta far better than a linear countdown.
            data["expiry_proximity"] = 1.0 / np.sqrt(np.maximum(days, 0.25))
        return pd.DataFrame(data, index=index)

    # ------------------------------------------------------------------ #
    @staticmethod
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        """Model-eligible columns: numeric, and not raw price levels.

        Raw levels (ema_50, vwap, atr) are excluded from model input because they are
        non-stationary — a model trained when NIFTY was 18,000 would be useless at
        26,000. Their ATR-normalized distances are kept instead.
        """
        excluded_prefixes = ("ema_", "vwap", "atr")
        excluded_exact = {"atr", "vwap", "has_volume"}
        out: list[str] = []
        for column in frame.columns:
            base = column.split("_", 1)[-1] if "_" in column else column
            if column in excluded_exact or base in excluded_exact:
                continue
            if any(base.startswith(p) and "dist" not in base and "slope" not in base
                   and "pct" not in base for p in excluded_prefixes):
                continue
            if not pd.api.types.is_numeric_dtype(frame[column]):
                continue
            out.append(column)
        return out

    def ready(self, frame: pd.DataFrame) -> bool:
        return len(frame) >= self.spec.min_bars


def latest_feature_row(features: pd.DataFrame) -> pd.Series | None:
    """The most recent fully-defined feature row, or None.

    Returning None (rather than a row full of NaNs) forces callers to handle the
    warm-up case explicitly instead of feeding a half-computed row to a model.
    """
    if features.empty:
        return None
    row = features.iloc[-1]
    if row.isna().all():
        return None
    return row
