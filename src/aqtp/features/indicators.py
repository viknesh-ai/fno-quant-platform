"""Indicator primitives (REQ 10).

Every function here is *causal*: the value at index `i` depends only on data at
indices <= `i`. There is no centering, no back-filling and no use of `shift(-n)`.
This property is what the leakage tests assert, and it is why the feature engine can
be run identically in backtest and live.

Implementations are pandas/numpy only — no TA library dependency, so behaviour is
inspectable and identical across environments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(values: pd.Series | np.ndarray) -> pd.Series:
    return values if isinstance(values, pd.Series) else pd.Series(values)


# --------------------------------------------------------------------------- #
# Moving averages / trend
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    # adjust=False gives the recursive form, which is what a live incremental
    # update would produce — important for backtest/live parity.
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights) / weights.sum()), raw=True
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's ATR (RMA smoothing), the convention Indian charting tools use."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (adx, plus_di, minus_di)."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr_ = atr(high, low, close, period)
    safe_atr = atr_.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / safe_atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / safe_atr

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator
    adx_ = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx_, plus_di, minus_di


def trend_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Least-squares slope over a rolling window, normalized by price level.

    Normalizing makes the slope comparable between a 200-rupee stock and a
    50,000-point index, which matters because one model scores both.
    """
    x = np.arange(period, dtype=float)
    x_centered = x - x.mean()
    denominator = float((x_centered ** 2).sum())

    def _slope(window: np.ndarray) -> float:
        y = window - window.mean()
        return float(np.dot(x_centered, y) / denominator)

    raw = series.rolling(period, min_periods=period).apply(_slope, raw=True)
    level = series.rolling(period, min_periods=period).mean().replace(0, np.nan)
    return raw / level


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # An all-gain window has infinite RS; RSI is 100 there, not NaN.
    return out.where(avg_loss != 0, 100.0).where(avg_gain != 0, out.fillna(50.0))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(period)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth: int = 3
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / span
    return k, k.rolling(smooth, min_periods=smooth).mean()


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.diff(period)


def acceleration(series: pd.Series, period: int = 5) -> pd.Series:
    """Second derivative of price — the rate at which momentum itself is changing."""
    return series.diff(period).diff(period)


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def realized_volatility(returns: pd.Series, period: int = 20, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(period, min_periods=period).std()
    if annualize:
        # 375 one-minute bars per NSE session x 252 sessions.
        vol = vol * np.sqrt(375 * 252)
    return vol


def rolling_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    return series.pct_change().rolling(period, min_periods=period).std()


def percentile_rank(series: pd.Series, period: int = 100) -> pd.Series:
    """Where the current value sits within its own recent history, in [0, 1].

    Uses only the trailing window, so it is causal — a percentile computed over the
    whole sample would be a classic leak.
    """
    def _rank(window: np.ndarray) -> float:
        current = window[-1]
        return float((window <= current).sum() - 1) / max(1, len(window) - 1)

    return series.rolling(period, min_periods=max(10, period // 4)).apply(_rank, raw=True)


def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    middle = sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    return middle + num_std * std, middle, middle - num_std * std


def bollinger_width(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    upper, middle, lower = bollinger(series, period, num_std)
    return (upper - lower) / middle.replace(0, np.nan)


def keltner_width(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, mult: float = 1.5
) -> pd.Series:
    middle = ema(close, period)
    band = mult * atr(high, low, close, period)
    return (2 * band) / middle.replace(0, np.nan)


def volatility_expansion(atr_series: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Ratio > 1 means volatility is expanding relative to its own recent baseline."""
    fast = atr_series.rolling(short, min_periods=short).mean()
    slow = atr_series.rolling(long, min_periods=long).mean()
    return fast / slow.replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #
def relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    average = volume.rolling(period, min_periods=period).mean().replace(0, np.nan)
    return volume / average


def volume_acceleration(volume: pd.Series, period: int = 5) -> pd.Series:
    return volume.rolling(period, min_periods=period).mean().pct_change(period)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


# --------------------------------------------------------------------------- #
# VWAP
# --------------------------------------------------------------------------- #
def vwap_deviation(close: pd.Series, vwap: pd.Series, atr_series: pd.Series) -> pd.Series:
    """Distance from VWAP expressed in ATR units, so it is comparable across
    instruments and across volatility regimes."""
    return (close - vwap) / atr_series.replace(0, np.nan)


def vwap_slope(vwap: pd.Series, period: int = 10) -> pd.Series:
    return trend_slope(vwap, period)


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #
def rolling_high(high: pd.Series, period: int) -> pd.Series:
    return high.rolling(period, min_periods=period).max()


def rolling_low(low: pd.Series, period: int) -> pd.Series:
    return low.rolling(period, min_periods=period).min()


def swing_highs(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    """Confirmed swing highs, shifted so they are only visible after confirmation.

    A pivot needs `right` bars *after* it to be confirmed. Marking it at its own
    index would let a strategy see a pivot before the market did — so the result is
    shifted forward by `right` bars. That shift is the whole point of this function.
    """
    window = left + right + 1
    is_max = high.rolling(window, min_periods=window).apply(
        lambda w: float(np.argmax(w) == left), raw=True
    )
    return is_max.fillna(0.0).astype(bool)


def swing_lows(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    window = left + right + 1
    is_min = low.rolling(window, min_periods=window).apply(
        lambda w: float(np.argmin(w) == left), raw=True
    )
    return is_min.fillna(0.0).astype(bool)


def last_swing_level(
    price: pd.Series, pivots: pd.Series, offset: int
) -> pd.Series:
    """Value of the most recent confirmed pivot, as seen at each bar."""
    levels = price.shift(offset).where(pivots)
    return levels.ffill()


def candle_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Body, wicks and range, normalized by the bar's own range."""
    high, low, open_, close = frame["high"], frame["low"], frame["open"], frame["close"]
    rng = (high - low).replace(0, np.nan)
    body = (close - open_)
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    return pd.DataFrame(
        {
            "candle_body": body / rng,
            "candle_body_abs": body.abs() / rng,
            "candle_range_pct": rng / close.replace(0, np.nan),
            "upper_wick": upper_wick / rng,
            "lower_wick": lower_wick / rng,
            "is_bullish": (close > open_).astype(float),
        },
        index=frame.index,
    )


def zscore(series: pd.Series, period: int = 50) -> pd.Series:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std().replace(0, np.nan)
    return (series - mean) / std
