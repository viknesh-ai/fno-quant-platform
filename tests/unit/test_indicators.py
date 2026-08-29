"""Indicator unit tests (REQ 64).

The central property asserted here is **causality**: an indicator value at bar `i`
must not change when future bars are appended. That is the structural guarantee
that makes backtest results meaningful, so it is tested directly rather than
inferred.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqtp.features import indicators as ind


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(5)
    return pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))


@pytest.fixture
def ohlc() -> pd.DataFrame:
    """Well-formed OHLC: low <= open, close <= high, as any real feed guarantees.

    Malformed bars are the DataQualityGate's responsibility (see
    tests/unit/test_data_quality.py); indicators may assume validated input.
    """
    rng = np.random.default_rng(6)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 300)))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + np.abs(rng.normal(0, 0.5, 300))
    low = pd.concat([open_, close], axis=1).min(axis=1) - np.abs(rng.normal(0, 0.5, 300))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(1000, 5000, 300).astype(float)}
    )


# --------------------------------------------------------------------------- #
# Causality — no indicator may use future data
# --------------------------------------------------------------------------- #
CAUSAL_SERIES_FUNCS = [
    ("sma", lambda s: ind.sma(s, 20)),
    ("ema", lambda s: ind.ema(s, 20)),
    ("rsi", lambda s: ind.rsi(s, 14)),
    ("roc", lambda s: ind.roc(s, 10)),
    ("momentum", lambda s: ind.momentum(s, 10)),
    ("trend_slope", lambda s: ind.trend_slope(s, 20)),
    ("percentile_rank", lambda s: ind.percentile_rank(s, 50)),
    ("zscore", lambda s: ind.zscore(s, 50)),
    ("bollinger_width", lambda s: ind.bollinger_width(s, 20)),
]


@pytest.mark.parametrize("name,func", CAUSAL_SERIES_FUNCS, ids=[n for n, _ in CAUSAL_SERIES_FUNCS])
def test_series_indicators_are_causal(series, name, func):
    """Truncating the input must not change earlier output values."""
    full = func(series)
    truncated = func(series.iloc[:200])
    overlap = full.iloc[:200]
    pd.testing.assert_series_equal(
        truncated.dropna(), overlap.loc[truncated.dropna().index], check_names=False,
        rtol=1e-9, atol=1e-9,
    )


@pytest.mark.parametrize(
    "name,func",
    [
        ("atr", lambda f: ind.atr(f["high"], f["low"], f["close"], 14)),
        ("adx", lambda f: ind.adx(f["high"], f["low"], f["close"], 14)[0]),
        ("rolling_high", lambda f: ind.rolling_high(f["high"], 20)),
        ("relative_volume", lambda f: ind.relative_volume(f["volume"], 20)),
    ],
)
def test_ohlc_indicators_are_causal(ohlc, name, func):
    full = func(ohlc)
    truncated = func(ohlc.iloc[:200])
    pd.testing.assert_series_equal(
        truncated.dropna(), full.iloc[:200].loc[truncated.dropna().index],
        check_names=False, rtol=1e-9, atol=1e-9,
    )


def test_swing_highs_are_only_marked_after_confirmation():
    """A pivot needs `right` bars after it; it must not be visible before then."""
    # Peak at index 5, with 3 bars on either side.
    values = pd.Series([1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1], dtype=float)
    pivots = ind.swing_highs(values, left=3, right=3)
    # The rolling window of size 7 ending at index 8 is the first that contains the
    # peak at its `left` offset, so confirmation cannot occur before index 8.
    assert not pivots.iloc[:8].any(), "pivot was marked before it could be confirmed"
    assert pivots.iloc[8]


def test_swing_lows_are_only_marked_after_confirmation():
    values = pd.Series([10, 9, 8, 7, 6, 1, 6, 7, 8, 9, 10], dtype=float)
    pivots = ind.swing_lows(values, left=3, right=3)
    assert not pivots.iloc[:8].any()
    assert pivots.iloc[8]


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def test_sma_matches_manual_mean(series):
    result = ind.sma(series, 10)
    assert result.iloc[9] == pytest.approx(series.iloc[:10].mean())
    assert pd.isna(result.iloc[8]), "SMA must be undefined before the window fills"


def test_rsi_is_bounded(series):
    result = ind.rsi(series, 14).dropna()
    assert result.between(0, 100).all()


def test_rsi_of_monotonic_rise_is_maximal():
    rising = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(rising, 14).iloc[-1] == pytest.approx(100.0)


def test_atr_is_non_negative(ohlc):
    assert (ind.atr(ohlc["high"], ohlc["low"], ohlc["close"], 14).dropna() >= 0).all()


def test_true_range_covers_gaps():
    """A gap must widen the true range beyond the bar's own high-low."""
    frame = pd.DataFrame(
        {"high": [100.0, 120.0], "low": [99.0, 118.0], "close": [99.5, 119.0]}
    )
    tr = ind.true_range(frame["high"], frame["low"], frame["close"])
    # Second bar gapped up from 99.5: TR is 120 - 99.5, not 120 - 118.
    assert tr.iloc[1] == pytest.approx(20.5)


def test_adx_is_bounded(ohlc):
    adx, plus_di, minus_di = ind.adx(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    assert adx.dropna().between(0, 100).all()
    assert plus_di.dropna().between(0, 100).all()
    assert minus_di.dropna().between(0, 100).all()


def test_adx_is_high_in_a_strong_trend():
    n = 200
    close = pd.Series(np.linspace(100, 200, n))
    high, low = close * 1.005, close * 0.995
    adx, plus_di, minus_di = ind.adx(high, low, close, 14)
    assert adx.iloc[-1] > 40, "a monotonic trend should produce a high ADX"
    assert plus_di.iloc[-1] > minus_di.iloc[-1]


def test_percentile_rank_is_bounded_and_ranks_extremes(series):
    result = ind.percentile_rank(series, 50).dropna()
    assert result.between(0, 1).all()
    rising = pd.Series(np.arange(100, dtype=float))
    assert ind.percentile_rank(rising, 50).iloc[-1] == pytest.approx(1.0)


def test_macd_histogram_is_line_minus_signal(series):
    line, signal, hist = ind.macd(series)
    pd.testing.assert_series_equal(hist.dropna(), (line - signal).dropna(), check_names=False)


def test_candle_features_are_normalized(ohlc):
    features = ind.candle_features(ohlc).dropna()
    assert features["candle_body"].between(-1.0001, 1.0001).all()
    assert (features["upper_wick"] >= -1e-9).all()
    assert (features["lower_wick"] >= -1e-9).all()


def test_zero_range_bar_does_not_divide_by_zero():
    """A bar with no range (all four prices equal) must not produce inf."""
    frame = pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]}
    )
    features = ind.candle_features(frame)
    assert not np.isinf(features.to_numpy(dtype=float)).any()


def test_volatility_expansion_detects_a_regime_shift():
    """Expansion is measured against the recent baseline, so the burst must be
    shorter than the long window for the ratio to register it."""
    quiet = np.full(40, 1.0)
    loud = np.full(6, 5.0)
    atr_series = pd.Series(np.concatenate([quiet, loud]))
    expansion = ind.volatility_expansion(atr_series, short=5, long=20)
    assert expansion.iloc[-1] > 1.5

    # Once the burst has filled the long window it is the new normal, not an
    # expansion — the ratio must return to ~1.
    sustained = pd.Series(np.concatenate([quiet, np.full(25, 5.0)]))
    assert ind.volatility_expansion(sustained, short=5, long=20).iloc[-1] == pytest.approx(1.0)
