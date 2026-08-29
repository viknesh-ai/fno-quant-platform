"""Data-leakage tests (REQ 18/19).

REQ 19 calls leakage prevention mandatory, so these are the highest-value tests in
the suite. Each one attacks a specific channel through which future information
could reach a model:

  * forming bars visible to features
  * higher-timeframe bars joined before they close
  * labels retained for bars whose outcome is not yet observable
  * train/validation/test windows overlapping in time
  * scalers fitted across a split boundary
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from aqtp.core.clock import IST
from aqtp.core.errors import LeakageError
from aqtp.core.types import Timeframe
from aqtp.data.candles import MultiTimeframeStore, candles_to_frame, resample
from aqtp.features.engine import FeatureEngine
from aqtp.ml.dataset import (
    Dataset,
    DatasetBuilder,
    assert_no_leakage,
    assert_split_ordering,
    chronological_split,
    walk_forward_splits,
)
from aqtp.ml.labeling import LabelSpec, triple_barrier_labels
from tests.conftest import make_candles


# --------------------------------------------------------------------------- #
# 1. The forming bar must never be visible
# --------------------------------------------------------------------------- #
def test_closed_frame_excludes_the_forming_bar():
    candles = make_candles(bars=100, seed=21)
    store = MultiTimeframeStore(
        base_timeframe=Timeframe.M1, timeframes=(Timeframe.M1, Timeframe.M5)
    )
    store.ingest_candles(candles)
    store.refresh_all()

    series = store.series(Timeframe.M5)
    now = candles[-1].timestamp
    closed = series.closed_frame(now)
    forming_bar_start = pd.Timestamp(series.bar_start(now))

    assert len(closed) > 0
    assert closed.index.max() < forming_bar_start, (
        "closed_frame() returned a bar that has not finished forming"
    )


def test_tick_updates_never_leak_into_closed_bars():
    """Folding a tick into the forming bar must not alter any closed bar."""
    candles = make_candles(bars=60, seed=22)
    store = MultiTimeframeStore(base_timeframe=Timeframe.M1, timeframes=(Timeframe.M1,))
    store.ingest_candles(candles)

    now = candles[-1].timestamp
    before = store.closed(Timeframe.M1, now).copy()
    store.ingest_tick(now + timedelta(seconds=30), price=999999.0, volume=10_000)
    after = store.closed(Timeframe.M1, now + timedelta(seconds=30))

    common = before.index.intersection(after.index)
    pd.testing.assert_frame_equal(before.loc[common], after.loc[common])


# --------------------------------------------------------------------------- #
# 2. Higher-timeframe features must lag by their own bar length
# --------------------------------------------------------------------------- #
def test_higher_timeframe_features_are_not_visible_before_they_close():
    """A 1h bar starting at 10:00 closes at 11:00 and must not inform a 10:30 row."""
    base = candles_to_frame(make_candles(bars=375 * 12, seed=23, momentum_decay=0.95))
    frames = {
        Timeframe.M5: resample(base, Timeframe.M5),
        Timeframe.H1: resample(base, Timeframe.H1),
    }
    engine = FeatureEngine()
    merged = engine.compute_multi_timeframe(frames, align_to=Timeframe.M5)

    hourly = engine.compute(frames[Timeframe.H1], prefix="1h_")
    column = "1h_return_1"
    assert column in merged.columns

    # For each merged row, the attached hourly value must come from a bar that had
    # already closed at that timestamp.
    sample = merged[[column]].dropna().iloc[200:260]
    for timestamp, row in sample.iterrows():
        eligible = hourly[hourly.index + pd.Timedelta(hours=1) <= timestamp]
        if eligible.empty:
            continue
        expected = eligible[column].iloc[-1]
        assert row[column] == pytest.approx(expected, rel=1e-9), (
            f"at {timestamp} the merged frame used an hourly value that had not closed"
        )


def test_multi_timeframe_never_mixes_a_faster_timeframe_down():
    """Aligning to 1h must not pull 5m features in — that would be future data
    relative to the hourly bar's own close."""
    base = candles_to_frame(make_candles(bars=375 * 10, seed=24))
    frames = {
        Timeframe.M5: resample(base, Timeframe.M5),
        Timeframe.H1: resample(base, Timeframe.H1),
    }
    merged = FeatureEngine().compute_multi_timeframe(frames, align_to=Timeframe.H1)
    assert not any(c.startswith("5m_") for c in merged.columns)


# --------------------------------------------------------------------------- #
# 3. Unresolvable labels must be dropped
# --------------------------------------------------------------------------- #
def test_trailing_labels_are_nan_and_get_dropped():
    frame = candles_to_frame(make_candles(bars=300, seed=25))
    atr = (frame["high"] - frame["low"]).rolling(14).mean()
    spec = LabelSpec(horizon_bars=10, target_atr_multiple=1.5, stop_atr_multiple=1.0,
                     min_movement_atr=0.25)
    labels = triple_barrier_labels(frame, atr, spec)

    # The final `horizon` rows cannot be resolved, so they must be NaN.
    assert labels["label"].iloc[-10:].isna().all(), (
        "labels were produced for bars whose outcome has not yet occurred"
    )


def test_dataset_excludes_bars_beyond_the_last_resolvable_label(multi_timeframe_frames):
    spec = LabelSpec(horizon_bars=12, target_atr_multiple=1.5, stop_atr_multiple=1.0,
                     min_movement_atr=0.25)
    dataset = DatasetBuilder().build(
        frames=multi_timeframe_frames, entry_timeframe=Timeframe.M5,
        label_spec=spec, symbol="TEST",
    )
    base = multi_timeframe_frames[Timeframe.M5]
    assert not dataset.is_empty
    last_resolvable = base.index[-(spec.horizon_bars + 1)]
    assert dataset.features.index[-1] <= last_resolvable


def test_assert_no_leakage_rejects_unresolved_labels():
    index = pd.date_range("2026-01-01", periods=10, freq="5min", tz=IST)
    dataset = Dataset(
        features=pd.DataFrame({"a": np.arange(10, dtype=float)}, index=index),
        labels=pd.Series([1.0] * 9 + [np.nan], index=index),
        feature_names=["a"],
    )
    with pytest.raises(LeakageError, match="labels contain NaN"):
        assert_no_leakage(dataset)


def test_assert_no_leakage_rejects_unordered_index():
    index = pd.DatetimeIndex(
        ["2026-01-01 10:00", "2026-01-01 09:00", "2026-01-01 11:00"], tz=IST
    )
    dataset = Dataset(
        features=pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=index),
        labels=pd.Series([1.0, 0.0, 1.0], index=index),
        feature_names=["a"],
    )
    with pytest.raises(LeakageError, match="chronologically ordered"):
        assert_no_leakage(dataset)


def test_assert_no_leakage_rejects_nan_features():
    index = pd.date_range("2026-01-01", periods=5, freq="5min", tz=IST)
    dataset = Dataset(
        features=pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0]}, index=index),
        labels=pd.Series([1.0] * 5, index=index),
        feature_names=["a"],
    )
    with pytest.raises(LeakageError, match="NaN"):
        assert_no_leakage(dataset)


# --------------------------------------------------------------------------- #
# 4. Splits must be chronological and embargoed
# --------------------------------------------------------------------------- #
def _toy_dataset(periods: int = 2000) -> Dataset:
    index = pd.date_range("2026-01-01 09:15", periods=periods, freq="5min", tz=IST)
    rng = np.random.default_rng(31)
    return Dataset(
        features=pd.DataFrame(
            {"a": rng.normal(size=periods), "b": rng.normal(size=periods)}, index=index
        ),
        labels=pd.Series(rng.integers(0, 2, periods).astype(float), index=index),
        metadata=pd.DataFrame({"realized_r": rng.normal(size=periods)}, index=index),
        feature_names=["a", "b"],
    )


def test_chronological_split_orders_and_embargoes():
    dataset = _toy_dataset()
    split = chronological_split(dataset, embargo=timedelta(minutes=60))

    assert split.train.features.index.max() < split.validation.features.index.min()
    assert split.validation.features.index.max() < split.test.features.index.min()
    assert_split_ordering(split, min_embargo=timedelta(minutes=59))


def test_walk_forward_folds_never_overlap():
    dataset = _toy_dataset(periods=6000)
    splits = walk_forward_splits(
        dataset, train_days=5, validation_days=2, test_days=2, step_days=2,
        embargo=timedelta(minutes=60),
    )
    assert splits, "expected at least one walk-forward fold"

    for split in splits:
        assert_split_ordering(split, min_embargo=timedelta(minutes=59))
        train_index = set(split.train.features.index)
        test_index = set(split.test.features.index)
        assert not (train_index & test_index), "train and test windows share rows"
        if not split.validation.is_empty:
            assert not (train_index & set(split.validation.features.index))


def test_assert_split_ordering_rejects_an_insufficient_embargo():
    dataset = _toy_dataset(periods=400)
    split = chronological_split(dataset, embargo=timedelta(minutes=5))
    with pytest.raises(LeakageError, match="embargo"):
        assert_split_ordering(split, min_embargo=timedelta(minutes=120))


# --------------------------------------------------------------------------- #
# 5. Transformations must be fitted on training data only
# --------------------------------------------------------------------------- #
def test_scaler_is_fitted_on_training_data_only():
    """The pipeline's scaler must not have seen test-set statistics.

    Fitting a scaler on the full sample leaks the test distribution into training —
    a subtle leak that inflates results without any obvious symptom.
    """
    from aqtp.ml.models import CalibratedModel, build_model

    dataset = _toy_dataset(periods=1000)
    split = chronological_split(dataset, embargo=timedelta(minutes=60))

    model = CalibratedModel(build_model("logistic_regression", 42), method="none")
    model.fit(split.train.features, split.train.labels)

    scaler = model.estimator.named_steps["scale"]
    train_mean = split.train.features.mean().to_numpy()
    full_mean = dataset.features.mean().to_numpy()

    assert np.allclose(scaler.mean_, train_mean, atol=1e-9)
    assert not np.allclose(scaler.mean_, full_mean, atol=1e-9), (
        "the scaler appears to have been fitted on the full dataset"
    )


def test_calibrator_is_fitted_on_validation_not_training():
    """Calibrating on training predictions fits memorized outputs, not real ones."""
    from aqtp.ml.models import CalibratedModel, build_model

    dataset = _toy_dataset(periods=1500)
    split = chronological_split(dataset, embargo=timedelta(minutes=60))

    with_validation = CalibratedModel(build_model("logistic_regression", 42), method="isotonic")
    with_validation.fit(
        split.train.features, split.train.labels,
        X_validation=split.validation.features, y_validation=split.validation.labels,
    )
    assert with_validation._calibrator is not None

    without_validation = CalibratedModel(build_model("logistic_regression", 42), method="isotonic")
    without_validation.fit(split.train.features, split.train.labels)
    assert without_validation._calibrator is None, (
        "a calibrator was fitted without a held-out validation set"
    )


# --------------------------------------------------------------------------- #
# 6. End-to-end: features at bar t must not change when future bars arrive
# --------------------------------------------------------------------------- #
def test_feature_row_is_stable_when_future_bars_arrive():
    candles = make_candles(bars=800, seed=41, momentum_decay=0.95)
    full = candles_to_frame(candles)
    partial = candles_to_frame(candles[:600])

    engine = FeatureEngine()
    full_features = engine.compute(full)
    partial_features = engine.compute(partial)

    target = partial.index[-1]
    columns = [
        c for c in engine.feature_columns(full_features)
        if not pd.isna(partial_features.loc[target, c])
    ]
    assert columns

    for column in columns:
        assert full_features.loc[target, column] == pytest.approx(
            partial_features.loc[target, column], rel=1e-9, abs=1e-9
        ), f"feature {column} at {target} changed once future bars were appended"
