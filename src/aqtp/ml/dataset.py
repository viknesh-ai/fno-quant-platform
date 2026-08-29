"""Leakage-safe dataset construction (REQ 18/19).

REQ 19 calls leakage prevention mandatory, so it is enforced structurally rather
than by convention:

  * Features come only from `MultiTimeframeStore.closed()` — a forming bar is never
    visible.
  * Labels are computed from future bars, then the trailing `horizon` rows are
    **dropped**, because their outcome could not have been observed.
  * Features are shifted one bar relative to labels: the label for bar *t* is paired
    with features computed at bar *t*, and the decision is executable at *t+1*.
  * Splits are chronological with an **embargo** gap. Without an embargo, the last
    training samples overlap in time with the first test samples (their labels are
    still resolving), which leaks.
  * Scalers/imputers are fit on training folds only and applied to validation/test.

`assert_no_leakage()` re-checks these invariants on a built dataset and raises
`LeakageError` if any is violated. It is called by the trainer and by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.errors import LeakageError
from ..core.logging import get_logger
from ..core.types import Timeframe
from ..features.engine import FEATURE_VERSION, FeatureEngine
from .labeling import LabelSpec, fixed_horizon_labels, triple_barrier_labels

logger = get_logger(__name__)


@dataclass
class Dataset:
    """A model-ready dataset with its provenance attached (REQ 50)."""

    features: pd.DataFrame
    labels: pd.Series
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)
    feature_version: str = FEATURE_VERSION
    label_spec: LabelSpec | None = None
    symbol: str = ""
    timeframe: str = ""

    def __len__(self) -> int:
        return len(self.features)

    @property
    def is_empty(self) -> bool:
        return self.features.empty

    @property
    def start(self) -> datetime | None:
        return self.features.index[0].to_pydatetime() if len(self.features) else None

    @property
    def end(self) -> datetime | None:
        return self.features.index[-1].to_pydatetime() if len(self.features) else None

    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else float("nan")

    def slice(self, start: datetime, end: datetime) -> "Dataset":
        mask = (self.features.index >= pd.Timestamp(start)) & (self.features.index < pd.Timestamp(end))
        return Dataset(
            features=self.features[mask],
            labels=self.labels[mask],
            metadata=self.metadata[mask] if not self.metadata.empty else self.metadata,
            feature_names=list(self.feature_names),
            feature_version=self.feature_version,
            label_spec=self.label_spec,
            symbol=self.symbol,
            timeframe=self.timeframe,
        )

    def describe(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": len(self),
            "features": len(self.feature_names),
            "feature_version": self.feature_version,
            "label": self.label_spec.name if self.label_spec else "",
            "start": str(self.start),
            "end": str(self.end),
            "positive_rate": round(self.positive_rate(), 4),
        }


class DatasetBuilder:
    """Builds datasets that are safe to train on."""

    def __init__(self, feature_engine: FeatureEngine | None = None) -> None:
        self.feature_engine = feature_engine or FeatureEngine()

    def build(
        self,
        *,
        frames: Mapping[Timeframe, pd.DataFrame],
        entry_timeframe: Timeframe,
        label_spec: LabelSpec,
        symbol: str = "",
        labeling: str = "triple_barrier",
        expiry_map: Mapping[pd.Timestamp, object] | None = None,
        min_rows: int = 100,
    ) -> Dataset:
        base = frames.get(entry_timeframe)
        if base is None or base.empty:
            return Dataset(pd.DataFrame(), pd.Series(dtype=float), symbol=symbol)

        features = self.feature_engine.compute_multi_timeframe(frames, align_to=entry_timeframe)
        if features.empty:
            return Dataset(pd.DataFrame(), pd.Series(dtype=float), symbol=symbol)

        time_features = self.feature_engine.time_features(features.index)
        if not time_features.empty:
            features = features.join(time_features, how="left")

        atr_column = f"{entry_timeframe.value}_atr"
        if atr_column not in features.columns:
            raise ValueError(f"expected ATR column {atr_column!r} in computed features")
        atr = features[atr_column]

        aligned = base.reindex(features.index)
        if labeling == "triple_barrier":
            labels_frame = triple_barrier_labels(aligned, atr, label_spec)
        else:
            labels_frame = fixed_horizon_labels(aligned, atr, label_spec)

        feature_names = self.feature_engine.feature_columns(features)
        matrix = features[feature_names]

        # --- the two structural drops -------------------------------------
        # 1. Rows whose label could not be resolved (the trailing horizon).
        valid_label = labels_frame["label"].notna()
        # 2. Rows where features are not yet fully warmed up.
        valid_features = matrix.notna().all(axis=1)

        mask = valid_label & valid_features
        matrix = matrix[mask]
        labels = labels_frame.loc[mask, "label"].astype(float)
        metadata = labels_frame.loc[mask, ["realized_r", "bars_to_exit", "outcome", "exit_price"]]
        metadata = metadata.join(aligned.loc[mask, ["close"]].rename(columns={"close": "entry_price"}))
        metadata["atr"] = atr[mask]

        if len(matrix) < min_rows:
            logger.warning(
                "%s: only %d usable rows after leakage filtering (needed %d)",
                symbol, len(matrix), min_rows,
            )

        dataset = Dataset(
            features=matrix,
            labels=labels,
            metadata=metadata,
            feature_names=feature_names,
            label_spec=label_spec,
            symbol=symbol,
            timeframe=entry_timeframe.value,
        )
        assert_no_leakage(dataset, base_frame=base, label_spec=label_spec)
        return dataset

    def build_multi_symbol(
        self,
        per_symbol: Mapping[str, Mapping[Timeframe, pd.DataFrame]],
        *,
        entry_timeframe: Timeframe,
        label_spec: LabelSpec,
        labeling: str = "triple_barrier",
    ) -> Dataset:
        """Pool several instruments into one dataset.

        Pooling is what makes a stock-universe model trainable at all — one symbol
        rarely has enough intraday history. A `symbol` column is retained in the
        metadata so per-instrument performance stays measurable (REQ 39).
        """
        parts: list[Dataset] = []
        for symbol, frames in per_symbol.items():
            dataset = self.build(
                frames=frames,
                entry_timeframe=entry_timeframe,
                label_spec=label_spec,
                symbol=symbol,
                labeling=labeling,
                min_rows=0,
            )
            if not dataset.is_empty:
                dataset.metadata = dataset.metadata.assign(symbol=symbol)
                parts.append(dataset)

        if not parts:
            return Dataset(pd.DataFrame(), pd.Series(dtype=float))

        common = sorted(set.intersection(*(set(p.feature_names) for p in parts)))
        features = pd.concat([p.features[common] for p in parts])
        labels = pd.concat([p.labels for p in parts])
        metadata = pd.concat([p.metadata for p in parts])

        order = features.index.argsort(kind="stable")
        return Dataset(
            features=features.iloc[order],
            labels=labels.iloc[order],
            metadata=metadata.iloc[order],
            feature_names=common,
            label_spec=label_spec,
            symbol="|".join(sorted(per_symbol)),
            timeframe=entry_timeframe.value,
        )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Split:
    train: Dataset
    validation: Dataset
    test: Dataset
    fold: int = 0
    train_start: datetime | None = None
    train_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None

    def describe(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_rows": len(self.train),
            "validation_rows": len(self.validation),
            "test_rows": len(self.test),
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "test_start": str(self.test_start),
            "test_end": str(self.test_end),
        }


def chronological_split(
    dataset: Dataset,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: timedelta = timedelta(minutes=60),
) -> Split:
    """Single chronological split with an embargo between segments (REQ 19)."""
    if dataset.is_empty:
        return Split(dataset, dataset, dataset)

    index = dataset.features.index
    n = len(index)
    train_end_pos = int(n * train_fraction)
    validation_end_pos = int(n * (train_fraction + validation_fraction))

    train_end = index[max(0, train_end_pos - 1)]
    validation_start = train_end + pd.Timedelta(embargo)
    validation_end = index[max(0, validation_end_pos - 1)]
    test_start = validation_end + pd.Timedelta(embargo)

    train_mask = index <= train_end
    validation_mask = (index >= validation_start) & (index <= validation_end)
    test_mask = index >= test_start

    return Split(
        train=_subset(dataset, train_mask),
        validation=_subset(dataset, validation_mask),
        test=_subset(dataset, test_mask),
        train_start=index[0].to_pydatetime() if n else None,
        train_end=train_end.to_pydatetime(),
        test_start=test_start.to_pydatetime(),
        test_end=index[-1].to_pydatetime() if n else None,
    )


def walk_forward_splits(
    dataset: Dataset,
    *,
    train_days: int,
    validation_days: int,
    test_days: int,
    step_days: int,
    embargo: timedelta = timedelta(minutes=60),
) -> list[Split]:
    """Rolling walk-forward windows (REQ 20).

    Each fold trains on `train_days`, tunes on `validation_days`, and is evaluated
    on a `test_days` window that the model has never seen, then the whole window
    rolls forward by `step_days`. Performance is stored per fold so no single
    aggregate backtest can stand in for the evidence (REQ 20).
    """
    if dataset.is_empty:
        return []

    index = dataset.features.index
    start, end = index[0], index[-1]
    window = pd.Timedelta(days=train_days + validation_days + test_days)
    if end - start < window:
        logger.warning(
            "insufficient history for walk-forward: have %s, need %s",
            end - start, window,
        )
        return []

    splits: list[Split] = []
    fold = 0
    cursor = start
    embargo_delta = pd.Timedelta(embargo)

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(days=train_days)
        validation_start = train_end + embargo_delta
        validation_end = validation_start + pd.Timedelta(days=validation_days)
        test_start = validation_end + embargo_delta
        test_end = test_start + pd.Timedelta(days=test_days)
        if test_end > end:
            break

        train_mask = (index >= train_start) & (index < train_end)
        validation_mask = (index >= validation_start) & (index < validation_end)
        test_mask = (index >= test_start) & (index < test_end)

        if train_mask.sum() and test_mask.sum():
            splits.append(
                Split(
                    train=_subset(dataset, train_mask),
                    validation=_subset(dataset, validation_mask),
                    test=_subset(dataset, test_mask),
                    fold=fold,
                    train_start=train_start.to_pydatetime(),
                    train_end=train_end.to_pydatetime(),
                    test_start=test_start.to_pydatetime(),
                    test_end=test_end.to_pydatetime(),
                )
            )
            fold += 1
        cursor = cursor + pd.Timedelta(days=step_days)

    logger.info("built %d walk-forward folds", len(splits))
    return splits


def _subset(dataset: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(
        features=dataset.features[mask],
        labels=dataset.labels[mask],
        metadata=dataset.metadata[mask] if not dataset.metadata.empty else dataset.metadata,
        feature_names=list(dataset.feature_names),
        feature_version=dataset.feature_version,
        label_spec=dataset.label_spec,
        symbol=dataset.symbol,
        timeframe=dataset.timeframe,
    )


# --------------------------------------------------------------------------- #
def assert_no_leakage(
    dataset: Dataset, *, base_frame: pd.DataFrame | None = None, label_spec: LabelSpec | None = None
) -> None:
    """Re-verify the leakage invariants on a built dataset.

    Raises LeakageError rather than warning: a dataset that leaks produces a model
    whose backtest is meaningless, and the requirement treats this as mandatory.
    """
    if dataset.is_empty:
        return

    index = dataset.features.index
    if not index.is_monotonic_increasing:
        raise LeakageError("dataset index is not chronologically ordered")
    if index.has_duplicates:
        raise LeakageError("dataset index contains duplicate timestamps")
    if len(dataset.features) != len(dataset.labels):
        raise LeakageError("features and labels have different lengths")
    if not dataset.features.index.equals(dataset.labels.index):
        raise LeakageError("features and labels are not aligned on the same index")
    if dataset.features.isna().any().any():
        raise LeakageError("features contain NaN; imputation must be fit on training data only")
    if dataset.labels.isna().any():
        raise LeakageError("labels contain NaN; unresolved outcomes must be dropped")

    # The trailing `horizon` bars of the source data must have been dropped.
    if base_frame is not None and label_spec is not None and not base_frame.empty:
        horizon = label_spec.horizon_bars
        if len(base_frame) > horizon:
            last_labelable = base_frame.index[-(horizon + 1)]
            if index[-1] > last_labelable:
                raise LeakageError(
                    f"dataset includes {index[-1]}, beyond the last resolvable label "
                    f"{last_labelable} for a {horizon}-bar horizon"
                )


def assert_split_ordering(split: Split, *, min_embargo: timedelta = timedelta(0)) -> None:
    """Verify a split is chronological and embargoed."""
    for earlier, later, names in (
        (split.train, split.validation, ("train", "validation")),
        (split.validation, split.test, ("validation", "test")),
        (split.train, split.test, ("train", "test")),
    ):
        if earlier.is_empty or later.is_empty:
            continue
        gap = later.features.index[0] - earlier.features.index[-1]
        if gap <= pd.Timedelta(0):
            raise LeakageError(
                f"{names[1]} set starts at {later.features.index[0]} which is not after "
                f"{names[0]} set end {earlier.features.index[-1]}"
            )
        if gap < pd.Timedelta(min_embargo):
            raise LeakageError(
                f"embargo between {names[0]} and {names[1]} is {gap}, below the required {min_embargo}"
            )
