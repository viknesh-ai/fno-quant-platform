"""Prediction targets and labeling (REQ 14/15).

REQ 15 forbids the target "will NIFTY go up". Every label here is defined by an
explicit (horizon, target threshold, stop threshold, minimum movement) tuple, with
volatility adjustment, so the label means the same thing in a quiet market and a
violent one.

The primary labeler is the triple-barrier method: for each bar, walk forward until
price touches the target barrier, the stop barrier, or the time barrier — whichever
comes first. That answers the question the trader actually faces ("does target come
before stop?") rather than the easier and less useful "is price higher in N bars?".

Look-ahead safety: labels use future data *by definition* — that is what a label is.
The invariant that must hold is that labels are only ever joined to features from
strictly earlier bars, and that the last `horizon` bars are dropped because their
outcome is not yet observable. Both are enforced in `dataset.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from ..configuration.schema import PredictionConfig
from ..core.logging import get_logger

logger = get_logger(__name__)


class BarrierOutcome(IntEnum):
    STOP_HIT = -1
    TIME_EXPIRED = 0
    TARGET_HIT = 1


@dataclass(frozen=True)
class LabelSpec:
    """A fully specified prediction target (REQ 15)."""

    horizon_bars: int
    target_atr_multiple: float
    stop_atr_multiple: float
    min_movement_atr: float
    volatility_adjusted: bool = True
    direction: str = "long"  # 'long' or 'short'

    @property
    def name(self) -> str:
        return (
            f"tb_{self.direction}_h{self.horizon_bars}"
            f"_t{self.target_atr_multiple:g}_s{self.stop_atr_multiple:g}"
        )

    def describe(self) -> str:
        return (
            f"{self.direction} triple-barrier: target {self.target_atr_multiple}xATR, "
            f"stop {self.stop_atr_multiple}xATR, within {self.horizon_bars} bars, "
            f"minimum movement {self.min_movement_atr}xATR"
        )


def triple_barrier_labels(
    frame: pd.DataFrame,
    atr: pd.Series,
    spec: LabelSpec,
) -> pd.DataFrame:
    """Label each bar by which barrier its future path touches first.

    Returns columns:
        outcome        : BarrierOutcome (-1 stop, 0 timeout, +1 target)
        label          : 1 if target hit before stop, else 0
        bars_to_exit   : how many bars until resolution
        exit_price     : price at resolution
        realized_r     : outcome in units of the risk taken
        target_price / stop_price : the barriers used

    Rows whose horizon extends past the end of the data are NaN and must be dropped
    by the caller — their outcome is genuinely unknown.
    """
    required = {"high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns {sorted(missing)}")
    if len(frame) != len(atr):
        raise ValueError("frame and atr must be aligned")

    n = len(frame)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)

    is_long = spec.direction == "long"
    horizon = int(spec.horizon_bars)

    outcome = np.full(n, np.nan)
    label = np.full(n, np.nan)
    bars_to_exit = np.full(n, np.nan)
    exit_price = np.full(n, np.nan)
    realized_r = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    stop_price = np.full(n, np.nan)

    for i in range(n):
        entry = close[i]
        width = atr_values[i]
        # An undefined ATR (warm-up) or a horizon running past the data end means the
        # label cannot be computed. Leaving NaN is correct; guessing would be a leak.
        if not np.isfinite(width) or width <= 0 or not np.isfinite(entry):
            continue
        end = i + horizon
        if end >= n:
            continue

        if not spec.volatility_adjusted:
            width = entry * 0.01  # fixed 1% barrier when volatility adjustment is off

        target_distance = spec.target_atr_multiple * width
        stop_distance = spec.stop_atr_multiple * width
        if is_long:
            upper, lower = entry + target_distance, entry - stop_distance
        else:
            upper, lower = entry + stop_distance, entry - target_distance
        target_price[i] = upper if is_long else lower
        stop_price[i] = lower if is_long else upper

        resolved = False
        for j in range(i + 1, end + 1):
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower
            if hit_upper and hit_lower:
                # Both barriers touched inside one bar. Without tick data we cannot
                # know the order, so we assume the adverse one — the conservative
                # choice, and the one that avoids an optimistic bias in training.
                outcome[i] = BarrierOutcome.STOP_HIT
                exit_price[i] = lower if is_long else upper
                resolved = True
            elif hit_upper:
                outcome[i] = BarrierOutcome.TARGET_HIT if is_long else BarrierOutcome.STOP_HIT
                exit_price[i] = upper
                resolved = True
            elif hit_lower:
                outcome[i] = BarrierOutcome.STOP_HIT if is_long else BarrierOutcome.TARGET_HIT
                exit_price[i] = lower
                resolved = True
            if resolved:
                bars_to_exit[i] = j - i
                break

        if not resolved:
            outcome[i] = BarrierOutcome.TIME_EXPIRED
            exit_price[i] = close[end]
            bars_to_exit[i] = horizon

        move = (exit_price[i] - entry) if is_long else (entry - exit_price[i])
        realized_r[i] = move / stop_distance if stop_distance > 0 else 0.0

        # A timeout only counts as a positive label if it actually delivered the
        # minimum movement; otherwise "didn't lose" would be scored as "won".
        if outcome[i] == BarrierOutcome.TARGET_HIT:
            label[i] = 1.0
        elif outcome[i] == BarrierOutcome.TIME_EXPIRED:
            label[i] = 1.0 if move >= spec.min_movement_atr * width else 0.0
        else:
            label[i] = 0.0

    return pd.DataFrame(
        {
            "outcome": outcome,
            "label": label,
            "bars_to_exit": bars_to_exit,
            "exit_price": exit_price,
            "realized_r": realized_r,
            "target_price": target_price,
            "stop_price": stop_price,
        },
        index=frame.index,
    )


def fixed_horizon_labels(
    frame: pd.DataFrame, atr: pd.Series, spec: LabelSpec
) -> pd.DataFrame:
    """Simpler alternative: sign of the return `horizon` bars ahead, with a
    volatility-scaled dead zone so tiny moves are not labelled as signal."""
    close = frame["close"]
    future = close.shift(-spec.horizon_bars)
    move = future - close
    threshold = spec.min_movement_atr * atr
    if spec.direction == "short":
        move = -move
    label = pd.Series(np.nan, index=frame.index)
    label[move > threshold] = 1.0
    label[move <= threshold] = 0.0
    label[future.isna()] = np.nan
    return pd.DataFrame(
        {
            "label": label,
            "realized_r": move / (spec.stop_atr_multiple * atr).replace(0, np.nan),
            "bars_to_exit": float(spec.horizon_bars),
            "exit_price": future,
        },
        index=frame.index,
    )


def volatility_target_labels(returns: pd.Series, horizon_bars: int) -> pd.Series:
    """Regression target for the volatility model (REQ 14).

    Realized volatility over the *next* `horizon_bars`. Used only for training; the
    final row set is truncated by the dataset builder.
    """
    future_vol = returns.shift(-horizon_bars).rolling(horizon_bars).std()
    return future_vol


def build_label_specs(config: PredictionConfig, bar_minutes: int) -> list[LabelSpec]:
    """Turn the configured horizons into concrete, bar-denominated label specs."""
    specs: list[LabelSpec] = []
    for horizon_minutes in config.horizons_minutes:
        bars = max(1, round(horizon_minutes / max(bar_minutes, 1)))
        for direction in ("long", "short"):
            specs.append(
                LabelSpec(
                    horizon_bars=bars,
                    target_atr_multiple=config.target_atr_multiple,
                    stop_atr_multiple=config.stop_atr_multiple,
                    min_movement_atr=config.min_movement_atr,
                    volatility_adjusted=config.volatility_adjusted,
                    direction=direction,
                )
            )
    return specs


def label_balance(labels: pd.Series) -> dict[str, float]:
    """Class balance report. A wildly imbalanced target usually means the barriers
    are misconfigured, not that the market is one-directional."""
    valid = labels.dropna()
    if valid.empty:
        return {"count": 0, "positive_rate": float("nan")}
    return {
        "count": int(len(valid)),
        "positive_rate": float(valid.mean()),
        "positives": int(valid.sum()),
        "negatives": int(len(valid) - valid.sum()),
    }
