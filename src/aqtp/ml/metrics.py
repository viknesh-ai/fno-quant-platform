"""Model evaluation metrics (REQ 16).

REQ 16 says explicitly: do not optimize solely for classification accuracy. So
accuracy is reported but never used as a selection criterion. The metrics that
decide whether a model is usable are:

  * **Log loss / Brier** — is the probability itself well-formed?
  * **Calibration error** — when it says 0.60, does it happen 60% of the time? This
    is the one that matters most, because position sizing and expected value are
    computed *from* the probability. A miscalibrated 0.8 is worse than an honest 0.55.
  * **Expected trading value** — does acting on it make money after costs?
  * **Drawdown impact** — what does acting on it cost in the worst stretch?
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    samples: int = 0
    positive_rate: float = float("nan")
    log_loss: float = float("nan")
    brier_score: float = float("nan")
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    accuracy: float = float("nan")
    calibration_error: float = float("nan")
    calibration_slope: float = float("nan")
    expected_value_per_trade: float = float("nan")
    expected_value_total: float = float("nan")
    max_drawdown_r: float = float("nan")
    trades_taken: int = 0
    hit_rate_at_threshold: float = float("nan")
    threshold: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def is_usable(self, *, max_brier: float, min_auc: float) -> tuple[bool, str]:
        """Gate a model on probability quality, not on accuracy (REQ 16)."""
        if not np.isfinite(self.brier_score) or self.brier_score > max_brier:
            return False, f"Brier score {self.brier_score:.4f} exceeds the {max_brier} limit"
        if np.isfinite(self.roc_auc) and self.roc_auc < min_auc:
            return False, f"ROC-AUC {self.roc_auc:.4f} below the {min_auc} minimum"
        if np.isfinite(self.calibration_error) and self.calibration_error > 0.12:
            return False, f"calibration error {self.calibration_error:.4f} is too large to size on"
        if np.isfinite(self.expected_value_per_trade) and self.expected_value_per_trade <= 0:
            return False, (
                f"expected value {self.expected_value_per_trade:+.4f}R per trade is not positive"
            )
        return True, ""


def evaluate_classification(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    threshold: float = 0.5,
    realized_r: np.ndarray | pd.Series | None = None,
    cost_r: float = 0.05,
) -> ClassificationMetrics:
    """Full evaluation of a probabilistic binary classifier.

    `realized_r` (P&L in R units per sample) turns the statistical metrics into the
    trading metrics REQ 16 requires. `cost_r` is the round-trip cost expressed in R,
    subtracted from every simulated trade so that expected value is net of costs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    metrics = ClassificationMetrics(samples=len(y_true), threshold=threshold)
    if len(y_true) == 0:
        return metrics

    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    metrics.positive_rate = float(y_true.mean())
    y_pred = (y_prob >= threshold).astype(float)
    metrics.accuracy = float((y_pred == y_true).mean())

    # These require both classes to be present; a single-class fold is a real
    # situation (quiet period) and must not crash the walk-forward run.
    both_classes = len(np.unique(y_true)) > 1
    try:
        metrics.log_loss = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        pass
    try:
        metrics.brier_score = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        pass
    if both_classes:
        try:
            metrics.roc_auc = float(roc_auc_score(y_true, y_prob))
            metrics.pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pass
        metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
        metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))

    metrics.calibration_error = expected_calibration_error(y_true, y_prob)
    metrics.calibration_slope = calibration_slope(y_true, y_prob)

    selected = y_prob >= threshold
    metrics.trades_taken = int(selected.sum())
    if metrics.trades_taken:
        metrics.hit_rate_at_threshold = float(y_true[selected].mean())

    if realized_r is not None:
        realized = np.asarray(realized_r, dtype=float)
        if len(realized) == len(y_true) and metrics.trades_taken:
            net = realized[selected] - cost_r
            metrics.expected_value_per_trade = float(np.nanmean(net))
            metrics.expected_value_total = float(np.nansum(net))
            metrics.max_drawdown_r = max_drawdown(np.nan_to_num(net))
    elif metrics.trades_taken:
        # Without realized R, approximate EV from the label and the configured
        # target/stop ratio implied by a 1:1 payoff. Clearly a proxy, so it is only
        # used when true P&L is unavailable.
        hit = metrics.hit_rate_at_threshold
        metrics.expected_value_per_trade = float(hit - (1 - hit) - cost_r)

    return metrics


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10
) -> float:
    """Mean absolute gap between predicted probability and observed frequency,
    weighted by bin population."""
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(y_prob, edges[1:-1], right=False)
    total_error, total_weight = 0.0, 0
    for b in range(bins):
        mask = indices == b
        count = int(mask.sum())
        if count == 0:
            continue
        total_error += abs(y_prob[mask].mean() - y_true[mask].mean()) * count
        total_weight += count
    return float(total_error / total_weight) if total_weight else float("nan")


def calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Slope of observed frequency against predicted probability.

    1.0 is perfect. Below 1 means the model is overconfident (its extremes are too
    extreme) — the most common failure mode for a tree ensemble on financial data,
    and the reason calibration is applied by default.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    logit = np.log(y_prob / (1 - y_prob))
    if np.allclose(logit, logit[0]):
        return float("nan")
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(fit_intercept=True, C=1e6, max_iter=1000)
        model.fit(logit.reshape(-1, 1), y_true)
        return float(model.coef_[0][0])
    except Exception:
        return float("nan")


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10) -> pd.DataFrame:
    """Per-bin predicted-vs-observed table; the input to a calibration plot."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(y_prob, edges[1:-1], right=False)
    rows = []
    for b in range(bins):
        mask = indices == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "count": int(mask.sum()),
                "predicted": float(y_prob[mask].mean()),
                "observed": float(y_true[mask].mean()),
                "gap": float(y_prob[mask].mean() - y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def compare_models(results: dict[str, ClassificationMetrics]) -> pd.DataFrame:
    """Side-by-side comparison table (REQ 16 requires model comparison).

    Sorted by expected trading value, then by Brier — deliberately not by accuracy
    or AUC, which do not tell you whether acting on the model makes money.
    """
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame({name: m.to_dict() for name, m in results.items()}).T
    frame.index.name = "model"
    sort_columns = [c for c in ("expected_value_per_trade", "brier_score") if c in frame.columns]
    if sort_columns:
        frame = frame.sort_values(
            sort_columns, ascending=[False, True][: len(sort_columns)]
        )
    return frame


@dataclass
class WalkForwardReport:
    """Per-fold results, kept separate so no aggregate can hide a bad period."""

    folds: list[ClassificationMetrics] = field(default_factory=list)
    fold_windows: list[dict] = field(default_factory=list)

    def add(self, metrics: ClassificationMetrics, window: dict) -> None:
        self.folds.append(metrics)
        self.fold_windows.append(window)

    def to_frame(self) -> pd.DataFrame:
        if not self.folds:
            return pd.DataFrame()
        frame = pd.DataFrame([m.to_dict() for m in self.folds])
        windows = pd.DataFrame(self.fold_windows)
        return pd.concat([windows, frame], axis=1)

    def summary(self) -> dict[str, float]:
        """Aggregate plus *consistency*: the fraction of folds that were profitable.

        A strategy that makes all its money in one fold out of ten is not robust,
        and the mean alone would hide that (REQ 20/40).
        """
        if not self.folds:
            return {}
        evs = np.array([f.expected_value_per_trade for f in self.folds], dtype=float)
        briers = np.array([f.brier_score for f in self.folds], dtype=float)
        aucs = np.array([f.roc_auc for f in self.folds], dtype=float)
        valid_ev = evs[np.isfinite(evs)]
        return {
            "folds": len(self.folds),
            "mean_expected_value": float(np.nanmean(evs)) if len(valid_ev) else float("nan"),
            "median_expected_value": float(np.nanmedian(evs)) if len(valid_ev) else float("nan"),
            "std_expected_value": float(np.nanstd(evs)) if len(valid_ev) else float("nan"),
            "profitable_fold_fraction": (
                float((valid_ev > 0).mean()) if len(valid_ev) else float("nan")
            ),
            "worst_fold_expected_value": float(np.nanmin(evs)) if len(valid_ev) else float("nan"),
            "mean_brier": float(np.nanmean(briers)),
            "mean_roc_auc": float(np.nanmean(aucs)),
            "total_samples": int(sum(f.samples for f in self.folds)),
        }

    def verdict(self, *, min_profitable_fraction: float = 0.6) -> tuple[bool, str]:
        """REQ 68: if the evidence does not support an edge, say FAILED."""
        summary = self.summary()
        if not summary:
            return False, "FAILED: no folds were evaluated"
        fraction = summary.get("profitable_fold_fraction", float("nan"))
        mean_ev = summary.get("mean_expected_value", float("nan"))
        if not np.isfinite(mean_ev) or mean_ev <= 0:
            return False, f"FAILED: mean out-of-sample expected value {mean_ev:+.4f}R is not positive"
        if not np.isfinite(fraction) or fraction < min_profitable_fraction:
            return False, (
                f"FAILED: only {fraction:.0%} of walk-forward folds were profitable, "
                f"below the {min_profitable_fraction:.0%} consistency requirement"
            )
        return True, (
            f"PASSED: mean {mean_ev:+.4f}R per trade across {summary['folds']} folds, "
            f"{fraction:.0%} of folds profitable"
        )
