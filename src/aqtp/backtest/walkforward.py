"""Walk-forward backtesting (REQ 17/20).

Distinct from `ml/train.py`, which walk-forwards a *model*. This module
walk-forwards the whole *strategy pipeline*: in each window a signal source is
(re)built from in-sample data and then evaluated on the following out-of-sample
window it never saw.

REQ 20 forbids treating one aggregate backtest as sufficient evidence, so the
report keeps every window separate and its verdict is driven by consistency across
windows, not by the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Instrument
from ..risk.costs import TransactionCostModel
from .engine import BacktestConfig, BacktestEngine, BacktestResult, SignalSource
from .metrics import ClosedTrade, compute_metrics

logger = get_logger(__name__)

# Builds a signal source from an in-sample frame.
SignalSourceFactory = Callable[[pd.DataFrame], SignalSource]


@dataclass
class WalkForwardWindow:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    result: BacktestResult

    @property
    def expectancy_r(self) -> float:
        return self.result.metrics.expectancy_r

    @property
    def net_pnl(self) -> float:
        return self.result.metrics.total_return

    def describe(self) -> dict:
        return {
            "window": self.index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
            "trades": self.result.metrics.total_trades,
            "net_pnl": round(self.net_pnl, 2),
            "expectancy_r": round(self.expectancy_r, 4)
            if np.isfinite(self.expectancy_r) else None,
            "win_rate": round(self.result.metrics.win_rate, 4)
            if np.isfinite(self.result.metrics.win_rate) else None,
            "max_dd_pct": round(self.result.metrics.max_drawdown_pct, 4),
        }


@dataclass
class WalkForwardBacktestReport:
    windows: list[WalkForwardWindow] = field(default_factory=list)
    combined_trades: list[ClosedTrade] = field(default_factory=list)
    initial_capital: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        if not self.windows:
            return pd.DataFrame()
        return pd.DataFrame([w.describe() for w in self.windows])

    def summary(self) -> dict:
        """Aggregate plus the consistency measures REQ 20 requires."""
        if not self.windows:
            return {}
        expectancies = np.array(
            [w.expectancy_r for w in self.windows if np.isfinite(w.expectancy_r)]
        )
        pnls = np.array([w.net_pnl for w in self.windows])
        traded = [w for w in self.windows if w.result.metrics.total_trades > 0]

        return {
            "windows": len(self.windows),
            "windows_with_trades": len(traded),
            "total_trades": int(sum(w.result.metrics.total_trades for w in self.windows)),
            "total_net_pnl": float(pnls.sum()),
            "mean_expectancy_r": float(expectancies.mean()) if len(expectancies) else float("nan"),
            "median_expectancy_r": float(np.median(expectancies)) if len(expectancies) else float("nan"),
            "std_expectancy_r": float(expectancies.std()) if len(expectancies) else float("nan"),
            "profitable_windows": int((pnls > 0).sum()),
            "profitable_window_fraction": float((pnls > 0).mean()) if len(pnls) else float("nan"),
            "worst_window_pnl": float(pnls.min()) if len(pnls) else float("nan"),
            "best_window_pnl": float(pnls.max()) if len(pnls) else float("nan"),
        }

    def combined_metrics(self):
        """Metrics over the concatenated out-of-sample trades only.

        This is the honest headline number: it contains no in-sample results at all.
        """
        if not self.combined_trades:
            return compute_metrics([], pd.Series(dtype=float), initial_capital=self.initial_capital)
        equity = self.initial_capital + np.cumsum([t.net_pnl for t in self.combined_trades])
        curve = pd.Series(
            equity, index=pd.DatetimeIndex([t.exit_time for t in self.combined_trades])
        )
        return compute_metrics(
            self.combined_trades, curve, initial_capital=self.initial_capital
        )

    def verdict(self, *, min_profitable_fraction: float = 0.6, min_windows: int = 3) -> tuple[bool, str]:
        """REQ 68: state PASSED or FAILED explicitly, on out-of-sample evidence."""
        summary = self.summary()
        if not summary:
            return False, "FAILED: no walk-forward windows were produced"
        if summary["windows"] < min_windows:
            return False, (
                f"INCONCLUSIVE: only {summary['windows']} windows (need {min_windows}); "
                "insufficient evidence either way"
            )
        if summary["total_trades"] < 30:
            return False, (
                f"INCONCLUSIVE: only {summary['total_trades']} out-of-sample trades; "
                "the sample is too small to support a conclusion"
            )
        if summary["total_net_pnl"] <= 0:
            return False, (
                f"FAILED: out-of-sample P&L {summary['total_net_pnl']:+,.0f} is not positive"
            )
        fraction = summary["profitable_window_fraction"]
        if fraction < min_profitable_fraction:
            return False, (
                f"FAILED: only {fraction:.0%} of windows were profitable, below the "
                f"{min_profitable_fraction:.0%} consistency requirement — the result depends "
                "on a small number of favourable periods"
            )
        return True, (
            f"PASSED: {summary['total_net_pnl']:+,.0f} across {summary['windows']} windows, "
            f"{fraction:.0%} profitable, mean {summary['mean_expectancy_r']:+.3f}R"
        )

    def summary_lines(self) -> list[str]:
        summary = self.summary()
        if not summary:
            return ["no walk-forward windows"]
        passed, verdict = self.verdict()
        return [
            f"Windows          : {summary['windows']} ({summary['windows_with_trades']} produced trades)",
            f"OOS trades       : {summary['total_trades']}",
            f"OOS net P&L      : {summary['total_net_pnl']:+,.0f}",
            f"Mean expectancy  : {summary['mean_expectancy_r']:+.3f}R "
            f"(median {summary['median_expectancy_r']:+.3f}R)",
            f"Profitable windows: {summary['profitable_windows']}/{summary['windows']} "
            f"({summary['profitable_window_fraction']:.0%})",
            f"Worst window     : {summary['worst_window_pnl']:+,.0f}",
            "",
            verdict,
        ]


class WalkForwardBacktester:
    def __init__(
        self,
        config: BacktestConfig,
        cost_model: TransactionCostModel,
    ) -> None:
        self.config = config
        self.costs = cost_model

    def run(
        self,
        *,
        instrument: Instrument,
        frame: pd.DataFrame,
        source_factory: SignalSourceFactory,
        train_days: int,
        test_days: int,
        step_days: int | None = None,
        embargo_minutes: int = 60,
    ) -> WalkForwardBacktestReport:
        """Roll a train/test window through the data."""
        report = WalkForwardBacktestReport(initial_capital=self.config.initial_capital)
        if frame.empty:
            return report

        frame = frame.sort_index()
        step_days = step_days or test_days
        start, end = frame.index[0], frame.index[-1]
        embargo = pd.Timedelta(minutes=embargo_minutes)

        cursor = start
        window_index = 0
        engine = BacktestEngine(self.config, self.costs)

        while True:
            train_start = cursor
            train_end = train_start + pd.Timedelta(days=train_days)
            test_start = train_end + embargo
            test_end = test_start + pd.Timedelta(days=test_days)
            if test_end > end:
                break

            train_frame = frame[(frame.index >= train_start) & (frame.index < train_end)]
            test_frame = frame[(frame.index >= test_start) & (frame.index < test_end)]

            if len(train_frame) < 50 or len(test_frame) < 10:
                cursor = cursor + pd.Timedelta(days=step_days)
                continue

            # The signal source is built from training data only, then frozen for
            # the test window. Rebuilding it inside the test window would be
            # in-sample fitting wearing a walk-forward costume.
            source = source_factory(train_frame)
            result = engine.run(
                instrument=instrument, frame=test_frame, signal_source=source
            )

            window = WalkForwardWindow(
                index=window_index,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
                result=result,
            )
            report.windows.append(window)
            report.combined_trades.extend(result.trades)
            window_index += 1
            cursor = cursor + pd.Timedelta(days=step_days)

        logger.info(
            "walk-forward backtest: %d windows, %d out-of-sample trades",
            len(report.windows), len(report.combined_trades),
        )
        return report
