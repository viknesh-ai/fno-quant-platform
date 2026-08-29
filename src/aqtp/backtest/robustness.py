"""Robustness testing (REQ 40).

REQ 40's premise is that a strategy must demonstrate robustness rather than produce
one beautiful backtest. Each test here attacks a different way a single backtest can
be misleading:

  Monte Carlo trade-order      — was the equity curve's shape luck of sequencing?
  Parameter perturbation       — does the edge survive slightly different settings,
                                 or does it sit on a knife-edge (i.e. it is fitted)?
  Cost sensitivity             — how much does the edge depend on optimistic costs?
  Slippage sensitivity         — same, for execution quality.
  Delayed entry/exit           — does the edge survive acting a bar late?
  Missing data                 — what happens when bars are absent?
  Spread widening              — does it survive a stressed book?
  Execution failure            — what if a fraction of orders simply don't fill?

`run_all()` aggregates these into a single verdict. A strategy that fails any of the
critical tests is reported as NOT ROBUST — REQ 68 requires marking such a strategy
FAILED rather than tuning until it passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from .metrics import ClosedTrade, drawdown_analysis

logger = get_logger(__name__)

# A callable that re-runs the backtest under modified conditions and returns trades.
BacktestRunner = Callable[..., Sequence[ClosedTrade]]


@dataclass
class TestOutcome:
    name: str
    passed: bool
    detail: str
    statistics: dict[str, float] = field(default_factory=dict)
    critical: bool = True

    def describe(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class RobustnessReport:
    outcomes: list[TestOutcome] = field(default_factory=list)

    def add(self, outcome: TestOutcome) -> None:
        self.outcomes.append(outcome)

    @property
    def critical_failures(self) -> list[TestOutcome]:
        return [o for o in self.outcomes if o.critical and not o.passed]

    @property
    def is_robust(self) -> bool:
        # Zero passing tests is never evidence of robustness, however the individual
        # failures were classified — a run where nothing could be evaluated must not
        # report ROBUST just because every failure was non-critical.
        return bool(self.outcomes) and any(o.passed for o in self.outcomes) and not self.critical_failures

    def verdict(self) -> str:
        if not self.outcomes:
            return "INCONCLUSIVE: no robustness tests were run"
        passed = sum(1 for o in self.outcomes if o.passed)
        if passed == 0:
            return (
                f"INCONCLUSIVE: none of the {len(self.outcomes)} robustness tests could be "
                "evaluated (typically too few trades). This is not evidence of robustness."
            )
        if self.is_robust:
            return f"ROBUST: {passed}/{len(self.outcomes)} tests passed"
        failures = "; ".join(o.name for o in self.critical_failures)
        return (
            f"NOT ROBUST — FAILED: {len(self.critical_failures)} critical test(s) failed "
            f"({failures}). Per REQ 68, do not tune the strategy until this passes."
        )

    def to_frame(self) -> pd.DataFrame:
        if not self.outcomes:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {"test": o.name, "passed": o.passed, "critical": o.critical, "detail": o.detail}
                for o in self.outcomes
            ]
        )

    def summary_lines(self) -> list[str]:
        return [o.describe() for o in self.outcomes] + ["", self.verdict()]


# --------------------------------------------------------------------------- #
def monte_carlo_trade_order(
    trades: Sequence[ClosedTrade],
    *,
    runs: int = 1000,
    initial_capital: float = 500_000.0,
    seed: int = 42,
    max_acceptable_drawdown_pct: float = 0.25,
    min_profitable_fraction: float = 0.90,
) -> TestOutcome:
    """Reshuffle trade order to see how much of the result was sequencing luck.

    The set of trades is held fixed and only their order is randomized. Total P&L is
    therefore identical in every run — what changes is the *path*, and hence the
    drawdown. A strategy whose 5th-percentile drawdown is catastrophic is one whose
    real-world survival depended on a favourable ordering.
    """
    if len(trades) < 10:
        return TestOutcome(
            "monte_carlo_trade_order", False,
            f"only {len(trades)} trades — too few for a meaningful resample",
            critical=False,
        )

    rng = np.random.default_rng(seed)
    pnls = np.array([t.net_pnl for t in trades])
    drawdowns: list[float] = []
    finals: list[float] = []

    for _ in range(runs):
        shuffled = rng.permutation(pnls)
        equity = initial_capital + np.cumsum(shuffled)
        peak = np.maximum.accumulate(np.concatenate([[initial_capital], equity]))[1:]
        drawdown_pct = np.max((peak - equity) / np.maximum(peak, 1e-9))
        drawdowns.append(float(drawdown_pct))
        finals.append(float(equity[-1]))

    drawdowns_array = np.array(drawdowns)
    p95_drawdown = float(np.percentile(drawdowns_array, 95))
    median_drawdown = float(np.median(drawdowns_array))
    profitable_fraction = float(np.mean(np.array(finals) > initial_capital))

    passed = (
        p95_drawdown <= max_acceptable_drawdown_pct
        and profitable_fraction >= min_profitable_fraction
    )
    return TestOutcome(
        "monte_carlo_trade_order",
        passed,
        (
            f"95th-percentile drawdown {p95_drawdown:.2%} (median {median_drawdown:.2%}), "
            f"{profitable_fraction:.0%} of orderings ended profitable"
        ),
        statistics={
            "p95_drawdown": p95_drawdown,
            "median_drawdown": median_drawdown,
            "p99_drawdown": float(np.percentile(drawdowns_array, 99)),
            "profitable_fraction": profitable_fraction,
        },
    )


def monte_carlo_resample(
    trades: Sequence[ClosedTrade],
    *,
    runs: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> TestOutcome:
    """Bootstrap the expectancy to get a confidence interval.

    Sampling *with* replacement (unlike the order test) varies which trades occur,
    which is what produces a genuine interval around the expectancy. If the lower
    bound of that interval is below zero, the observed edge is not distinguishable
    from noise at this sample size.
    """
    if len(trades) < 20:
        return TestOutcome(
            "monte_carlo_expectancy", False,
            f"only {len(trades)} trades — cannot bootstrap an expectancy interval",
            critical=False,
        )

    rng = np.random.default_rng(seed)
    r_values = np.array([t.r_multiple for t in trades])
    means = np.array([rng.choice(r_values, size=len(r_values), replace=True).mean() for _ in range(runs)])

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(means, alpha * 100))
    upper = float(np.percentile(means, (1 - alpha) * 100))
    observed = float(r_values.mean())
    passed = lower > 0

    return TestOutcome(
        "monte_carlo_expectancy",
        passed,
        (
            f"expectancy {observed:+.3f}R, {confidence:.0%} CI [{lower:+.3f}, {upper:+.3f}]"
            + ("" if passed else " — the interval includes zero, so no edge is demonstrated")
        ),
        statistics={"observed_r": observed, "ci_lower": lower, "ci_upper": upper},
    )


def parameter_perturbation(
    runner: Callable[[Mapping[str, float]], Sequence[ClosedTrade]],
    base_params: Mapping[str, float],
    *,
    perturbation_pct: float = 0.15,
    steps: int = 2,
    min_profitable_fraction: float = 0.70,
) -> TestOutcome:
    """Re-run with each parameter nudged up and down.

    A genuine edge degrades gracefully as parameters move. An edge that collapses
    when a threshold shifts by 15% was fitted to the sample, and the backtest is
    measuring the fitting rather than the market.
    """
    results: dict[str, float] = {}
    baseline_trades = runner(base_params)
    baseline_expectancy = (
        float(np.mean([t.r_multiple for t in baseline_trades])) if baseline_trades else 0.0
    )

    for name, value in base_params.items():
        if not isinstance(value, (int, float)) or value == 0:
            continue
        for step in range(1, steps + 1):
            for sign in (-1, 1):
                factor = 1 + sign * perturbation_pct * step / steps
                variant = dict(base_params)
                variant[name] = value * factor
                trades = runner(variant)
                expectancy = (
                    float(np.mean([t.r_multiple for t in trades])) if trades else 0.0
                )
                results[f"{name}x{factor:.2f}"] = expectancy

    if not results:
        return TestOutcome(
            "parameter_perturbation", False, "no numeric parameters to perturb", critical=False
        )

    values = np.array(list(results.values()))
    profitable_fraction = float(np.mean(values > 0))
    worst = float(values.min())
    passed = profitable_fraction >= min_profitable_fraction and worst > -0.1

    return TestOutcome(
        "parameter_perturbation",
        passed,
        (
            f"{profitable_fraction:.0%} of {len(results)} perturbed variants stayed profitable "
            f"(baseline {baseline_expectancy:+.3f}R, worst {worst:+.3f}R)"
        ),
        statistics={
            "baseline_r": baseline_expectancy,
            "profitable_fraction": profitable_fraction,
            "worst_r": worst,
            "mean_r": float(values.mean()),
        },
    )


def cost_sensitivity(
    trades: Sequence[ClosedTrade],
    *,
    multipliers: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    min_profitable_multiplier: float = 1.5,
) -> TestOutcome:
    """Re-price the same trades under higher cost assumptions.

    Recomputing analytically (rather than re-running) isolates the cost effect
    exactly. A strategy that only works at 1.0x modelled costs has no margin for the
    day the spread is wider than usual.
    """
    if not trades:
        return TestOutcome("cost_sensitivity", False, "no trades to evaluate", critical=False)

    gross = np.array([t.gross_pnl for t in trades])
    costs = np.array([t.costs for t in trades])
    risk = np.array(
        [abs(t.net_pnl / t.r_multiple) if t.r_multiple else 0.0 for t in trades]
    )

    outcomes: dict[str, float] = {}
    breakeven_multiplier = float("inf")
    for multiplier in multipliers:
        net = gross - costs * multiplier
        total = float(net.sum())
        outcomes[f"{multiplier:.1f}x"] = total
        if total <= 0 and breakeven_multiplier == float("inf"):
            breakeven_multiplier = multiplier

    passed = outcomes.get(f"{min_profitable_multiplier:.1f}x", -1) > 0
    return TestOutcome(
        "cost_sensitivity",
        passed,
        (
            f"net P&L at cost multipliers: "
            + ", ".join(f"{k} {v:+,.0f}" for k, v in outcomes.items())
            + (
                f"; turns unprofitable at {breakeven_multiplier:.1f}x costs"
                if breakeven_multiplier != float("inf")
                else "; profitable across all tested cost levels"
            )
        ),
        statistics={"breakeven_cost_multiplier": breakeven_multiplier, **outcomes},
    )


def slippage_sensitivity(
    trades: Sequence[ClosedTrade],
    *,
    multipliers: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    min_profitable_multiplier: float = 2.0,
) -> TestOutcome:
    """Same idea, isolating slippage — the cost that varies most with market stress."""
    if not trades:
        return TestOutcome("slippage_sensitivity", False, "no trades to evaluate", critical=False)

    net_before_slippage = np.array([t.net_pnl + t.slippage for t in trades])
    slippage = np.array([t.slippage for t in trades])

    outcomes: dict[str, float] = {}
    for multiplier in multipliers:
        outcomes[f"{multiplier:.1f}x"] = float((net_before_slippage - slippage * multiplier).sum())

    passed = outcomes.get(f"{min_profitable_multiplier:.1f}x", -1) > 0
    return TestOutcome(
        "slippage_sensitivity",
        passed,
        "net P&L at slippage multipliers: "
        + ", ".join(f"{k} {v:+,.0f}" for k, v in outcomes.items()),
        statistics=outcomes,
    )


def execution_failure_simulation(
    trades: Sequence[ClosedTrade],
    *,
    failure_rates: Sequence[float] = (0.05, 0.10, 0.20),
    runs: int = 200,
    seed: int = 42,
) -> TestOutcome:
    """Drop a random fraction of trades to simulate rejected or unfilled orders.

    The risk this probes is concentration: if the edge lives in a handful of trades,
    missing a few of them destroys it. A robust strategy loses roughly its failure
    rate in P&L, not all of it.
    """
    if len(trades) < 20:
        return TestOutcome(
            "execution_failure", False, f"only {len(trades)} trades to sample", critical=False
        )

    rng = np.random.default_rng(seed)
    pnls = np.array([t.net_pnl for t in trades])
    baseline = float(pnls.sum())
    statistics: dict[str, float] = {"baseline": baseline}
    all_passed = True

    for rate in failure_rates:
        totals = []
        for _ in range(runs):
            keep = rng.random(len(pnls)) >= rate
            totals.append(float(pnls[keep].sum()))
        median = float(np.median(totals))
        profitable = float(np.mean(np.array(totals) > 0))
        statistics[f"fail_{rate:.0%}_median"] = median
        statistics[f"fail_{rate:.0%}_profitable"] = profitable
        if profitable < 0.85:
            all_passed = False

    return TestOutcome(
        "execution_failure",
        all_passed,
        "; ".join(
            f"{rate:.0%} order failures -> {statistics[f'fail_{rate:.0%}_profitable']:.0%} of runs profitable"
            for rate in failure_rates
        ),
        statistics=statistics,
    )


def spread_widening_simulation(
    trades: Sequence[ClosedTrade],
    *,
    widening_factors: Sequence[float] = (1.5, 2.0, 3.0),
) -> TestOutcome:
    """Charge a wider spread on every trade, both legs."""
    if not trades:
        return TestOutcome("spread_widening", False, "no trades to evaluate", critical=False)

    outcomes: dict[str, float] = {}
    for factor in widening_factors:
        extra = sum(
            t.spread_at_entry * (factor - 1) * t.quantity for t in trades
        )
        outcomes[f"{factor:.1f}x"] = float(sum(t.net_pnl for t in trades) - extra)

    passed = outcomes.get("2.0x", -1) > 0
    return TestOutcome(
        "spread_widening",
        passed,
        "net P&L at spread multipliers: "
        + ", ".join(f"{k} {v:+,.0f}" for k, v in outcomes.items()),
        statistics=outcomes,
    )


def delayed_action_test(
    baseline: Sequence[ClosedTrade],
    delayed: Mapping[int, Sequence[ClosedTrade]],
    *,
    max_acceptable_degradation: float = 0.5,
) -> TestOutcome:
    """Compare results when entries/exits are delayed by N bars.

    A strategy that loses most of its edge from a one-bar delay is capturing a move
    too fast to execute reliably, which will not survive real latency.
    """
    if not baseline:
        return TestOutcome("delayed_action", False, "no baseline trades", critical=False)

    baseline_r = float(np.mean([t.r_multiple for t in baseline]))
    statistics = {"baseline_r": baseline_r}
    passed = True
    details: list[str] = []

    for bars, trades in sorted(delayed.items()):
        if not trades:
            details.append(f"{bars}-bar delay produced no trades")
            passed = False
            continue
        delayed_r = float(np.mean([t.r_multiple for t in trades]))
        statistics[f"delay_{bars}_r"] = delayed_r
        degradation = (
            (baseline_r - delayed_r) / abs(baseline_r) if baseline_r != 0 else 1.0
        )
        details.append(f"{bars}-bar delay: {delayed_r:+.3f}R ({degradation:+.0%} change)")
        if bars <= 1 and degradation > max_acceptable_degradation:
            passed = False

    return TestOutcome(
        "delayed_action", passed, "; ".join(details), statistics=statistics
    )


def missing_data_simulation(
    frame: pd.DataFrame,
    runner: Callable[[pd.DataFrame], Sequence[ClosedTrade]],
    *,
    drop_rates: Sequence[float] = (0.01, 0.05),
    seed: int = 42,
) -> TestOutcome:
    """Remove random bars and confirm the system still behaves sanely.

    The pass condition is deliberately not "P&L stays the same" — losing data should
    change results. What must not happen is a crash or a wild swing, either of which
    means the pipeline is silently mishandling gaps.
    """
    if frame.empty:
        return TestOutcome("missing_data", False, "no data supplied", critical=False)

    rng = np.random.default_rng(seed)
    baseline = runner(frame)
    baseline_total = float(sum(t.net_pnl for t in baseline)) if baseline else 0.0
    statistics = {"baseline": baseline_total}
    details: list[str] = []
    passed = True

    for rate in drop_rates:
        keep = rng.random(len(frame)) >= rate
        degraded_frame = frame[keep]
        try:
            trades = runner(degraded_frame)
        except Exception as exc:  # a crash is a genuine failure of this test
            return TestOutcome(
                "missing_data", False, f"pipeline raised {type(exc).__name__} with {rate:.0%} of bars missing"
            )
        total = float(sum(t.net_pnl for t in trades)) if trades else 0.0
        statistics[f"drop_{rate:.0%}"] = total
        details.append(f"{rate:.0%} bars missing -> {total:+,.0f} ({len(trades)} trades)")
        if baseline_total > 0 and total < baseline_total * -0.5:
            passed = False

    return TestOutcome("missing_data", passed, "; ".join(details), statistics=statistics)


# --------------------------------------------------------------------------- #
def run_all(
    trades: Sequence[ClosedTrade],
    *,
    initial_capital: float,
    monte_carlo_runs: int = 1000,
    cost_multipliers: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    slippage_multipliers: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
    seed: int = 42,
) -> RobustnessReport:
    """Run the tests that need only the trade list (REQ 40)."""
    report = RobustnessReport()
    report.add(
        monte_carlo_trade_order(
            trades, runs=monte_carlo_runs, initial_capital=initial_capital, seed=seed
        )
    )
    report.add(monte_carlo_resample(trades, runs=monte_carlo_runs, seed=seed))
    report.add(cost_sensitivity(trades, multipliers=cost_multipliers))
    report.add(slippage_sensitivity(trades, multipliers=slippage_multipliers))
    report.add(execution_failure_simulation(trades, seed=seed))
    report.add(spread_widening_simulation(trades))
    return report
