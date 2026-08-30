"""Cross-asset context: correlation, breadth, dispersion and volatility regime.

A signal on one symbol is worth different amounts depending on what the rest of
the market is doing at the same moment. A BANKNIFTY long when breadth is negative
and every index is correlated at 0.9 is not a diversifying trade — it is the same
trade the portfolio already has, sized twice.

This tracker keeps a rolling return matrix across every underlying the scanner
touches and answers three questions: how correlated is everything right now, is
participation broad or narrow, and is the volatility regime rising or falling.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Mapping

import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CrossAssetReport:
    """Market-wide state at one instant."""

    timestamp: datetime | None = None
    symbols_tracked: int = 0
    average_correlation: float | None = None
    max_correlation: float | None = None
    dispersion: float | None = None            # cross-sectional return spread
    breadth: float | None = None               # [-1, 1]: share advancing
    breadth_above_vwap: float | None = None
    relative_strength: dict[str, float] = field(default_factory=dict)
    leaders: list[str] = field(default_factory=list)
    laggards: list[str] = field(default_factory=list)
    benchmark: str = ""
    benchmark_return: float | None = None
    vix: float | None = None
    vix_change_pct: float | None = None
    vix_percentile: float | None = None
    reasons: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def regime(self) -> str:
        """A one-word description of the market backdrop."""
        if self.vix_percentile is not None and self.vix_percentile > 0.80:
            return "stressed"
        if self.average_correlation is not None and self.average_correlation > 0.80:
            return "risk_on_off"      # everything moving together — macro-driven
        if self.dispersion is not None and self.dispersion > 0.015:
            return "stock_pickers"
        return "normal"

    @property
    def diversification_score(self) -> float:
        """[0, 1] — how much an additional position actually diversifies.

        Near zero, adding a second name adds risk without adding independence,
        and the portfolio's correlated-exposure cap should bind harder.
        """
        if self.average_correlation is None:
            return 0.5
        return float(np.clip(1.0 - self.average_correlation, 0.0, 1.0))

    def alignment_for(self, symbol: str, direction_sign: float) -> float:
        """[-1, 1] — does the market backdrop support a trade in this direction?

        Combines breadth with the symbol's own relative strength, so a long in a
        leading name during broad participation scores high and the same long in
        a laggard during narrow, negative breadth scores negative.
        """
        votes: list[float] = []
        if self.breadth is not None:
            votes.append(float(np.clip(self.breadth * direction_sign, -1.0, 1.0)))
        strength = self.relative_strength.get(symbol)
        if strength is not None:
            votes.append(float(np.clip(strength * direction_sign / 0.01, -1.0, 1.0)))
        if self.vix_change_pct is not None:
            # Rising VIX supports downside, falling VIX supports upside.
            votes.append(float(np.clip(-self.vix_change_pct * direction_sign / 0.05, -1.0, 1.0)))
        return float(np.mean(votes)) if votes else 0.0

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"xa_diversification": self.diversification_score}
        for name in (
            "average_correlation", "max_correlation", "dispersion", "breadth",
            "breadth_above_vwap", "vix", "vix_change_pct", "vix_percentile",
            "benchmark_return",
        ):
            value = getattr(self, name)
            if value is not None:
                out[f"xa_{name}"] = float(value)
        return out

    def explain(self) -> list[str]:
        return list(self.reasons)


# --------------------------------------------------------------------------- #
class CrossAssetTracker:
    """Rolling cross-sectional state across the scanned universe."""

    def __init__(self, *, history: int = 120, benchmark: str = "NIFTY") -> None:
        self._history = history
        self._benchmark = benchmark
        self._returns: dict[str, Deque[float]] = {}
        self._prices: dict[str, float] = {}
        self._vix: Deque[float] = deque(maxlen=500)
        self._last_report: CrossAssetReport | None = None

    # ------------------------------------------------------------------ #
    def observe(self, symbol: str, price: float) -> None:
        """Record one price observation, converting it to a log return."""
        if price is None or not math.isfinite(price) or price <= 0:
            return
        previous = self._prices.get(symbol)
        self._prices[symbol] = float(price)
        if previous is None or previous <= 0:
            return
        series = self._returns.setdefault(symbol, deque(maxlen=self._history))
        series.append(math.log(price / previous))

    def observe_vix(self, value: float) -> None:
        if value and math.isfinite(value) and value > 0:
            self._vix.append(float(value))

    def observations(self, symbol: str) -> int:
        return len(self._returns.get(symbol, ()))

    @property
    def last_report(self) -> CrossAssetReport | None:
        return self._last_report

    # ------------------------------------------------------------------ #
    def build(
        self,
        *,
        timestamp: datetime,
        session_returns: Mapping[str, float] | None = None,
        above_vwap: Mapping[str, bool] | None = None,
        min_observations: int = 20,
    ) -> CrossAssetReport:
        """Compute the market-wide report.

        `session_returns` is each symbol's return since the open — used for
        breadth and relative strength, which are day-level measures rather than
        bar-level ones.
        """
        report = CrossAssetReport(timestamp=timestamp, benchmark=self._benchmark)
        usable = {
            symbol: np.asarray(series, dtype=float)
            for symbol, series in self._returns.items()
            if len(series) >= min_observations
        }
        report.symbols_tracked = len(usable)

        if len(usable) >= 3:
            length = min(len(series) for series in usable.values())
            matrix = np.vstack([series[-length:] for series in usable.values()])
            # Drop constant rows: a symbol that has not moved has no correlation.
            variable = matrix[matrix.std(axis=1) > 1e-12]
            if variable.shape[0] >= 3:
                correlation = np.corrcoef(variable)
                off_diagonal = correlation[np.triu_indices_from(correlation, k=1)]
                off_diagonal = off_diagonal[np.isfinite(off_diagonal)]
                if off_diagonal.size:
                    report.average_correlation = float(np.mean(np.abs(off_diagonal)))
                    report.max_correlation = float(np.max(np.abs(off_diagonal)))
                # Dispersion: cross-sectional spread of the latest bar's returns.
                report.dispersion = float(np.std(variable[:, -1], ddof=1))

        if session_returns:
            values = np.array(
                [v for v in session_returns.values() if v is not None and math.isfinite(v)],
                dtype=float,
            )
            if values.size:
                advancing = float(np.sum(values > 0))
                report.breadth = float((2.0 * advancing / values.size) - 1.0)
                report.relative_strength = {
                    symbol: float(value)
                    for symbol, value in session_returns.items()
                    if value is not None and math.isfinite(value)
                }
                ranked = sorted(report.relative_strength.items(), key=lambda kv: kv[1], reverse=True)
                report.leaders = [symbol for symbol, _ in ranked[:3]]
                report.laggards = [symbol for symbol, _ in ranked[-3:]][::-1]
            report.benchmark_return = session_returns.get(self._benchmark)

        if above_vwap:
            flags = list(above_vwap.values())
            if flags:
                report.breadth_above_vwap = float(sum(1 for f in flags if f) / len(flags))

        if self._vix:
            report.vix = float(self._vix[-1])
            if len(self._vix) >= 2 and self._vix[0] > 0:
                report.vix_change_pct = float((self._vix[-1] - self._vix[-2]) / self._vix[-2])
            if len(self._vix) >= 30:
                history = np.asarray(self._vix, dtype=float)
                report.vix_percentile = float(np.mean(history <= history[-1]))

        _narrate(report)
        self._last_report = report
        return report


def _narrate(report: CrossAssetReport) -> None:
    if report.average_correlation is not None:
        report.reasons.append(
            f"average pairwise correlation {report.average_correlation:.2f} across "
            f"{report.symbols_tracked} symbols — diversification score "
            f"{report.diversification_score:.2f}"
        )
    if report.breadth is not None:
        share = (report.breadth + 1.0) / 2.0
        report.reasons.append(
            f"breadth {share:.0%} advancing"
            + (f", {report.breadth_above_vwap:.0%} above VWAP" if report.breadth_above_vwap is not None else "")
        )
    if report.leaders:
        report.reasons.append(
            f"leaders {', '.join(report.leaders)} | laggards {', '.join(report.laggards)}"
        )
    if report.vix is not None:
        line = f"India VIX {report.vix:.2f}"
        if report.vix_change_pct is not None:
            line += f" ({report.vix_change_pct:+.2%})"
        if report.vix_percentile is not None:
            line += f", {report.vix_percentile:.0%} percentile"
        report.reasons.append(line)
    report.reasons.append(f"market regime: {report.regime.replace('_', ' ')}")
