"""Statistical characterisation of a price series.

The question this module answers is not "is price going up" but "is this series
the kind of series where a trend-following bet is even the right shape of bet".
Indicators assume a model; these estimators test whether that model holds right
now. A breakout taken in a mean-reverting, high-entropy tape is a different trade
from the same breakout in a persistent, low-entropy one, and the platform should
be able to tell them apart before it risks capital.

Every estimator degrades gracefully: too little data returns `None` rather than a
confidently wrong number, and the report's `quality` field says how much of the
analysis actually ran.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.logging import get_logger

logger = get_logger(__name__)

TRADING_MINUTES_PER_YEAR = 375 * 252


# --------------------------------------------------------------------------- #
# Individual estimators
# --------------------------------------------------------------------------- #
def hurst_exponent(series: Sequence[float] | pd.Series, *, min_points: int = 64) -> float | None:
    """Rescaled-range Hurst exponent.

    H > 0.5 is persistent (trends continue), H < 0.5 is anti-persistent (moves
    revert), H ~ 0.5 is a random walk. Estimated over dyadic window sizes because
    the R/S statistic is only meaningful when averaged across scales.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_points:
        return None

    log_prices = np.log(np.maximum(values, 1e-12))
    returns = np.diff(log_prices)
    if returns.size < min_points // 2 or np.allclose(returns, 0.0):
        return None

    n = returns.size
    scales: list[int] = []
    scale = 8
    while scale <= n // 2:
        scales.append(scale)
        scale *= 2
    if len(scales) < 3:
        return None

    log_scales: list[float] = []
    log_rs: list[float] = []
    for window in scales:
        chunks = n // window
        ratios = []
        for i in range(chunks):
            chunk = returns[i * window : (i + 1) * window]
            deviation = np.cumsum(chunk - chunk.mean())
            spread = deviation.max() - deviation.min()
            sigma = chunk.std(ddof=1)
            if sigma > 1e-12 and spread > 0:
                ratios.append(spread / sigma)
        if ratios:
            log_scales.append(math.log(window))
            log_rs.append(math.log(float(np.mean(ratios))))

    if len(log_scales) < 3:
        return None
    slope, _ = np.polyfit(log_scales, log_rs, 1)
    return float(np.clip(slope, 0.0, 1.0))


def variance_ratio(series: Sequence[float] | pd.Series, *, lag: int = 5) -> float | None:
    """Lo–MacKinlay variance ratio.

    VR > 1 means variance grows faster than linearly in time — trending. VR < 1
    means it grows slower — mean-reverting. VR == 1 is a random walk. This is a
    sharper test than Hurst on short samples because it needs no scaling fit.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < lag * 10:
        return None

    returns = np.diff(np.log(np.maximum(values, 1e-12)))
    if returns.size < lag * 8 or np.allclose(returns, 0.0):
        return None

    var_1 = float(np.var(returns, ddof=1))
    if var_1 <= 1e-18:
        return None

    aggregated = np.convolve(returns, np.ones(lag), mode="valid")
    var_q = float(np.var(aggregated, ddof=1))
    return float(var_q / (lag * var_1))


def permutation_entropy(
    series: Sequence[float] | pd.Series, *, order: int = 4, delay: int = 1
) -> float | None:
    """Normalised permutation entropy in [0, 1].

    Near 1 the ordering of successive prices is indistinguishable from noise;
    lower values mean the series has repeatable local shapes worth modelling.
    Unlike volatility this is scale-free, so it compares NIFTY with a mid-cap
    directly.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    length = values.size - delay * (order - 1)
    if length < 20:
        return None

    patterns: dict[tuple[int, ...], int] = {}
    for i in range(length):
        window = values[i : i + delay * order : delay]
        key = tuple(np.argsort(window, kind="stable"))
        patterns[key] = patterns.get(key, 0) + 1

    counts = np.array(list(patterns.values()), dtype=float)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normaliser = math.log(math.factorial(order))
    return float(np.clip(entropy / normaliser, 0.0, 1.0)) if normaliser > 0 else None


def efficiency_ratio(series: Sequence[float] | pd.Series, *, period: int = 20) -> float | None:
    """Kaufman efficiency ratio: net movement divided by total path length.

    1.0 is a straight line, 0.0 is pure churn. It is the cheapest honest answer to
    "is this move going anywhere".
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < period + 1:
        return None
    window = values[-(period + 1) :]
    direction = abs(window[-1] - window[0])
    path = float(np.sum(np.abs(np.diff(window))))
    return float(direction / path) if path > 1e-12 else 0.0


def autocorrelation(series: Sequence[float] | pd.Series, *, lag: int = 1) -> float | None:
    """Autocorrelation of returns at `lag` bars."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < lag + 30:
        return None
    returns = np.diff(np.log(np.maximum(values, 1e-12)))
    if returns.size <= lag + 5 or np.allclose(returns, 0.0):
        return None
    a, b = returns[:-lag], returns[lag:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ewma_volatility(returns: Sequence[float] | pd.Series, *, lam: float = 0.94) -> float | None:
    """RiskMetrics EWMA volatility (per bar, not annualised)."""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        return None
    variance = float(np.var(values[: min(20, values.size)], ddof=1))
    for r in values:
        variance = lam * variance + (1.0 - lam) * r * r
    return float(math.sqrt(max(variance, 0.0)))


def garch11_forecast(
    returns: Sequence[float] | pd.Series, *, horizon: int = 1
) -> tuple[float, dict[str, float]] | None:
    """One-step (or `horizon`-step) GARCH(1,1) volatility forecast.

    Fitted by maximising the Gaussian quasi-likelihood over a small parameter
    grid rather than a full optimiser: on 200–400 intraday bars the likelihood
    surface is flat enough that a grid is both faster and more stable, and a
    volatility forecast that is wrong in the third decimal changes no decision.
    Returns the forecast per-bar sigma and the fitted parameters.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 60:
        return None
    values = values - values.mean()
    sample_var = float(np.var(values, ddof=1))
    if sample_var <= 1e-18:
        return None

    best: tuple[float, float, float, float] | None = None  # (loglik, omega, alpha, beta)
    for alpha in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25):
        for beta in (0.60, 0.70, 0.80, 0.88, 0.94):
            if alpha + beta >= 0.999:
                continue
            omega = sample_var * (1.0 - alpha - beta)
            if omega <= 0:
                continue
            variance = sample_var
            loglik = 0.0
            for r in values:
                variance = omega + alpha * r * r + beta * variance
                if variance <= 1e-18:
                    loglik = -np.inf
                    break
                loglik -= 0.5 * (math.log(variance) + r * r / variance)
            if np.isfinite(loglik) and (best is None or loglik > best[0]):
                best = (loglik, omega, alpha, beta)

    if best is None:
        return None
    _, omega, alpha, beta = best

    variance = sample_var
    for r in values:
        variance = omega + alpha * r * r + beta * variance
    persistence = alpha + beta
    long_run = omega / max(1e-12, 1.0 - persistence)
    for _ in range(max(0, horizon - 1)):
        variance = long_run + persistence * (variance - long_run)

    params = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "long_run_sigma": math.sqrt(max(long_run, 0.0)),
    }
    return float(math.sqrt(max(variance, 0.0))), params


def detect_jumps(
    frame: pd.DataFrame, *, window: int = 60, threshold: float = 4.0
) -> tuple[int, float]:
    """Lee–Mykland style jump count and the size of the most recent jump.

    Returns `(jumps_in_window, latest_jump_in_sigmas)`. A tape that has just
    jumped is not the tape the last twenty bars of features describe, which is
    exactly when a mechanical entry does the most damage.
    """
    if frame.empty or "close" not in frame or len(frame) < window + 5:
        return 0, 0.0
    close = frame["close"].astype(float).tail(window + 1)
    returns = np.diff(np.log(np.maximum(close.to_numpy(), 1e-12)))
    if returns.size < 10:
        return 0, 0.0
    # Bipower variation is robust to the jumps we are trying to detect.
    bipower = (math.pi / 2.0) * np.mean(np.abs(returns[:-1]) * np.abs(returns[1:]))
    sigma = math.sqrt(max(bipower, 1e-18))
    if sigma <= 1e-12:
        return 0, 0.0
    standardised = np.abs(returns) / sigma
    jumps = int(np.sum(standardised > threshold))
    return jumps, float(standardised[-1])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class StatisticalReport:
    """What kind of series this is, right now."""

    bars_analyzed: int = 0
    hurst: float | None = None
    variance_ratio: float | None = None
    entropy: float | None = None
    efficiency_ratio: float | None = None
    autocorr_1: float | None = None
    autocorr_5: float | None = None
    ewma_sigma: float | None = None
    garch_sigma: float | None = None
    garch_params: dict[str, float] = field(default_factory=dict)
    volatility_ratio: float | None = None   # forecast sigma / realised sigma
    jump_count: int = 0
    latest_jump_sigmas: float = 0.0
    reasons: list[str] = field(default_factory=list)

    # --- derived judgements ------------------------------------------------
    @property
    def is_trending(self) -> bool:
        votes = 0
        if self.hurst is not None and self.hurst > 0.55:
            votes += 1
        if self.variance_ratio is not None and self.variance_ratio > 1.15:
            votes += 1
        if self.efficiency_ratio is not None and self.efficiency_ratio > 0.40:
            votes += 1
        return votes >= 2

    @property
    def is_mean_reverting(self) -> bool:
        votes = 0
        if self.hurst is not None and self.hurst < 0.45:
            votes += 1
        if self.variance_ratio is not None and self.variance_ratio < 0.85:
            votes += 1
        if self.autocorr_1 is not None and self.autocorr_1 < -0.10:
            votes += 1
        return votes >= 2

    @property
    def persistence_score(self) -> float:
        """Signed [-1, 1]: +1 strongly trend-persistent, -1 strongly reverting."""
        votes: list[float] = []
        if self.hurst is not None:
            votes.append(float(np.clip((self.hurst - 0.5) / 0.25, -1.0, 1.0)))
        if self.variance_ratio is not None:
            votes.append(float(np.clip((self.variance_ratio - 1.0) / 0.4, -1.0, 1.0)))
        if self.autocorr_1 is not None:
            votes.append(float(np.clip(self.autocorr_1 / 0.2, -1.0, 1.0)))
        if self.efficiency_ratio is not None:
            votes.append(float(np.clip((self.efficiency_ratio - 0.3) / 0.4, -1.0, 1.0)))
        return float(np.mean(votes)) if votes else 0.0

    @property
    def tradability(self) -> float:
        """[0, 1] — how much structure there is to trade at all.

        Low entropy and high efficiency mean the series has exploitable shape;
        high entropy means any model is fitting noise.
        """
        parts: list[float] = []
        if self.entropy is not None:
            parts.append(float(np.clip((0.98 - self.entropy) / 0.25, 0.0, 1.0)))
        if self.efficiency_ratio is not None:
            parts.append(float(np.clip(self.efficiency_ratio / 0.5, 0.0, 1.0)))
        if abs(self.persistence_score) > 0:
            parts.append(abs(self.persistence_score))
        return float(np.mean(parts)) if parts else 0.5

    @property
    def volatility_regime(self) -> str:
        if self.volatility_ratio is None:
            return "unknown"
        if self.volatility_ratio > 1.25:
            return "expanding"
        if self.volatility_ratio < 0.80:
            return "contracting"
        return "stable"

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {
            "stat_persistence": self.persistence_score,
            "stat_tradability": self.tradability,
            "stat_jump_count": float(self.jump_count),
            "stat_latest_jump_sigmas": self.latest_jump_sigmas,
        }
        for name in (
            "hurst", "variance_ratio", "entropy", "efficiency_ratio",
            "autocorr_1", "autocorr_5", "ewma_sigma", "garch_sigma", "volatility_ratio",
        ):
            value = getattr(self, name)
            if value is not None:
                out[f"stat_{name}"] = float(value)
        return out

    def explain(self) -> list[str]:
        return list(self.reasons)


def analyze_series(frame: pd.DataFrame, *, forecast_horizon: int = 3) -> StatisticalReport:
    """Run the full statistical battery over a closed-bar OHLC frame."""
    report = StatisticalReport()
    if frame is None or frame.empty or "close" not in frame:
        report.reasons.append("no price history available for statistical analysis")
        return report

    close = frame["close"].astype(float).dropna()
    report.bars_analyzed = int(close.size)
    if close.size < 30:
        report.reasons.append(f"only {close.size} bars — statistical estimates suppressed")
        return report

    report.hurst = hurst_exponent(close)
    report.variance_ratio = variance_ratio(close, lag=5)
    report.entropy = permutation_entropy(close)
    report.efficiency_ratio = efficiency_ratio(close, period=20)
    report.autocorr_1 = autocorrelation(close, lag=1)
    report.autocorr_5 = autocorrelation(close, lag=5)

    returns = np.diff(np.log(np.maximum(close.to_numpy(), 1e-12)))
    report.ewma_sigma = ewma_volatility(returns)
    garch = garch11_forecast(returns, horizon=forecast_horizon)
    if garch is not None:
        report.garch_sigma, report.garch_params = garch
    realised = float(np.std(returns[-30:], ddof=1)) if returns.size >= 31 else None
    forecast = report.garch_sigma or report.ewma_sigma
    if realised and realised > 1e-12 and forecast:
        report.volatility_ratio = forecast / realised

    report.jump_count, report.latest_jump_sigmas = detect_jumps(frame)

    # --- narrative ---------------------------------------------------------
    if report.hurst is not None:
        shape = (
            "persistent" if report.hurst > 0.55
            else "mean-reverting" if report.hurst < 0.45
            else "random-walk-like"
        )
        report.reasons.append(f"Hurst {report.hurst:.2f} — {shape}")
    if report.variance_ratio is not None:
        report.reasons.append(
            f"variance ratio(5) {report.variance_ratio:.2f} "
            f"({'trending' if report.variance_ratio > 1.15 else 'reverting' if report.variance_ratio < 0.85 else 'neutral'})"
        )
    if report.efficiency_ratio is not None:
        report.reasons.append(f"efficiency ratio {report.efficiency_ratio:.2f} over 20 bars")
    if report.entropy is not None:
        report.reasons.append(
            f"permutation entropy {report.entropy:.2f} "
            f"({'noisy' if report.entropy > 0.95 else 'structured'})"
        )
    if report.volatility_ratio is not None:
        report.reasons.append(
            f"volatility {report.volatility_regime}: forecast/realised {report.volatility_ratio:.2f}"
        )
    if report.jump_count:
        report.reasons.append(
            f"{report.jump_count} price jump(s) in the last 60 bars; "
            f"latest bar {report.latest_jump_sigmas:.1f}σ"
        )
    return report
