"""MarketRegimeEngine (REQ 11).

Output is a *probability distribution* over regimes, not a label. That matters
because REQ 11 requires the system to refuse to trade when regime uncertainty is
high — which is only expressible if uncertainty is quantified.

Method: each regime gets an evidence score built from measured, bounded features.
Scores are turned into probabilities with a tempered softmax. The temperature is
what controls how decisively the engine commits: weak, conflicting evidence
produces a flat distribution, which raises UNCERTAIN and blocks trading.

A statistical/ML regime classifier can be swapped in behind the same interface
(`RegimeClassifier`), which is why `detect()` returns a `RegimeState` rather than
exposing its internals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Regime, Timeframe

logger = get_logger(__name__)


@dataclass(frozen=True)
class RegimeState:
    """Probabilistic regime assessment at one instant."""

    probabilities: Mapping[Regime, float]
    dominant: Regime
    confidence: float
    entropy: float
    timestamp: datetime | None = None
    evidence: Mapping[str, float] = field(default_factory=dict)

    def probability(self, regime: Regime) -> float:
        return float(self.probabilities.get(regime, 0.0))

    @property
    def is_uncertain(self) -> bool:
        return self.dominant is Regime.UNCERTAIN

    @property
    def directional_bias(self) -> float:
        """Net directional lean in [-1, 1] implied by the distribution."""
        return self.probability(Regime.TRENDING_UP) - self.probability(Regime.TRENDING_DOWN)

    def top(self, n: int = 3) -> list[tuple[Regime, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def describe(self) -> str:
        return ", ".join(f"{r.value} {p:.2f}" for r, p in self.top(4))


class RegimeClassifier(Protocol):
    """Pluggable back-end so an HMM/clustering model can replace the rule engine."""

    def score(self, features: pd.Series) -> Mapping[Regime, float]: ...


@dataclass
class RegimeThresholds:
    """All thresholds are configuration. They are starting points to be validated
    empirically (REQ 9), not claims about the market."""

    adx_trending: float = 22.0
    adx_strong_trend: float = 32.0
    adx_ranging: float = 18.0
    slope_significant: float = 0.0004
    vol_percentile_high: float = 0.75
    vol_percentile_low: float = 0.25
    vol_expansion_ratio: float = 1.25
    vol_compression_ratio: float = 0.80
    breakout_range_position: float = 0.92
    breakout_volume_ratio: float = 1.5
    range_position_band: float = 0.25
    expiry_days_threshold: int = 1
    # Softmax temperature: lower = more decisive. Tuned so that a single weak
    # signal cannot by itself produce a high-confidence regime.
    temperature: float = 0.55
    min_confidence: float = 0.40
    max_entropy_to_trade: float = 0.80


class MarketRegimeEngine:
    """Rule-and-evidence regime detector with a probabilistic output."""

    ALL_REGIMES: tuple[Regime, ...] = (
        Regime.TRENDING_UP,
        Regime.TRENDING_DOWN,
        Regime.RANGE,
        Regime.BREAKOUT,
        Regime.HIGH_VOLATILITY,
        Regime.LOW_VOLATILITY,
        Regime.VOLATILITY_EXPANSION,
        Regime.VOLATILITY_COMPRESSION,
        Regime.EXPIRY_DRIVEN,
        Regime.EVENT_DRIVEN,
    )

    def __init__(
        self,
        thresholds: RegimeThresholds | None = None,
        *,
        classifier: RegimeClassifier | None = None,
    ) -> None:
        self.thresholds = thresholds or RegimeThresholds()
        self.classifier = classifier
        self._history: list[RegimeState] = []

    # ------------------------------------------------------------------ #
    def detect(
        self,
        features: pd.Series,
        *,
        timestamp: datetime | None = None,
        days_to_expiry: int | None = None,
        event_active: bool = False,
        prefix: str = "",
    ) -> RegimeState:
        """Assess the regime from one row of computed features."""
        get = lambda name, default=np.nan: _value(features, f"{prefix}{name}", default)  # noqa: E731

        scores: dict[Regime, float] = {r: 0.0 for r in self.ALL_REGIMES}
        evidence: dict[str, float] = {}
        t = self.thresholds

        adx = get("adx")
        di_spread = get("di_spread")
        slope20 = get("trend_slope_20")
        slope50 = get("trend_slope_50")
        ema_spread = get("ema_spread_atr")
        vol_pct = get("vol_percentile")
        vol_exp = get("vol_expansion")
        bb_pct = get("bb_width_percentile")
        squeeze = get("is_squeeze", 0.0)
        range_pos = get("range_position_20")
        rel_volume = get("rel_volume_20", 1.0)
        atr_pct = get("atr_pct")

        # --- trend ---------------------------------------------------------
        if not math.isnan(adx):
            evidence["adx"] = adx
            trend_strength = _ramp(adx, t.adx_ranging, t.adx_strong_trend)
            direction = 0.0
            if not math.isnan(di_spread):
                direction += np.tanh(di_spread / 15.0)
            if not math.isnan(slope20):
                direction += np.tanh(slope20 / max(t.slope_significant, 1e-9))
            if not math.isnan(ema_spread):
                direction += np.tanh(ema_spread / 1.5)
            direction /= 3.0
            evidence["trend_direction"] = float(direction)

            if direction > 0:
                scores[Regime.TRENDING_UP] += 2.2 * trend_strength * direction
                scores[Regime.TRENDING_DOWN] -= 0.8 * trend_strength * direction
            else:
                scores[Regime.TRENDING_DOWN] += 2.2 * trend_strength * abs(direction)
                scores[Regime.TRENDING_UP] -= 0.8 * trend_strength * abs(direction)

            # Longer-horizon agreement reinforces; disagreement is a range tell.
            if not math.isnan(slope50) and not math.isnan(slope20):
                if np.sign(slope50) == np.sign(slope20) and abs(slope50) > t.slope_significant:
                    target = Regime.TRENDING_UP if slope50 > 0 else Regime.TRENDING_DOWN
                    scores[target] += 0.6
                elif np.sign(slope50) != np.sign(slope20):
                    scores[Regime.RANGE] += 0.5

            # --- range -----------------------------------------------------
            range_strength = 1.0 - _ramp(adx, t.adx_ranging, t.adx_trending)
            scores[Regime.RANGE] += 1.8 * range_strength
            if not math.isnan(range_pos):
                # Price parked mid-range is the clearest range evidence.
                centrality = 1.0 - abs(range_pos - 0.5) * 2.0
                scores[Regime.RANGE] += 1.0 * max(0.0, centrality) * range_strength
                evidence["range_position"] = float(range_pos)

        # --- breakout --------------------------------------------------------
        if not math.isnan(range_pos):
            at_edge = range_pos >= t.breakout_range_position or range_pos <= (1 - t.breakout_range_position)
            if at_edge:
                breakout_score = 1.4
                if not math.isnan(rel_volume) and rel_volume >= t.breakout_volume_ratio:
                    # Volume confirmation is what separates a breakout from a poke.
                    breakout_score += 1.0 * min(2.0, rel_volume / t.breakout_volume_ratio)
                if not math.isnan(vol_exp) and vol_exp >= t.vol_expansion_ratio:
                    breakout_score += 0.8
                scores[Regime.BREAKOUT] += breakout_score
                scores[Regime.RANGE] -= 0.8
                evidence["breakout_edge"] = float(range_pos)

        bos_up, bos_down = get("bos_up", 0.0), get("bos_down", 0.0)
        if bos_up > 0 or bos_down > 0:
            scores[Regime.BREAKOUT] += 0.9
            scores[Regime.TRENDING_UP if bos_up > 0 else Regime.TRENDING_DOWN] += 0.5

        # --- volatility level -------------------------------------------------
        if not math.isnan(vol_pct):
            evidence["vol_percentile"] = float(vol_pct)
            if vol_pct >= t.vol_percentile_high:
                scores[Regime.HIGH_VOLATILITY] += 2.0 * _ramp(vol_pct, t.vol_percentile_high, 1.0)
            if vol_pct <= t.vol_percentile_low:
                scores[Regime.LOW_VOLATILITY] += 2.0 * (1.0 - _ramp(vol_pct, 0.0, t.vol_percentile_low))

        # --- volatility dynamics ---------------------------------------------
        if not math.isnan(vol_exp):
            evidence["vol_expansion"] = float(vol_exp)
            if vol_exp >= t.vol_expansion_ratio:
                scores[Regime.VOLATILITY_EXPANSION] += 2.0 * min(2.0, vol_exp / t.vol_expansion_ratio)
            elif vol_exp <= t.vol_compression_ratio:
                scores[Regime.VOLATILITY_COMPRESSION] += 2.0 * (t.vol_compression_ratio / max(vol_exp, 0.1))

        if squeeze > 0.5:
            scores[Regime.VOLATILITY_COMPRESSION] += 1.2
            scores[Regime.LOW_VOLATILITY] += 0.5
        if not math.isnan(bb_pct):
            if bb_pct <= 0.15:
                scores[Regime.VOLATILITY_COMPRESSION] += 0.8
            elif bb_pct >= 0.85:
                scores[Regime.VOLATILITY_EXPANSION] += 0.8

        # --- expiry ------------------------------------------------------------
        if days_to_expiry is not None:
            evidence["days_to_expiry"] = float(days_to_expiry)
            if days_to_expiry <= t.expiry_days_threshold:
                # Expiry doesn't replace the price regime; it overlays gamma/theta
                # dynamics on top of it, so it is additive rather than exclusive.
                scores[Regime.EXPIRY_DRIVEN] += 2.5 - 0.8 * days_to_expiry

        # --- event -------------------------------------------------------------
        if event_active:
            scores[Regime.EVENT_DRIVEN] += 2.5
            evidence["event_active"] = 1.0

        if self.classifier is not None:
            for regime, value in self.classifier.score(features).items():
                scores[regime] = scores.get(regime, 0.0) + float(value)

        probabilities = self._to_probabilities(scores)
        state = self._finalize(probabilities, evidence, timestamp)
        self._history.append(state)
        if len(self._history) > 500:
            del self._history[:100]
        return state

    # ------------------------------------------------------------------ #
    def _to_probabilities(self, scores: Mapping[Regime, float]) -> dict[Regime, float]:
        """Tempered softmax over evidence scores.

        Scores are clipped before exponentiating so one runaway feature cannot
        saturate the distribution to a false certainty.
        """
        regimes = list(scores.keys())
        raw = np.array([max(0.0, scores[r]) for r in regimes], dtype=float)
        if raw.sum() <= 0:
            # No evidence at all: everything is UNCERTAIN by construction.
            return {r: 0.0 for r in regimes} | {Regime.UNCERTAIN: 1.0}

        clipped = np.clip(raw, 0.0, 6.0)
        exponent = np.exp(clipped / max(self.thresholds.temperature, 1e-3))
        probabilities = exponent / exponent.sum()
        return {regime: float(p) for regime, p in zip(regimes, probabilities)}

    def _finalize(
        self,
        probabilities: dict[Regime, float],
        evidence: dict[str, float],
        timestamp: datetime | None,
    ) -> RegimeState:
        if Regime.UNCERTAIN in probabilities and probabilities[Regime.UNCERTAIN] >= 1.0:
            return RegimeState(
                probabilities={Regime.UNCERTAIN: 1.0},
                dominant=Regime.UNCERTAIN,
                confidence=0.0,
                entropy=1.0,
                timestamp=timestamp,
                evidence=evidence,
            )

        values = np.array(list(probabilities.values()))
        # Normalized Shannon entropy: 0 = one regime certain, 1 = uniform.
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum())
        entropy /= math.log(len(values)) if len(values) > 1 else 1.0

        dominant = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        confidence = probabilities[dominant]

        # REQ 11: refuse to name a regime we do not actually have evidence for.
        if confidence < self.thresholds.min_confidence or entropy > self.thresholds.max_entropy_to_trade:
            probabilities = dict(probabilities)
            probabilities[Regime.UNCERTAIN] = max(
                probabilities.get(Regime.UNCERTAIN, 0.0), 1.0 - confidence
            )
            total = sum(probabilities.values())
            probabilities = {r: p / total for r, p in probabilities.items()}
            dominant = Regime.UNCERTAIN
            confidence = probabilities[Regime.UNCERTAIN]

        return RegimeState(
            probabilities=probabilities,
            dominant=dominant,
            confidence=confidence,
            entropy=entropy,
            timestamp=timestamp,
            evidence=evidence,
        )

    # ------------------------------------------------------------------ #
    def tradable(self, state: RegimeState) -> tuple[bool, str]:
        """REQ 11: do not trade when regime uncertainty is too high."""
        if state.dominant is Regime.UNCERTAIN:
            return False, f"regime uncertain (confidence {state.confidence:.2f})"
        if state.entropy > self.thresholds.max_entropy_to_trade:
            return False, f"regime entropy {state.entropy:.2f} above {self.thresholds.max_entropy_to_trade}"
        if state.confidence < self.thresholds.min_confidence:
            return False, f"regime confidence {state.confidence:.2f} below {self.thresholds.min_confidence}"
        return True, ""

    def stability(self, lookback: int = 10) -> float:
        """Fraction of recent assessments agreeing with the latest dominant regime.

        A regime that flips every bar is not a regime; strategies keyed to it should
        be discounted, which the ensemble does using this value.
        """
        if len(self._history) < 2:
            return 1.0
        recent = self._history[-lookback:]
        current = recent[-1].dominant
        return sum(1 for s in recent if s.dominant is current) / len(recent)

    def changed(self, since: RegimeState | None) -> bool:
        """REQ 35: a regime change is an exit reason."""
        if since is None or not self._history:
            return False
        return self._history[-1].dominant is not since.dominant

    @property
    def current(self) -> RegimeState | None:
        return self._history[-1] if self._history else None


# --------------------------------------------------------------------------- #
def _value(features: pd.Series, name: str, default: float = np.nan) -> float:
    try:
        value = features.get(name, default)
    except (AttributeError, TypeError):
        return default
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0->1 ramp between two thresholds, clipped outside."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Regime/strategy compatibility (REQ 12/13)
# --------------------------------------------------------------------------- #
REGIME_STRATEGY_FIT: dict[str, dict[Regime, float]] = {
    "trend_following": {
        Regime.TRENDING_UP: 1.0, Regime.TRENDING_DOWN: 1.0, Regime.BREAKOUT: 0.7,
        Regime.RANGE: 0.1, Regime.HIGH_VOLATILITY: 0.5, Regime.LOW_VOLATILITY: 0.4,
        Regime.VOLATILITY_EXPANSION: 0.7, Regime.VOLATILITY_COMPRESSION: 0.3,
        Regime.EXPIRY_DRIVEN: 0.4, Regime.EVENT_DRIVEN: 0.2, Regime.UNCERTAIN: 0.0,
    },
    "momentum": {
        Regime.TRENDING_UP: 0.9, Regime.TRENDING_DOWN: 0.9, Regime.BREAKOUT: 1.0,
        Regime.RANGE: 0.2, Regime.HIGH_VOLATILITY: 0.7, Regime.LOW_VOLATILITY: 0.3,
        Regime.VOLATILITY_EXPANSION: 0.9, Regime.VOLATILITY_COMPRESSION: 0.2,
        Regime.EXPIRY_DRIVEN: 0.5, Regime.EVENT_DRIVEN: 0.3, Regime.UNCERTAIN: 0.0,
    },
    "breakout": {
        Regime.TRENDING_UP: 0.7, Regime.TRENDING_DOWN: 0.7, Regime.BREAKOUT: 1.0,
        Regime.RANGE: 0.3, Regime.HIGH_VOLATILITY: 0.6, Regime.LOW_VOLATILITY: 0.4,
        Regime.VOLATILITY_EXPANSION: 1.0, Regime.VOLATILITY_COMPRESSION: 0.6,
        Regime.EXPIRY_DRIVEN: 0.4, Regime.EVENT_DRIVEN: 0.3, Regime.UNCERTAIN: 0.0,
    },
    "mean_reversion": {
        # REQ 12.4: mean reversion must disable itself in strong trends.
        Regime.TRENDING_UP: 0.0, Regime.TRENDING_DOWN: 0.0, Regime.BREAKOUT: 0.0,
        Regime.RANGE: 1.0, Regime.HIGH_VOLATILITY: 0.3, Regime.LOW_VOLATILITY: 0.8,
        Regime.VOLATILITY_EXPANSION: 0.1, Regime.VOLATILITY_COMPRESSION: 0.7,
        Regime.EXPIRY_DRIVEN: 0.3, Regime.EVENT_DRIVEN: 0.1, Regime.UNCERTAIN: 0.0,
    },
    "volatility_expansion": {
        Regime.TRENDING_UP: 0.5, Regime.TRENDING_DOWN: 0.5, Regime.BREAKOUT: 0.9,
        Regime.RANGE: 0.3, Regime.HIGH_VOLATILITY: 0.6, Regime.LOW_VOLATILITY: 0.5,
        Regime.VOLATILITY_EXPANSION: 1.0, Regime.VOLATILITY_COMPRESSION: 0.9,
        Regime.EXPIRY_DRIVEN: 0.5, Regime.EVENT_DRIVEN: 0.4, Regime.UNCERTAIN: 0.0,
    },
    "market_structure": {
        Regime.TRENDING_UP: 0.9, Regime.TRENDING_DOWN: 0.9, Regime.BREAKOUT: 0.8,
        Regime.RANGE: 0.5, Regime.HIGH_VOLATILITY: 0.5, Regime.LOW_VOLATILITY: 0.5,
        Regime.VOLATILITY_EXPANSION: 0.6, Regime.VOLATILITY_COMPRESSION: 0.5,
        Regime.EXPIRY_DRIVEN: 0.4, Regime.EVENT_DRIVEN: 0.2, Regime.UNCERTAIN: 0.0,
    },
    "vwap": {
        Regime.TRENDING_UP: 0.8, Regime.TRENDING_DOWN: 0.8, Regime.BREAKOUT: 0.6,
        Regime.RANGE: 0.8, Regime.HIGH_VOLATILITY: 0.5, Regime.LOW_VOLATILITY: 0.6,
        Regime.VOLATILITY_EXPANSION: 0.5, Regime.VOLATILITY_COMPRESSION: 0.6,
        Regime.EXPIRY_DRIVEN: 0.5, Regime.EVENT_DRIVEN: 0.2, Regime.UNCERTAIN: 0.0,
    },
    "options_flow": {
        Regime.TRENDING_UP: 0.7, Regime.TRENDING_DOWN: 0.7, Regime.BREAKOUT: 0.6,
        Regime.RANGE: 0.7, Regime.HIGH_VOLATILITY: 0.6, Regime.LOW_VOLATILITY: 0.5,
        Regime.VOLATILITY_EXPANSION: 0.6, Regime.VOLATILITY_COMPRESSION: 0.6,
        Regime.EXPIRY_DRIVEN: 0.9, Regime.EVENT_DRIVEN: 0.3, Regime.UNCERTAIN: 0.0,
    },
    "expiry": {
        Regime.TRENDING_UP: 0.3, Regime.TRENDING_DOWN: 0.3, Regime.BREAKOUT: 0.4,
        Regime.RANGE: 0.4, Regime.HIGH_VOLATILITY: 0.4, Regime.LOW_VOLATILITY: 0.4,
        Regime.VOLATILITY_EXPANSION: 0.5, Regime.VOLATILITY_COMPRESSION: 0.4,
        Regime.EXPIRY_DRIVEN: 1.0, Regime.EVENT_DRIVEN: 0.2, Regime.UNCERTAIN: 0.0,
    },
}


def regime_fit(strategy_name: str, state: RegimeState) -> float:
    """Expected compatibility of a strategy with the current regime distribution.

    Weighting the fit by the regime *probabilities* (rather than by the single
    dominant label) means a strategy is not fully switched off by a marginal
    regime call.
    """
    table = REGIME_STRATEGY_FIT.get(strategy_name)
    if table is None:
        return 0.5
    return float(sum(table.get(regime, 0.3) * p for regime, p in state.probabilities.items()))
