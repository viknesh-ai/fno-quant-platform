"""AnalysisEngine — one deep read of one underlying.

Given everything already collected for a symbol in a cycle (frames, features,
quote, option chains, regime), this produces a single `AnalysisReport`: the
sub-reports from every analysis module, translated into the eight independent
dimensions the confluence engine combines, plus the vetoes that mean *do not
trade regardless of score*.

The engine is stateful only in the ways it must be — the microstructure tracker
needs quote history and the cross-asset tracker needs a return matrix. Both are
bounded and both are owned here so the orchestrator does not have to thread them
through the pipeline by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Direction, OptionChain, Quote, Timeframe
from ..regime.engine import RegimeState
from .confluence import ConfluenceEngine, ConfluenceReport, Dimension
from .crossasset import CrossAssetReport, CrossAssetTracker
from .microstructure import MicrostructureReport, MicrostructureTracker
from .options_analytics import OptionsAnalyticsReport, analyze_chain
from .statistics import StatisticalReport, analyze_series
from .volume_profile import VolumeProfile, build_volume_profile

logger = get_logger(__name__)


@dataclass
class AnalysisReport:
    """Everything the analysis layer knows about one underlying, right now."""

    underlying: str
    timestamp: datetime
    price: float = 0.0
    statistical: StatisticalReport = field(default_factory=StatisticalReport)
    volume_profile: VolumeProfile = field(default_factory=VolumeProfile)
    microstructure: MicrostructureReport = field(default_factory=MicrostructureReport)
    options: OptionsAnalyticsReport | None = None
    crossasset: CrossAssetReport | None = None
    confluence: ConfluenceReport = field(default_factory=ConfluenceReport)
    timeframe_alignment: float = 0.0     # signed [-1, 1] across the TF hierarchy
    duration_ms: float = 0.0

    # ---------------------------------------------------------------- #
    @property
    def bias(self) -> Direction:
        return self.confluence.bias

    @property
    def conviction(self) -> float:
        return self.confluence.conviction

    @property
    def vetoed(self) -> bool:
        return self.confluence.vetoed

    @property
    def vetoes(self) -> list[str]:
        return self.confluence.vetoes

    def supports(self, direction: Direction) -> float:
        """[0, 1] — how strongly the analysis backs this direction."""
        return self.confluence.supports(direction)

    def feature_dict(self) -> dict[str, float]:
        """Flat numeric view for the ML feature vector and the journal."""
        out: dict[str, float] = {"analysis_tf_alignment": self.timeframe_alignment}
        out.update(self.statistical.to_dict())
        out.update(self.volume_profile.to_dict())
        out.update(self.microstructure.to_dict())
        if self.options is not None:
            out.update(self.options.to_dict())
        if self.crossasset is not None:
            out.update(self.crossasset.to_dict())
        out.update(self.confluence.to_dict())
        return {k: float(v) for k, v in out.items() if v is not None and math.isfinite(float(v))}

    def render(self) -> str:
        """The full human-readable analysis, section by section."""
        lines = [
            f"═══ ANALYSIS {self.underlying} @ {self.price:,.2f} "
            f"({self.timestamp:%Y-%m-%d %H:%M:%S}) ═══",
            "",
            "CONFLUENCE",
        ]
        lines.extend(f"  {line}" for line in self.confluence.explain())
        lines.append("")
        lines.append(f"MULTI-TIMEFRAME  alignment {self.timeframe_alignment:+.2f}")
        for title, reasons in (
            ("STATISTICAL", self.statistical.explain()),
            ("VOLUME PROFILE", self.volume_profile.explain()),
            ("MICROSTRUCTURE", self.microstructure.explain()),
            ("OPTIONS POSITIONING", self.options.explain() if self.options else []),
            ("CROSS-ASSET", self.crossasset.explain() if self.crossasset else []),
        ):
            if not reasons:
                continue
            lines.append("")
            lines.append(title)
            lines.extend(f"  - {reason}" for reason in reasons)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
class AnalysisEngine:
    """Runs every analysis module and combines the results."""

    def __init__(
        self,
        config=None,
        *,
        confluence: ConfluenceEngine | None = None,
        microstructure: MicrostructureTracker | None = None,
        crossasset: CrossAssetTracker | None = None,
    ) -> None:
        self.config = config
        weights = getattr(config, "dimension_weights", None) if config else None
        self.confluence = confluence or ConfluenceEngine(
            weights=weights,
            min_dimensions=getattr(config, "min_dimensions", 4) if config else 4,
            conflict_penalty=getattr(config, "conflict_penalty", 0.5) if config else 0.5,
        )
        self.microstructure = microstructure or MicrostructureTracker(
            history=getattr(config, "microstructure_history", 120) if config else 120
        )
        self.crossasset = crossasset or CrossAssetTracker(
            benchmark=getattr(config, "benchmark", "NIFTY") if config else "NIFTY"
        )
        self._last: dict[str, AnalysisReport] = {}

    # ------------------------------------------------------------------ #
    def last(self, underlying: str) -> AnalysisReport | None:
        return self._last.get(underlying)

    def observe_market(self, underlying: str, price: float) -> None:
        """Feed the cross-asset return matrix. Called for every scanned symbol."""
        self.crossasset.observe(underlying, price)

    def observe_vix(self, value: float) -> None:
        self.crossasset.observe_vix(value)

    def build_market_context(
        self,
        *,
        timestamp: datetime,
        session_returns: Mapping[str, float] | None = None,
        above_vwap: Mapping[str, bool] | None = None,
    ) -> CrossAssetReport:
        """Build the market-wide report once per cycle, before scoring symbols."""
        return self.crossasset.build(
            timestamp=timestamp, session_returns=session_returns, above_vwap=above_vwap
        )

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        *,
        underlying: str,
        timestamp: datetime,
        quote: Quote,
        frames: Mapping[Timeframe, pd.DataFrame],
        features_by_timeframe: Mapping[Timeframe, pd.Series],
        entry_timeframe: Timeframe,
        setup_timeframe: Timeframe,
        regime_timeframe: Timeframe,
        regime: RegimeState | None = None,
        chains: list[OptionChain] | None = None,
        lot_size: int = 1,
        iv_rank: float | None = None,
        previous_atm_iv: float | None = None,
        test_quantity: int = 0,
        quote_age_seconds: float | None = None,
    ) -> AnalysisReport:
        """Produce the full analysis for one underlying."""
        import time as _time

        started = _time.perf_counter()
        report = AnalysisReport(
            underlying=underlying,
            timestamp=timestamp,
            price=float(quote.last_price or 0.0),
        )

        entry_frame = frames.get(entry_timeframe, pd.DataFrame())
        entry_features = features_by_timeframe.get(entry_timeframe)

        # --- sub-reports ---------------------------------------------------
        report.statistical = analyze_series(entry_frame)
        report.volume_profile = build_volume_profile(
            entry_frame,
            bins=getattr(self.config, "volume_profile_bins", 60) if self.config else 60,
            value_area_pct=getattr(self.config, "value_area_pct", 0.70) if self.config else 0.70,
            price=report.price,
        )
        report.microstructure = self.microstructure.analyze(
            quote, frame=entry_frame, test_quantity=test_quantity
        )
        if chains:
            try:
                report.options = analyze_chain(
                    chains[0],
                    now=timestamp,
                    lot_size=lot_size,
                    iv_rank=iv_rank,
                    next_expiry_chain=chains[1] if len(chains) > 1 else None,
                    previous_atm_iv=previous_atm_iv,
                )
            except Exception as exc:      # analysis must never break a cycle
                logger.warning("option analytics failed for %s: %s", underlying, exc)
        report.crossasset = self.crossasset.last_report

        # --- reference drift ------------------------------------------------
        # Non-directional dimensions (statistics, volatility) are scored as
        # "does this support the move already in progress", so they need a
        # reference direction that is not derived from them.
        drift = self._reference_drift(entry_features, entry_frame)
        report.timeframe_alignment = self._timeframe_alignment(
            features_by_timeframe, entry_timeframe, setup_timeframe, regime_timeframe
        )

        dimensions = self._dimensions(
            report=report,
            entry_features=entry_features,
            features_by_timeframe=features_by_timeframe,
            entry_timeframe=entry_timeframe,
            drift=drift,
            underlying=underlying,
        )
        vetoes, warnings = self._vetoes(report, regime=regime, quote_age_seconds=quote_age_seconds)

        report.confluence = self.confluence.combine(
            dimensions,
            underlying=underlying,
            timestamp=timestamp,
            vetoes=vetoes,
            warnings=warnings,
        )
        report.duration_ms = (_time.perf_counter() - started) * 1000.0
        self._last[underlying] = report
        return report

    # ================================================================== #
    # Dimensions
    # ================================================================== #
    def _dimensions(
        self,
        *,
        report: AnalysisReport,
        entry_features: pd.Series | None,
        features_by_timeframe: Mapping[Timeframe, pd.Series],
        entry_timeframe: Timeframe,
        drift: float,
        underlying: str,
    ) -> list[Dimension]:
        dimensions: list[Dimension] = [
            self._trend_dimension(entry_features, report.timeframe_alignment),
            self._momentum_dimension(entry_features),
            self._structure_dimension(report.volume_profile, entry_features),
            self._orderflow_dimension(report.microstructure),
            self._positioning_dimension(report.options),
            self._statistical_dimension(report.statistical, drift),
            self._volatility_dimension(report.statistical, entry_features, drift),
            self._crossasset_dimension(report.crossasset, underlying),
        ]
        return dimensions

    @staticmethod
    def _feature(features: pd.Series | None, name: str, default: float = float("nan")) -> float:
        if features is None or name not in features.index:
            return default
        try:
            value = float(features[name])
        except (TypeError, ValueError):
            return default
        return default if not math.isfinite(value) else value

    # ------------------------------------------------------------------ #
    def _trend_dimension(self, features: pd.Series | None, alignment: float) -> Dimension:
        reasons: list[str] = []
        votes: list[float] = []

        di_spread = self._feature(features, "di_spread")
        adx = self._feature(features, "adx")
        if math.isfinite(di_spread) and math.isfinite(adx):
            # ADX gates the strength; DI spread carries the direction.
            strength = float(np.clip((adx - 15.0) / 25.0, 0.0, 1.0))
            votes.append(float(np.clip(di_spread / 20.0, -1.0, 1.0)) * strength)
            reasons.append(f"ADX {adx:.1f} with DI spread {di_spread:+.1f}")

        ema_spread = self._feature(features, "ema_spread_atr")
        if math.isfinite(ema_spread):
            votes.append(float(np.clip(ema_spread / 1.5, -1.0, 1.0)))
            reasons.append(f"fast/slow EMA spread {ema_spread:+.2f} ATR")

        slope = self._feature(features, "trend_slope_20")
        if math.isfinite(slope):
            votes.append(float(np.clip(slope * 50.0, -1.0, 1.0)))

        vwap_distance = self._feature(features, "vwap_dist_atr")
        if math.isfinite(vwap_distance):
            votes.append(float(np.clip(vwap_distance / 2.0, -1.0, 1.0)))
            reasons.append(f"{vwap_distance:+.2f} ATR from session VWAP")

        if abs(alignment) > 0.05:
            votes.append(alignment)
            reasons.append(f"multi-timeframe alignment {alignment:+.2f}")

        score = float(np.mean(votes)) if votes else 0.0
        confidence = float(np.clip(len(votes) / 4.0, 0.0, 1.0))
        return Dimension("trend", score, confidence, reasons=reasons)

    def _momentum_dimension(self, features: pd.Series | None) -> Dimension:
        reasons: list[str] = []
        votes: list[float] = []

        rsi = self._feature(features, "rsi_centered")
        if math.isfinite(rsi):
            votes.append(float(np.clip(rsi / 25.0, -1.0, 1.0)))
            reasons.append(f"RSI {rsi + 50:.1f}")

        macd_hist = self._feature(features, "macd_hist")
        if math.isfinite(macd_hist):
            votes.append(float(np.clip(macd_hist / 0.5, -1.0, 1.0)))
            reasons.append(f"MACD histogram {macd_hist:+.2f} ATR")

        roc = self._feature(features, "roc_10")
        if math.isfinite(roc):
            votes.append(float(np.clip(roc / 0.01, -1.0, 1.0)))

        acceleration = self._feature(features, "acceleration_5")
        if math.isfinite(acceleration):
            votes.append(float(np.clip(acceleration / 0.004, -1.0, 1.0)))
            reasons.append(f"price acceleration {acceleration:+.4f}")

        obv_slope = self._feature(features, "obv_slope")
        if math.isfinite(obv_slope) and abs(obv_slope) > 0:
            votes.append(float(np.clip(obv_slope, -1.0, 1.0)))

        score = float(np.mean(votes)) if votes else 0.0
        confidence = float(np.clip(len(votes) / 4.0, 0.0, 1.0))
        return Dimension("momentum", score, confidence, reasons=reasons)

    def _structure_dimension(
        self, profile: VolumeProfile, features: pd.Series | None
    ) -> Dimension:
        reasons: list[str] = []
        votes: list[float] = []

        if profile.poc is not None:
            votes.append(profile.structure_score)
            reasons.append(
                f"price {profile.location.replace('_', ' ')} "
                f"(POC {profile.poc:,.2f})"
            )

        bos_up = self._feature(features, "bos_up", 0.0)
        bos_down = self._feature(features, "bos_down", 0.0)
        if bos_up > 0 or bos_down > 0:
            votes.append(1.0 if bos_up > 0 else -1.0)
            reasons.append("break of structure " + ("up" if bos_up > 0 else "down"))

        range_position = self._feature(features, "range_position_20")
        if math.isfinite(range_position):
            votes.append(float(np.clip((range_position - 0.5) * 2.0, -1.0, 1.0)))
            reasons.append(f"{range_position:.0%} of the 20-bar range")

        higher_high = self._feature(features, "higher_high", 0.0)
        lower_low = self._feature(features, "lower_low", 0.0)
        if higher_high > 0 or lower_low > 0:
            votes.append(0.6 * (higher_high - lower_low))

        score = float(np.mean(votes)) if votes else 0.0
        confidence = float(np.clip(len(votes) / 3.0, 0.0, 1.0))
        return Dimension("structure", score, confidence, reasons=reasons)

    def _orderflow_dimension(self, micro: MicrostructureReport) -> Dimension:
        reasons = [r for r in micro.reasons if "spread" not in r.lower()][:4]
        score = micro.pressure
        # Confidence rises with how many quote updates backed the sequential
        # measurements; a single snapshot is a weak read of flow.
        confidence = float(np.clip(micro.quote_updates / 20.0, 0.15, 1.0))
        if micro.quote_updates == 0:
            confidence = 0.15
        return Dimension("orderflow", score, confidence, reasons=reasons)

    def _positioning_dimension(self, options: OptionsAnalyticsReport | None) -> Dimension:
        if options is None:
            return Dimension("positioning", 0.0, 0.0, reasons=["no option chain available"])
        reasons = options.reasons[:5]
        score = options.positioning_bias
        confidence = float(np.clip(options.strikes_analyzed / 15.0, 0.0, 1.0))
        return Dimension("positioning", score, confidence, reasons=reasons)

    def _statistical_dimension(self, stats: StatisticalReport, drift: float) -> Dimension:
        reasons = stats.reasons[:4]
        if stats.bars_analyzed < 30 or drift == 0.0:
            return Dimension("statistical", 0.0, 0.1, reasons=reasons)
        # Persistence says the drift continues; anti-persistence says it reverses.
        score = stats.persistence_score * float(np.sign(drift))
        confidence = float(np.clip(stats.bars_analyzed / 120.0, 0.2, 1.0)) * stats.tradability
        return Dimension("statistical", score, confidence, reasons=reasons)

    def _volatility_dimension(
        self, stats: StatisticalReport, features: pd.Series | None, drift: float
    ) -> Dimension:
        reasons: list[str] = []
        expansion = self._feature(features, "vol_expansion")
        squeeze = self._feature(features, "is_squeeze", 0.0)

        magnitude = 0.0
        if math.isfinite(expansion):
            # Expansion above 1.0 means short-term ATR is outrunning long-term:
            # directional moves carry further.
            magnitude = float(np.clip((expansion - 1.0) / 0.5, -1.0, 1.0))
            reasons.append(f"volatility expansion ratio {expansion:.2f}")
        if stats.volatility_regime != "unknown":
            reasons.append(f"volatility {stats.volatility_regime}")
            if stats.volatility_regime == "expanding":
                magnitude = max(magnitude, 0.4)
            elif stats.volatility_regime == "contracting":
                magnitude = min(magnitude, 0.0)
        if squeeze > 0:
            reasons.append("in a Bollinger/Keltner squeeze — compression before expansion")

        score = magnitude * float(np.sign(drift)) if drift else 0.0
        confidence = 0.6 if reasons else 0.0
        return Dimension("volatility", score, confidence, reasons=reasons)

    def _crossasset_dimension(
        self, crossasset: CrossAssetReport | None, underlying: str
    ) -> Dimension:
        if crossasset is None or crossasset.symbols_tracked < 3:
            return Dimension(
                "crossasset", 0.0, 0.0,
                reasons=["not enough symbols tracked yet for cross-asset context"],
            )
        score = crossasset.alignment_for(underlying, 1.0)
        confidence = float(np.clip(crossasset.symbols_tracked / 10.0, 0.2, 1.0))
        return Dimension("crossasset", score, confidence, reasons=crossasset.reasons[:4])

    # ================================================================== #
    # Reference drift and timeframe alignment
    # ================================================================== #
    def _reference_drift(self, features: pd.Series | None, frame: pd.DataFrame) -> float:
        """The direction of the move currently in progress, in ATR units."""
        distance = self._feature(features, "dist_ema_20_atr")
        if math.isfinite(distance) and abs(distance) > 0.05:
            return distance
        spread = self._feature(features, "ema_spread_atr")
        if math.isfinite(spread):
            return spread
        if frame is not None and len(frame) >= 6 and "close" in frame:
            close = frame["close"].astype(float)
            change = float(close.iloc[-1] - close.iloc[-5])
            return change
        return 0.0

    def _timeframe_alignment(
        self,
        features_by_timeframe: Mapping[Timeframe, pd.Series],
        entry: Timeframe,
        setup: Timeframe,
        regime: Timeframe,
    ) -> float:
        """Signed [-1, 1] agreement across the timeframe hierarchy.

        Weighted toward the higher timeframes: an entry-timeframe push against a
        hostile regime timeframe is a fade, not a trend trade, and the weighting
        is what makes the difference visible in one number.
        """
        weights = {entry: 0.2, setup: 0.35, regime: 0.45}
        votes: list[tuple[float, float]] = []
        for timeframe, weight in weights.items():
            features = features_by_timeframe.get(timeframe)
            if features is None:
                continue
            reads: list[float] = []
            spread = self._feature(features, "ema_spread_atr")
            if math.isfinite(spread):
                reads.append(float(np.clip(spread / 1.5, -1.0, 1.0)))
            di_spread = self._feature(features, "di_spread")
            if math.isfinite(di_spread):
                reads.append(float(np.clip(di_spread / 20.0, -1.0, 1.0)))
            distance = self._feature(features, "dist_ema_20_atr")
            if math.isfinite(distance):
                reads.append(float(np.clip(distance / 2.0, -1.0, 1.0)))
            if reads:
                votes.append((float(np.mean(reads)), weight))
        if not votes:
            return 0.0
        total_weight = sum(weight for _, weight in votes)
        return float(
            np.clip(sum(value * weight for value, weight in votes) / total_weight, -1.0, 1.0)
        )

    # ================================================================== #
    # Vetoes
    # ================================================================== #
    def _vetoes(
        self,
        report: AnalysisReport,
        *,
        regime: RegimeState | None,
        quote_age_seconds: float | None,
    ) -> tuple[list[str], list[str]]:
        """Conditions that mean *do not trade*, and softer ones that only warn."""
        vetoes: list[str] = []
        warnings: list[str] = []
        config = self.config

        micro = report.microstructure
        max_spread = getattr(config, "max_spread_pct", 0.02) if config else 0.02
        if micro.spread_pct is not None and micro.spread_pct > max_spread:
            vetoes.append(
                f"spread {micro.spread_pct:.2%} exceeds the {max_spread:.2%} analysis limit"
            )
        if micro.is_toxic:
            vetoes.append(
                f"toxic order flow (toxicity {micro.flow_toxicity:.2f}, "
                f"absorption {micro.absorption:.2f})"
            )

        stats = report.statistical
        jump_limit = getattr(config, "max_jump_sigmas", 5.0) if config else 5.0
        if stats.latest_jump_sigmas > jump_limit:
            vetoes.append(
                f"the last bar moved {stats.latest_jump_sigmas:.1f}σ — features describe a "
                "market that no longer exists"
            )
        elif stats.jump_count >= 3:
            warnings.append(f"{stats.jump_count} jumps in the last 60 bars — unstable tape")

        min_tradability = getattr(config, "min_tradability", 0.15) if config else 0.15
        if stats.bars_analyzed >= 60 and stats.tradability < min_tradability:
            vetoes.append(
                f"series has no exploitable structure (tradability {stats.tradability:.2f}, "
                f"entropy {stats.entropy:.2f})" if stats.entropy is not None else
                f"series has no exploitable structure (tradability {stats.tradability:.2f})"
            )

        max_age = getattr(config, "max_quote_age_seconds", 20.0) if config else 20.0
        if quote_age_seconds is not None and quote_age_seconds > max_age:
            vetoes.append(f"quote is {quote_age_seconds:.0f}s old (limit {max_age:.0f}s)")

        options = report.options
        if options is not None:
            max_pin = getattr(config, "max_pin_risk", 0.75) if config else 0.75
            if options.pin_risk > max_pin:
                vetoes.append(
                    f"pin risk {options.pin_risk:.2f} at max pain {options.max_pain:,.0f} — "
                    "directional moves get pulled back into the pin"
                )
            if options.suppresses_movement:
                warnings.append(
                    "dealers are long gamma: expect mean reversion, not follow-through"
                )
            if options.iv_rank is not None and options.iv_rank > 0.85:
                warnings.append(
                    f"IV rank {options.iv_rank:.0%} — long premium is expensive here"
                )

        if regime is not None and regime.is_uncertain:
            warnings.append(f"regime is uncertain (confidence {regime.confidence:.2f})")

        crossasset = report.crossasset
        if crossasset is not None and crossasset.regime == "stressed":
            warnings.append(
                "market is in a stressed volatility regime — correlations converge to one"
            )
        return vetoes, warnings
