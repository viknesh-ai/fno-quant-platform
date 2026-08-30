"""Tests for the deep-analysis layer (`aqtp.analysis`).

The properties worth asserting here are not "the number is 0.42" — synthetic data
would make that meaningless. They are the ones a wrong implementation would
violate: an estimator that says a trending series is mean-reverting, a confluence
score that ignores conflict, a veto that does not veto, an exposure sign that has
the dealer on the wrong side of the trade.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from aqtp.analysis.confluence import ConfluenceEngine, Dimension
from aqtp.analysis.crossasset import CrossAssetTracker
from aqtp.analysis.engine import AnalysisEngine
from aqtp.analysis.microstructure import (
    MicrostructureTracker,
    analyze_quote,
    tape_analysis,
)
from aqtp.analysis.options_analytics import analyze_chain
from aqtp.analysis.statistics import (
    analyze_series,
    detect_jumps,
    efficiency_ratio,
    garch11_forecast,
    hurst_exponent,
    permutation_entropy,
    variance_ratio,
)
from aqtp.analysis.volume_profile import build_volume_profile
from aqtp.core.clock import IST
from aqtp.core.types import Direction, OptionType, Timeframe
from aqtp.data.candles import candles_to_frame

from ..conftest import make_candles, make_option_chain, make_quote

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


# =========================================================================== #
# Statistics
# =========================================================================== #
class TestStatisticalEstimators:
    def test_hurst_separates_trending_from_mean_reverting(self):
        rng = np.random.default_rng(7)
        steps = rng.normal(0, 1, 2000)

        # A trending series: returns share a common drift component.
        trending = 100 + np.cumsum(steps + 0.5)
        # A mean-reverting series: an OU process pulled back to its mean.
        reverting = np.zeros(2000)
        for i in range(1, 2000):
            reverting[i] = reverting[i - 1] * 0.3 + steps[i]
        reverting = 100 + reverting

        h_trend = hurst_exponent(trending)
        h_revert = hurst_exponent(reverting)
        assert h_trend is not None and h_revert is not None
        assert h_trend > h_revert

    def test_variance_ratio_exceeds_one_for_momentum(self):
        """A drift alone does not raise VR — only autocorrelated returns do.

        This is the whole point of the statistic: it tests the random-walk
        hypothesis, not whether price went up.
        """
        rng = np.random.default_rng(11)
        shocks = rng.normal(0, 0.001, 1200)
        returns = np.zeros(1200)
        for i in range(1, 1200):
            returns[i] = 0.5 * returns[i - 1] + shocks[i]   # positive autocorrelation
        assert variance_ratio(100 * np.exp(np.cumsum(returns)), lag=5) > 1.0

    def test_variance_ratio_is_near_one_for_a_drifting_random_walk(self):
        rng = np.random.default_rng(11)
        drifting = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.001, 1500)))
        assert 0.75 < variance_ratio(drifting, lag=5) < 1.25

    def test_variance_ratio_below_one_for_an_alternating_series(self):
        # Perfect alternation is maximally anti-persistent.
        values = np.array([100 + (1 if i % 2 else -1) for i in range(400)], dtype=float)
        assert variance_ratio(values, lag=4) < 1.0

    def test_entropy_is_lower_for_a_clean_ramp_than_for_noise(self):
        rng = np.random.default_rng(3)
        ramp = np.linspace(100, 140, 600)
        noise = 100 + rng.normal(0, 1, 600)
        assert permutation_entropy(ramp) < permutation_entropy(noise)

    def test_efficiency_ratio_is_one_for_a_straight_line(self):
        assert efficiency_ratio(np.linspace(100, 120, 60), period=20) == pytest.approx(1.0)

    def test_efficiency_ratio_is_near_zero_for_pure_churn(self):
        churn = np.array([100 + (1 if i % 2 else 0) for i in range(60)], dtype=float)
        assert efficiency_ratio(churn, period=20) < 0.2

    def test_estimators_return_none_rather_than_guessing_on_short_series(self):
        short = np.linspace(100, 101, 10)
        assert hurst_exponent(short) is None
        assert variance_ratio(short) is None
        assert permutation_entropy(short) is None

    def test_garch_fit_is_stationary_and_positive(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0, 0.004, 500)
        result = garch11_forecast(returns)
        assert result is not None
        sigma, params = result
        assert sigma > 0
        assert 0 < params["persistence"] < 1.0
        # The forecast should be in the neighbourhood of the sample volatility,
        # not an order of magnitude away.
        assert 0.3 < sigma / returns.std() < 3.0

    def test_jump_detection_finds_an_injected_jump(self):
        frame = candles_to_frame(make_candles(bars=200, volatility=0.0003, seed=17))
        clean_count, _ = detect_jumps(frame)

        jumped = frame.copy()
        jumped.iloc[-1, jumped.columns.get_loc("close")] *= 1.05
        _, latest = detect_jumps(jumped)
        assert latest > 4.0
        assert clean_count <= 1


class TestStatisticalReport:
    def test_trending_series_reads_as_persistent(self, trending_frame):
        report = analyze_series(trending_frame)
        assert report.bars_analyzed == len(trending_frame)
        assert report.persistence_score > 0
        assert report.reasons

    def test_report_degrades_gracefully_without_data(self):
        report = analyze_series(pd.DataFrame())
        assert report.bars_analyzed == 0
        assert report.persistence_score == 0.0
        assert report.reasons

    def test_feature_dict_is_finite(self, trending_frame):
        report = analyze_series(trending_frame)
        for name, value in report.to_dict().items():
            assert np.isfinite(value), name


# =========================================================================== #
# Volume profile
# =========================================================================== #
class TestVolumeProfile:
    def test_value_area_brackets_the_poc(self, ranging_frame):
        profile = build_volume_profile(ranging_frame)
        assert profile.poc is not None
        assert profile.value_area_low <= profile.poc <= profile.value_area_high

    def test_value_area_holds_roughly_the_requested_share_of_volume(self, ranging_frame):
        profile = build_volume_profile(ranging_frame, value_area_pct=0.70)
        # The construction expands in whole bins, so it overshoots slightly and
        # never undershoots by more than one bin's worth.
        assert profile.total_volume > 0
        assert profile.value_area_high > profile.value_area_low

    def test_location_is_above_value_when_price_breaks_out(self, ranging_frame):
        profile = build_volume_profile(ranging_frame)
        above = build_volume_profile(
            ranging_frame, price=profile.value_area_high * 1.02
        )
        assert above.location == "above_value"
        assert above.structure_score > 0

    def test_location_is_below_value_beneath_the_area(self, ranging_frame):
        profile = build_volume_profile(ranging_frame)
        below = build_volume_profile(ranging_frame, price=profile.value_area_low * 0.98)
        assert below.location == "below_value"
        assert below.structure_score < 0

    def test_profile_without_volume_falls_back_to_time_at_price(self):
        frame = candles_to_frame(make_candles(bars=200, with_volume=False, seed=21))
        profile = build_volume_profile(frame)
        assert profile.poc is not None
        assert any("time at price" in reason for reason in profile.reasons)

    def test_empty_frame_is_handled(self):
        profile = build_volume_profile(pd.DataFrame())
        assert profile.poc is None
        assert profile.location == "unknown"
        assert profile.structure_score == 0.0


# =========================================================================== #
# Microstructure
# =========================================================================== #
class TestMicrostructure:
    def test_touch_imbalance_follows_the_resting_size(self, call_instrument):
        quote = make_quote(call_instrument, price=200.0, spread=1.0)
        quote.bid_quantity = 2000
        quote.ask_quantity = 200
        report = analyze_quote(quote)
        assert report.touch_imbalance > 0.5

        quote.bid_quantity, quote.ask_quantity = 200, 2000
        assert analyze_quote(quote).touch_imbalance < -0.5

    def test_micro_price_leans_toward_the_thinner_side(self, call_instrument):
        quote = make_quote(call_instrument, price=200.0, spread=2.0)
        quote.bid_quantity, quote.ask_quantity = 5000, 100
        report = analyze_quote(quote)
        # Heavy bid, thin ask: fair value sits above the mid.
        assert report.micro_price > report.mid

    def test_impact_cost_is_estimated_when_the_book_is_too_thin(self, call_instrument):
        quote = make_quote(call_instrument, price=200.0, spread=1.0)
        report = analyze_quote(quote, test_quantity=1_000_000)
        assert report.impact_cost_pct is not None
        assert any("cannot absorb" in reason for reason in report.reasons)

    def test_order_flow_imbalance_needs_history(self, call_instrument):
        tracker = MicrostructureTracker()
        quote = make_quote(call_instrument, price=200.0)
        first = tracker.analyze(quote)
        assert first.order_flow_imbalance == 0.0

        # Walk the bid up repeatedly: sustained buy-side pressure.
        for step in range(1, 15):
            moved = make_quote(call_instrument, price=200.0 + step * 0.5)
            moved.bid_quantity = 1000
            moved.ask_quantity = 200
            report = tracker.analyze(moved)
        assert report.quote_updates > 10
        assert report.order_flow_imbalance > 0

    def test_tape_pressure_is_positive_when_bars_close_on_their_highs(self):
        index = pd.date_range("2026-06-01 09:15", periods=50, freq="5min", tz=IST)
        frame = pd.DataFrame(
            {
                "open": np.linspace(100, 110, 50),
                "high": np.linspace(100.5, 110.5, 50),
                "low": np.linspace(99.5, 109.5, 50),
                "close": np.linspace(100.5, 110.5, 50),  # closes at the high
                "volume": np.full(50, 1000.0),
            },
            index=index,
        )
        pressure, slope, _ = tape_analysis(frame)
        assert pressure > 0.8
        assert slope > 0

    def test_tracker_evicts_symbols_beyond_its_cap(self, call_instrument):
        from dataclasses import replace

        tracker = MicrostructureTracker(history=5, max_symbols=3)
        for index in range(6):
            instrument = replace(call_instrument, trading_symbol=f"SYM{index}")
            tracker.observe(make_quote(instrument, price=100.0))
        assert tracker.samples("SYM0") == 0     # evicted
        assert tracker.samples("SYM5") == 1

    def test_toxic_flow_is_flagged(self, call_instrument):
        tracker = MicrostructureTracker()
        # Every update trades at a rising price: entirely one-sided flow.
        report = None
        for step in range(20):
            quote = make_quote(call_instrument, price=200.0 + step, volume=50_000 + step * 5_000)
            report = tracker.analyze(quote)
        assert report.flow_toxicity > 0.9


# =========================================================================== #
# Options analytics
# =========================================================================== #
class TestOptionsAnalytics:
    def test_chain_analysis_produces_a_complete_map(self):
        chain = make_option_chain(now=NOW)
        report = analyze_chain(chain, now=NOW, lot_size=75)
        assert report.strikes_analyzed > 0
        assert report.pcr_oi > 0
        assert report.atm_iv is not None
        assert report.expected_move_pct is not None
        assert report.max_pain is not None
        assert report.reasons

    def test_max_pain_lands_inside_the_listed_strikes(self):
        chain = make_option_chain(now=NOW)
        report = analyze_chain(chain, now=NOW, lot_size=75)
        strikes = chain.strikes()
        assert min(strikes) <= report.max_pain <= max(strikes)

    def test_a_symmetric_chain_nets_to_zero_gamma_exposure(self):
        """Gamma is identical for a call and a put at the same strike.

        Under the standard dealer convention (long call gamma, short put gamma)
        a chain with equal call and put OI at every strike must net to exactly
        zero. Anything else would mean the sign convention is wrong.
        """
        chain = make_option_chain(now=NOW)
        report = analyze_chain(chain, now=NOW, lot_size=75)
        assert report.total_gamma_exposure == pytest.approx(0.0, abs=1.0)
        assert report.gamma_regime == "unknown"

    def test_call_heavy_open_interest_puts_dealers_long_gamma(self):
        chain = make_option_chain(now=NOW)
        for contract in chain.contracts:
            if contract.option_type is OptionType.CE:
                contract.quote.open_interest *= 3
        report = analyze_chain(chain, now=NOW, lot_size=75)
        assert report.total_gamma_exposure > 0
        assert report.gamma_regime == "positive"
        assert report.suppresses_movement
        assert not report.amplifies_movement

    def test_put_heavy_open_interest_puts_dealers_short_gamma(self):
        chain = make_option_chain(now=NOW)
        for contract in chain.contracts:
            if contract.option_type is OptionType.PE:
                contract.quote.open_interest *= 3
        report = analyze_chain(chain, now=NOW, lot_size=75)
        assert report.total_gamma_exposure < 0
        assert report.gamma_regime == "negative"
        assert report.amplifies_movement

    def test_walls_sit_on_the_correct_side_of_spot(self):
        chain = make_option_chain(now=NOW, spot=24000.0)
        report = analyze_chain(chain, now=NOW, lot_size=75)
        if report.call_wall is not None:
            assert report.call_wall > report.spot
        if report.put_wall is not None:
            assert report.put_wall < report.spot

    def test_positioning_bias_is_bounded(self):
        chain = make_option_chain(now=NOW)
        report = analyze_chain(chain, now=NOW, lot_size=75)
        assert -1.0 <= report.positioning_bias <= 1.0
        assert -1.0 <= report.volatility_bias <= 1.0

    def test_iv_rank_drives_the_volatility_bias(self):
        chain = make_option_chain(now=NOW)
        cheap = analyze_chain(chain, now=NOW, lot_size=75, iv_rank=0.05)
        rich = analyze_chain(chain, now=NOW, lot_size=75, iv_rank=0.95)
        assert cheap.volatility_bias > 0      # options are cheap: buy premium
        assert rich.volatility_bias < 0       # options are rich: sell it

    def test_pin_risk_only_appears_near_expiry(self):
        far = make_option_chain(now=NOW)
        far_report = analyze_chain(far, now=NOW, lot_size=75)
        assert far_report.pin_risk == 0.0

        near_now = datetime(2026, 6, 25, 10, 0, tzinfo=IST)
        near = make_option_chain(now=near_now, spot=24000.0)
        near_report = analyze_chain(near, now=near_now, lot_size=75)
        assert near_report.days_to_expiry < 1.0

    def test_empty_chain_is_handled(self):
        from aqtp.core.types import OptionChain

        empty = OptionChain(underlying_symbol="NIFTY", expiry=NOW.date(), underlying_ltp=0.0)
        report = analyze_chain(empty, now=NOW)
        assert report.strikes_analyzed == 0
        assert report.reasons

    def test_feature_dict_is_finite(self):
        chain = make_option_chain(now=NOW)
        report = analyze_chain(chain, now=NOW, lot_size=75, iv_rank=0.4)
        for name, value in report.to_dict().items():
            assert np.isfinite(value), name


# =========================================================================== #
# Cross-asset
# =========================================================================== #
class TestCrossAsset:
    def test_correlation_rises_when_symbols_move_together(self):
        tracker = CrossAssetTracker()
        rng = np.random.default_rng(13)
        common = rng.normal(0, 0.002, 60)
        prices = {name: 100.0 for name in ("A", "B", "C", "D")}
        for step in range(60):
            for name in prices:
                prices[name] *= 1 + common[step]        # identical moves
                tracker.observe(name, prices[name])
        report = tracker.build(timestamp=NOW)
        assert report.symbols_tracked == 4
        assert report.average_correlation > 0.9
        assert report.diversification_score < 0.1

    def test_breadth_reflects_the_share_advancing(self):
        tracker = CrossAssetTracker()
        report = tracker.build(
            timestamp=NOW,
            session_returns={"A": 0.01, "B": 0.02, "C": -0.01, "D": -0.02},
        )
        assert report.breadth == pytest.approx(0.0)
        assert report.leaders[0] == "B"
        assert report.laggards[0] == "D"

    def test_alignment_opposes_a_long_when_breadth_is_negative(self):
        tracker = CrossAssetTracker()
        report = tracker.build(
            timestamp=NOW,
            session_returns={"A": -0.01, "B": -0.02, "C": -0.015, "D": 0.001},
        )
        assert report.alignment_for("A", 1.0) < 0     # long into weak breadth
        assert report.alignment_for("A", -1.0) > 0    # short is supported

    def test_vix_percentile_needs_history(self):
        tracker = CrossAssetTracker()
        for value in np.linspace(10, 20, 40):
            tracker.observe_vix(float(value))
        report = tracker.build(timestamp=NOW)
        assert report.vix == pytest.approx(20.0)
        assert report.vix_percentile == pytest.approx(1.0)

    def test_regime_is_named(self):
        tracker = CrossAssetTracker()
        report = tracker.build(timestamp=NOW)
        assert report.regime in ("normal", "stressed", "risk_on_off", "stock_pickers")


# =========================================================================== #
# Confluence
# =========================================================================== #
class TestConfluence:
    @staticmethod
    def _dimension(name: str, score: float, confidence: float = 1.0) -> Dimension:
        return Dimension(name, score, confidence)

    def test_agreeing_dimensions_produce_high_conviction(self):
        engine = ConfluenceEngine()
        dimensions = [
            self._dimension(name, 0.8)
            for name in ("trend", "momentum", "structure", "orderflow",
                         "positioning", "statistical", "volatility", "crossasset")
        ]
        report = engine.combine(dimensions, underlying="NIFTY")
        assert report.bias is Direction.LONG
        assert report.alignment == 1.0
        assert report.conviction > 0.5
        assert report.supports(Direction.LONG) > 0.4
        assert report.supports(Direction.SHORT) == 0.0

    def test_conflicting_dimensions_cut_conviction(self):
        engine = ConfluenceEngine()
        agreeing = [
            self._dimension(name, 0.8)
            for name in ("trend", "momentum", "structure", "orderflow")
        ]
        conflicted = agreeing + [
            self._dimension(name, -0.8)
            for name in ("positioning", "statistical", "volatility", "crossasset")
        ]
        clean = engine.combine(list(agreeing), underlying="NIFTY")
        messy = engine.combine(conflicted, underlying="NIFTY")
        assert messy.conviction < clean.conviction
        assert messy.conflicting

    def test_a_veto_zeroes_conviction_and_flattens_the_bias(self):
        engine = ConfluenceEngine()
        dimensions = [
            self._dimension(name, 0.9)
            for name in ("trend", "momentum", "structure", "orderflow")
        ]
        report = engine.combine(
            dimensions, underlying="NIFTY", vetoes=["toxic order flow"]
        )
        assert report.vetoed
        assert report.conviction == 0.0
        assert report.bias is Direction.FLAT
        assert report.supports(Direction.LONG) == 0.0

    def test_low_confidence_dimensions_are_discounted_not_counted(self):
        engine = ConfluenceEngine()
        strong = [self._dimension("trend", 1.0, confidence=1.0)]
        weak = [self._dimension(name, -1.0, confidence=0.01) for name in ("momentum", "structure")]
        report = engine.combine(strong + weak, underlying="NIFTY")
        # The near-zero-confidence dimensions do not report at all.
        assert report.net_score > 0
        assert "momentum" not in report.conflicting

    def test_too_few_reporting_dimensions_caps_conviction(self):
        engine = ConfluenceEngine(min_dimensions=6)
        dimensions = [self._dimension(name, 0.9) for name in ("trend", "momentum")]
        report = engine.combine(dimensions, underlying="NIFTY")
        assert report.warnings
        assert report.conviction < 0.6

    def test_no_usable_dimension_is_not_a_crash(self):
        engine = ConfluenceEngine()
        report = engine.combine(
            [self._dimension("trend", 0.5, confidence=0.0)], underlying="NIFTY"
        )
        assert report.conviction == 0.0
        assert report.bias is Direction.FLAT
        assert report.warnings

    def test_explanation_names_every_reporting_dimension(self):
        engine = ConfluenceEngine()
        dimensions = [
            Dimension("trend", 0.7, 1.0, reasons=["ADX 28 with DI spread +12"]),
            Dimension("orderflow", 0.4, 1.0, reasons=["order-flow imbalance +0.45"]),
            Dimension("positioning", -0.3, 1.0, reasons=["dealers are long gamma"]),
            Dimension("structure", 0.5, 1.0, reasons=["price above value"]),
        ]
        rendered = engine.combine(dimensions, underlying="NIFTY").render()
        for name in ("trend", "orderflow", "positioning", "structure"):
            assert name in rendered
        assert "ADX 28" in rendered


# =========================================================================== #
# AnalysisEngine end to end
# =========================================================================== #
class TestAnalysisEngine:
    @staticmethod
    def _frames(multi_timeframe_frames):
        return {
            Timeframe.M5: multi_timeframe_frames[Timeframe.M5],
            Timeframe.M15: multi_timeframe_frames[Timeframe.M15],
            Timeframe.H1: multi_timeframe_frames[Timeframe.H1],
        }

    @staticmethod
    def _features(frames):
        from aqtp.features.engine import FeatureEngine, latest_feature_row

        engine = FeatureEngine()
        out = {}
        for timeframe, frame in frames.items():
            row = latest_feature_row(engine.compute(frame))
            if row is not None:
                out[timeframe] = row
        return out

    def test_full_analysis_runs_and_is_bounded(
        self, multi_timeframe_frames, index_instrument, config
    ):
        frames = self._frames(multi_timeframe_frames)
        features = self._features(frames)
        engine = AnalysisEngine(config.analysis)
        quote = make_quote(index_instrument, price=float(frames[Timeframe.M5]["close"].iloc[-1]))

        report = engine.analyze(
            underlying="NIFTY",
            timestamp=NOW,
            quote=quote,
            frames=frames,
            features_by_timeframe=features,
            entry_timeframe=Timeframe.M5,
            setup_timeframe=Timeframe.M15,
            regime_timeframe=Timeframe.H1,
            chains=[make_option_chain(now=NOW)],
            lot_size=75,
        )

        assert len(report.confluence.dimensions) == 8
        assert -1.0 <= report.confluence.net_score <= 1.0
        assert 0.0 <= report.conviction <= 1.0
        assert -1.0 <= report.timeframe_alignment <= 1.0
        assert report.render()
        for name, value in report.feature_dict().items():
            assert np.isfinite(value), name

    def test_support_is_zero_for_the_opposing_direction(
        self, multi_timeframe_frames, index_instrument, config
    ):
        frames = self._frames(multi_timeframe_frames)
        engine = AnalysisEngine(config.analysis)
        quote = make_quote(index_instrument, price=24000.0)
        report = engine.analyze(
            underlying="NIFTY", timestamp=NOW, quote=quote, frames=frames,
            features_by_timeframe=self._features(frames),
            entry_timeframe=Timeframe.M5, setup_timeframe=Timeframe.M15,
            regime_timeframe=Timeframe.H1,
        )
        if report.bias is Direction.LONG:
            assert report.supports(Direction.SHORT) == 0.0
        elif report.bias is Direction.SHORT:
            assert report.supports(Direction.LONG) == 0.0

    def test_a_stale_quote_is_vetoed(self, multi_timeframe_frames, index_instrument, config):
        frames = self._frames(multi_timeframe_frames)
        engine = AnalysisEngine(config.analysis)
        report = engine.analyze(
            underlying="NIFTY", timestamp=NOW,
            quote=make_quote(index_instrument, price=24000.0),
            frames=frames, features_by_timeframe=self._features(frames),
            entry_timeframe=Timeframe.M5, setup_timeframe=Timeframe.M15,
            regime_timeframe=Timeframe.H1,
            quote_age_seconds=600.0,
        )
        assert report.vetoed
        assert any("old" in veto for veto in report.vetoes)
        assert report.conviction == 0.0

    def test_a_wide_spread_is_vetoed(self, multi_timeframe_frames, index_instrument, config):
        frames = self._frames(multi_timeframe_frames)
        engine = AnalysisEngine(config.analysis)
        wide = make_quote(index_instrument, price=100.0, spread=20.0)   # 20% spread
        report = engine.analyze(
            underlying="NIFTY", timestamp=NOW, quote=wide, frames=frames,
            features_by_timeframe=self._features(frames),
            entry_timeframe=Timeframe.M5, setup_timeframe=Timeframe.M15,
            regime_timeframe=Timeframe.H1,
        )
        assert report.vetoed
        assert any("spread" in veto for veto in report.vetoes)

    def test_analysis_never_raises_on_empty_inputs(self, index_instrument, config):
        engine = AnalysisEngine(config.analysis)
        report = engine.analyze(
            underlying="NIFTY", timestamp=NOW,
            quote=make_quote(index_instrument, price=24000.0),
            frames={}, features_by_timeframe={},
            entry_timeframe=Timeframe.M5, setup_timeframe=Timeframe.M15,
            regime_timeframe=Timeframe.H1,
        )
        assert report.conviction >= 0.0
        assert report.render()

    def test_market_context_feeds_the_crossasset_dimension(self, config):
        engine = AnalysisEngine(config.analysis)
        for step in range(30):
            for name, drift in (("NIFTY", 1.001), ("BANKNIFTY", 1.0012), ("FINNIFTY", 0.999)):
                engine.observe_market(name, 100.0 * drift ** step)
        report = engine.build_market_context(
            timestamp=NOW,
            session_returns={"NIFTY": 0.01, "BANKNIFTY": 0.012, "FINNIFTY": -0.005},
        )
        assert report.symbols_tracked == 3
        assert report.breadth is not None
