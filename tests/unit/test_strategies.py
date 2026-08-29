"""Strategy and regime tests (REQ 11/12/13/64)."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from aqtp.core.clock import IST
from aqtp.core.types import Direction, Regime, Timeframe
from aqtp.data.candles import candles_to_frame, resample
from aqtp.features.engine import FeatureEngine, latest_feature_row
from aqtp.regime.engine import MarketRegimeEngine, RegimeState, regime_fit
from aqtp.signals.ensemble import SignalEnsemble
from aqtp.strategies.base import StrategyContext, StrategySignal, no_signal
from aqtp.strategies.market_structure import MarketStructureStrategy
from aqtp.strategies.mean_reversion import MeanReversionStrategy
from aqtp.strategies.registry import available_strategies, build_strategies
from tests.conftest import make_candles


# --------------------------------------------------------------------------- #
def build_context(frame: pd.DataFrame, *, regime: RegimeState | None = None, **overrides):
    """Assemble a StrategyContext from a price frame."""
    engine = FeatureEngine()
    features = engine.compute(frame)
    row = latest_feature_row(features)
    assert row is not None

    if regime is None:
        regime = MarketRegimeEngine().detect(row)

    defaults = dict(
        underlying="NIFTY",
        instrument=None,
        timestamp=frame.index[-1].to_pydatetime(),
        last_price=float(frame["close"].iloc[-1]),
        features=row,
        features_by_timeframe={Timeframe.M5: row, Timeframe.M15: row, Timeframe.H1: row},
        frames={Timeframe.M5: frame, Timeframe.M15: frame, Timeframe.H1: frame},
        regime=regime,
        entry_timeframe=Timeframe.M5,
        setup_timeframe=Timeframe.M15,
        regime_timeframe=Timeframe.H1,
        atr=float(row.get("atr", 1.0)),
        session_context={"minutes_until_close": 180.0, "minutes_since_open": 105.0},
    )
    defaults.update(overrides)

    from aqtp.core.types import Exchange, Instrument, InstrumentType, Segment

    if defaults["instrument"] is None:
        defaults["instrument"] = Instrument(
            trading_symbol="NIFTY", exchange=Exchange.NSE, segment=Segment.CASH,
            instrument_type=InstrumentType.IDX,
        )
    return StrategyContext(**defaults)


# =========================================================================== #
# Regime engine (REQ 11)
# =========================================================================== #
class TestRegimeEngine:
    def test_probabilities_form_a_distribution(self, trending_frame):
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        state = MarketRegimeEngine().detect(row)
        assert sum(state.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert all(0.0 <= p <= 1.0 for p in state.probabilities.values())

    def test_uptrend_is_classified_as_trending_up(self, trending_frame):
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        state = MarketRegimeEngine().detect(row)
        assert state.dominant is Regime.TRENDING_UP
        assert state.probability(Regime.TRENDING_UP) > state.probability(Regime.TRENDING_DOWN)

    def test_downtrend_is_classified_as_trending_down(self, downtrend_frame):
        row = latest_feature_row(FeatureEngine().compute(downtrend_frame))
        state = MarketRegimeEngine().detect(row)
        assert state.dominant is Regime.TRENDING_DOWN

    def test_no_evidence_produces_uncertain(self):
        empty = pd.Series(dtype=float)
        state = MarketRegimeEngine().detect(empty)
        assert state.dominant is Regime.UNCERTAIN
        assert state.confidence == 0.0

    def test_uncertain_regime_is_not_tradable(self):
        engine = MarketRegimeEngine()
        state = engine.detect(pd.Series(dtype=float))
        tradable, reason = engine.tradable(state)
        assert not tradable and "uncertain" in reason.lower()

    def test_expiry_day_raises_the_expiry_regime(self, ranging_frame):
        row = latest_feature_row(FeatureEngine().compute(ranging_frame))
        engine = MarketRegimeEngine()
        far = engine.detect(row, days_to_expiry=20)
        near = MarketRegimeEngine().detect(row, days_to_expiry=0)
        assert near.probability(Regime.EXPIRY_DRIVEN) > far.probability(Regime.EXPIRY_DRIVEN)

    def test_entropy_is_low_when_one_regime_dominates(self, trending_frame):
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        state = MarketRegimeEngine().detect(row)
        assert 0.0 <= state.entropy <= 1.0
        assert state.entropy < 0.6

    def test_regime_fit_penalizes_mean_reversion_in_a_trend(self, trending_frame):
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        state = MarketRegimeEngine().detect(row)
        assert regime_fit("mean_reversion", state) < 0.2
        assert regime_fit("trend_following", state) > 0.7


# =========================================================================== #
# Strategies (REQ 12)
# =========================================================================== #
class TestStrategyContract:
    @pytest.mark.parametrize("name", available_strategies())
    def test_every_strategy_returns_a_signal_and_never_raises(self, name, ranging_frame):
        """A strategy must degrade to a FLAT signal, never propagate an exception."""
        from aqtp.strategies.registry import get_strategy_class

        strategy = get_strategy_class(name)()
        context = build_context(ranging_frame)
        signal = strategy.generate(context)
        assert isinstance(signal, StrategySignal)
        assert signal.strategy == name
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.reasoning, "every signal must carry reasoning (REQ 48)"

    @pytest.mark.parametrize("name", available_strategies())
    def test_strategies_survive_degenerate_input(self, name):
        """Empty/garbage input must produce FLAT, not a crash."""
        from aqtp.strategies.registry import get_strategy_class

        frame = candles_to_frame(make_candles(bars=20, seed=99))
        strategy = get_strategy_class(name)()
        context = build_context(frame)
        signal = strategy.generate(context)
        assert signal.direction is Direction.FLAT

    def test_unknown_parameter_is_rejected(self):
        from aqtp.strategies.trend import TrendFollowingStrategy

        with pytest.raises(ValueError, match="unknown parameters"):
            TrendFollowingStrategy(params={"not_a_real_parameter": 1})

    def test_registry_rejects_unknown_strategy_names(self):
        from aqtp.configuration.schema import StrategyConfig

        with pytest.raises(KeyError, match="unknown strategies"):
            build_strategies({"does_not_exist": StrategyConfig()})


class TestMeanReversionLockout:
    """REQ 12.4: mean reversion must disable itself in strong trends."""

    def test_disabled_in_a_trending_regime(self, trending_frame):
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        regime = MarketRegimeEngine().detect(row)
        assert regime.dominant is Regime.TRENDING_UP

        signal = MeanReversionStrategy().generate(build_context(trending_frame, regime=regime))
        assert signal.direction is Direction.FLAT
        assert "disabled" in signal.reasoning[0].lower()

    def test_disabled_by_high_adx_even_if_regime_says_range(self, trending_frame):
        """The ADX guard is a second, independent lockout."""
        row = latest_feature_row(FeatureEngine().compute(trending_frame))
        # Force a RANGE regime to prove the strategy still refuses on its own.
        forced = RegimeState(
            probabilities={Regime.RANGE: 1.0}, dominant=Regime.RANGE,
            confidence=1.0, entropy=0.0,
        )
        signal = MeanReversionStrategy().generate(build_context(trending_frame, regime=forced))
        assert signal.direction is Direction.FLAT
        assert "adx" in signal.reasoning[0].lower() or "slope" in signal.reasoning[0].lower()


class TestMarketStructure:
    """REQ 12.6: discretionary terms must be measurable rules."""

    def test_break_of_structure_is_detected(self):
        # Build a clean swing high at index 20, then break it at the end.
        prices = [100.0] * 10 + list(np.linspace(100, 110, 10)) + list(np.linspace(110, 104, 10)) + [112.0]
        frame = pd.DataFrame(
            {
                "open": prices, "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices], "close": prices,
                "volume": [1000.0] * len(prices),
            },
            index=pd.date_range("2026-06-01 09:15", periods=len(prices), freq="5min", tz=IST),
        )
        params = MarketStructureStrategy.default_params()
        result = MarketStructureStrategy.analyze_structure(frame, params, atr=1.0)
        assert result["event"] in ("break_of_structure", "liquidity_sweep", "failed_breakout")
        assert result["direction"] is not Direction.FLAT

    def test_quiet_market_produces_no_structure_event(self):
        prices = [100.0] * 40
        frame = pd.DataFrame(
            {"open": prices, "high": [100.2] * 40, "low": [99.8] * 40, "close": prices,
             "volume": [1000.0] * 40},
            index=pd.date_range("2026-06-01 09:15", periods=40, freq="5min", tz=IST),
        )
        result = MarketStructureStrategy.analyze_structure(
            frame, MarketStructureStrategy.default_params(), atr=0.4
        )
        assert result["event"] is None


# =========================================================================== #
# Ensemble (REQ 13/54)
# =========================================================================== #
def _signal(strategy: str, direction: Direction, confidence: float, fit: float = 1.0):
    entry, stop, target = (
        (100.0, 98.0, 104.0) if direction is Direction.LONG else (100.0, 102.0, 96.0)
    )
    return StrategySignal(
        strategy=strategy, underlying="NIFTY", direction=direction, confidence=confidence,
        expected_move_atr=2.0, proposed_entry=entry, proposed_stop=stop, proposed_target=target,
        expected_holding_minutes=30, regime_fit=fit, reasoning=[f"{strategy} says so"],
    )


def _regime(dominant=Regime.TRENDING_UP, confidence=0.8) -> RegimeState:
    return RegimeState(
        probabilities={dominant: confidence, Regime.RANGE: 1 - confidence},
        dominant=dominant, confidence=confidence, entropy=0.3,
    )


class TestSignalEnsemble:
    def test_agreeing_strategies_produce_a_decision(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        result = ensemble.combine(
            signals=[
                _signal("trend_following", Direction.LONG, 0.8),
                _signal("momentum", Direction.LONG, 0.75),
                _signal("options_flow", Direction.LONG, 0.7),
            ],
            prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability,
        )
        assert result.is_tradable
        assert result.direction is Direction.LONG
        assert len(result.agreeing_strategies) == 3

    def test_too_few_agreeing_strategies_is_no_trade(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        result = ensemble.combine(
            signals=[_signal("trend_following", Direction.LONG, 0.9)],
            prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability,
        )
        assert not result.is_tradable
        assert "agree" in result.rejection_reason

    def test_uncertain_regime_blocks_everything(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        uncertain = RegimeState(
            probabilities={Regime.UNCERTAIN: 1.0}, dominant=Regime.UNCERTAIN,
            confidence=0.2, entropy=0.95,
        )
        result = ensemble.combine(
            signals=[
                _signal("trend_following", Direction.LONG, 0.9),
                _signal("momentum", Direction.LONG, 0.9),
            ],
            prediction=None, regime=uncertain,
            min_model_probability=config.prediction.min_model_probability,
        )
        assert not result.is_tradable
        assert "uncertain" in result.rejection_reason

    def test_correlated_strategies_are_discounted(self, config):
        """REQ 13: three correlated strategies must not out-vote three independent
        ones by simple addition."""
        ensemble = SignalEnsemble(config.ensemble)
        # trend_following, market_structure and vwap are one correlation family.
        correlated = ensemble._side_score(
            [
                (_signal("trend_following", Direction.LONG, 0.8), 0.8),
                (_signal("market_structure", Direction.LONG, 0.8), 0.8),
                (_signal("vwap", Direction.LONG, 0.8), 0.8),
            ]
        )
        independent = ensemble._side_score(
            [
                (_signal("trend_following", Direction.LONG, 0.8), 0.8),
                (_signal("mean_reversion", Direction.LONG, 0.8), 0.8),
                (_signal("options_flow", Direction.LONG, 0.8), 0.8),
            ]
        )
        assert correlated < independent, "correlated evidence was counted at full weight"
        assert correlated == pytest.approx(0.8 * (1 + 0.6 + 0.36), rel=1e-6)

    def test_stop_is_the_tightest_proposal(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        loose = _signal("trend_following", Direction.LONG, 0.8)
        loose.proposed_stop = 90.0
        tight = _signal("momentum", Direction.LONG, 0.8)
        tight.proposed_stop = 98.0

        result = ensemble.combine(
            signals=[loose, tight], prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability,
        )
        assert result.is_tradable
        assert result.proposed_stop == pytest.approx(98.0)

    def test_ml_disagreement_blocks_the_trade(self, config):
        from aqtp.ml.predict import Prediction

        ensemble = SignalEnsemble(config.ensemble)
        opposing = Prediction(
            direction=Direction.SHORT, probability=0.8, confidence=0.8, uncertainty=0.05,
            expected_move_atr=1.5, horizon_minutes=30,
        )
        result = ensemble.combine(
            signals=[
                _signal("trend_following", Direction.LONG, 0.9),
                _signal("momentum", Direction.LONG, 0.9),
            ],
            prediction=opposing, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability,
        )
        assert not result.is_tradable
        assert "favours SHORT" in result.rejection_reason

    def test_low_ml_probability_blocks_the_trade(self, config):
        from aqtp.ml.predict import Prediction

        ensemble = SignalEnsemble(config.ensemble)
        weak = Prediction(
            direction=Direction.LONG, probability=0.52, confidence=0.5, uncertainty=0.1,
            expected_move_atr=1.5, horizon_minutes=30,
        )
        result = ensemble.combine(
            signals=[
                _signal("trend_following", Direction.LONG, 0.9),
                _signal("momentum", Direction.LONG, 0.9),
            ],
            prediction=weak, regime=_regime(),
            min_model_probability=0.58,
        )
        assert not result.is_tradable
        assert "below the required" in result.rejection_reason

    def test_confidence_is_not_the_probability(self, config):
        """REQ 54: confidence must be a distinct, calibrated quantity."""
        from aqtp.ml.predict import Prediction

        ensemble = SignalEnsemble(config.ensemble)
        prediction = Prediction(
            direction=Direction.LONG, probability=0.75, confidence=0.4, uncertainty=0.25,
            expected_move_atr=1.5, horizon_minutes=30,
        )
        result = ensemble.combine(
            signals=[
                _signal("trend_following", Direction.LONG, 0.9),
                _signal("momentum", Direction.LONG, 0.9),
            ],
            prediction=prediction, regime=_regime(),
            min_model_probability=0.58,
        )
        if result.is_tradable:
            assert result.confidence != pytest.approx(result.ml_probability)
