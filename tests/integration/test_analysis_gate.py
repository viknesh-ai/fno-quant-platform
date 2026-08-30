"""The analysis layer's authority over the pipeline.

The analysis layer is deliberately given three powers and no more: it can veto,
it can refuse a direction it opposes, and it can scale a score within a bounded
influence. These tests pin all three, and — just as importantly — pin what it
*cannot* do: manufacture a trade the strategies and the model did not agree on.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from aqtp.analysis.confluence import ConfluenceEngine, Dimension
from aqtp.analysis.engine import AnalysisReport
from aqtp.configuration.schema import AnalysisConfig
from aqtp.core.clock import IST
from aqtp.core.types import Direction, Regime
from aqtp.regime.engine import RegimeState
from aqtp.signals.ensemble import SignalEnsemble
from aqtp.strategies.base import StrategySignal

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


def _signal(strategy: str, direction: Direction, confidence: float) -> StrategySignal:
    entry, stop, target = (24000.0, 23900.0, 24250.0) if direction is Direction.LONG \
        else (24000.0, 24100.0, 23750.0)
    return StrategySignal(
        strategy=strategy, underlying="NIFTY", direction=direction, confidence=confidence,
        expected_move_atr=2.0, proposed_entry=entry, proposed_stop=stop,
        proposed_target=target, expected_holding_minutes=30, regime_fit=0.9,
        reasoning=[f"{strategy} says so"],
    )


def _regime(dominant=Regime.TRENDING_UP, confidence=0.85) -> RegimeState:
    return RegimeState(
        probabilities={dominant: confidence, Regime.RANGE: 1 - confidence},
        dominant=dominant, confidence=confidence, entropy=0.3,
    )


def _analysis(
    *, score: float, confidence: float = 1.0, vetoes: list[str] | None = None
) -> AnalysisReport:
    """An AnalysisReport whose confluence points where the test needs it to."""
    dimensions = [
        Dimension(name, score, confidence)
        for name in ("trend", "momentum", "structure", "orderflow",
                     "positioning", "statistical", "volatility", "crossasset")
    ]
    report = AnalysisReport(underlying="NIFTY", timestamp=NOW, price=24000.0)
    report.confluence = ConfluenceEngine().combine(
        dimensions, underlying="NIFTY", timestamp=NOW, vetoes=vetoes or []
    )
    return report


def _long_signals() -> list[StrategySignal]:
    return [
        _signal("trend_following", Direction.LONG, 0.85),
        _signal("momentum", Direction.LONG, 0.80),
        _signal("options_flow", Direction.LONG, 0.75),
    ]


class TestAnalysisVeto:
    def test_a_veto_blocks_an_otherwise_valid_signal(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        baseline = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
        )
        assert baseline.is_tradable    # the same inputs trade without the analysis

        vetoed = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.9, vetoes=["toxic order flow"]),
            analysis_config=config.analysis,
        )
        assert not vetoed.is_tradable
        assert "toxic order flow" in vetoed.rejection_reason

    def test_veto_can_be_switched_off(self, config):
        settings = AnalysisConfig(veto_enabled=False, require_direction_agreement=False,
                                  min_conviction=0.0)
        result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.9, vetoes=["toxic order flow"]),
            analysis_config=settings,
        )
        assert result.is_tradable


class TestDirectionAgreement:
    def test_opposing_analysis_blocks_the_trade(self, config):
        result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=-0.9), analysis_config=config.analysis,
        )
        assert not result.is_tradable
        assert "against a LONG signal" in result.rejection_reason

    def test_a_flat_analysis_does_not_block_it(self, config):
        settings = AnalysisConfig(min_conviction=0.0)
        result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.0), analysis_config=settings,
        )
        assert result.is_tradable


class TestConviction:
    def test_low_conviction_blocks_the_trade(self, config):
        settings = AnalysisConfig(min_conviction=0.60)
        result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.15), analysis_config=settings,
        )
        assert not result.is_tradable
        assert "conviction" in result.rejection_reason

    def test_strong_support_raises_the_score_and_weak_support_lowers_it(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        settings = AnalysisConfig(min_conviction=0.0, influence=0.5)

        strong = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.95), analysis_config=settings,
        )
        weak = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.15), analysis_config=settings,
        )
        assert strong.is_tradable and weak.is_tradable
        assert strong.score > weak.score
        assert strong.analysis_adjustment > 1.0
        assert weak.analysis_adjustment < 1.0

    def test_influence_zero_leaves_the_score_untouched(self, config):
        ensemble = SignalEnsemble(config.ensemble)
        advisory = AnalysisConfig(min_conviction=0.0, influence=0.0)

        without = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
        )
        with_analysis = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.95), analysis_config=advisory,
        )
        assert with_analysis.score == pytest.approx(without.score)
        assert with_analysis.analysis_adjustment == pytest.approx(1.0)

    def test_the_adjustment_is_bounded_by_influence(self, config):
        """Influence 1.0 is the widest the analysis may swing a score: [0.5x, 2x]."""
        ensemble = SignalEnsemble(config.ensemble)
        settings = AnalysisConfig(min_conviction=0.0, influence=1.0)
        result = ensemble.combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=1.0), analysis_config=settings,
        )
        assert 0.0 <= result.analysis_adjustment <= 2.0
        assert 0.0 <= result.score <= 1.0


class TestAnalysisCannotCreateTrades:
    def test_perfect_analysis_cannot_rescue_a_single_strategy(self, config):
        """The analysis is evidence. It does not get a vote on the agreement gate."""
        result = SignalEnsemble(config.ensemble).combine(
            signals=[_signal("trend_following", Direction.LONG, 0.99)],
            prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=1.0), analysis_config=config.analysis,
        )
        assert not result.is_tradable
        assert "agree" in result.rejection_reason

    def test_perfect_analysis_cannot_rescue_an_uncertain_regime(self, config):
        uncertain = RegimeState(
            probabilities={Regime.UNCERTAIN: 1.0}, dominant=Regime.UNCERTAIN,
            confidence=0.2, entropy=0.95,
        )
        result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=uncertain,
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=1.0), analysis_config=config.analysis,
        )
        assert not result.is_tradable


class TestScannerIntegration:
    def test_analysis_support_is_a_ranked_component(self, config, index_instrument):
        from aqtp.scanner.opportunity import OpportunityScanner

        ensemble_result = SignalEnsemble(config.ensemble).combine(
            signals=_long_signals(), prediction=None, regime=_regime(),
            min_model_probability=config.prediction.min_model_probability, timestamp=NOW,
            analysis=_analysis(score=0.9),
            analysis_config=AnalysisConfig(min_conviction=0.0),
        )
        scanner = OpportunityScanner(config.scanner)

        from tests.conftest import make_quote

        quote = make_quote(index_instrument, price=24000.0)
        endorsed = scanner.score(
            underlying="NIFTY", ensemble=ensemble_result, prediction=None,
            regime=_regime(), quote=quote, instrument=index_instrument,
            timestamp=NOW, analysis=_analysis(score=0.9),
        )
        unread = scanner.score(
            underlying="NIFTY", ensemble=ensemble_result, prediction=None,
            regime=_regime(), quote=quote, instrument=index_instrument,
            timestamp=NOW, analysis=None,
        )
        assert "analysis" in endorsed.components
        assert endorsed.components["analysis"] > unread.components["analysis"]
        assert endorsed.score > unread.score
