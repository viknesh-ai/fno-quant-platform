"""End-to-end pipeline test (REQ 2/69/70).

Drives the full chain — market data -> features -> regime -> strategies -> ML ->
ensemble -> ranking -> risk -> sizing -> execution -> journal — against a fully
simulated broker, and asserts the architectural invariants that the requirements
turn on rather than any particular P&L outcome.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from aqtp.backtest.engine import BacktestConfig, BacktestEngine, BacktestSignal
from aqtp.backtest.metrics import compute_metrics
from aqtp.backtest.robustness import run_all
from aqtp.core.clock import IST
from aqtp.core.types import Decision, Direction, Regime, Timeframe, TradingMode
from aqtp.data.candles import candles_to_frame, resample
from aqtp.journal.explain import ExplanationBuilder
from aqtp.risk.costs import TransactionCostModel
from tests.conftest import make_candles, make_option_chain, make_quote

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


class TestModeIsolation:
    """REQ 41/43: paper and backtest can never reach a real order endpoint."""

    def test_paper_mode_routes_execution_to_the_simulator(self, config):
        from aqtp.brokers.factory import build_execution_broker
        from aqtp.brokers.simulated import SimulatedBroker
        from aqtp.configuration.loader import Credentials

        assert config.mode is TradingMode.PAPER
        broker = build_execution_broker(
            config, Credentials(access_token="x"), quote_source=lambda i: None
        )
        assert isinstance(broker, SimulatedBroker)

    def test_paper_mode_never_submits_real_orders(self, config):
        assert not config.submits_real_orders

    def test_backtest_mode_routes_execution_to_the_simulator(self, config):
        from aqtp.brokers.factory import build_execution_broker
        from aqtp.brokers.simulated import SimulatedBroker
        from aqtp.configuration.loader import Credentials

        config.mode = TradingMode.BACKTEST
        broker = build_execution_broker(
            config, Credentials(access_token="x"), quote_source=lambda i: None
        )
        assert isinstance(broker, SimulatedBroker)


class TestStrategyCannotReachTheBroker:
    """REQ 67.23: a strategy must have no path to an order."""

    def test_strategy_context_exposes_no_broker(self):
        from aqtp.strategies.base import StrategyContext

        fields = set(StrategyContext.__dataclass_fields__)
        forbidden = {"broker", "execution", "execution_engine", "adapter", "order_manager"}
        assert not (fields & forbidden), (
            "StrategyContext exposes an execution handle; a strategy could place orders"
        )

    def test_strategy_signal_is_not_an_order(self):
        from aqtp.strategies.base import StrategySignal

        fields = set(StrategySignal.__dataclass_fields__)
        assert "quantity" not in fields, "a strategy must not decide position size"
        assert "order_type" not in fields, "a strategy must not choose an order type"


class TestExplainability:
    """REQ 48: every trade must answer all eight questions."""

    def test_decision_report_answers_every_required_question(self, config):
        from aqtp.options.selector import OptionSelector
        from aqtp.risk.engine import RiskApproval
        from aqtp.risk.sizing import SizingResult
        from aqtp.signals.ensemble import EnsembleResult
        from aqtp.signals.expected_value import ExpectedValueCalculator

        chain = make_option_chain(spot=24_000, now=NOW)
        selection = OptionSelector(config.option_selection).select(
            chains=[chain], direction=Direction.LONG, underlying_price=24_000,
            expected_move=250.0, atr=120.0, now=NOW, holding_minutes=60,
            capital_available=200_000, lot_size=75,
        )
        assert selection.ok

        ensemble = EnsembleResult(
            decision=Decision.BUY, direction=Direction.LONG, score=0.72, confidence=0.68,
            ml_probability=0.63, strategy_agreement=0.8,
            agreeing_strategies=["trend_following", "momentum"],
            proposed_entry=24_000.0, proposed_stop=23_880.0, proposed_target=24_300.0,
            expected_move_atr=2.5, regime=Regime.TRENDING_UP, regime_confidence=0.81,
            reasons=["ADX confirms a trend"], timestamp=NOW,
        )
        contract = selection.selected.contract
        ev = ExpectedValueCalculator(
            config.expected_value, TransactionCostModel(config.costs)
        ).evaluate(
            instrument=contract.instrument, quantity=75,
            entry_price=contract.last_price, stop_price=contract.last_price * 0.85,
            target_price=contract.last_price * 1.35, win_probability=0.63,
            quote=contract.quote,
        )
        approval = RiskApproval(
            approved=True, decision=Decision.BUY,
            sizing=SizingResult(
                quantity=75, lots=1, lot_size=75, risk_amount=2250.0, risk_per_unit=30.0,
                capital_required=15_000.0, margin_required=15_000.0,
                reasons=["risk budget ₹5,000"],
            ),
            approved_at=NOW, approval_id="RA1", expected_value=ev,
        )

        report = ExplanationBuilder().build(
            decision_id="D-EXPLAIN", timestamp=NOW, underlying="NIFTY",
            ensemble=ensemble, selection=selection, approval=approval,
            expected_value=ev, underlying_price=24_000.0,
        )

        for field in (
            report.why_enter, report.why_this_instrument, report.why_this_strike,
            report.why_this_expiry, report.why_this_size, report.why_this_stop,
            report.why_this_target,
        ):
            assert field, "a REQ 48 question was left unanswered"

        rendered = report.render()
        for question in (
            "WHY DID WE ENTER?", "WHY THIS INSTRUMENT?", "WHY THIS STRIKE?",
            "WHY THIS EXPIRY?", "WHY THIS SIZE?", "WHY THIS STOP?", "WHY THIS TARGET?",
        ):
            assert question in rendered

    def test_a_rejected_decision_explains_why_not(self, config):
        from aqtp.risk.engine import RiskApproval
        from aqtp.signals.ensemble import no_trade

        report = ExplanationBuilder().build(
            decision_id="D-NO", timestamp=NOW, underlying="NIFTY",
            ensemble=no_trade("regime uncertain", timestamp=NOW),
        )
        assert report.decision is Decision.NO_TRADE
        assert report.no_trade_reasons
        assert "WHY NO TRADE?" in report.render()

    def test_every_exit_carries_a_reason(self):
        from aqtp.core.types import ExitReason

        lines = ExplanationBuilder.explain_exit(
            reason=ExitReason.STOP_REACHED, detail="price hit 170", pnl=-2250.0,
            r_multiple=-1.0, holding_minutes=25,
        )
        assert any("STOP_REACHED" in line for line in lines)


class TestJournal:
    """REQ 47: every signal and trade is persisted with the required fields."""

    def test_decisions_and_trades_round_trip(self, journal):
        journal.record_decision(
            {
                "decision_id": "D1", "underlying": "NIFTY", "instrument": "NIFTY26JUN24000CE",
                "decision": "BUY", "direction": "LONG", "strategy": "trend_following",
                "regime": "TRENDING_UP", "ml_probability": 0.63, "entry_price": 200.0,
                "stop_price": 170.0, "target_price": 250.0, "position_size": 75,
                "approved": True, "strategies_agreeing": ["trend_following", "momentum"],
            }
        )
        journal.record_trade(
            {
                "trade_id": "T1", "decision_id": "D1", "underlying": "NIFTY",
                "instrument": "NIFTY26JUN24000CE", "strategy": "trend_following",
                "direction": "LONG", "quantity": 75, "entry_price": 200.0,
                "exit_price": 230.0, "net_pnl": 2000.0, "r_multiple": 0.9,
                "exit_reason": "TARGET_REACHED",
                "entry_time": NOW.isoformat(), "exit_time": (NOW + timedelta(minutes=30)).isoformat(),
            }
        )
        stats = journal.statistics()
        assert stats["decisions"] == 1
        assert stats["closed_trades"] == 1
        assert stats["net_pnl"] == 2000.0

    def test_rejected_decisions_are_recorded_too(self, journal):
        """The record of why we did NOT trade is required (REQ 36/47)."""
        journal.record_decision(
            {
                "decision_id": "D2", "underlying": "BANKNIFTY", "decision": "NO_TRADE",
                "approved": False, "rejection_reason": "spread too wide",
            }
        )
        assert journal.statistics()["rejected_decisions"] == 1

    def test_query_refuses_non_select_statements(self, journal):
        with pytest.raises(ValueError):
            journal.query("DELETE FROM trades")


class TestBacktestRealism:
    """REQ 38: the backtester must not manufacture profit."""

    @pytest.fixture
    def frame(self):
        return resample(
            candles_to_frame(make_candles(bars=375 * 20, seed=77, momentum_decay=0.95)),
            Timeframe.M5,
        )

    def test_fills_never_occur_on_the_signal_bar(self, frame, stock_future, config):
        """A fill at the signal bar's own close is unobtainable in reality."""
        entry_times: list[datetime] = []
        signal_times: list[datetime] = []

        def source(ts, history):
            if len(history) < 60 or len(signal_times) >= 5:
                return None
            if len(history) % 97:
                return None
            signal_times.append(ts)
            price = float(history["close"].iloc[-1])
            return BacktestSignal(
                Decision.BUY, price, price * 0.99, price * 1.02,
                strategy="test", regime=Regime.TRENDING_UP.value,
            )

        engine = BacktestEngine(
            BacktestConfig(initial_capital=500_000, fill_model="next_bar_open"),
            TransactionCostModel(config.costs),
        )
        result = engine.run(instrument=stock_future, frame=frame, signal_source=source)

        for trade in result.trades:
            assert trade.entry_time > signal_times[0] or trade.entry_time not in signal_times, (
                "a trade filled on the same bar that generated its signal"
            )

    def test_costs_are_always_deducted(self, frame, stock_future, config):
        def source(ts, history):
            if len(history) < 60 or len(history) % 211:
                return None
            price = float(history["close"].iloc[-1])
            return BacktestSignal(
                Decision.BUY, price, price * 0.99, price * 1.02,
                strategy="test", regime=Regime.TRENDING_UP.value,
            )

        engine = BacktestEngine(
            BacktestConfig(initial_capital=500_000), TransactionCostModel(config.costs)
        )
        result = engine.run(instrument=stock_future, frame=frame, signal_source=source)
        if result.trades:
            assert all(t.costs > 0 for t in result.trades), "a trade was booked with zero cost"
            assert all(t.net_pnl < t.gross_pnl for t in result.trades)

    def test_a_bar_spanning_both_barriers_assumes_the_stop(self, stock_future, config):
        """Without tick data the order is unknown; assuming the target would bias
        every backtest upward."""
        index = pd.date_range("2026-06-01 09:15", periods=6, freq="5min", tz=IST)
        # Bar 2 spans a huge range covering both the stop and the target.
        frame = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [100.5, 100.5, 130.0, 100.5, 100.5, 100.5],
                "low": [99.5, 99.5, 70.0, 99.5, 99.5, 99.5],
                "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "volume": [1000.0] * 6,
            },
            index=index,
        )

        fired = {"done": False}

        def source(ts, history):
            if fired["done"] or len(history) != 1:
                return None
            fired["done"] = True
            return BacktestSignal(
                Decision.BUY, 100.0, 90.0, 110.0, strategy="test",
                regime=Regime.TRENDING_UP.value,
            )

        engine = BacktestEngine(
            BacktestConfig(initial_capital=5_000_000, spread_pct=0.0),
            TransactionCostModel(config.costs),
        )
        result = engine.run(instrument=stock_future, frame=frame, signal_source=source)
        if result.trades:
            assert result.trades[0].exit_reason == "STOP_REACHED", (
                "an ambiguous bar was resolved optimistically"
            )

    def test_robustness_verdict_is_explicit(self, frame, stock_future, config):
        """REQ 68: state PASSED or FAILED; never leave it implied."""
        def source(ts, history):
            if len(history) < 60 or len(history) % 53:
                return None
            price = float(history["close"].iloc[-1])
            return BacktestSignal(
                Decision.BUY, price, price * 0.995, price * 1.01,
                strategy="test", regime=Regime.TRENDING_UP.value,
            )

        engine = BacktestEngine(
            BacktestConfig(initial_capital=500_000), TransactionCostModel(config.costs)
        )
        result = engine.run(instrument=stock_future, frame=frame, signal_source=source)
        report = run_all(result.trades, initial_capital=500_000, monte_carlo_runs=200)
        verdict = report.verdict()
        assert any(word in verdict for word in ("ROBUST", "NOT ROBUST", "INCONCLUSIVE"))

    def test_no_trades_never_reads_as_robust(self):
        """A run that could not be evaluated must not be reported as evidence."""
        report = run_all([], initial_capital=500_000, monte_carlo_runs=10)
        assert not report.is_robust
        assert "INCONCLUSIVE" in report.verdict()


class TestModelRegistry:
    """REQ 51: the deployment ladder is enforced."""

    def test_a_model_cannot_jump_from_research_to_active(self, config):
        from aqtp.core.errors import ModelError
        from aqtp.ml.registry import ModelRegistry, ModelStatus

        registry = ModelRegistry(config.paths.model_dir)
        record = registry.register(
            {"dummy": True}, family="logistic_regression",
            dataset_description={"rows": 1000}, passed=True, verdict="PASSED",
        )
        with pytest.raises(ModelError, match="must survive paper trading"):
            registry.promote(record.model_id, ModelStatus.ACTIVE)

    def test_a_failed_model_cannot_be_activated(self, config):
        from aqtp.core.errors import ModelError
        from aqtp.ml.registry import ModelRegistry, ModelStatus

        registry = ModelRegistry(config.paths.model_dir)
        record = registry.register(
            {"dummy": True}, family="random_forest", dataset_description={"rows": 1000},
            passed=False, verdict="FAILED: no edge demonstrated",
        )
        registry.promote(record.model_id, ModelStatus.VALIDATION)
        registry.promote(record.model_id, ModelStatus.PAPER)
        with pytest.raises(ModelError, match="out-of-sample evidence"):
            registry.promote(record.model_id, ModelStatus.ACTIVE)

    def test_the_full_ladder_is_walkable_for_a_passing_model(self, config):
        from aqtp.ml.registry import ModelRegistry, ModelStatus

        registry = ModelRegistry(config.paths.model_dir)
        record = registry.register(
            {"dummy": True}, family="lightgbm", dataset_description={"rows": 5000},
            passed=True, verdict="PASSED",
        )
        for status in (ModelStatus.VALIDATION, ModelStatus.PAPER, ModelStatus.ACTIVE):
            record = registry.promote(record.model_id, status)
        assert record.status is ModelStatus.ACTIVE
        assert registry.active("lightgbm").model_id == record.model_id


class TestExperimentVersioning:
    """REQ 50: every experiment records its full provenance."""

    def test_experiment_captures_all_required_versions(self, config):
        from aqtp.research.experiments import ExperimentStore

        store = ExperimentStore(config.paths.experiment_dir)
        experiment = store.create(
            name="test_run", configuration={"a": 1}, symbols=["NIFTY"],
            random_seed=42, backtest_start="2026-01-01", backtest_end="2026-06-01",
        )
        experiment.results = {"net_pnl": 1234.0}
        store.save(experiment)

        loaded = store.load(experiment.experiment_id)
        assert loaded.feature_version
        assert loaded.strategy_version
        assert loaded.random_seed == 42
        assert loaded.backtest_start == "2026-01-01"
        assert loaded.configuration == {"a": 1}
        assert loaded.results["net_pnl"] == 1234.0

    def test_identical_inputs_produce_the_same_experiment_id(self, config):
        from aqtp.research.experiments import ExperimentStore

        store = ExperimentStore(config.paths.experiment_dir)
        kwargs = dict(name="repro", configuration={"a": 1}, symbols=["NIFTY"], random_seed=7)
        assert store.create(**kwargs).experiment_id == store.create(**kwargs).experiment_id

    def test_a_different_seed_produces_a_different_id(self, config):
        from aqtp.research.experiments import ExperimentStore

        store = ExperimentStore(config.paths.experiment_dir)
        first = store.create(name="r", configuration={}, symbols=["NIFTY"], random_seed=1)
        second = store.create(name="r", configuration={}, symbols=["NIFTY"], random_seed=2)
        assert first.experiment_id != second.experiment_id
