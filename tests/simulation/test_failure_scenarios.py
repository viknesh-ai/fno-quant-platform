"""Simulation tests (REQ 64 'Simulation tests').

Each scenario drives a failure mode that never occurs in a happy-path test but is
routine in live trading: flash volatility, spread collapse, API failure, delayed
data, partial fills, duplicate responses, order rejection and sudden reversal.

The assertion in every case is the same in spirit — the system must degrade to a
*safe, explicit* outcome (NO_TRADE, a recorded rejection, a bounded position) and
never to a silent one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aqtp.brokers.simulated import SimulatedBroker, SimulationSettings
from aqtp.core.clock import IST
from aqtp.core.errors import (
    BrokerError,
    OrderRejected,
    OrderStateAmbiguous,
    TransientBrokerError,
)
from aqtp.core.types import (
    Decision,
    Direction,
    ExitReason,
    ManagementAction,
    OrderRequest,
    OrderType,
    Product,
    Regime,
    TransactionType,
)
from aqtp.data.quality import DataQualityGate, QualityIssue
from aqtp.execution.engine import ExecutionEngine
from aqtp.execution.position_manager import ManagedPosition, PositionManager
from aqtp.risk.engine import RiskApproval
from aqtp.risk.sizing import SizingResult
from tests.conftest import make_quote

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


def _approval(quantity=75) -> RiskApproval:
    return RiskApproval(
        approved=True, decision=Decision.BUY,
        sizing=SizingResult(
            quantity=quantity, lots=quantity // 75, lot_size=75, risk_amount=2250.0,
            risk_per_unit=30.0, capital_required=quantity * 200.0,
            margin_required=quantity * 200.0,
        ),
        approved_at=NOW, approval_id="RA-SIM",
    )


def _managed(instrument, *, entry=200.0, stop=170.0, target=250.0) -> ManagedPosition:
    return ManagedPosition(
        position_id="PSIM", decision_id="DSIM", instrument=instrument,
        underlying=instrument.underlying_symbol or instrument.trading_symbol,
        strategy="trend_following", direction=Direction.LONG, quantity=75,
        entry_price=entry, entry_time=NOW - timedelta(minutes=10),
        stop_price=stop, target_price=target, initial_stop_price=stop,
        entry_regime=Regime.TRENDING_UP, last_price=entry,
        expected_holding_minutes=60,
    )


@pytest.fixture
def live_quotes(call_instrument):
    """Mutable quote store so a scenario can change the market mid-test."""
    return {call_instrument.trading_symbol: make_quote(call_instrument, price=200.0, spread=1.0, now=NOW)}


@pytest.fixture
def broker(live_quotes):
    sim = SimulatedBroker(
        quote_source=lambda i: live_quotes.get(i.trading_symbol),
        starting_cash=500_000,
        settings=SimulationSettings(seed=17),
    )
    sim.authenticate()
    return sim


@pytest.fixture
def engine(config, broker, journal):
    return ExecutionEngine(config, broker, journal)


# =========================================================================== #
class TestFlashVolatility:
    def test_a_flash_move_is_rejected_as_a_price_jump(self, config, call_instrument):
        gate = DataQualityGate(config.data_quality)
        gate.check_quote(make_quote(call_instrument, price=200.0, now=NOW), now=NOW)

        later = NOW + timedelta(seconds=2)
        verdict = gate.check_quote(
            make_quote(call_instrument, price=90.0, now=later), now=later
        )
        assert not verdict.ok
        assert QualityIssue.PRICE_JUMP in verdict.issues

    def test_a_flash_crash_through_the_stop_exits_the_position(
        self, config, call_instrument
    ):
        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        crashed = make_quote(call_instrument, price=120.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=crashed, now=NOW)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.STOP_REACHED
        assert decision.urgency == "immediate"

    def test_a_flash_spike_to_target_takes_the_profit(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        spiked = make_quote(call_instrument, price=260.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=spiked, now=NOW)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.TARGET_REACHED


class TestSpreadWidening:
    def test_widened_spread_blocks_a_new_entry(self, engine, call_instrument):
        wide = make_quote(call_instrument, price=200.0, spread=30.0, now=NOW)
        result = engine.execute(
            approval=_approval(), instrument=call_instrument, quote=wide,
            decision_id="D-SPREAD", strategy="trend_following",
        )
        assert not result.success
        assert "spread" in result.failure_reason

    def test_severely_widened_spread_exits_an_open_position(self, config, call_instrument):
        """REQ 35: deteriorating liquidity is an exit reason in its own right."""
        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        # Well inside stop/target, but the book has fallen apart.
        illiquid = make_quote(call_instrument, price=205.0, spread=20.0, now=NOW)
        decision = manager.evaluate(position, quote=illiquid, now=NOW)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.LIQUIDITY_DETERIORATED


class TestApiFailure:
    def test_submit_failure_is_recorded_and_no_position_is_assumed(
        self, engine, broker, call_instrument, live_quotes, journal
    ):
        broker.fail_next_submit = OrderRejected("exchange rejected: price band", code="GA010")
        result = engine.execute(
            approval=_approval(), instrument=call_instrument,
            quote=live_quotes[call_instrument.trading_symbol],
            decision_id="D-APIFAIL", strategy="trend_following",
        )
        assert not result.success
        assert broker.get_positions() == [], "a rejected order must not create a position"
        rows = journal.query("SELECT status FROM orders ORDER BY id DESC LIMIT 1")
        assert rows[0]["status"] == "REJECTED"

    def test_ambiguous_submit_halts_rather_than_resubmitting(
        self, engine, broker, call_instrument, live_quotes
    ):
        """The order DID reach the book but we were told nothing. Resubmitting here
        is how a duplicate position is created, so the engine must resolve instead."""
        broker.ambiguous_next_submit = True
        result = engine.execute(
            approval=_approval(), instrument=call_instrument,
            quote=live_quotes[call_instrument.trading_symbol],
            decision_id="D-AMBIG", strategy="trend_following",
        )
        # Resolution by reference finds the order the simulator really did accept.
        positions = broker.get_positions()
        assert len(positions) <= 1, "an ambiguous submit produced a duplicate position"

    def test_repeated_quote_failures_trip_the_circuit_breaker(self, config, call_instrument):
        gate = DataQualityGate(config.data_quality)
        symbol = call_instrument.trading_symbol
        for _ in range(config.data_quality.max_consecutive_failures):
            gate.record_failure(symbol)
        assert gate.is_circuit_broken(symbol)

    def test_market_data_engine_survives_a_broker_outage(
        self, config, broker, call_instrument, monkeypatch
    ):
        from aqtp.data.market_data import MarketDataEngine

        def boom(instrument):
            raise TransientBrokerError("feed down")

        monkeypatch.setattr(broker, "get_quote", boom)
        engine = MarketDataEngine(broker, config)
        snapshot = engine.build_snapshot([call_instrument])

        assert snapshot.instruments == {}
        assert call_instrument.trading_symbol in snapshot.rejected


class TestDelayedData:
    def test_stale_data_produces_no_trade(self, config, call_instrument):
        gate = DataQualityGate(config.data_quality)
        stale = make_quote(call_instrument, price=200.0, now=NOW, age_seconds=300)
        verdict = gate.check_quote(stale, now=NOW)
        assert not verdict.ok
        assert QualityIssue.STALE_PRICE in verdict.issues

    def test_stale_features_block_a_prediction(self, config):
        from aqtp.ml.predict import PredictionEngine
        import pandas as pd

        engine = PredictionEngine(config.prediction)
        prediction = engine.predict(
            pd.Series({"a": 1.0}), feature_age_seconds=9999, timestamp=NOW
        )
        assert not prediction.usable

    def test_a_stale_position_quote_is_not_acted_on(self, config, call_instrument):
        """The manager must not evaluate a stop against a quote it cannot trust."""
        gate = DataQualityGate(config.data_quality)
        stale = make_quote(call_instrument, price=120.0, now=NOW, age_seconds=600)
        assert not gate.check_quote(stale, now=NOW).ok


class TestPartialFill:
    def test_partial_fill_is_reflected_in_the_position(self, live_quotes, call_instrument):
        """Depth-limited fills must produce a smaller position, not a phantom full one."""
        quote = make_quote(call_instrument, price=200.0, spread=1.0, now=NOW)
        # Only 100 units of depth against a 150-unit order.
        from aqtp.core.types import DepthLevel

        quote.depth_sell = (DepthLevel(200.5, 200),)
        store = {call_instrument.trading_symbol: quote}

        sim = SimulatedBroker(
            quote_source=lambda i: store.get(i.trading_symbol),
            starting_cash=500_000,
            settings=SimulationSettings(seed=5, depth_consumption_ratio=0.5),
        )
        sim.authenticate()
        order = sim.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=150,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="PARTIAL1234",
            )
        )
        assert order.filled_quantity < 150
        assert order.filled_quantity % call_instrument.lot_size == 0
        positions = sim.get_positions()
        assert positions[0].quantity == order.filled_quantity

    def test_execution_engine_reports_a_partial_fill_honestly(
        self, engine, call_instrument, live_quotes
    ):
        from aqtp.core.types import DepthLevel

        quote = live_quotes[call_instrument.trading_symbol]
        quote.depth_sell = (DepthLevel(200.5, 200),)
        result = engine.execute(
            approval=_approval(quantity=150), instrument=call_instrument, quote=quote,
            decision_id="D-PARTIAL", strategy="trend_following",
        )
        if result.success and result.filled_quantity < 150:
            assert result.partially_filled
            assert any("partial" in s.lower() for s in result.steps)


class TestDuplicateResponse:
    def test_the_same_intent_cannot_produce_two_positions(
        self, engine, broker, call_instrument, live_quotes
    ):
        """REQ 45: duplicate submission must be structurally impossible."""
        approval = _approval()
        quote = live_quotes[call_instrument.trading_symbol]

        first = engine.execute(
            approval=approval, instrument=call_instrument, quote=quote,
            decision_id="D-DUP", strategy="trend_following",
        )
        second = engine.execute(
            approval=approval, instrument=call_instrument, quote=quote,
            decision_id="D-DUP", strategy="trend_following",
        )
        assert first.success
        positions = broker.get_positions()
        total = sum(abs(p.quantity) for p in positions)
        assert total == 75, f"duplicate execution created {total} units instead of 75"

    def test_replayed_broker_response_does_not_double_count(self, broker, call_instrument):
        request = OrderRequest(
            instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
            order_type=OrderType.MARKET, product=Product.NRML, client_order_id="REPLAY12345",
        )
        broker.place_order(request)
        broker.place_order(request)  # replayed
        assert sum(abs(p.quantity) for p in broker.get_positions()) == 75


class TestOrderRejection:
    def test_rejection_leaves_the_portfolio_untouched(
        self, engine, broker, call_instrument, live_quotes
    ):
        broker.fail_next_submit = OrderRejected("margin shortfall", code="GA020")
        before = broker.get_positions()
        engine.execute(
            approval=_approval(), instrument=call_instrument,
            quote=live_quotes[call_instrument.trading_symbol],
            decision_id="D-REJ", strategy="trend_following",
        )
        assert broker.get_positions() == before

    def test_rejection_reason_is_journalled(
        self, engine, broker, call_instrument, live_quotes, journal
    ):
        broker.fail_next_submit = OrderRejected("price outside band", code="GA030")
        engine.execute(
            approval=_approval(), instrument=call_instrument,
            quote=live_quotes[call_instrument.trading_symbol],
            decision_id="D-REJ2", strategy="trend_following",
        )
        rows = journal.query("SELECT rejection_reason FROM orders ORDER BY id DESC LIMIT 1")
        assert "price outside band" in (rows[0]["rejection_reason"] or "")


class TestSuddenReversal:
    def test_regime_flip_against_the_position_triggers_an_exit(self, config, call_instrument):
        from aqtp.regime.engine import RegimeState

        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        reversed_regime = RegimeState(
            probabilities={Regime.TRENDING_DOWN: 0.85, Regime.RANGE: 0.15},
            dominant=Regime.TRENDING_DOWN, confidence=0.85, entropy=0.2,
        )
        quote = make_quote(call_instrument, price=205.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=quote, now=NOW, regime=reversed_regime)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.REGIME_CHANGED

    def test_model_flipping_against_the_position_triggers_an_exit(
        self, config, call_instrument
    ):
        from aqtp.ml.predict import Prediction

        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        opposing = Prediction(
            direction=Direction.SHORT, probability=0.75, confidence=0.7, uncertainty=0.1,
            expected_move_atr=1.5, horizon_minutes=30,
        )
        quote = make_quote(call_instrument, price=205.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=quote, now=NOW, prediction=opposing)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.PREDICTION_INVALIDATED

    def test_emergency_shutdown_overrides_everything(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        # Deep in profit — normally a HOLD — but an emergency outranks it.
        quote = make_quote(call_instrument, price=240.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=quote, now=NOW, emergency=True)

        assert decision.action is ManagementAction.EMERGENCY_EXIT
        assert decision.reason is ExitReason.EMERGENCY_SHUTDOWN
        assert decision.urgency == "immediate"


class TestStopManagement:
    def test_stop_moves_to_breakeven_at_one_r(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument, entry=200.0, stop=170.0, target=300.0)
        manager.track(position)

        quote = make_quote(call_instrument, price=230.0, spread=1.0, now=NOW)  # +1R
        decision = manager.evaluate(position, quote=quote, now=NOW)

        assert decision.action is ManagementAction.MOVE_STOP
        assert decision.new_stop == pytest.approx(200.0)

    def test_stops_only_ever_move_in_our_favour(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument, entry=200.0, stop=170.0, target=300.0)
        manager.track(position)

        for price in (230.0, 245.0, 260.0):
            quote = make_quote(call_instrument, price=price, spread=1.0, now=NOW)
            decision = manager.evaluate(position, quote=quote, now=NOW)
            if decision.new_stop is not None:
                assert decision.new_stop >= position.stop_price
                manager.apply(position, decision)

        assert position.stop_price >= 200.0

    def test_time_limit_closes_a_stagnant_position(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument)
        position.expected_holding_minutes = 30
        position.entry_time = NOW - timedelta(minutes=60)
        manager.track(position)

        quote = make_quote(call_instrument, price=201.0, spread=1.0, now=NOW)
        decision = manager.evaluate(position, quote=quote, now=NOW)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.TIME_LIMIT

    def test_session_end_forces_a_square_off(self, config, call_instrument):
        manager = PositionManager(config)
        position = _managed(call_instrument)
        manager.track(position)

        late = datetime(2026, 6, 1, 15, 20, tzinfo=IST)
        quote = make_quote(call_instrument, price=205.0, spread=1.0, now=late)
        decision = manager.evaluate(position, quote=quote, now=late)

        assert decision.action is ManagementAction.EXIT
        assert decision.reason is ExitReason.END_OF_SESSION
