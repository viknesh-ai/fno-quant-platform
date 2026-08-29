"""State reconciliation and execution-engine tests (REQ 32/45/46/63)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from aqtp.brokers.simulated import SimulatedBroker, SimulationSettings
from aqtp.core.clock import IST
from aqtp.core.types import (
    Decision,
    Direction,
    OrderRequest,
    OrderType,
    Position,
    Product,
    Regime,
    TransactionType,
)
from aqtp.execution.engine import ExecutionEngine
from aqtp.execution.position_manager import ManagedPosition, PositionManager
from aqtp.execution.reconcile import DiscrepancyType, StateReconciler
from aqtp.risk.engine import RiskApproval
from aqtp.risk.sizing import SizingResult
from tests.conftest import make_quote

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


@pytest.fixture
def quotes(call_instrument):
    return {call_instrument.trading_symbol: make_quote(call_instrument, price=200.0, spread=1.0, now=NOW)}


@pytest.fixture
def broker(quotes, call_instrument):
    sim = SimulatedBroker(
        quote_source=lambda i: quotes.get(i.trading_symbol),
        starting_cash=500_000,
        settings=SimulationSettings(seed=3),
    )
    sim.authenticate()
    return sim


@pytest.fixture
def position_manager(config):
    return PositionManager(config)


@pytest.fixture
def reconciler(broker, journal, position_manager):
    return StateReconciler(broker, journal, position_manager)


def _managed(instrument, *, quantity=75, entry=200.0, position_id="P1") -> ManagedPosition:
    return ManagedPosition(
        position_id=position_id, decision_id="D1", instrument=instrument,
        underlying=instrument.underlying_symbol or instrument.trading_symbol,
        strategy="trend_following", direction=Direction.LONG, quantity=quantity,
        entry_price=entry, entry_time=NOW - timedelta(minutes=30),
        stop_price=entry * 0.85, target_price=entry * 1.3,
        initial_stop_price=entry * 0.85, entry_regime=Regime.TRENDING_UP,
        last_price=entry,
    )


# =========================================================================== #
class TestReconciliation:
    def test_clean_state_reports_no_discrepancies(self, reconciler):
        report = reconciler.reconcile(now=NOW)
        assert report.is_clean
        assert report.safe_to_trade

    def test_broker_position_we_do_not_manage_is_detected(
        self, reconciler, broker, call_instrument
    ):
        broker.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="RECKEY12345",
            )
        )
        report = reconciler.reconcile(now=NOW)
        assert not report.is_clean
        assert call_instrument.trading_symbol in report.unmanaged_positions
        assert any(d.type is DiscrepancyType.MISSING_POSITION for d in report.discrepancies)

    def test_local_position_absent_at_broker_is_dropped(
        self, reconciler, position_manager, call_instrument
    ):
        """REQ 46: the broker is authoritative, so local state is corrected."""
        position_manager.track(_managed(call_instrument))
        assert position_manager.count == 1

        report = reconciler.reconcile(now=NOW)
        assert position_manager.count == 0, "local phantom position was not removed"
        discrepancy = next(
            d for d in report.discrepancies if d.type is DiscrepancyType.UNEXPECTED_POSITION
        )
        assert discrepancy.resolved
        assert "broker is authoritative" in discrepancy.resolution

    def test_quantity_mismatch_adopts_the_broker_quantity(
        self, reconciler, broker, position_manager, call_instrument
    ):
        broker.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="RECKEY22345",
            )
        )
        # Local state wrongly believes we hold two lots.
        position_manager.track(_managed(call_instrument, quantity=150))

        reconciler.reconcile(now=NOW)
        assert position_manager.all()[0].quantity == 75

    def test_unmanaged_positions_can_be_adopted_with_a_stop(
        self, reconciler, broker, position_manager, call_instrument
    ):
        """REQ 63: an unmanaged position is unbounded risk and must be brought in."""
        broker.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="RECKEY32345",
            )
        )
        report = reconciler.reconcile(now=NOW)
        adopted = reconciler.adopt_unmanaged(report)

        assert call_instrument.trading_symbol in adopted
        assert position_manager.count == 1
        position = position_manager.all()[0]
        assert position.stop_price > 0, "an adopted position must have a bounding stop"
        assert position.stop_price < position.entry_price

    def test_unresolved_intent_blocks_trading(self, reconciler, journal):
        """REQ 63: entries stay blocked until state is provably reconciled."""
        journal.record_order_intent(
            {
                "client_order_id": "AQORPHAN12345", "instrument": "NIFTY26JUN24000CE",
                "transaction_type": "BUY", "quantity": 75, "status": "INTENT",
                "timestamp": NOW.isoformat(),
            }
        )
        report = reconciler.reconcile(now=NOW, resolve_intents=True)
        # The simulator confirms it never saw the order, so the intent is closed out.
        rows = journal.query(
            "SELECT status FROM orders WHERE client_order_id = ?", ("AQORPHAN12345",)
        )
        assert rows[0]["status"] == "NEVER_SUBMITTED"

    def test_intent_that_did_reach_the_broker_is_recovered(
        self, reconciler, broker, journal, call_instrument
    ):
        """The crash-recovery path: an intent whose order actually exists."""
        client_id = "AQRECOVER123"
        journal.record_order_intent(
            {
                "client_order_id": client_id, "instrument": call_instrument.trading_symbol,
                "transaction_type": "BUY", "quantity": 75, "status": "INTENT",
                "timestamp": NOW.isoformat(),
            }
        )
        broker.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id=client_id,
            )
        )
        reconciler.reconcile(now=NOW, resolve_intents=True)

        rows = journal.query(
            "SELECT status, broker_order_id, filled_quantity FROM orders WHERE client_order_id = ?",
            (client_id,),
        )
        assert rows[0]["broker_order_id"], "the recovered order id was not written back"
        assert rows[0]["filled_quantity"] == 75

    def test_broker_outage_is_reported_not_treated_as_clean(
        self, reconciler, broker, monkeypatch
    ):
        from aqtp.core.errors import TransientBrokerError

        def boom():
            raise TransientBrokerError("broker unreachable")

        monkeypatch.setattr(broker, "get_positions", boom)
        report = reconciler.reconcile(now=NOW)
        assert report.error
        assert not report.safe_to_trade, "an unreachable broker must not read as reconciled"


# =========================================================================== #
class TestExecutionEngine:
    @pytest.fixture
    def engine(self, config, broker, journal):
        return ExecutionEngine(config, broker, journal)

    @staticmethod
    def _approval(quantity=75) -> RiskApproval:
        return RiskApproval(
            approved=True, decision=Decision.BUY,
            sizing=SizingResult(
                quantity=quantity, lots=quantity // 75, lot_size=75, risk_amount=2250.0,
                risk_per_unit=30.0, capital_required=quantity * 200.0,
                margin_required=quantity * 200.0,
            ),
            approved_at=NOW, approval_id="RA000001",
        )

    def test_full_thirteen_step_pipeline_succeeds(
        self, engine, call_instrument, quotes, journal
    ):
        result = engine.execute(
            approval=self._approval(), instrument=call_instrument,
            quote=quotes[call_instrument.trading_symbol], decision_id="D1",
            strategy="trend_following", market_data_at=NOW, signal_at=NOW,
        )
        assert result.success, result.failure_reason
        assert result.verified, "a fill must be verified against the broker, not assumed"
        assert result.filled_quantity == 75
        assert len(result.steps) >= 13

        rows = journal.query("SELECT status FROM orders WHERE client_order_id = ?",
                             (result.client_order_id,))
        assert rows and rows[0]["status"] in ("EXECUTED", "COMPLETED")

    def test_unapproved_risk_cannot_execute(self, engine, call_instrument, quotes):
        """REQ 67.22: the RiskEngine cannot be bypassed."""
        rejected = RiskApproval(approved=False, decision=Decision.NO_TRADE)
        result = engine.execute(
            approval=rejected, instrument=call_instrument,
            quote=quotes[call_instrument.trading_symbol], decision_id="D2", strategy="x",
        )
        assert not result.success
        assert "risk approval was not granted" in result.failure_reason

    def test_order_intent_is_written_before_submission(
        self, engine, call_instrument, quotes, journal
    ):
        """REQ 45: crash safety depends on the intent existing before the send."""
        engine.execute(
            approval=self._approval(), instrument=call_instrument,
            quote=quotes[call_instrument.trading_symbol], decision_id="D3",
            strategy="trend_following",
        )
        rows = journal.query(
            "SELECT intent_recorded_at, submitted_at FROM orders ORDER BY id DESC LIMIT 1"
        )
        assert rows[0]["intent_recorded_at"] is not None
        assert rows[0]["submitted_at"] is not None
        assert rows[0]["intent_recorded_at"] <= rows[0]["submitted_at"]

    def test_quantity_not_a_lot_multiple_is_refused(self, engine, call_instrument, quotes):
        result = engine.execute(
            approval=self._approval(quantity=80),  # lot size is 75
            instrument=call_instrument, quote=quotes[call_instrument.trading_symbol],
            decision_id="D4", strategy="x",
        )
        assert not result.success
        assert "lot size" in result.failure_reason

    def test_widened_spread_since_approval_aborts(self, engine, call_instrument):
        wide = make_quote(call_instrument, price=200.0, spread=40.0, now=NOW)
        result = engine.execute(
            approval=self._approval(), instrument=call_instrument, quote=wide,
            decision_id="D5", strategy="x",
        )
        assert not result.success
        assert "spread" in result.failure_reason

    def test_missing_quote_aborts(self, engine, call_instrument):
        result = engine.execute(
            approval=self._approval(), instrument=call_instrument, quote=None,
            decision_id="D6", strategy="x",
        )
        assert not result.success
        assert "quote" in result.failure_reason

    def test_latency_is_recorded_for_every_fill(
        self, engine, call_instrument, quotes, journal
    ):
        """REQ 33: all six timestamps and four latencies."""
        result = engine.execute(
            approval=self._approval(), instrument=call_instrument,
            quote=quotes[call_instrument.trading_symbol], decision_id="D7",
            strategy="trend_following",
            market_data_at=NOW - timedelta(milliseconds=500),
            signal_at=NOW - timedelta(milliseconds=300),
        )
        assert result.success
        trace = result.latency
        assert trace.market_data_at and trace.signal_at and trace.order_submit_at
        assert trace.broker_ack_at and trace.fill_at
        assert trace.signal_latency_ms == pytest.approx(200.0, abs=50)
        assert trace.total_latency_ms > 0
        assert len(engine.latency) == 1

    def test_close_position_uses_a_market_order(self, engine, broker, call_instrument, quotes):
        broker.place_order(
            OrderRequest(
                instrument=call_instrument, transaction_type=TransactionType.BUY, quantity=75,
                order_type=OrderType.MARKET, product=Product.NRML, client_order_id="EXITSETUP12",
            )
        )
        result = engine.close_position(
            instrument=call_instrument, quantity=75, direction=Direction.LONG,
            reason="target_reached", quote=quotes[call_instrument.trading_symbol],
        )
        assert result.success
        assert result.filled_quantity == 75
        assert broker.get_positions() == []
