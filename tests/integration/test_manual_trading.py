"""Operator-initiated trading (`aqtp trade ...`).

The property that matters most here is that a manual order is not a back door.
It is sized by the sizer, checked by the risk engine, journalled with a decision
id and tracked by the position manager — and `--force` narrows exactly one of
those, never the kill switches.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from aqtp.brokers.simulated import SimulatedBroker, SimulationSettings
from aqtp.core.clock import IST, SimulatedClock
from aqtp.core.types import (
    Exchange,
    Instrument,
    InstrumentType,
    Quote,
    Segment,
    TransactionType,
)
from aqtp.execution.manual import ManualTradeError, ManualTrader
from aqtp.journal.store import TradeJournal
from aqtp.orchestrator import TradingOrchestrator
from aqtp.risk.killswitch import KillScope

from tests.conftest import make_quote

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)
EXPIRY = date(2026, 6, 25)


# --------------------------------------------------------------------------- #
# A small, complete instrument master
# --------------------------------------------------------------------------- #
def _instruments() -> list[Instrument]:
    out = [
        Instrument(
            trading_symbol="NIFTY", exchange=Exchange.NSE, segment=Segment.CASH,
            instrument_type=InstrumentType.IDX, name="NIFTY 50",
            underlying_symbol="NIFTY", lot_size=1,
        ),
        Instrument(
            trading_symbol="NIFTY26JUNFUT", exchange=Exchange.NSE, segment=Segment.FNO,
            instrument_type=InstrumentType.FUT, underlying_symbol="NIFTY",
            lot_size=75, expiry_date=EXPIRY,
        ),
    ]
    for strike in range(23500, 24600, 100):
        for option_type in (InstrumentType.CE, InstrumentType.PE):
            out.append(
                Instrument(
                    trading_symbol=f"NIFTY26JUN{strike}{option_type.value}",
                    exchange=Exchange.NSE, segment=Segment.FNO,
                    instrument_type=option_type, underlying_symbol="NIFTY",
                    lot_size=75, tick_size=0.05, freeze_quantity=1800,
                    expiry_date=EXPIRY, strike_price=float(strike),
                )
            )
    return out


def _price_for(instrument: Instrument) -> float:
    """A plausible price: 24,000 for the index, intrinsic + 120 for the options."""
    if instrument.instrument_type is InstrumentType.IDX:
        return 24_000.0
    if instrument.instrument_type is InstrumentType.FUT:
        return 24_050.0
    strike = instrument.strike_price or 24_000.0
    intrinsic = (
        max(0.0, 24_000.0 - strike) if instrument.instrument_type is InstrumentType.CE
        else max(0.0, strike - 24_000.0)
    )
    return round(intrinsic + 120.0, 2)


@pytest.fixture
def live_session(config, tmp_path):
    """A started orchestrator wired end to end against the simulator."""
    instruments = _instruments()

    def quote_source(instrument: Instrument) -> Quote:
        return make_quote(
            instrument,
            price=_price_for(instrument),
            spread=max(0.05, _price_for(instrument) * 0.002),
            volume=200_000,
            open_interest=500_000,
            now=NOW,
        )

    broker = SimulatedBroker(
        quote_source=quote_source,
        starting_cash=config.capital.available_capital,
        settings=SimulationSettings(latency_ms=0, seed=7),
        instruments=instruments,
    )
    journal = TradeJournal(tmp_path / "manual.sqlite", mode=config.mode.value)
    # A fixed clock: the instrument master below expires in June 2026, and the
    # universe filter drops anything already expired.
    orchestrator = TradingOrchestrator(
        config, data_broker=broker, execution_broker=broker, journal=journal,
        clock=SimulatedClock(current=NOW),
    )
    assert orchestrator.start(warm_up_days=0)
    return orchestrator, ManualTrader(orchestrator)


# =========================================================================== #
# Resolution
# =========================================================================== #
class TestSymbolResolution:
    def test_an_exact_trading_symbol_resolves(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY26JUN24000CE")
        assert contract.instrument.trading_symbol == "NIFTY26JUN24000CE"
        assert contract.quote is not None

    def test_an_underlying_plus_option_resolves_to_atm_by_default(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        assert contract.instrument.instrument_type is InstrumentType.CE
        # Spot is 24,000, so ATM is the 24,000 strike.
        assert contract.instrument.strike_price == pytest.approx(24_000.0)
        assert any("ATM" in note for note in contract.notes)

    def test_an_explicit_strike_is_honoured(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="PE", strike=23_800.0)
        assert contract.instrument.strike_price == pytest.approx(23_800.0)
        assert contract.instrument.instrument_type is InstrumentType.PE

    def test_an_unlisted_strike_snaps_to_the_nearest_and_says_so(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE", strike=23_977.0)
        assert contract.instrument.strike_price == pytest.approx(24_000.0)
        assert any("nearest available" in note for note in contract.notes)

    def test_omitting_the_option_type_resolves_to_the_future(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY")
        assert contract.instrument.instrument_type is InstrumentType.FUT

    def test_an_unknown_symbol_is_a_clear_error(self, live_session):
        _, trader = live_session
        with pytest.raises(ManualTradeError, match="neither a tradable symbol"):
            trader.resolve("NOTASYMBOL")

    def test_a_bad_option_type_is_a_clear_error(self, live_session):
        _, trader = live_session
        with pytest.raises(ManualTradeError, match="CE or PE"):
            trader.resolve("NIFTY", option_type="CALL")

    def test_an_untradable_expiry_lists_the_alternatives(self, live_session):
        _, trader = live_session
        with pytest.raises(ManualTradeError, match="not a tradable expiry"):
            trader.resolve("NIFTY", option_type="CE", expiry=date(2027, 1, 1))


# =========================================================================== #
# Entry
# =========================================================================== #
class TestManualEntry:
    def test_a_sound_manual_buy_fills_and_is_tracked(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=1)

        assert result.success, result.reason
        assert result.quantity == 75
        assert result.position_id
        assert orchestrator.positions.count == 1

        position = orchestrator.positions.all()[0]
        assert position.strategy == "manual"
        assert position.stop_price < position.entry_price < position.target_price

    def test_the_order_is_journalled_with_a_decision_id(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=1)
        assert result.success

        rows = orchestrator.journal.query(
            "SELECT decision_id, strategy, instrument FROM trades WHERE trade_id = ?",
            (result.position_id,),
        )
        assert rows and rows[0]["decision_id"] == result.decision_id
        assert rows[0]["strategy"] == "manual"

    def test_an_oversized_request_is_refused_not_silently_trimmed(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=500)
        assert not result.success
        assert "risk sizing allows" in result.reason

    def test_explicit_stops_and_targets_are_used_verbatim(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=1, stop_price=80.0, target_price=200.0)
        assert result.success
        position = orchestrator.positions.all()[0]
        assert position.stop_price == pytest.approx(80.0)
        assert position.target_price == pytest.approx(200.0)

    def test_a_kill_switch_blocks_a_manual_order(self, live_session):
        orchestrator, trader = live_session
        orchestrator.kill_switches.engage(
            KillScope.APPLICATION, reason="test", engaged_by="test"
        )
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=1)
        assert not result.success
        assert "kill switch" in result.reason

    def test_force_does_not_bypass_a_kill_switch(self, live_session):
        """--force narrows the sizing checks. It is not a master key."""
        orchestrator, trader = live_session
        orchestrator.kill_switches.engage(
            KillScope.APPLICATION, reason="test", engaged_by="test"
        )
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=1, force=True)
        assert not result.success
        assert "kill switch" in result.reason

    def test_force_requires_an_explicit_size(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        # Force with no size: there is nothing to override *to*.
        result = trader._enter(
            contract, TransactionType.BUY, force=True,
            stop_price=1e9,   # guarantees the risk engine rejects it
        )
        assert not result.success
        assert "--force requires" in result.reason

    def test_force_is_journalled_as_a_forced_order(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        result = trader.buy(contract, lots=20, force=True)
        assert result.success, result.reason
        assert result.quantity == 20 * 75

        rows = orchestrator.journal.query(
            "SELECT message FROM events WHERE category = 'manual' ORDER BY id DESC LIMIT 1"
        )
        assert rows and "forced" in rows[0]["message"]

    def test_force_does_not_bypass_the_exchange_freeze_limit(self, live_session):
        """--force overrides *our* sizing, never the exchange's own limits."""
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        assert contract.instrument.freeze_quantity == 1800

        result = trader.buy(contract, lots=200, force=True)   # 15,000 units
        assert not result.success
        assert "freeze" in result.reason

    def test_a_quote_free_instrument_is_refused(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        contract.quote = None
        result = trader.buy(contract, lots=1)
        assert not result.success
        assert "cannot price" in result.reason


# =========================================================================== #
# Exit
# =========================================================================== #
class TestManualExit:
    def test_exit_closes_and_untracks_the_position(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        entry = trader.buy(contract, lots=1)
        assert entry.success

        result = trader.exit_position(entry.position_id)
        assert result.success, result.reason
        assert orchestrator.positions.count == 0

    def test_a_partial_exit_leaves_the_remainder_open(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        entry = trader.buy(contract, lots=2, force=True)
        assert entry.success and entry.quantity == 150

        result = trader.exit_position(entry.position_id, quantity=75)
        assert result.success, result.reason
        assert orchestrator.positions.count == 1
        assert orchestrator.positions.all()[0].quantity == 75

    def test_exiting_an_unknown_position_is_a_clear_error(self, live_session):
        _, trader = live_session
        result = trader.exit_position("P-nope")
        assert not result.success
        assert "no tracked position" in result.reason

    def test_an_out_of_range_quantity_is_refused(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        entry = trader.buy(contract, lots=1)
        result = trader.exit_position(entry.position_id, quantity=9_999)
        assert not result.success
        assert "between 1 and" in result.reason

    def test_close_all_flattens_everything(self, live_session):
        orchestrator, trader = live_session
        for strike in (23_900.0, 24_000.0):
            contract = trader.resolve("NIFTY", option_type="CE", strike=strike)
            assert trader.buy(contract, lots=1).success
        assert orchestrator.positions.count == 2

        results = trader.close_all()
        assert len(results) == 2
        assert all(r.success for r in results)
        assert orchestrator.positions.count == 0


# =========================================================================== #
# Level management
# =========================================================================== #
class TestLevelManagement:
    def test_set_stop_moves_the_managed_level(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        entry = trader.buy(contract, lots=1)
        result = trader.set_stop(entry.position_id, stop_price=95.0, target_price=250.0)
        assert result.success

        position = orchestrator.positions.all()[0]
        assert position.stop_price == pytest.approx(95.0)
        assert position.target_price == pytest.approx(250.0)

    def test_set_stop_needs_something_to_set(self, live_session):
        _, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        entry = trader.buy(contract, lots=1)
        result = trader.set_stop(entry.position_id)
        assert not result.success
        assert "--stop" in result.reason


# =========================================================================== #
# Order management
# =========================================================================== #
class TestOrderManagement:
    def test_cancelling_an_unknown_order_is_a_clear_error(self, live_session):
        _, trader = live_session
        result = trader.cancel("NOPE")
        assert not result.success
        assert "no order" in result.reason

    def test_modifying_with_nothing_to_change_is_refused(self, live_session):
        orchestrator, trader = live_session
        contract = trader.resolve("NIFTY", option_type="CE")
        trader.buy(contract, lots=1)
        orders = orchestrator.execution_broker.list_orders()
        assert orders
        result = trader.modify(orders[0].broker_order_id)
        assert not result.success
        # Either "not open" (it already filled) or "nothing to modify".
        assert "not open" in result.reason or "nothing to modify" in result.reason

    def test_cancel_all_on_an_empty_book_is_harmless(self, live_session):
        _, trader = live_session
        assert trader.cancel_all() == []
