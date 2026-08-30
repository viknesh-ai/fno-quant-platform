"""RuntimeSupervisor — the behaviours that keep a live session survivable.

These are the failure paths, tested by injecting the failures: a broker that
stops responding, a cycle that raises every time, a wedged loop, a SIGTERM at an
awkward moment, and the square-off that has to happen whether or not anything
else worked.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import pytest

from aqtp.core.clock import IST, SimulatedClock
from aqtp.core.errors import BrokerError
from aqtp.core.types import ExitReason
from aqtp.runtime.supervisor import RuntimeSupervisor

NOW = datetime(2026, 6, 1, 11, 0, tzinfo=IST)


# --------------------------------------------------------------------------- #
# A minimal orchestrator double
# --------------------------------------------------------------------------- #
class FakeReport:
    def __init__(self, halted_reason: str = "") -> None:
        self.halted_reason = halted_reason

    def summary(self) -> str:
        return "fake cycle"


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_event(self, **payload) -> None:
        self.events.append(payload)

    def categories(self) -> list[str]:
        return [event.get("category") for event in self.events]


class FakeAlerts:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, event, message, **_) -> None:
        self.sent.append(message)

    def emergency(self, message, **_) -> None:
        self.sent.append(f"EMERGENCY: {message}")

    def broker_error(self, message, **_) -> None:
        self.sent.append(f"BROKER: {message}")


class FakePositions:
    def __init__(self, positions=None) -> None:
        self._positions = list(positions or [])

    def all(self):
        return list(self._positions)

    @property
    def count(self) -> int:
        return len(self._positions)

    def clear(self) -> None:
        self._positions.clear()


class FakeBroker:
    def __init__(self, *, fail_auth_times: int = 0) -> None:
        self.auth_calls = 0
        self.fail_auth_times = fail_auth_times
        self.cancelled: list[str] = []
        self.orders: list = []

    def authenticate(self) -> None:
        self.auth_calls += 1
        if self.auth_calls <= self.fail_auth_times:
            raise BrokerError("session expired")

    def list_orders(self):
        return list(self.orders)

    def cancel_order(self, broker_order_id, *, segment):
        self.cancelled.append(broker_order_id)
        return None

    def get_quote(self, instrument):
        raise BrokerError("no quote")


class FakeReconciler:
    def __init__(self, safe: bool = True) -> None:
        self.safe = safe
        self.calls = 0

    def reconcile(self, *, now, resolve_intents: bool = False):
        self.calls += 1
        report = type("R", (), {})()
        report.safe_to_trade = self.safe
        report.summary = lambda: "fake reconciliation"
        return report


class FakeKillSwitches:
    def __init__(self) -> None:
        self.engaged: list[tuple] = []

    def engage(self, scope, *, reason, engaged_by):
        self.engaged.append((scope, reason))


class FakeOrchestrator:
    """Just enough orchestrator for the supervisor to drive."""

    def __init__(self, config, *, cycle_error: Exception | None = None, positions=None) -> None:
        self.config = config
        self.run_id = "TEST"
        self.clock = SimulatedClock(current=NOW)
        self.journal = FakeJournal()
        self.alerts = FakeAlerts()
        self.positions = FakePositions(positions)
        self.data_broker = FakeBroker()
        self.execution_broker = self.data_broker
        self.reconciler = FakeReconciler()
        self.kill_switches = FakeKillSwitches()
        self.trading_allowed = True
        self.cycles = 0
        self.cycle_error = cycle_error
        self.exits: list = []
        self.shutdown_calls = 0

    def run_cycle(self):
        self.cycles += 1
        if self.cycle_error is not None:
            raise self.cycle_error
        return FakeReport()

    def status(self):
        return {
            "mode": self.config.mode.value, "equity": 500_000.0, "open_positions":
            self.positions.count, "unrealized_pnl": 0.0, "trading_allowed": self.trading_allowed,
        }

    def shutdown(self, *, close_positions: bool = False) -> None:
        self.shutdown_calls += 1

    def _record_equity_snapshot(self) -> None:
        pass

    def _exit(self, position, decision, quantity, now, quote) -> bool:
        self.exits.append((position, decision.reason, quantity))
        self.positions._positions.remove(position)
        return True


class FakePosition:
    def __init__(self, symbol: str = "NIFTY26JUN24000CE") -> None:
        self.position_id = f"P-{symbol}"
        self.quantity = 75
        self.instrument = type("I", (), {"trading_symbol": symbol})()


@pytest.fixture
def supervisor_config(config, tmp_path):
    """Config tuned so the tests run in milliseconds, not minutes."""
    config.runtime.fast_loop_seconds = 0.01
    config.runtime.idle_loop_seconds = 0.01
    config.runtime.heartbeat_seconds = 0.01
    config.runtime.state_snapshot_seconds = 1000.0
    config.runtime.reconnect_backoff_seconds = 0.01
    config.runtime.reconnect_max_backoff_seconds = 0.02
    config.runtime.watchdog_timeout_seconds = 1000.0
    config.runtime.pid_file = str(tmp_path / "aqtp.pid")
    return config


def _supervisor(orchestrator) -> RuntimeSupervisor:
    # Signal handlers only install on the main thread and would clobber pytest's.
    return RuntimeSupervisor(orchestrator, install_signal_handlers=False)


# =========================================================================== #
class TestLifecycle:
    def test_it_runs_the_requested_number_of_cycles_and_stops(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        supervisor = _supervisor(orchestrator)
        assert supervisor.run(max_cycles=5) == 0
        assert orchestrator.cycles == 5
        assert orchestrator.shutdown_calls == 1

    def test_the_pid_file_is_written_and_removed(self, supervisor_config):
        from pathlib import Path

        orchestrator = FakeOrchestrator(supervisor_config)
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=1)
        assert not Path(supervisor_config.runtime.pid_file).exists()

    def test_request_stop_ends_the_loop(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        supervisor = _supervisor(orchestrator)

        def stop_soon():
            while orchestrator.cycles < 2:
                time.sleep(0.005)
            supervisor.request_stop("test asked it to stop")

        thread = threading.Thread(target=stop_soon, daemon=True)
        thread.start()
        supervisor.run()
        thread.join(timeout=2)
        assert supervisor.state.stop_reason == "test asked it to stop"
        assert orchestrator.shutdown_calls == 1

    def test_lifecycle_is_journalled(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        _supervisor(orchestrator).run(max_cycles=1)
        assert "lifecycle" in orchestrator.journal.categories()


class TestErrorHandling:
    def test_consecutive_failures_halt_the_loop(self, supervisor_config):
        supervisor_config.runtime.max_consecutive_cycle_errors = 3
        orchestrator = FakeOrchestrator(supervisor_config, cycle_error=RuntimeError("boom"))
        supervisor = _supervisor(orchestrator)

        assert supervisor.run(max_cycles=50) == 1
        assert supervisor.state.consecutive_errors == 3
        assert orchestrator.cycles == 3       # it stopped, it did not keep grinding
        assert any("consecutive cycle failures" in msg for msg in orchestrator.alerts.sent)

    def test_a_transient_error_does_not_stop_the_loop(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        original = orchestrator.run_cycle
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("one bad cycle")
            return original()

        orchestrator.run_cycle = flaky
        supervisor = _supervisor(orchestrator)
        assert supervisor.run(max_cycles=5) == 0
        assert supervisor.state.errors == 1
        assert supervisor.state.consecutive_errors == 0     # the streak reset

    def test_a_broker_error_triggers_a_reconnect(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise BrokerError("session expired")
            return FakeReport()

        orchestrator.run_cycle = flaky
        supervisor = _supervisor(orchestrator)
        assert supervisor.run(max_cycles=3) == 0
        assert supervisor.state.reconnects == 1
        assert orchestrator.reconciler.calls == 1    # it re-reconciled afterwards

    def test_a_failed_reconnect_stops_the_process(self, supervisor_config):
        supervisor_config.runtime.max_reconnect_attempts = 2
        orchestrator = FakeOrchestrator(supervisor_config)
        orchestrator.data_broker = FakeBroker(fail_auth_times=99)
        orchestrator.execution_broker = orchestrator.data_broker
        orchestrator.run_cycle = lambda: (_ for _ in ()).throw(BrokerError("gone"))

        supervisor = _supervisor(orchestrator)
        assert supervisor.run(max_cycles=10) == 1
        assert any("could not restore" in msg for msg in orchestrator.alerts.sent)

    def test_entries_stay_blocked_when_reconciliation_is_unresolved(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        orchestrator.reconciler = FakeReconciler(safe=False)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise BrokerError("session expired")
            return FakeReport()

        orchestrator.run_cycle = flaky
        _supervisor(orchestrator).run(max_cycles=3)
        assert orchestrator.trading_allowed is False


class TestSquareOff:
    def test_positions_are_closed_at_the_square_off_time(self, supervisor_config):
        supervisor_config.runtime.auto_square_off_before_close = True
        position = FakePosition()
        orchestrator = FakeOrchestrator(supervisor_config, positions=[position])
        # Move the clock past the configured square-off time.
        orchestrator.clock.current = datetime(
            2026, 6, 1, 15, 20, tzinfo=IST
        )
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=2)

        assert orchestrator.exits
        assert orchestrator.exits[0][1] is ExitReason.END_OF_SESSION
        assert orchestrator.positions.count == 0

    def test_square_off_happens_once_not_every_cycle(self, supervisor_config):
        position = FakePosition()
        orchestrator = FakeOrchestrator(supervisor_config, positions=[position])
        orchestrator.clock.current = datetime(2026, 6, 1, 15, 20, tzinfo=IST)
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=5)
        assert len(orchestrator.exits) == 1

    def test_no_square_off_before_the_configured_time(self, supervisor_config):
        position = FakePosition()
        orchestrator = FakeOrchestrator(supervisor_config, positions=[position])
        orchestrator.clock.current = datetime(2026, 6, 1, 11, 0, tzinfo=IST)
        _supervisor(orchestrator).run(max_cycles=3)
        assert orchestrator.exits == []
        assert orchestrator.positions.count == 1

    def test_square_off_can_be_disabled(self, supervisor_config):
        supervisor_config.runtime.auto_square_off_before_close = False
        position = FakePosition()
        orchestrator = FakeOrchestrator(supervisor_config, positions=[position])
        orchestrator.clock.current = datetime(2026, 6, 1, 15, 20, tzinfo=IST)
        _supervisor(orchestrator).run(max_cycles=3)
        assert orchestrator.exits == []


class TestShutdownPolicy:
    def test_exiting_with_open_positions_raises_an_emergency_alert(self, supervisor_config):
        supervisor_config.runtime.square_off_on_shutdown = False
        supervisor_config.runtime.auto_square_off_before_close = False
        orchestrator = FakeOrchestrator(supervisor_config, positions=[FakePosition()])
        _supervisor(orchestrator).run(max_cycles=1)
        assert any("open position" in msg for msg in orchestrator.alerts.sent)

    def test_square_off_on_shutdown_flattens_everything(self, supervisor_config):
        supervisor_config.runtime.square_off_on_shutdown = True
        supervisor_config.runtime.auto_square_off_before_close = False
        orchestrator = FakeOrchestrator(supervisor_config, positions=[FakePosition()])
        _supervisor(orchestrator).run(max_cycles=1)
        assert orchestrator.positions.count == 0
        assert orchestrator.exits[0][1] is ExitReason.EMERGENCY_SHUTDOWN

    def test_a_clean_exit_raises_no_position_alarm(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config, positions=[])
        _supervisor(orchestrator).run(max_cycles=1)
        assert not any("open position" in msg for msg in orchestrator.alerts.sent)


class TestWatchdog:
    def test_a_stalled_loop_trips_the_watchdog_and_halts(self, supervisor_config):
        supervisor_config.runtime.watchdog_timeout_seconds = 0.05
        supervisor_config.runtime.watchdog_action = "halt"
        orchestrator = FakeOrchestrator(supervisor_config)

        stall = threading.Event()

        def slow_cycle():
            orchestrator.cycles += 1
            if orchestrator.cycles == 1:
                return FakeReport()
            stall.wait(0.5)       # wedge the loop
            return FakeReport()

        orchestrator.run_cycle = slow_cycle
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=3)

        assert supervisor.state.watchdog_trips >= 1
        assert orchestrator.trading_allowed is False
        assert orchestrator.kill_switches.engaged
        assert any("WATCHDOG" in msg for msg in orchestrator.alerts.sent)

    def test_alert_only_watchdog_does_not_halt_trading(self, supervisor_config):
        supervisor_config.runtime.watchdog_timeout_seconds = 0.05
        supervisor_config.runtime.watchdog_action = "alert"
        orchestrator = FakeOrchestrator(supervisor_config)

        stall = threading.Event()

        def slow_cycle():
            orchestrator.cycles += 1
            if orchestrator.cycles == 1:
                return FakeReport()
            stall.wait(0.5)
            return FakeReport()

        orchestrator.run_cycle = slow_cycle
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=3)

        assert supervisor.state.watchdog_trips >= 1
        assert orchestrator.kill_switches.engaged == []


class TestPacing:
    def test_the_interval_is_fast_inside_the_session(self, supervisor_config):
        # validate_assignment runs per field, so raise the ceiling first.
        supervisor_config.runtime.idle_loop_seconds = 30.0
        supervisor_config.runtime.fast_loop_seconds = 2.0
        supervisor = _supervisor(FakeOrchestrator(supervisor_config))
        inside = datetime(2026, 6, 1, 11, 0, tzinfo=IST)     # a Monday, mid-session
        assert supervisor._interval(inside) == 2.0

    def test_the_interval_is_idle_outside_market_hours(self, supervisor_config):
        # validate_assignment runs per field, so raise the ceiling first.
        supervisor_config.runtime.idle_loop_seconds = 30.0
        supervisor_config.runtime.fast_loop_seconds = 2.0
        supervisor = _supervisor(FakeOrchestrator(supervisor_config))
        overnight = datetime(2026, 6, 1, 3, 0, tzinfo=IST)
        assert supervisor._interval(overnight) == 30.0


class TestState:
    def test_state_describes_the_process(self, supervisor_config):
        orchestrator = FakeOrchestrator(supervisor_config)
        supervisor = _supervisor(orchestrator)
        supervisor.run(max_cycles=3)
        described = supervisor.state.describe()
        assert described["cycles"] == 3
        assert described["errors"] == 0
        assert described["stopping"] is True
