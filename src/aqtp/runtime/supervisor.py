"""RuntimeSupervisor — the process wrapper around the trading loop.

The orchestrator's own `run_forever` is a plain loop: fine for a backtest or a
console session, not enough for a process that holds real positions all day. This
supervisor adds the five things a production run actually needs:

1. **Signal handling.** SIGTERM (systemd, docker stop) and SIGINT stop the loop
   at a cycle boundary and run the configured shutdown policy, rather than
   killing the process mid-order.
2. **Reconnection.** A dropped or expired broker session is re-authenticated with
   exponential backoff. Entries stay blocked until reconciliation succeeds.
3. **A watchdog.** If no cycle completes within the timeout, the configured
   action fires — alert, halt entries, or flatten. A wedged loop with open
   positions is the worst state this system can be in, so it is the one state it
   refuses to sit in silently.
4. **Adaptive pacing.** Fast inside the trading window, idle outside it. There is
   no reason to hammer the broker's rate limit at 03:00.
5. **Square-off.** Positions are closed at the configured time whether or not any
   strategy asked for it.

Every one of these writes to the journal, so a post-mortem can reconstruct what
the process was doing without the log file.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.clock import is_market_open, to_ist
from ..core.errors import BrokerError
from ..core.logging import get_logger
from ..core.types import ExitReason, ManagementAction
from ..monitoring.alerts import AlertEvent
from ..risk.killswitch import KillScope

logger = get_logger(__name__)


@dataclass
class SupervisorState:
    """Live process health, surfaced to `aqtp status` and the dashboard."""

    started_at: datetime | None = None
    cycles: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    reconnects: int = 0
    last_cycle_at: float = 0.0
    last_heartbeat_at: float = 0.0
    watchdog_trips: int = 0
    stopping: bool = False
    stop_reason: str = ""
    squared_off: bool = False
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "cycles": self.cycles,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "reconnects": self.reconnects,
            "watchdog_trips": self.watchdog_trips,
            "seconds_since_last_cycle": (
                round(time.monotonic() - self.last_cycle_at, 1) if self.last_cycle_at else None
            ),
            "degraded": self.degraded,
            "stopping": self.stopping,
            "stop_reason": self.stop_reason,
        }


class RuntimeSupervisor:
    """Owns the process lifecycle around a `TradingOrchestrator`."""

    def __init__(self, orchestrator, *, install_signal_handlers: bool = True) -> None:
        self.orchestrator = orchestrator
        self.config = orchestrator.config
        self.runtime = self.config.runtime
        self.state = SupervisorState()
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None
        self._install_handlers = install_signal_handlers
        self._pid_path: Path | None = None

    # ================================================================== #
    # Process plumbing
    # ================================================================== #
    def _write_pid_file(self) -> None:
        if not self.runtime.pid_file:
            return
        path = Path(self.runtime.pid_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{os.getpid()}\n", encoding="utf-8")
            self._pid_path = path
        except OSError as exc:
            logger.warning("could not write the pid file %s: %s", path, exc)

    def _remove_pid_file(self) -> None:
        if self._pid_path is None:
            return
        try:
            self._pid_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _handle_signal(self, signum: int, _frame) -> None:
        name = signal.Signals(signum).name
        logger.warning("received %s — stopping at the end of this cycle", name)
        self.request_stop(f"received {name}")

    def request_stop(self, reason: str) -> None:
        self.state.stopping = True
        self.state.stop_reason = reason
        self._stop.set()

    # ================================================================== #
    # Watchdog
    # ================================================================== #
    def _watchdog_loop(self) -> None:
        timeout = self.runtime.watchdog_timeout_seconds
        while not self._stop.wait(timeout / 3.0):
            if not self.state.last_cycle_at:
                continue
            stalled = time.monotonic() - self.state.last_cycle_at
            if stalled <= timeout:
                continue
            self.state.watchdog_trips += 1
            self.state.degraded = True
            message = (
                f"WATCHDOG: no cycle completed for {stalled:.0f}s "
                f"(limit {timeout:.0f}s), action={self.runtime.watchdog_action}"
            )
            logger.critical(message)
            self.orchestrator.alerts.emergency(message)
            self.orchestrator.journal.record_event(
                level="CRITICAL", category="watchdog", message=message,
                payload={"stalled_seconds": round(stalled, 1)},
            )

            action = self.runtime.watchdog_action
            if action == "halt":
                self.orchestrator.trading_allowed = False
                self.orchestrator.kill_switches.engage(
                    KillScope.APPLICATION, reason="watchdog timeout", engaged_by="watchdog"
                )
            elif action == "flatten":
                self.orchestrator.kill_switches.engage(
                    KillScope.APPLICATION, reason="watchdog timeout", engaged_by="watchdog"
                )
                self.request_stop("watchdog timeout")
            # `alert` deliberately does nothing beyond the alert above.
            return

    # ================================================================== #
    # Broker session
    # ================================================================== #
    def _reconnect(self) -> bool:
        """Re-authenticate with exponential backoff. Entries stay blocked until
        reconciliation confirms the state afterwards."""
        backoff = self.runtime.reconnect_backoff_seconds
        for attempt in range(1, self.runtime.max_reconnect_attempts + 1):
            if self._stop.is_set():
                return False
            logger.warning(
                "reconnect attempt %d/%d in %.1fs",
                attempt, self.runtime.max_reconnect_attempts, backoff,
            )
            if self._stop.wait(backoff):
                return False
            try:
                self.orchestrator.data_broker.authenticate()
                if self.orchestrator.execution_broker is not self.orchestrator.data_broker:
                    self.orchestrator.execution_broker.authenticate()
                self.state.reconnects += 1
                logger.warning("broker session restored after %d attempt(s)", attempt)
                self.orchestrator.journal.record_event(
                    level="WARNING", category="broker",
                    message="broker session restored", payload={"attempts": attempt},
                )
                report = self.orchestrator.reconciler.reconcile(
                    now=self.orchestrator.clock.now(), resolve_intents=True
                )
                self.orchestrator.last_reconciliation = report
                self.orchestrator.trading_allowed = report.safe_to_trade
                if not report.safe_to_trade:
                    logger.error("post-reconnect reconciliation is unresolved; entries stay blocked")
                self.state.degraded = not report.safe_to_trade
                return True
            except Exception as exc:
                logger.error("reconnect attempt %d failed: %s", attempt, exc)
                backoff = min(backoff * 2.0, self.runtime.reconnect_max_backoff_seconds)

        self.orchestrator.alerts.emergency(
            f"could not restore the broker session after "
            f"{self.runtime.max_reconnect_attempts} attempts — stopping"
        )
        return False

    # ================================================================== #
    # Session policy
    # ================================================================== #
    def _interval(self, now: datetime) -> float:
        """Fast inside the trading window, idle outside it."""
        if not is_market_open(now):
            return self.runtime.idle_loop_seconds
        moment = to_ist(now).time()
        session = self.config.session
        if session.trading_start <= moment <= session.trading_end:
            return self.runtime.fast_loop_seconds
        return self.runtime.idle_loop_seconds

    def _maybe_square_off(self, now: datetime) -> None:
        """Close everything at the configured square-off time.

        This runs regardless of what any strategy thinks: carrying an intraday
        option position past the square-off time is a decision nobody made.
        """
        if not self.runtime.auto_square_off_before_close or self.state.squared_off:
            return
        moment = to_ist(now)
        if moment.time() < self.config.session.square_off_time:
            return
        if not is_market_open(now):
            return

        open_positions = list(self.orchestrator.positions.all())
        self.state.squared_off = True
        if not open_positions:
            return

        logger.warning(
            "square-off time %s reached — closing %d position(s)",
            self.config.session.square_off_time, len(open_positions),
        )
        self.orchestrator.alerts.send(
            AlertEvent.RISK_LIMIT,
            f"square-off: closing {len(open_positions)} position(s) at "
            f"{self.config.session.square_off_time:%H:%M}",
        )
        self.flatten(reason=ExitReason.END_OF_SESSION)

    def flatten(self, *, reason: ExitReason = ExitReason.EMERGENCY_SHUTDOWN) -> int:
        """Close every tracked position at market. Returns how many closed."""
        from ..execution.position_manager import ManagementDecision

        closed = 0
        now = self.orchestrator.clock.now()
        for position in list(self.orchestrator.positions.all()):
            try:
                quote = self.orchestrator.data_broker.get_quote(position.instrument)
            except BrokerError:
                quote = None
            decision = ManagementDecision(
                ManagementAction.EMERGENCY_EXIT, reason, "flatten requested by the supervisor"
            )
            try:
                if self.orchestrator._exit(position, decision, position.quantity, now, quote):
                    closed += 1
            except Exception as exc:
                logger.exception(
                    "failed to flatten %s: %s", position.instrument.trading_symbol, exc
                )
                self.orchestrator.alerts.emergency(
                    f"COULD NOT FLATTEN {position.instrument.trading_symbol}: {exc} — "
                    "this position is still open at the broker"
                )
        return closed

    def _heartbeat(self, now: datetime) -> None:
        if time.monotonic() - self.state.last_heartbeat_at < self.runtime.heartbeat_seconds:
            return
        self.state.last_heartbeat_at = time.monotonic()
        status = self.orchestrator.status()
        logger.info(
            "heartbeat: mode=%s cycles=%d equity=%.0f positions=%d unrealized=%+.0f "
            "trading_allowed=%s errors=%d",
            status["mode"], self.state.cycles, status["equity"], status["open_positions"],
            status["unrealized_pnl"], status["trading_allowed"], self.state.errors,
        )
        self.orchestrator.journal.record_event(
            level="INFO", category="heartbeat",
            message="process heartbeat",
            payload={**self.state.describe(), "equity": status["equity"],
                     "open_positions": status["open_positions"]},
        )

    # ================================================================== #
    # Main loop
    # ================================================================== #
    def run(self, *, max_cycles: int | None = None) -> int:
        """Run until stopped. Returns a process exit code."""
        self.state.started_at = self.orchestrator.clock.now()
        self._write_pid_file()

        if self._install_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, self._handle_signal)
                except (ValueError, OSError):
                    # Not the main thread, or a platform without the signal.
                    logger.debug("could not install a handler for %s", sig)

        if self.runtime.watchdog_timeout_seconds > 0:
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="aqtp-watchdog", daemon=True
            )
            self._watchdog.start()

        banner = self.config.mode_banner()
        logger.warning("%s — supervisor started (pid %d)", banner, os.getpid())
        self.orchestrator.journal.record_event(
            level="WARNING", category="lifecycle",
            message=f"supervisor started in {self.config.mode.value} mode",
            payload={"pid": os.getpid(), "run_id": self.orchestrator.run_id},
        )

        exit_code = 0
        last_snapshot = 0.0
        try:
            while not self._stop.is_set():
                if max_cycles is not None and self.state.cycles >= max_cycles:
                    self.request_stop("reached the requested cycle count")
                    break

                cycle_started = time.perf_counter()
                now = self.orchestrator.clock.now()

                try:
                    self._maybe_square_off(now)
                    report = self.orchestrator.run_cycle()
                    self.state.cycles += 1
                    self.state.last_cycle_at = time.monotonic()
                    self.state.consecutive_errors = 0
                    self.state.degraded = not self.orchestrator.trading_allowed
                    if report.halted_reason:
                        logger.debug("cycle: %s", report.summary())
                    else:
                        logger.info("cycle: %s", report.summary())

                except BrokerError as exc:
                    self.state.errors += 1
                    self.state.consecutive_errors += 1
                    self.state.degraded = True
                    logger.error("broker error in cycle: %s", exc)
                    self.orchestrator.alerts.broker_error(str(exc))
                    if not self._reconnect():
                        self.request_stop("broker session could not be restored")
                        exit_code = 1

                except Exception as exc:
                    self.state.errors += 1
                    self.state.consecutive_errors += 1
                    logger.exception("cycle failed: %s", exc)
                    self.orchestrator.journal.record_event(
                        level="ERROR", category="cycle", message=str(exc)
                    )
                    if self.state.consecutive_errors >= self.runtime.max_consecutive_cycle_errors:
                        message = (
                            f"{self.state.consecutive_errors} consecutive cycle failures — "
                            "halting rather than trading blind"
                        )
                        logger.critical(message)
                        self.orchestrator.alerts.emergency(message)
                        self.request_stop(message)
                        exit_code = 1

                self._heartbeat(now)

                if time.monotonic() - last_snapshot >= self.runtime.state_snapshot_seconds:
                    last_snapshot = time.monotonic()
                    try:
                        self.orchestrator._record_equity_snapshot()
                    except Exception:
                        logger.debug("equity snapshot failed", exc_info=True)

                elapsed = time.perf_counter() - cycle_started
                self._stop.wait(max(0.0, self._interval(now) - elapsed))

        except KeyboardInterrupt:
            self.request_stop("keyboard interrupt")
        finally:
            self._shutdown()

        return exit_code

    # ------------------------------------------------------------------ #
    def _shutdown(self) -> None:
        self._stop.set()
        reason = self.state.stop_reason or "shutdown"
        logger.warning("shutting down: %s", reason)

        if self.runtime.cancel_open_orders_on_shutdown and self.config.submits_real_orders:
            try:
                cancelled = self._cancel_open_orders()
                if cancelled:
                    logger.warning("cancelled %d open order(s) on shutdown", cancelled)
            except Exception as exc:
                logger.error("could not cancel open orders: %s", exc)

        if self.runtime.square_off_on_shutdown:
            try:
                closed = self.flatten(reason=ExitReason.EMERGENCY_SHUTDOWN)
                logger.warning("closed %d position(s) on shutdown", closed)
            except Exception as exc:
                logger.error("square-off on shutdown failed: %s", exc)

        open_positions = self.orchestrator.positions.count
        if open_positions and not self.runtime.square_off_on_shutdown:
            # Say it loudly. A process that exits holding positions is a fact the
            # operator has to know before they walk away from the terminal.
            logger.critical(
                "EXITING WITH %d OPEN POSITION(S) — they remain live at the broker. "
                "Set runtime.square_off_on_shutdown: true to close them automatically, "
                "or run `aqtp trade close-all`.", open_positions,
            )
            self.orchestrator.alerts.emergency(
                f"process exiting with {open_positions} open position(s) still at the broker"
            )

        try:
            self.orchestrator.shutdown(close_positions=False)
        except Exception as exc:
            logger.error("orchestrator shutdown raised: %s", exc)

        self.orchestrator.journal.record_event(
            level="WARNING", category="lifecycle", message=f"supervisor stopped: {reason}",
            payload={**self.state.describe(), "open_positions": open_positions},
        )
        self._remove_pid_file()
        logger.warning("supervisor stopped after %d cycle(s)", self.state.cycles)

    def _cancel_open_orders(self) -> int:
        cancelled = 0
        try:
            orders = self.orchestrator.execution_broker.list_orders()
        except Exception as exc:
            logger.warning("could not list orders on shutdown: %s", exc)
            return 0
        for order in orders:
            if not order.status.is_open:
                continue
            try:
                self.orchestrator.execution_broker.cancel_order(
                    order.broker_order_id, segment=order.instrument.segment
                )
                cancelled += 1
            except Exception as exc:
                logger.error("could not cancel %s: %s", order.broker_order_id, exc)
        return cancelled
