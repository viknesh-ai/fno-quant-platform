"""TradingOrchestrator — the pipeline of REQ 2 and REQ 69.

    Market Data -> Feature Engine -> Market Regime -> Strategy Analysis
    -> ML Prediction -> Signal Aggregation -> Opportunity Ranking -> Risk Engine
    -> Position Sizing -> Execution Engine -> Broker API

The orchestrator only *sequences* these stages; it contains no trading logic of its
own. That is what keeps the layers independent (REQ 2) and makes the ordering
auditable: a strategy cannot reach the broker because there is no code path from
one to the other except through the RiskEngine.

Startup follows REQ 63 exactly: load state, connect, reconcile, detect unmanaged
positions and outstanding orders, restore monitoring, and only then accept signals.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .brokers.base import BrokerAdapter, Capability
from .configuration.loader import Credentials
from .configuration.schema import AppConfig
from .core.clock import Clock, LiveClock, is_market_open, minutes_until_close, to_ist
from .core.errors import BrokerError
from .core.ids import new_run_id
from .core.logging import get_logger
from .core.types import (
    Decision,
    Direction,
    ExitReason,
    Instrument,
    ManagementAction,
    Regime,
    Timeframe,
    TradingMode,
)
from .data.instruments import InstrumentDiscovery, InstrumentUniverse
from .data.market_data import MarketDataEngine, MarketSnapshot
from .events.risk import EventRiskEngine
from .execution.engine import ExecutionEngine
from .execution.latency import LatencyMonitor
from .execution.position_manager import ManagedPosition, PositionManager
from .execution.reconcile import ReconciliationReport, StateReconciler
from .features.engine import FeatureEngine, latest_feature_row
from .features.options_features import IVHistory, compute_chain_features
from .journal.explain import ExplanationBuilder
from .journal.store import TradeJournal
from .ml.predict import PredictionEngine
from .monitoring.alerts import AlertEvent, AlertManager
from .options.selector import OptionSelector
from .regime.engine import MarketRegimeEngine, RegimeState
from .risk.costs import TransactionCostModel
from .risk.drawdown import DrawdownProtection
from .risk.engine import RiskEngine
from .risk.killswitch import KillScope, KillSwitchManager
from .risk.portfolio import PortfolioManager
from .risk.sizing import PositionSizer, effective_equity
from .scanner.opportunity import Opportunity, OpportunityScanner, ScanResult
from .signals.ensemble import SignalEnsemble, no_trade
from .signals.expected_value import ExpectedValueCalculator, blended_win_probability
from .strategies.base import StrategyContext
from .strategies.health import StrategyHealthMonitor, TradeOutcome
from .strategies.registry import build_strategies

logger = get_logger(__name__)


@dataclass
class CycleReport:
    """What one decision cycle did — surfaced to the dashboard and the CLI."""

    timestamp: datetime
    universe_size: int = 0
    symbols_evaluated: int = 0
    data_rejected: int = 0
    opportunities: list[Opportunity] = field(default_factory=list)
    trades_entered: int = 0
    positions_managed: int = 0
    positions_exited: int = 0
    no_trade_reasons: dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    halted_reason: str = ""

    def summary(self) -> str:
        if self.halted_reason:
            return f"cycle halted: {self.halted_reason}"
        return (
            f"evaluated {self.symbols_evaluated}/{self.universe_size}, "
            f"{len(self.opportunities)} opportunities, {self.trades_entered} entered, "
            f"{self.positions_exited} exited ({self.duration_ms:.0f}ms)"
        )


class TradingOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        *,
        data_broker: BrokerAdapter,
        execution_broker: BrokerAdapter,
        journal: TradeJournal,
        clock: Clock | None = None,
        prediction_engine: PredictionEngine | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or LiveClock()
        self.run_id = new_run_id()
        self.data_broker = data_broker
        self.execution_broker = execution_broker
        self.journal = journal

        # --- pipeline components (REQ 69) ---------------------------------
        self.market_data = MarketDataEngine(data_broker, config, clock=self.clock)
        self.discovery = InstrumentDiscovery(config.universe)
        self.features = FeatureEngine()
        self.regime_engine = MarketRegimeEngine()
        self.strategies = build_strategies(config.strategies)
        self.health = StrategyHealthMonitor(config.strategy_health)
        self.prediction = prediction_engine or PredictionEngine(config.prediction)
        self.ensemble = SignalEnsemble(config.ensemble, health_monitor=self.health)
        self.scanner = OpportunityScanner(config.scanner)
        self.option_selector = OptionSelector(config.option_selection)
        self.iv_history = IVHistory()

        self.costs = TransactionCostModel(config.costs)
        self.expected_value = ExpectedValueCalculator(config.expected_value, self.costs)
        self.portfolio = PortfolioManager()
        self.drawdown = DrawdownProtection(config.risk)
        self.kill_switches = KillSwitchManager(
            Path(config.paths.data_dir) / "killswitches.json"
        )
        self.sizer = PositionSizer(config.capital, config.risk)
        self.event_risk = EventRiskEngine(config.event_risk)
        self.risk = RiskEngine(
            config,
            portfolio=self.portfolio,
            drawdown=self.drawdown,
            kill_switches=self.kill_switches,
            sizer=self.sizer,
            cost_model=self.costs,
            event_risk=self.event_risk,
            health=self.health,
        )

        self.positions = PositionManager(config)
        self.latency = LatencyMonitor()
        self.execution = ExecutionEngine(
            config, execution_broker, journal, clock=self.clock, latency_monitor=self.latency
        )
        self.reconciler = StateReconciler(execution_broker, journal, self.positions)
        self.alerts = AlertManager(
            config.monitoring, alert_file=Path(config.paths.log_dir) / "alerts.jsonl"
        )
        self.explainer = ExplanationBuilder()

        # --- runtime state --------------------------------------------------
        self.universe: InstrumentUniverse | None = None
        self.started = False
        self.trading_allowed = False
        self.last_cycle: CycleReport | None = None
        self.last_reconciliation: ReconciliationReport | None = None
        self._equity = config.capital.available_capital
        self._decision_counter = 0
        self._position_counter = 0
        self._session_date: date | None = None
        self._last_reconcile_at: float = 0.0
        self._entry_timeframe = Timeframe(config.timeframes.entry_timeframe)
        self._setup_timeframe = Timeframe(config.timeframes.setup_timeframe)
        self._regime_timeframe = Timeframe(config.timeframes.regime_timeframe)

    # ================================================================== #
    # Startup (REQ 63)
    # ================================================================== #
    def start(self, *, warm_up_days: int = 10, adopt_positions: bool = True) -> bool:
        """Bring the system up in the order REQ 63 mandates."""
        banner = self.config.mode_banner()
        logger.warning("=" * 60)
        logger.warning("  %s", banner)
        logger.warning("  run_id=%s  broker=%s", self.run_id, self.config.broker.name)
        logger.warning("=" * 60)
        self.journal.record_event(
            level="INFO", category="lifecycle",
            message=f"starting in {self.config.mode.value} mode", payload={"run_id": self.run_id},
        )

        # 1. Load persisted state (kill switches load in their constructor).
        if self.kill_switches.is_shutdown:
            logger.critical(
                "a GLOBAL kill switch is engaged from a previous run; refusing to start. "
                "Clear it with `aqtp resume --scope global` after investigating."
            )
            return False

        # 2. Connect to the broker.
        try:
            self.data_broker.authenticate()
            if self.execution_broker is not self.data_broker:
                self.execution_broker.authenticate()
        except Exception as exc:
            logger.critical("broker authentication failed: %s", exc)
            self.alerts.send(AlertEvent.API_DISCONNECT, f"authentication failed: {exc}")
            return False
        logger.info("broker authenticated")

        # 3. Discover the universe.
        if not self.refresh_universe():
            logger.critical("instrument discovery failed; cannot start")
            return False

        # 4-5. Reconcile broker state; detect unmanaged positions and open orders.
        report = self.reconciler.reconcile(now=self.clock.now(), resolve_intents=True)
        self.last_reconciliation = report
        if report.unmanaged_positions and adopt_positions:
            adopted = self.reconciler.adopt_unmanaged(report)
            if adopted:
                self.alerts.send(
                    AlertEvent.RECONCILIATION,
                    f"adopted {len(adopted)} unmanaged position(s): {', '.join(adopted)}",
                )
        if report.orphan_orders:
            self.alerts.send(
                AlertEvent.RECONCILIATION,
                f"{len(report.orphan_orders)} untracked open order(s) at the broker: "
                f"{', '.join(report.orphan_orders[:5])}",
            )

        # 6. Restore monitoring / equity baseline.
        self._refresh_equity()
        self._session_date = to_ist(self.clock.now()).date()
        self.drawdown.start_session(self._equity, session_date=self._session_date)

        # 7. Only now allow new signals.
        self.trading_allowed = report.safe_to_trade
        if not self.trading_allowed:
            logger.error(
                "state is not fully reconciled; entries are BLOCKED until it is. %s",
                report.summary(),
            )
            self.alerts.send(
                AlertEvent.RECONCILIATION,
                "entries blocked: unresolved state discrepancies at startup",
            )

        # Warm up features with history so the first cycle is not all NaN.
        if warm_up_days > 0 and self.universe:
            underlyings = self._underlying_instruments()
            loaded = self.market_data.warm_up(underlyings, days=warm_up_days)
            logger.info("warmed up %d instruments with history", len(loaded))

        self.started = True
        logger.warning("%s — startup complete, trading_allowed=%s", banner, self.trading_allowed)
        return True

    def refresh_universe(self) -> bool:
        try:
            instruments = self.data_broker.fetch_instruments(force_refresh=False)
        except BrokerError as exc:
            logger.error("could not fetch the instrument master: %s", exc)
            return False

        previous = self.universe
        self.universe = self.discovery.refresh(instruments, as_of=self.clock.now())
        if previous is not None:
            changes = self.discovery.detect_changes(previous)
            if changes:
                logger.info("universe changes: %s", changes)
                self.journal.record_event(
                    level="INFO", category="universe", message="universe changed", payload=changes
                )
        return len(self.universe) > 0

    # ================================================================== #
    # The decision cycle
    # ================================================================== #
    def run_cycle(self) -> CycleReport:
        """One full pass of the pipeline."""
        started = time.perf_counter()
        now = self.clock.now()
        report = CycleReport(timestamp=now, universe_size=len(self.universe or []))

        if not self.started:
            report.halted_reason = "orchestrator has not been started"
            return report

        # Session rollover.
        today = to_ist(now).date()
        if self._session_date != today:
            self._session_date = today
            self.market_data.reset_session()
            self._refresh_equity()
            self.drawdown.start_session(self._equity, session_date=today)
            logger.info("new session: %s", today)

        # Periodic universe refresh (REQ 4).
        if self.discovery.needs_refresh():
            self.refresh_universe()
            report.universe_size = len(self.universe or [])

        # Periodic reconciliation (REQ 46).
        if time.monotonic() - self._last_reconcile_at > self.config.execution.reconcile_interval_seconds:
            self.last_reconciliation = self.reconciler.reconcile(now=now)
            self._last_reconcile_at = time.monotonic()
            if not self.last_reconciliation.safe_to_trade:
                self.trading_allowed = False

        self._refresh_equity()
        self.risk.on_equity_update(self._equity, now=now)

        # --- manage existing positions FIRST ------------------------------
        # Exits are evaluated before entries every cycle. Freeing capital and
        # honouring stops must not wait behind a scan of forty symbols.
        report.positions_managed, report.positions_exited = self._manage_positions(now)

        # --- gate new entries -----------------------------------------------
        blocked = self._entry_blocked(now)
        if blocked:
            report.halted_reason = blocked
            report.duration_ms = (time.perf_counter() - started) * 1000
            self.last_cycle = report
            return report

        # --- scan for opportunities -------------------------------------------
        scan = self._scan(now, report)
        report.opportunities = scan.opportunities

        # --- act on the best -------------------------------------------------
        capacity = self.config.risk.max_open_positions - self.positions.count
        for opportunity in scan.top(max(0, capacity)):
            if self.positions.count >= self.config.risk.max_open_positions:
                break
            if self._enter(opportunity, now):
                report.trades_entered += 1

        report.duration_ms = (time.perf_counter() - started) * 1000
        self.last_cycle = report
        self._record_equity_snapshot()
        return report

    # ------------------------------------------------------------------ #
    def _entry_blocked(self, now: datetime) -> str:
        """Every reason new entries are not permitted right now (REQ 36)."""
        if not self.trading_allowed:
            return "state not reconciled; entries blocked"
        if self.kill_switches.is_shutdown:
            return "global kill switch is engaged"

        allowed, reason = self.kill_switches.can_trade()
        if not allowed:
            return reason

        can_trade, drawdown_reason = self.drawdown.can_trade(now)
        if not can_trade:
            return drawdown_reason

        if self.config.submits_real_orders and not is_market_open(now):
            return "market is closed"

        moment = to_ist(now)
        session = self.config.session
        if moment.time() < session.trading_start:
            return f"before the configured trading start ({session.trading_start:%H:%M})"
        if moment.time() >= session.no_new_entries_after:
            return f"past the no-new-entries time ({session.no_new_entries_after:%H:%M})"

        if self.positions.count >= self.config.risk.max_open_positions:
            return f"at the maximum of {self.config.risk.max_open_positions} open positions"

        return ""

    # ------------------------------------------------------------------ #
    def _scan(self, now: datetime, report: CycleReport) -> ScanResult:
        """Market data -> features -> regime -> strategies -> ML -> ensemble -> rank."""
        assert self.universe is not None
        opportunities: list[Opportunity] = []

        instruments = self._underlying_instruments()
        chains_wanted = self._chains_wanted(now)
        snapshot = self.market_data.build_snapshot(instruments, chains_for=chains_wanted)
        report.data_rejected = len(snapshot.rejected)

        for symbol, instrument_snapshot in snapshot.instruments.items():
            underlying = self._underlying_for(symbol)
            if underlying is None:
                continue
            report.symbols_evaluated += 1

            opportunity = self._evaluate_underlying(
                underlying=underlying,
                snapshot=snapshot,
                symbol=symbol,
                now=now,
            )
            if opportunity is None:
                continue
            opportunities.append(opportunity)
            if not opportunity.is_actionable:
                key = opportunity.rejection_reason[:60]
                report.no_trade_reasons[key] = report.no_trade_reasons.get(key, 0) + 1

        return self.scanner.rank(opportunities, timestamp=now)

    def _evaluate_underlying(
        self, *, underlying: str, snapshot: MarketSnapshot, symbol: str, now: datetime
    ) -> Opportunity | None:
        instrument_snapshot = snapshot.get(symbol)
        if instrument_snapshot is None or not instrument_snapshot.usable:
            return None

        # --- features across timeframes (REQ 9/10) ------------------------
        frames: dict[Timeframe, pd.DataFrame] = {}
        features_by_timeframe: dict[Timeframe, pd.Series] = {}
        for timeframe in (self._entry_timeframe, self._setup_timeframe, self._regime_timeframe):
            frame = self.market_data.frame(symbol, timeframe)
            if frame.empty:
                continue
            frames[timeframe] = frame
            computed = self.features.compute(frame)
            row = latest_feature_row(computed)
            if row is not None:
                features_by_timeframe[timeframe] = row

        entry_features = features_by_timeframe.get(self._entry_timeframe)
        if entry_features is None:
            return None
        if not self.features.ready(frames.get(self._entry_timeframe, pd.DataFrame())):
            return None

        atr = float(entry_features.get("atr", float("nan")))
        if not np.isfinite(atr) or atr <= 0:
            return None

        # --- option chain features ------------------------------------------
        expiries = self.discovery.tradable_expiries(underlying, as_of=to_ist(now).date())
        option_features = None
        chains = []
        if expiries:
            for expiry in expiries[: self.config.option_selection.consider_expiries]:
                chain = snapshot.chain(underlying, expiry)
                if chain is not None:
                    chains.append(chain)
            if chains:
                option_features = compute_chain_features(chains[0])
                if option_features.atm_iv:
                    self.iv_history.record(underlying, now, option_features.atm_iv)

        days_to_expiry = (expiries[0] - to_ist(now).date()).days if expiries else None

        # --- regime (REQ 11) --------------------------------------------------
        regime_features = features_by_timeframe.get(self._regime_timeframe, entry_features)
        event_window = self.event_risk.assess(now, underlying=underlying)
        regime = self.regime_engine.detect(
            regime_features,
            timestamp=now,
            days_to_expiry=days_to_expiry,
            event_active=event_window.active,
        )

        # --- strategies (REQ 12) -----------------------------------------------
        context = StrategyContext(
            underlying=underlying,
            instrument=instrument_snapshot.instrument,
            timestamp=now,
            last_price=instrument_snapshot.last_price,
            features=entry_features,
            features_by_timeframe=features_by_timeframe,
            frames=frames,
            regime=regime,
            entry_timeframe=self._entry_timeframe,
            setup_timeframe=self._setup_timeframe,
            regime_timeframe=self._regime_timeframe,
            atr=atr,
            option_features=option_features,
            days_to_expiry=days_to_expiry,
            session_context={
                "minutes_until_close": minutes_until_close(now),
                "minutes_since_open": instrument_snapshot.context.minutes_since_open,
            },
        )

        signals = []
        for strategy in self.strategies:
            can_trade, _ = self.health.can_trade(strategy.name, now=now)
            if not can_trade:
                continue
            try:
                signals.append(strategy.generate(context))
            except Exception as exc:
                # A broken strategy must not take down the cycle.
                logger.exception("strategy %s raised: %s", strategy.name, exc)

        # --- ML prediction (REQ 14) ----------------------------------------------
        feature_age = (now - instrument_snapshot.quote.timestamp).total_seconds()
        prediction = self.prediction.predict(
            entry_features,
            regime=regime.dominant,
            regime_confidence=regime.confidence,
            timestamp=now,
            feature_age_seconds=feature_age,
        )

        # --- ensemble (REQ 13) ------------------------------------------------------
        ensemble = self.ensemble.combine(
            signals=signals,
            prediction=prediction,
            regime=regime,
            min_model_probability=self.config.prediction.min_model_probability,
            timestamp=now,
        )

        return self.scanner.score(
            underlying=underlying,
            ensemble=ensemble,
            prediction=prediction,
            regime=regime,
            quote=instrument_snapshot.quote,
            instrument=instrument_snapshot.instrument,
            available_capital=self._equity,
            timestamp=now,
        )

    # ------------------------------------------------------------------ #
    def _enter(self, opportunity: Opportunity, now: datetime) -> bool:
        """Select a contract, size it, risk-check it, and execute."""
        self._decision_counter += 1
        decision_id = f"D{self.run_id}-{self._decision_counter:05d}"
        ensemble = opportunity.ensemble
        assert ensemble is not None

        underlying = opportunity.underlying
        underlying_price = opportunity.quote.last_price if opportunity.quote else 0.0
        expiries = self.discovery.tradable_expiries(underlying, as_of=to_ist(now).date())

        # --- option contract selection (REQ 6) ------------------------------
        chains = []
        for expiry in expiries[: self.config.option_selection.consider_expiries]:
            chain = self.market_data.fetch_option_chain(underlying, expiry)
            if chain is not None:
                chains.append(chain)

        expected_move = abs(ensemble.proposed_target - ensemble.proposed_entry)
        iv_percentile = None
        if chains:
            features = compute_chain_features(chains[0])
            if features.atm_iv:
                iv_percentile = self.iv_history.percentile(underlying, features.atm_iv)

        selection = self.option_selector.select(
            chains=chains,
            direction=ensemble.direction,
            underlying_price=underlying_price,
            expected_move=expected_move,
            atr=expected_move / max(ensemble.expected_move_atr, 0.1),
            now=now,
            holding_minutes=ensemble.expected_holding_minutes,
            capital_available=self._equity * self.config.risk.max_position_size_pct,
            lot_size=self._lot_size_for(underlying),
            iv_percentile=iv_percentile,
        )

        if not selection.ok or selection.selected is None:
            self._record_no_trade(
                decision_id, opportunity, now,
                reason=f"no suitable contract: {selection.reason}",
                selection=selection,
            )
            return False

        contract = selection.selected.contract
        instrument = contract.instrument
        contract_quote = contract.quote
        entry_price = contract.last_price or 0.0

        # Translate the underlying-level stop/target into contract prices via delta,
        # since the position is held in the option, not the underlying.
        greeks = contract.greeks
        delta = abs(greeks.delta) if greeks else 0.5
        underlying_risk = abs(ensemble.proposed_entry - ensemble.proposed_stop)
        underlying_reward = abs(ensemble.proposed_target - ensemble.proposed_entry)
        contract_risk = max(entry_price * 0.15, underlying_risk * delta)
        contract_reward = underlying_reward * delta
        stop_price = max(0.05, entry_price - contract_risk)
        target_price = entry_price + contract_reward

        # --- expected value (REQ 55) ------------------------------------------
        win_probability = blended_win_probability(
            ml_probability=ensemble.ml_probability,
            ensemble_confidence=ensemble.confidence,
        )
        lot_size = instrument.lot_size
        expected_value = self.expected_value.evaluate(
            instrument=instrument,
            quantity=lot_size,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            win_probability=win_probability,
            quote=contract_quote,
        )
        if not expected_value.passes:
            self._record_no_trade(
                decision_id, opportunity, now,
                reason=f"expected value: {expected_value.reason}",
                selection=selection, expected_value=expected_value,
            )
            return False

        # --- risk engine (REQ 25) — the final authority --------------------------
        portfolio_state = self._portfolio_state(now)
        margin = self._margin()
        approval = self.risk.evaluate(
            decision=ensemble.decision,
            instrument=instrument,
            underlying=underlying,
            strategy=ensemble.agreeing_strategies[0] if ensemble.agreeing_strategies else "ensemble",
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            portfolio_state=portfolio_state,
            margin=margin,
            quote=contract_quote,
            expected_value=expected_value,
            greeks=greeks,
            underlying_price=underlying_price,
            now=now,
        )

        explanation = self.explainer.build(
            decision_id=decision_id,
            timestamp=now,
            underlying=underlying,
            ensemble=ensemble,
            selection=selection,
            approval=approval,
            expected_value=expected_value,
            underlying_price=underlying_price,
        )

        if not approval.approved:
            self._write_decision(
                decision_id, opportunity, now, instrument=instrument,
                approval=approval, expected_value=expected_value, explanation=explanation,
                approved=False,
            )
            self.alerts.rejected(
                f"{underlying}: {approval.rejection_reasons[0] if approval.rejection_reasons else 'risk rejected'}",
                decision_id=decision_id,
            )
            return False

        # --- execute (REQ 32) ----------------------------------------------------
        result = self.execution.execute(
            approval=approval,
            instrument=instrument,
            quote=contract_quote,
            decision_id=decision_id,
            strategy=ensemble.agreeing_strategies[0] if ensemble.agreeing_strategies else "ensemble",
            market_data_at=contract_quote.timestamp if contract_quote else now,
            signal_at=ensemble.timestamp or now,
        )

        self._write_decision(
            decision_id, opportunity, now, instrument=instrument,
            approval=approval, expected_value=expected_value, explanation=explanation,
            approved=result.success,
        )

        if not result.success:
            logger.warning("execution failed for %s: %s", underlying, result.failure_reason)
            if result.ambiguous:
                # An ambiguous execution means we may hold an unknown position.
                # Stop entering until reconciliation clears it.
                self.trading_allowed = False
                self.alerts.emergency(
                    f"ambiguous order state for {instrument.trading_symbol}; entries halted "
                    "pending reconciliation"
                )
            return False

        # --- track the position ---------------------------------------------------
        self._position_counter += 1
        fill_price = result.average_fill_price or entry_price
        managed = ManagedPosition(
            position_id=f"P{self.run_id}-{self._position_counter:04d}",
            decision_id=decision_id,
            instrument=instrument,
            underlying=underlying,
            strategy=ensemble.agreeing_strategies[0] if ensemble.agreeing_strategies else "ensemble",
            direction=Direction.LONG,  # long the option; direction is in CE vs PE
            quantity=result.filled_quantity,
            entry_price=fill_price,
            entry_time=now,
            stop_price=max(0.05, fill_price - contract_risk),
            target_price=fill_price + contract_reward,
            initial_stop_price=max(0.05, fill_price - contract_risk),
            entry_regime=ensemble.regime,
            entry_ml_probability=ensemble.ml_probability,
            expected_holding_minutes=ensemble.expected_holding_minutes,
            underlying_entry_price=underlying_price,
            last_price=fill_price,
        )
        self.positions.track(managed)

        self.journal.record_trade(
            {
                "trade_id": managed.position_id,
                "decision_id": decision_id,
                "entry_order_id": result.client_order_id,
                "underlying": underlying,
                "instrument": instrument.trading_symbol,
                "expiry": instrument.expiry_date.isoformat() if instrument.expiry_date else None,
                "strike": instrument.strike_price,
                "option_type": instrument.instrument_type.value,
                "strategy": managed.strategy,
                "regime": ensemble.regime.value,
                "direction": ensemble.direction.value,
                "quantity": result.filled_quantity,
                "entry_time": now.isoformat(),
                "entry_price": fill_price,
                "stop_price": managed.stop_price,
                "target_price": managed.target_price,
                "ml_probability": ensemble.ml_probability,
                "explanation": explanation.render(),
            }
        )
        self.alerts.trade_entry(
            f"{ensemble.decision.value} {instrument.trading_symbol} x{result.filled_quantity} "
            f"@ {fill_price:.2f}",
            decision_id=decision_id, underlying=underlying,
        )
        logger.info("ENTERED %s", explanation.render().splitlines()[0])
        return True

    # ------------------------------------------------------------------ #
    def _manage_positions(self, now: datetime) -> tuple[int, int]:
        """Evaluate and act on every open position (REQ 34/35)."""
        managed_count = 0
        exited_count = 0
        emergency = self.kill_switches.is_shutdown or self.drawdown.state.is_shutdown

        for position in list(self.positions.all()):
            try:
                quote = self.data_broker.get_quote(position.instrument)
            except BrokerError as exc:
                logger.warning("cannot price %s: %s", position.instrument.trading_symbol, exc)
                self.alerts.broker_error(f"quote failed for {position.instrument.trading_symbol}: {exc}")
                continue

            verdict = self.market_data.quality.check_quote(quote, now=now)
            if not verdict.ok and not emergency:
                logger.warning(
                    "holding %s: quote failed validation (%s)",
                    position.instrument.trading_symbol, verdict.reason,
                )
                continue

            managed_count += 1
            decision = self.positions.evaluate(
                position,
                quote=quote,
                now=now,
                regime=self.regime_engine.current,
                emergency=emergency,
                risk_limit_breached=not self.drawdown.state.can_open_new
                and self.drawdown.state.is_shutdown,
            )

            if decision.action is ManagementAction.HOLD:
                continue

            if decision.is_exit:
                quantity = (
                    decision.exit_quantity
                    if decision.action is ManagementAction.PARTIAL_EXIT
                    else position.quantity
                )
                if self._exit(position, decision, quantity, now, quote):
                    if decision.action is ManagementAction.PARTIAL_EXIT:
                        self.positions.apply(position, decision)
                    else:
                        exited_count += 1
            else:
                self.positions.apply(position, decision)

        return managed_count, exited_count

    def _exit(self, position, decision, quantity: int, now: datetime, quote) -> bool:
        result = self.execution.close_position(
            instrument=position.instrument,
            quantity=quantity,
            direction=position.direction,
            reason=decision.reason.value if decision.reason else decision.action.value,
            decision_id=position.decision_id,
            strategy=position.strategy,
            quote=quote,
            use_market_order=True,
        )
        if not result.success:
            logger.error(
                "EXIT FAILED for %s: %s", position.instrument.trading_symbol, result.failure_reason
            )
            self.alerts.broker_error(
                f"exit failed for {position.instrument.trading_symbol}: {result.failure_reason}"
            )
            return False

        exit_price = result.average_fill_price or position.last_price
        gross = (exit_price - position.entry_price) * result.filled_quantity * position.direction.sign
        costs = self.costs.round_trip(
            instrument=position.instrument,
            quantity=result.filled_quantity,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quote=quote,
        )
        net = gross - costs.total
        risk_total = position.risk_per_unit * result.filled_quantity
        r_multiple = net / risk_total if risk_total > 0 else 0.0
        reason = decision.reason or ExitReason.TARGET_REACHED

        self.journal.record_trade(
            {
                "trade_id": position.position_id,
                "decision_id": position.decision_id,
                "exit_order_id": result.client_order_id,
                "underlying": position.underlying,
                "instrument": position.instrument.trading_symbol,
                "strategy": position.strategy,
                "regime": position.entry_regime.value,
                "direction": position.direction.value,
                "quantity": result.filled_quantity,
                "entry_time": position.entry_time.isoformat(),
                "entry_price": position.entry_price,
                "exit_time": now.isoformat(),
                "exit_price": exit_price,
                "stop_price": position.stop_price,
                "target_price": position.target_price,
                "gross_pnl": gross,
                "costs": costs.total,
                "slippage": costs.slippage,
                "net_pnl": net,
                "r_multiple": r_multiple,
                "exit_reason": reason.value,
                "holding_minutes": position.holding_minutes(now),
                "ml_probability": position.entry_ml_probability,
                "mae_r": position.mae_r,
                "mfe_r": position.mfe_r,
                "explanation": "\n".join(
                    self.explainer.explain_exit(
                        reason=reason, detail=decision.detail, pnl=net,
                        r_multiple=r_multiple, holding_minutes=position.holding_minutes(now),
                    )
                ),
            }
        )

        won = net > 0
        self.risk.on_trade_closed(won=won, now=now)
        health = self.health.record(
            TradeOutcome(
                strategy=position.strategy,
                underlying=position.underlying,
                timestamp=now,
                r_multiple=r_multiple,
                pnl=net,
                regime=position.entry_regime,
                won=won,
                holding_minutes=position.holding_minutes(now),
            )
        )
        if not health.can_trade:
            self.alerts.strategy_disabled(
                f"{position.strategy} is now {health.state.value}: {health.note}"
            )

        if decision.action is not ManagementAction.PARTIAL_EXIT:
            self.positions.untrack(position.position_id)

        self.alerts.trade_exit(
            f"{position.instrument.trading_symbol} exit {reason.value}: {net:+,.0f} "
            f"({r_multiple:+.2f}R)",
            position_id=position.position_id,
        )
        logger.info(
            "EXITED %s %s: %+.0f (%.2fR) — %s",
            position.instrument.trading_symbol, reason.value, net, r_multiple, decision.detail,
        )
        return True

    # ================================================================== #
    # Helpers
    # ================================================================== #
    def _underlying_instruments(self) -> list[Instrument]:
        """The spot/index instruments whose price drives the analysis."""
        if self.universe is None:
            return []
        out: list[Instrument] = []
        for group in self.universe.underlyings.values():
            if group.spot is not None:
                out.append(group.spot)
            else:
                # No spot in the master: fall back to the nearest future as the
                # underlying reference rather than skipping the symbol entirely.
                nearest = group.nearest_future(to_ist(self.clock.now()).date())
                if nearest is not None:
                    out.append(nearest)
        return out

    def _underlying_for(self, symbol: str) -> str | None:
        if self.universe is None:
            return None
        if symbol in self.universe.underlyings:
            return symbol
        for name, group in self.universe.underlyings.items():
            if group.spot is not None and group.spot.trading_symbol == symbol:
                return name
            if any(f.trading_symbol == symbol for f in group.futures):
                return name
        return None

    def _lot_size_for(self, underlying: str) -> int:
        group = self.universe.get(underlying) if self.universe else None
        return group.lot_size if group else 1

    def _chains_wanted(self, now: datetime) -> dict[str, date]:
        if self.universe is None:
            return {}
        today = to_ist(now).date()
        wanted: dict[str, date] = {}
        for name in self.universe.symbols():
            expiries = self.discovery.tradable_expiries(name, as_of=today)
            if expiries:
                wanted[name] = expiries[0]
        return wanted

    def _refresh_equity(self) -> None:
        margin = self._margin()
        equity, source = effective_equity(self.config.capital, margin)
        unrealized = self.positions.total_unrealized()
        self._equity = equity + unrealized
        logger.debug("equity %.0f from %s (unrealized %+.0f)", self._equity, source, unrealized)

    def _margin(self):
        try:
            return self.execution_broker.get_margin()
        except Exception as exc:
            logger.debug("margin unavailable: %s", exc)
            return None

    def _portfolio_state(self, now: datetime):
        try:
            broker_positions = self.execution_broker.get_positions()
        except BrokerError:
            broker_positions = []
        return self.portfolio.build_state(
            positions=broker_positions,
            equity=self._equity,
            now=now,
        )

    def _record_no_trade(
        self, decision_id: str, opportunity: Opportunity, now: datetime, *,
        reason: str, selection=None, expected_value=None,
    ) -> None:
        self.journal.record_decision(
            {
                "decision_id": decision_id,
                "timestamp": now.isoformat(),
                "run_id": self.run_id,
                "underlying": opportunity.underlying,
                "decision": Decision.NO_TRADE.value,
                "regime": opportunity.regime.dominant.value if opportunity.regime else None,
                "regime_confidence": opportunity.regime.confidence if opportunity.regime else None,
                "ensemble_score": opportunity.score,
                "approved": False,
                "rejection_reason": reason,
            }
        )

    def _write_decision(
        self, decision_id: str, opportunity: Opportunity, now: datetime, *,
        instrument: Instrument, approval, expected_value, explanation, approved: bool,
    ) -> None:
        ensemble = opportunity.ensemble
        sizing = approval.sizing
        self.journal.record_decision(
            {
                "decision_id": decision_id,
                "timestamp": now.isoformat(),
                "run_id": self.run_id,
                "underlying": opportunity.underlying,
                "instrument": instrument.trading_symbol,
                "expiry": instrument.expiry_date.isoformat() if instrument.expiry_date else None,
                "strike": instrument.strike_price,
                "option_type": instrument.instrument_type.value,
                "decision": ensemble.decision.value if ensemble else Decision.NO_TRADE.value,
                "direction": ensemble.direction.value if ensemble else None,
                "strategy": ensemble.agreeing_strategies[0] if ensemble and ensemble.agreeing_strategies else None,
                "strategies_agreeing": ensemble.agreeing_strategies if ensemble else [],
                "regime": ensemble.regime.value if ensemble else None,
                "regime_confidence": ensemble.regime_confidence if ensemble else None,
                "ml_probability": ensemble.ml_probability if ensemble else None,
                "ensemble_score": ensemble.score if ensemble else None,
                "ensemble_confidence": ensemble.confidence if ensemble else None,
                "expected_value": expected_value.expected_value_rupees if expected_value else None,
                "expected_value_r": expected_value.expected_value_r if expected_value else None,
                "entry_price": ensemble.proposed_entry if ensemble else None,
                "stop_price": ensemble.proposed_stop if ensemble else None,
                "target_price": ensemble.proposed_target if ensemble else None,
                "position_size": sizing.quantity if sizing else 0,
                "lots": sizing.lots if sizing else 0,
                "risk_amount": sizing.risk_amount if sizing else 0.0,
                "approved": approved,
                "rejection_reason": "; ".join(approval.rejection_reasons) or None,
                "risk_checks": approval.checklist(),
                "portfolio_state": approval.portfolio_after.describe() if approval.portfolio_after else None,
                "explanation": explanation.render(),
                "reasoning": ensemble.reasons if ensemble else [],
            }
        )

    def _record_equity_snapshot(self) -> None:
        margin = self._margin()
        self.journal.record_equity(
            equity=self._equity,
            unrealized_pnl=self.positions.total_unrealized(),
            open_positions=self.positions.count,
            drawdown_pct=self.drawdown.state.drawdown_pct,
            drawdown_level=self.drawdown.state.level.value,
            margin_used=margin.net_margin_used if margin else 0.0,
            margin_available=margin.available_margin if margin else 0.0,
        )

    # ================================================================== #
    def run_forever(self, *, max_cycles: int | None = None) -> None:
        """Main loop. Interruptible and safe to stop at any point."""
        cycles = 0
        interval = self.config.execution.loop_interval_seconds
        logger.warning("%s — entering main loop", self.config.mode_banner())
        try:
            while max_cycles is None or cycles < max_cycles:
                cycle_started = time.perf_counter()
                try:
                    report = self.run_cycle()
                    if report.halted_reason:
                        logger.debug("cycle: %s", report.summary())
                    else:
                        logger.info("cycle: %s", report.summary())
                except Exception as exc:
                    logger.exception("cycle failed: %s", exc)
                    self.journal.record_event(
                        level="ERROR", category="cycle", message=str(exc)
                    )
                cycles += 1
                elapsed = time.perf_counter() - cycle_started
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            logger.warning("interrupted by operator")
        finally:
            self.shutdown()

    def shutdown(self, *, close_positions: bool = False) -> None:
        logger.warning("shutting down (%s)", self.config.mode_banner())
        if close_positions:
            now = self.clock.now()
            for position in list(self.positions.all()):
                try:
                    quote = self.data_broker.get_quote(position.instrument)
                except BrokerError:
                    quote = None
                from .execution.position_manager import ManagementDecision

                self._exit(
                    position,
                    ManagementDecision(
                        ManagementAction.EMERGENCY_EXIT, ExitReason.EMERGENCY_SHUTDOWN,
                        "shutdown requested with position closure",
                    ),
                    position.quantity, now, quote,
                )
        self.journal.record_event(
            level="INFO", category="lifecycle", message="shutdown",
            payload={"run_id": self.run_id, "open_positions": self.positions.count},
        )
        self.started = False

    # ================================================================== #
    def status(self) -> dict[str, Any]:
        """Everything the dashboard and the `status` CLI command need."""
        return {
            "mode": self.config.mode.value,
            "run_id": self.run_id,
            "started": self.started,
            "trading_allowed": self.trading_allowed,
            "equity": round(self._equity, 2),
            "universe_size": len(self.universe or []),
            "open_positions": self.positions.count,
            "unrealized_pnl": round(self.positions.total_unrealized(), 2),
            "regime": self.regime_engine.current.dominant.value if self.regime_engine.current else None,
            "regime_confidence": (
                round(self.regime_engine.current.confidence, 3)
                if self.regime_engine.current else None
            ),
            "risk": self.risk.status(),
            "strategy_health": self.health.summary(),
            "prediction_engine": self.prediction.health(),
            "market_data": self.market_data.health(),
            "latency": self.latency.statistics(),
            "last_cycle": self.last_cycle.summary() if self.last_cycle else None,
            "last_reconciliation": (
                self.last_reconciliation.summary() if self.last_reconciliation else None
            ),
            "alerts": self.alerts.summary(),
        }
