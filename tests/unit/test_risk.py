"""Risk engine, sizing, drawdown and kill-switch tests (REQ 24/25/26/27/44)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from aqtp.core.clock import IST
from aqtp.core.types import (
    Decision,
    DrawdownLevel,
    Instrument,
    MarginInfo,
    Position,
    StrategyState,
)
from aqtp.events.risk import EventRiskEngine
from aqtp.risk.costs import TransactionCostModel
from aqtp.risk.drawdown import DrawdownProtection
from aqtp.risk.engine import RiskEngine
from aqtp.risk.killswitch import EmergencyPolicy, KillScope, KillSwitchManager
from aqtp.risk.portfolio import PortfolioManager
from aqtp.risk.sizing import PositionSizer, effective_equity
from aqtp.signals.expected_value import ExpectedValueCalculator
from aqtp.strategies.health import StrategyHealthMonitor, TradeOutcome
from tests.conftest import make_quote


# =========================================================================== #
# Position sizing (REQ 24)
# =========================================================================== #
class TestPositionSizing:
    def test_size_respects_the_risk_budget(self, config, call_instrument):
        sizer = PositionSizer(config.capital, config.risk)
        result = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=170.0,
            equity=500_000, available_margin=400_000,
        )
        assert result.ok
        # 1% of 500k = 5,000 budget; 30/unit risk; 75/lot -> 2 lots (4,500 risk).
        assert result.lots == 2
        assert result.quantity == 150
        assert result.risk_amount == pytest.approx(4_500)
        assert result.risk_amount <= 500_000 * config.risk.risk_per_trade_pct

    def test_never_rounds_up_past_the_budget(self, config, call_instrument):
        """One lot exceeding the budget must produce zero, not one."""
        sizer = PositionSizer(config.capital, config.risk)
        result = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=100.0,
            equity=500_000, available_margin=400_000,
        )
        assert not result.ok
        assert result.quantity == 0
        assert "exceeding" in result.rejection_reason

    def test_quantity_is_always_a_lot_multiple(self, config, call_instrument):
        sizer = PositionSizer(config.capital, config.risk)
        for equity in (200_000, 500_000, 1_000_000, 2_500_000):
            result = sizer.size(
                instrument=call_instrument, entry_price=200.0, stop_price=180.0,
                equity=equity, available_margin=equity,
            )
            if result.ok:
                assert result.quantity % call_instrument.lot_size == 0

    def test_freeze_quantity_caps_the_size(self, config):
        instrument = Instrument(
            trading_symbol="X", exchange=__import__("aqtp.core.types", fromlist=["Exchange"]).Exchange.NSE,
            segment=__import__("aqtp.core.types", fromlist=["Segment"]).Segment.FNO,
            instrument_type=__import__("aqtp.core.types", fromlist=["InstrumentType"]).InstrumentType.CE,
            lot_size=50, freeze_quantity=100, expiry_date=date(2026, 6, 25), strike_price=100.0,
        )
        sizer = PositionSizer(config.capital, config.risk)
        result = sizer.size(
            instrument=instrument, entry_price=10.0, stop_price=9.0,
            equity=50_000_000, available_margin=50_000_000,
        )
        assert result.quantity <= instrument.freeze_quantity
        assert "freeze" in result.capped_by.lower()

    def test_risk_multiplier_scales_size_down(self, config, call_instrument):
        sizer = PositionSizer(config.capital, config.risk)
        full = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=170.0,
            equity=1_000_000, available_margin=1_000_000,
        )
        half = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=170.0,
            equity=1_000_000, available_margin=1_000_000, risk_multiplier=0.5,
        )
        assert half.quantity < full.quantity
        assert half.risk_amount <= full.risk_amount / 2 + call_instrument.lot_size * 30

    def test_zero_stop_distance_is_rejected(self, config, call_instrument):
        sizer = PositionSizer(config.capital, config.risk)
        result = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=200.0,
            equity=500_000, available_margin=500_000,
        )
        assert not result.ok
        assert "stop distance is zero" in result.rejection_reason

    def test_margin_shortfall_caps_the_size(self, config, call_instrument):
        sizer = PositionSizer(config.capital, config.risk)
        result = sizer.size(
            instrument=call_instrument, entry_price=200.0, stop_price=170.0,
            equity=500_000, available_margin=20_000,
        )
        # 200 x 75 = 15,000 per lot; only one lot fits in 20,000 of margin.
        assert result.lots <= 1


class TestEffectiveEquity:
    def test_broker_equity_wins_when_lower(self, config):
        margin = MarginInfo(clear_cash=200_000, net_margin_used=0, collateral_available=0)
        equity, source = effective_equity(config.capital, margin)
        assert equity == 200_000
        assert "broker" in source

    def test_configured_capital_caps_a_larger_broker_balance(self, config):
        margin = MarginInfo(clear_cash=5_000_000, net_margin_used=0)
        equity, _ = effective_equity(config.capital, margin)
        assert equity == config.capital.available_capital


# =========================================================================== #
# Drawdown ladder (REQ 26)
# =========================================================================== #
class TestDrawdownProtection:
    def test_ladder_escalates_through_all_four_levels(self, config):
        """Drawdown is measured from the PEAK, independently of the daily loss limit.

        The session therefore opens at 1,000,000 and rises to a 1,200,000 peak first,
        so the subsequent decline is a drawdown without also being a daily loss —
        which is what isolates this ladder from the daily-loss control.
        """
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        protection.update_equity(1_200_000)
        peak = 1_200_000
        assert protection.state.level is DrawdownLevel.NORMAL

        protection.update_equity(peak * 0.955)   # -4.5% from peak
        assert protection.state.level is DrawdownLevel.REDUCED
        protection.update_equity(peak * 0.925)   # -7.5% from peak
        assert protection.state.level is DrawdownLevel.PAUSED
        protection.update_equity(peak * 0.895)   # -10.5% from peak
        assert protection.state.level is DrawdownLevel.SHUTDOWN

    def test_daily_loss_limit_pauses_independently_of_drawdown(self, config):
        """A 4.5% loss on the day trips the daily limit even at a shallow drawdown."""
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        protection.update_equity(955_000)
        assert protection.state.level is DrawdownLevel.PAUSED
        assert "daily loss" in protection.state.reason

    def test_shutdown_is_not_cleared_by_recovery(self, config):
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        protection.update_equity(880_000)
        assert protection.state.is_shutdown

        protection.update_equity(1_000_000)
        assert protection.state.is_shutdown, "shutdown must require an explicit reset"

        protection.reset_shutdown(operator_note="investigated")
        assert protection.state.level is DrawdownLevel.NORMAL

    def test_recovery_requires_clearing_the_threshold_with_a_buffer(self, config):
        """Hysteresis: hovering at the boundary must not flip the level each tick."""
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        protection.update_equity(1_200_000)  # establish the peak
        peak = 1_200_000

        protection.update_equity(peak * 0.958)  # -4.2% from peak, into REDUCED
        assert protection.state.level is DrawdownLevel.REDUCED

        protection.update_equity(peak * 0.961)  # -3.9%, still above 80% of the trigger
        assert protection.state.level is DrawdownLevel.REDUCED

        protection.update_equity(peak * 0.970)  # -3.0%, below the 3.2% buffer
        assert protection.state.level is DrawdownLevel.NORMAL

    def test_risk_is_never_increased_after_wins(self, config):
        """REQ 26 forbids raising risk after a winning streak."""
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        now = datetime(2026, 6, 1, 10, 0, tzinfo=IST)
        for _ in range(10):
            protection.record_trade(won=True, now=now)
        protection.update_equity(1_200_000)
        assert protection.state.risk_multiplier == 1.0

    def test_consecutive_losses_pause_trading(self, config):
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        now = datetime(2026, 6, 1, 10, 0, tzinfo=IST)
        for i in range(config.risk.max_consecutive_losses):
            protection.record_trade(won=False, now=now + timedelta(minutes=i))
        assert protection.state.level is DrawdownLevel.PAUSED

    def test_cooldown_blocks_trading_after_a_loss(self, config):
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        now = datetime(2026, 6, 1, 10, 0, tzinfo=IST)
        protection.record_trade(won=False, now=now)

        allowed, reason = protection.can_trade(now + timedelta(minutes=1))
        assert not allowed and "cooling down" in reason

        allowed, _ = protection.can_trade(
            now + timedelta(minutes=config.risk.cooldown_minutes_after_loss + 1)
        )
        assert allowed

    def test_daily_trade_limit_is_enforced(self, config):
        protection = DrawdownProtection(config.risk)
        protection.start_session(1_000_000, session_date=date(2026, 6, 1))
        now = datetime(2026, 6, 1, 10, 0, tzinfo=IST)
        for i in range(config.risk.max_trades_per_day):
            protection.record_trade(won=True, now=now + timedelta(minutes=i))
        allowed, reason = protection.can_trade(now + timedelta(hours=2))
        assert not allowed and "daily trade limit" in reason


# =========================================================================== #
# Kill switches (REQ 44)
# =========================================================================== #
class TestKillSwitches:
    def test_global_switch_blocks_every_scope(self, tmp_path):
        switches = KillSwitchManager(tmp_path / "ks.json")
        switches.engage(KillScope.GLOBAL, reason="test")
        for scope in KillScope:
            allowed, reason = switches.check(scope, target="anything")
            assert not allowed and "GLOBAL" in reason

    def test_targeted_switch_blocks_only_its_target(self, tmp_path):
        switches = KillSwitchManager(tmp_path / "ks.json")
        switches.engage(KillScope.STRATEGY, target="momentum", reason="underperforming")

        allowed, _ = switches.can_trade(strategy="momentum")
        assert not allowed
        allowed, _ = switches.can_trade(strategy="trend_following")
        assert allowed

    def test_switches_survive_a_restart(self, tmp_path):
        path = tmp_path / "ks.json"
        KillSwitchManager(path).engage(KillScope.GLOBAL, reason="crash test")
        restarted = KillSwitchManager(path)
        assert restarted.is_shutdown, "kill switches must persist across a restart"

    def test_emergency_shutdown_records_its_policy(self, tmp_path):
        switches = KillSwitchManager(tmp_path / "ks.json")
        switches.emergency_shutdown("margin call", policy=EmergencyPolicy.CLOSE_POSITIONS)
        assert switches.is_shutdown
        assert switches.emergency_policy is EmergencyPolicy.CLOSE_POSITIONS


# =========================================================================== #
# Strategy health (REQ 27/28)
# =========================================================================== #
class TestStrategyHealth:
    @staticmethod
    def _record(monitor, strategy, r_values, start=datetime(2026, 6, 1, 10, 0, tzinfo=IST)):
        from aqtp.core.types import Regime

        for i, r in enumerate(r_values):
            monitor.record(
                TradeOutcome(
                    strategy=strategy, underlying="NIFTY", timestamp=start + timedelta(minutes=i),
                    r_multiple=r, pnl=r * 1000, regime=Regime.TRENDING_UP, won=r > 0,
                )
            )

    def test_too_few_trades_leaves_state_active(self, config):
        monitor = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor, "momentum", [-1.0] * 5)
        assert monitor.health("momentum").state is StrategyState.ACTIVE

    def test_poor_expectancy_reduces_then_pauses_then_disables(self, config):
        monitor = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor, "s_reduce", [-0.1] * 15)
        assert monitor.health("s_reduce").state is StrategyState.REDUCED

        monitor2 = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor2, "s_pause", [-0.25] * 15)
        assert monitor2.health("s_pause").state is StrategyState.PAUSED

        monitor3 = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor3, "s_disable", [-0.5] * 15)
        assert monitor3.health("s_disable").state is StrategyState.DISABLED

    def test_disabled_strategy_cannot_trade(self, config):
        monitor = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor, "bad", [-0.6] * 15)
        allowed, reason = monitor.can_trade("bad")
        assert not allowed and "DISABLED" in reason
        assert monitor.risk_multiplier("bad") == 0.0

    def test_allocation_weights_are_shrunk_toward_equal(self, config):
        """REQ 53: do not over-allocate on a small recent sample."""
        monitor = StrategyHealthMonitor(config.strategy_health)
        self._record(monitor, "hot", [1.5] * 15)
        self._record(monitor, "cold", [0.05] * 15)
        weights = monitor.allocation_weights(["hot", "cold"])

        assert weights["hot"] > weights["cold"]
        # Shrinkage keeps the hot strategy well below a naive proportional share.
        assert weights["hot"] < 0.75
        assert sum(weights.values()) == pytest.approx(1.0)


# =========================================================================== #
# RiskEngine (REQ 25)
# =========================================================================== #
@pytest.fixture
def risk_engine(config):
    kill_switches = KillSwitchManager(None)
    drawdown = DrawdownProtection(config.risk)
    drawdown.start_session(500_000, session_date=date(2026, 6, 1))
    return RiskEngine(
        config,
        portfolio=PortfolioManager(),
        drawdown=drawdown,
        kill_switches=kill_switches,
        sizer=PositionSizer(config.capital, config.risk),
        cost_model=TransactionCostModel(config.costs),
        event_risk=EventRiskEngine(config.event_risk),
        health=StrategyHealthMonitor(config.strategy_health),
    )


def _evaluate(engine, config, instrument, *, quote=None, now=None, **overrides):
    from aqtp.risk.portfolio import PortfolioManager

    now = now or datetime(2026, 6, 1, 11, 0, tzinfo=IST)
    quote = quote or make_quote(instrument, price=200.0, spread=1.0, now=now)
    state = PortfolioManager().build_state(positions=[], equity=500_000, now=now)
    ev = ExpectedValueCalculator(
        config.expected_value, TransactionCostModel(config.costs)
    ).evaluate(
        instrument=instrument, quantity=150, entry_price=200.0, stop_price=170.0,
        target_price=250.0, win_probability=0.65, quote=quote,
    )
    kwargs = dict(
        decision=Decision.BUY, instrument=instrument, underlying="NIFTY",
        strategy="trend_following", entry_price=200.0, stop_price=170.0, target_price=250.0,
        portfolio_state=state, margin=MarginInfo(clear_cash=400_000), quote=quote,
        expected_value=ev, now=now,
    )
    kwargs.update(overrides)
    return engine.evaluate(**kwargs)


class TestRiskEngine:
    def test_a_sound_trade_is_approved(self, risk_engine, config, call_instrument):
        approval = _evaluate(risk_engine, config, call_instrument)
        assert approval.approved, approval.rejection_reasons
        assert approval.quantity > 0
        assert approval.approval_id

    def test_every_control_is_evaluated_not_just_the_first_failure(
        self, risk_engine, config, call_instrument
    ):
        approval = _evaluate(risk_engine, config, call_instrument)
        names = {c.name for c in approval.checks}
        for expected in (
            "kill_switch", "drawdown_protection", "daily_loss_limit", "max_open_positions",
            "max_spread", "min_liquidity", "expected_value", "position_sizing",
            "max_risk_per_trade", "max_exposure_per_instrument", "max_correlated_exposure",
            "max_portfolio_delta", "max_portfolio_gamma",
        ):
            assert expected in names, f"risk control {expected} was not evaluated"

    def test_kill_switch_blocks_approval(self, risk_engine, config, call_instrument):
        risk_engine.kill_switches.engage(KillScope.GLOBAL, reason="test halt")
        approval = _evaluate(risk_engine, config, call_instrument)
        assert not approval.approved
        assert any("kill_switch" in r for r in approval.rejection_reasons)

    def test_wide_spread_is_rejected(self, risk_engine, config, call_instrument):
        wide = make_quote(call_instrument, price=200.0, spread=20.0)
        approval = _evaluate(risk_engine, config, call_instrument, quote=wide)
        assert not approval.approved
        assert any("max_spread" in r for r in approval.rejection_reasons)

    def test_thin_liquidity_is_rejected(self, risk_engine, config, call_instrument):
        thin = make_quote(call_instrument, price=200.0, spread=1.0, volume=10, open_interest=5)
        approval = _evaluate(risk_engine, config, call_instrument, quote=thin)
        assert not approval.approved
        reasons = " ".join(approval.rejection_reasons)
        assert "min_liquidity" in reasons or "min_open_interest" in reasons

    def test_drawdown_shutdown_blocks_approval(self, risk_engine, config, call_instrument):
        risk_engine.drawdown.update_equity(400_000)  # -20% from the 500k peak
        approval = _evaluate(risk_engine, config, call_instrument)
        assert not approval.approved

    def test_position_limit_blocks_approval(self, risk_engine, config, call_instrument):
        from aqtp.risk.portfolio import PortfolioManager

        now = datetime(2026, 6, 1, 11, 0, tzinfo=IST)
        positions = [
            Position(instrument=call_instrument, quantity=75, average_price=200.0, last_price=200.0)
            for _ in range(config.risk.max_open_positions)
        ]
        state = PortfolioManager().build_state(positions=positions, equity=500_000, now=now)
        approval = _evaluate(risk_engine, config, call_instrument, portfolio_state=state)
        assert not approval.approved
        assert any("max_open_positions" in r for r in approval.rejection_reasons)

    def test_failing_expected_value_blocks_approval(self, risk_engine, config, call_instrument):
        now = datetime(2026, 6, 1, 11, 0, tzinfo=IST)
        quote = make_quote(call_instrument, price=200.0, spread=1.0, now=now)
        bad_ev = ExpectedValueCalculator(
            config.expected_value, TransactionCostModel(config.costs)
        ).evaluate(
            instrument=call_instrument, quantity=150, entry_price=200.0, stop_price=170.0,
            target_price=250.0, win_probability=0.30, quote=quote,
        )
        approval = _evaluate(risk_engine, config, call_instrument, expected_value=bad_ev)
        assert not approval.approved
        assert any("expected_value" in r for r in approval.rejection_reasons)

    def test_no_trade_decision_is_refused_outright(self, risk_engine, config, call_instrument):
        approval = _evaluate(risk_engine, config, call_instrument, decision=Decision.NO_TRADE)
        assert not approval.approved
