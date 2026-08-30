"""Command-line interface (REQ 58/59).

Every command REQ 59 lists is implemented, plus the operator commands a live desk
actually needs: `aqtp start` to run the production loop, `aqtp analyze` for the
full analysis of one underlying, and `aqtp trade ...` to place, modify and close
orders by hand through the same risk pipeline the bot uses.

The interface always shows the current trading mode. LIVE is the default mode and
is guarded at three levels: the config interlocks, a typed confirmation phrase,
and a prominent banner.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..configuration.loader import Credentials, describe_config, load_config
from ..configuration.schema import AppConfig
from ..core.errors import AQTPError, ConfigurationError
from ..core.logging import setup_logging
from ..core.types import TradingMode
from ..journal.store import TradeJournal
from ..risk.killswitch import EmergencyPolicy, KillScope, KillSwitchManager

app = typer.Typer(
    name="aqtp",
    help="Adaptive Quantitative Trading Platform for Indian F&O markets.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

CONFIG_OPTION = typer.Option("config/default.yaml", "--config", "-c", help="Path to config YAML.")
MODE_OPTION = typer.Option(None, "--mode", "-m", help="Override the trading mode.")

_MODE_STYLE = {
    TradingMode.PAPER: "bold cyan",
    TradingMode.DEMO: "bold yellow",
    TradingMode.LIVE: "bold white on red",
    TradingMode.BACKTEST: "dim",
}


def _load(config_path: str, mode: str | None = None) -> AppConfig:
    overrides = {"mode": mode.upper()} if mode else None
    try:
        return load_config(config_path, overrides=overrides)
    except ConfigurationError as exc:
        console.print(f"[bold red]Configuration error[/]\n{exc}")
        raise typer.Exit(2)


def _load_disarmed(config_path: str, mode: str | None = None, *, quiet: bool = False) -> AppConfig:
    overrides = {"mode": mode.upper()} if mode else None
    try:
        return load_config(config_path, overrides=overrides)
    except ConfigurationError as exc:
        if "interlock" not in str(exc):
            console.print(f"[bold red]Configuration error[/]\n{exc}")
            raise typer.Exit(2)
        config = _load(config_path, mode="PAPER")
        if not quiet:
            console.print(
                "[yellow]LIVE is configured but not armed — showing this as PAPER for "
                "inspection. Run `aqtp arm` to see the interlocks.[/]"
            )
        return config


def _banner(config: AppConfig) -> None:
    """REQ 42/43: the mode must be prominently visible at all times."""
    style = _MODE_STYLE.get(config.mode, "bold")
    extra = ""
    if config.mode is TradingMode.LIVE:
        extra = "\n[bold red]REAL ORDERS WILL BE PLACED WITH REAL MONEY[/]"
    elif config.mode is TradingMode.PAPER:
        extra = "\n[dim]Real market data, simulated execution. No broker orders.[/]"
    elif config.mode is TradingMode.DEMO:
        extra = "\n[dim]Orders are routed to the broker's demo environment.[/]"
    console.print(
        Panel(
            f"[{style}]TRADING MODE: {config.mode.value}[/]{extra}",
            subtitle=f"capital ₹{config.capital.available_capital:,.0f} · "
                     f"risk {config.risk.risk_per_trade_pct:.2%}/trade · "
                     f"broker {config.broker.name}",
            border_style="red" if config.mode is TradingMode.LIVE else "cyan",
        )
    )


def _journal(config: AppConfig) -> TradeJournal:
    return TradeJournal(config.paths.journal_db, mode=config.mode.value)


def _build(config: AppConfig):
    """Construct a fully-wired orchestrator."""
    from ..brokers.factory import build_data_broker, build_execution_broker
    from ..orchestrator import TradingOrchestrator

    credentials = Credentials.from_env()
    credentials.validate_for(config.mode)
    data_broker = build_data_broker(config, credentials)

    def quote_source(instrument):
        try:
            return data_broker.get_quote(instrument)
        except Exception:
            return None

    execution_broker = build_execution_broker(
        config, credentials, quote_source=quote_source, data_broker=data_broker
    )
    journal = _journal(config)
    return TradingOrchestrator(
        config,
        data_broker=data_broker,
        execution_broker=execution_broker,
        journal=journal,
    )


# =========================================================================== #
# Configuration and diagnostics
# =========================================================================== #
@app.command("config")
def show_config(
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = MODE_OPTION,
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show one section only."),
    validate_only: bool = typer.Option(False, "--validate", help="Validate and exit."),
) -> None:
    """Show and validate the effective configuration (REQ 58)."""
    config = _load(config_path, mode)
    _banner(config)
    if validate_only:
        console.print("[green]configuration is valid[/]")
        raise typer.Exit(0)

    data = describe_config(config)
    if section:
        if section not in data:
            console.print(f"[red]no such section {section!r}[/] — available: {sorted(data)}")
            raise typer.Exit(1)
        data = {section: data[section]}
    console.print_json(json.dumps(data, default=str))


@app.command("doctor")
def doctor(config_path: str = CONFIG_OPTION) -> None:
    """Check configuration, credentials and broker connectivity."""
    config = _load(config_path)
    _banner(config)

    table = Table("check", "result", "detail", box=None)

    table.add_row("configuration", "[green]ok[/]", "validated")

    credentials = Credentials.from_env()
    try:
        credentials.validate_for(config.mode)
        table.add_row("credentials", "[green]ok[/]", repr(credentials))
    except ConfigurationError as exc:
        table.add_row("credentials", "[red]fail[/]", str(exc))
        console.print(table)
        raise typer.Exit(1)

    if config.mode is TradingMode.LIVE:
        table.add_row("live interlocks", "[green]ok[/]", "all three confirmed")

    try:
        from ..brokers.factory import build_data_broker

        broker = build_data_broker(config, credentials)
        broker.authenticate()
        table.add_row("broker auth", "[green]ok[/]", broker.name)
        instruments = broker.fetch_instruments()
        table.add_row("instruments", "[green]ok[/]", f"{len(instruments):,} contracts")
        margin = broker.get_margin()
        table.add_row("margin", "[green]ok[/]", f"₹{margin.available_margin:,.0f} available")
    except Exception as exc:
        table.add_row("broker", "[red]fail[/]", str(exc)[:100])

    console.print(table)


# =========================================================================== #
# Status and inspection (REQ 59)
# =========================================================================== #
@app.command("status")
def status(config_path: str = CONFIG_OPTION) -> None:
    """Show system status."""
    config = _load_disarmed(config_path)
    _banner(config)
    journal = _journal(config)
    stats = journal.statistics()

    table = Table("metric", "value", box=None)
    for key, value in stats.items():
        table.add_row(key.replace("_", " "), str(value))

    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    table.add_row("kill switches", str(len(switches.active())))
    if switches.is_shutdown:
        table.add_row("[red]GLOBAL SHUTDOWN[/]", "[red]engaged[/]")
    console.print(table)


@app.command("portfolio")
def portfolio(config_path: str = CONFIG_OPTION) -> None:
    """Show portfolio exposure and aggregate greeks."""
    config = _load(config_path)
    _banner(config)
    try:
        orchestrator = _build(config)
    except AQTPError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    state = orchestrator._portfolio_state(orchestrator.clock.now())
    table = Table("metric", "value", box=None)
    for key, value in state.describe().items():
        table.add_row(key.replace("_", " "), f"{value:,.2f}" if isinstance(value, float) else str(value))
    console.print(table)

    if state.exposure_by_underlying:
        exposure = Table("underlying", "exposure", "% of equity", box=None)
        for name, value in sorted(
            state.exposure_by_underlying.items(), key=lambda kv: kv[1], reverse=True
        ):
            exposure.add_row(name, f"{value:,.0f}", f"{state.underlying_exposure_pct(name):.1%}")
        console.print(exposure)


@app.command("positions")
def positions(config_path: str = CONFIG_OPTION) -> None:
    """Show open positions reported by the broker."""
    config = _load(config_path)
    _banner(config)
    orchestrator = _build(config)
    orchestrator.execution_broker.authenticate()
    rows = orchestrator.execution_broker.get_positions()
    if not rows:
        console.print("[dim]no open positions[/]")
        return
    table = Table("instrument", "qty", "avg price", "last", "unrealized", box=None)
    for position in rows:
        table.add_row(
            position.instrument.trading_symbol,
            str(position.quantity),
            f"{position.average_price:,.2f}",
            f"{position.last_price:,.2f}",
            f"{position.unrealized_pnl:+,.2f}",
        )
    console.print(table)


@app.command("orders")
def orders(config_path: str = CONFIG_OPTION, limit: int = 20) -> None:
    """Show recent orders from the journal."""
    config = _load_disarmed(config_path)
    _banner(config)
    rows = _journal(config).query(
        "SELECT timestamp, instrument, transaction_type, quantity, price, status, "
        "filled_quantity, average_fill_price, total_latency_ms FROM orders "
        "ORDER BY id DESC LIMIT ?", (limit,)
    )
    if not rows:
        console.print("[dim]no orders recorded[/]")
        return
    table = Table("time", "instrument", "side", "qty", "price", "status", "filled", "latency ms", box=None)
    for row in rows:
        table.add_row(
            (row["timestamp"] or "")[11:19], row["instrument"] or "", row["transaction_type"] or "",
            str(row["quantity"] or ""), f"{row['price'] or 0:.2f}", row["status"] or "",
            f"{row['filled_quantity'] or 0}", f"{row['total_latency_ms'] or 0:.0f}",
        )
    console.print(table)


@app.command("signals")
def signals(config_path: str = CONFIG_OPTION, limit: int = 20) -> None:
    """Show recent decisions, including NO_TRADE reasons."""
    config = _load_disarmed(config_path)
    _banner(config)
    rows = _journal(config).recent_decisions(limit)
    if not rows:
        console.print("[dim]no decisions recorded[/]")
        return
    table = Table("time", "underlying", "decision", "score", "ml p", "status", "reason", box=None)
    for row in rows:
        approved = "[green]approved[/]" if row["approved"] else "[yellow]rejected[/]"
        table.add_row(
            (row["timestamp"] or "")[11:19], row["underlying"] or "", row["decision"] or "",
            f"{row['ensemble_score'] or 0:.3f}", f"{row['ml_probability'] or 0:.3f}",
            approved, (row["rejection_reason"] or "")[:60],
        )
    console.print(table)


@app.command("opportunities")
def opportunities(config_path: str = CONFIG_OPTION) -> None:
    """Run one scan cycle and show ranked opportunities."""
    config = _load(config_path)
    _banner(config)
    orchestrator = _build(config)
    if not orchestrator.start(warm_up_days=5):
        console.print("[red]startup failed[/]")
        raise typer.Exit(1)
    report = orchestrator.run_cycle()
    console.print(f"[dim]{report.summary()}[/]")

    if not report.opportunities:
        console.print("[dim]no actionable opportunities[/]")
        if report.no_trade_reasons:
            table = Table("no-trade reason", "count", box=None)
            for reason, count in sorted(
                report.no_trade_reasons.items(), key=lambda kv: kv[1], reverse=True
            )[:10]:
                table.add_row(reason, str(count))
            console.print(table)
        return

    table = Table("rank", "underlying", "decision", "score", "R:R", "components", box=None)
    for rank, opportunity in enumerate(report.opportunities, 1):
        table.add_row(
            str(rank), opportunity.underlying, opportunity.decision.value,
            f"{opportunity.score:.3f}", f"{opportunity.risk_reward:.2f}",
            ", ".join(f"{k}={v:.2f}" for k, v in sorted(opportunity.components.items())),
        )
    console.print(table)


@app.command("strategies")
def strategies(config_path: str = CONFIG_OPTION) -> None:
    """List strategies and their configured state."""
    config = _load_disarmed(config_path)
    _banner(config)
    from ..strategies.registry import available_strategies

    table = Table("strategy", "enabled", "weight", "params", box=None)
    for name in available_strategies():
        entry = config.strategies.get(name)
        enabled = "[green]yes[/]" if (entry is None or entry.enabled) else "[dim]no[/]"
        weight = f"{entry.weight:.2f}" if entry else "1.00"
        params = json.dumps(entry.params) if entry and entry.params else "defaults"
        table.add_row(name, enabled, weight, params[:50])
    console.print(table)


@app.command("models")
def models(config_path: str = CONFIG_OPTION) -> None:
    """List registered ML models (REQ 51)."""
    config = _load_disarmed(config_path)
    _banner(config)
    from ..ml.registry import ModelRegistry

    registry = ModelRegistry(config.paths.model_dir)
    rows = registry.summary()
    if not rows:
        console.print("[dim]no models registered — run `aqtp train`[/]")
        return
    table = Table("model id", "family", "status", "trained", "passed", "verdict", box=None)
    for row in rows:
        table.add_row(
            row["model_id"], row["family"], row["status"], row["trained"],
            "[green]yes[/]" if row["passed"] else "[red]no[/]", row["verdict"],
        )
    console.print(table)


@app.command("risk")
def risk(config_path: str = CONFIG_OPTION) -> None:
    """Show risk limits and current utilization."""
    config = _load_disarmed(config_path)
    _banner(config)
    table = Table("control", "limit", box=None)
    for key, value in config.risk.model_dump().items():
        formatted = f"{value:.2%}" if isinstance(value, float) and key.endswith("_pct") else str(value)
        table.add_row(key.replace("_", " "), formatted)
    console.print(table)

    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    if switches.active():
        console.print("\n[bold red]ACTIVE KILL SWITCHES[/]")
        for record in switches.active():
            console.print(f"  [red]{record.describe()}[/]")


@app.command("performance")
def performance(config_path: str = CONFIG_OPTION, days: int = 30) -> None:
    """Show realized performance from the journal."""
    config = _load_disarmed(config_path)
    _banner(config)
    since = (datetime.now() - timedelta(days=days)).isoformat()
    rows = _journal(config).query(
        "SELECT strategy, COUNT(*) n, SUM(net_pnl) pnl, AVG(r_multiple) avg_r, "
        "SUM(CASE WHEN net_pnl>0 THEN 1 ELSE 0 END) wins, SUM(costs) costs "
        "FROM trades WHERE exit_time IS NOT NULL AND entry_time >= ? "
        "GROUP BY strategy ORDER BY pnl DESC", (since,)
    )
    if not rows:
        console.print(f"[dim]no closed trades in the last {days} days[/]")
        return
    table = Table("strategy", "trades", "win rate", "net P&L", "avg R", "costs", box=None)
    total_pnl = total_trades = 0
    for row in rows:
        n = row["n"] or 0
        total_trades += n
        total_pnl += row["pnl"] or 0
        table.add_row(
            row["strategy"] or "—", str(n),
            f"{(row['wins'] or 0) / n:.1%}" if n else "—",
            f"{row['pnl'] or 0:+,.0f}", f"{row['avg_r'] or 0:+.3f}",
            f"{row['costs'] or 0:,.0f}",
        )
    table.add_section()
    table.add_row("[bold]TOTAL[/]", str(total_trades), "", f"[bold]{total_pnl:+,.0f}[/]", "", "")
    console.print(table)


# =========================================================================== #
# Running (REQ 41/42/43)
# =========================================================================== #
def _run(config: AppConfig, cycles: int | None, dashboard: bool) -> None:
    """Start the production loop under the runtime supervisor.

    The supervisor — not the bare orchestrator loop — is what runs here, because
    a live session needs signal handling, broker reconnection, a watchdog and an
    enforced square-off. See `aqtp/runtime/supervisor.py`.
    """
    from ..runtime.supervisor import RuntimeSupervisor

    setup_logging(config.log_level, log_dir=Path(config.paths.log_dir), json_format=config.json_logs)
    _banner(config)
    orchestrator = _build(config)

    dash = None
    if dashboard and config.monitoring.dashboard_enabled:
        from ..monitoring.dashboard import Dashboard

        dash = Dashboard(config.monitoring, orchestrator.status)
        dash.start()
        console.print(
            f"[dim]dashboard: http://{config.monitoring.dashboard_host}:"
            f"{config.monitoring.dashboard_port}[/]"
        )

    if not orchestrator.start():
        console.print("[bold red]startup failed — see the log for details[/]")
        if dash is not None:
            dash.stop()
        raise typer.Exit(1)

    supervisor = RuntimeSupervisor(orchestrator)
    try:
        code = supervisor.run(max_cycles=cycles)
    finally:
        if dash is not None:
            dash.stop()
    if code:
        raise typer.Exit(code)


@app.command("paper-start")
def paper_start(
    config_path: str = CONFIG_OPTION,
    cycles: Optional[int] = typer.Option(None, "--cycles", help="Stop after N cycles."),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard"),
) -> None:
    """Start PAPER trading: real market data, simulated execution (REQ 41)."""
    config = _load(config_path, mode="PAPER")
    _run(config, cycles, dashboard)


@app.command("paper-stop")
def paper_stop(config_path: str = CONFIG_OPTION) -> None:
    """Stop paper trading by engaging the application kill switch."""
    config = _load(config_path)
    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    switches.engage(KillScope.APPLICATION, reason="paper-stop requested", engaged_by="cli")
    console.print("[yellow]application kill switch engaged; the running loop will stop entering[/]")


@app.command("demo-start")
def demo_start(
    config_path: str = CONFIG_OPTION,
    cycles: Optional[int] = typer.Option(None, "--cycles"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard"),
) -> None:
    """Start DEMO trading against the broker's demo environment (REQ 42)."""
    config = _load(config_path, mode="DEMO")
    console.print("[yellow]DEMO mode submits orders to the broker's demo environment.[/]")
    if not typer.confirm("Continue?"):
        raise typer.Exit(0)
    _run(config, cycles, dashboard)


@app.command("demo-stop")
def demo_stop(config_path: str = CONFIG_OPTION) -> None:
    """Stop demo trading."""
    config = _load(config_path)
    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    switches.engage(KillScope.APPLICATION, reason="demo-stop requested", engaged_by="cli")
    console.print("[yellow]application kill switch engaged[/]")


@app.command("live-status")
def live_status(config_path: str = CONFIG_OPTION) -> None:
    """Show whether LIVE trading is currently permitted (REQ 43).

    This one deliberately loads disarmed: it is the command you run *because* you
    are not armed yet, so requiring the interlocks to run it would be circular.
    """
    from ..configuration.schema import LIVE_INTERLOCKS
    import os

    import yaml

    config = _load_disarmed(config_path, quiet=True)

    try:
        configured_mode = str(yaml.safe_load(Path(config_path).read_text()).get("mode", "PAPER"))
    except Exception:
        configured_mode = config.mode.value

    table = Table("interlock", "status", box=None)
    ready = True
    for key, expected in LIVE_INTERLOCKS.items():
        ok = os.environ.get(key, "") == expected
        ready = ready and ok
        table.add_row(key, "[green]set[/]" if ok else f"[red]missing[/] (expect {expected!r})")
    console.print(table)
    console.print(f"\nconfigured mode: [bold]{configured_mode}[/]")

    if ready and configured_mode == "LIVE":
        console.print(
            "[bold white on red] ARMED — `aqtp start` will place real orders "
            "with real money [/]"
        )
    elif ready:
        console.print(
            f"[yellow]interlocks are set, but the config says {configured_mode}.[/] "
            "Use `aqtp start --mode LIVE` or change the config to trade live."
        )
    else:
        console.print(
            "[green]LIVE is blocked.[/] The interlocks above are missing — "
            "run `aqtp arm` to see them."
        )


LIVE_PHRASE = "I ACCEPT THE RISK"


def _live_gate(config: AppConfig, *, yes: bool) -> None:
    """The one confirmation that stands between the operator and real orders."""
    _banner(config)
    console.print(
        "\n[bold white on red] LIVE TRADING — REAL ORDERS WITH REAL MONEY [/]\n"
        f"Capital at risk : ₹{config.capital.available_capital:,.0f}\n"
        f"Risk per trade  : ₹{config.capital.available_capital * config.risk.risk_per_trade_pct:,.0f} "
        f"({config.risk.risk_per_trade_pct:.1%})\n"
        f"Max daily loss  : ₹{config.capital.available_capital * config.risk.max_daily_loss_pct:,.0f} "
        f"({config.risk.max_daily_loss_pct:.1%})\n"
        f"Max positions   : {config.risk.max_open_positions}\n"
        f"Square-off at   : {config.session.square_off_time:%H:%M} IST"
    )
    if yes:
        # --yes exists so systemd and cron can start the process unattended. The
        # environment interlocks are still required, so this is not a way to
        # trade live by accident.
        console.print("[yellow]--yes supplied: skipping the typed confirmation[/]")
        return
    typed = typer.prompt(f"\nType exactly '{LIVE_PHRASE}' to proceed")
    if typed.strip() != LIVE_PHRASE:
        console.print("[green]aborted — no orders were placed[/]")
        raise typer.Exit(0)


@app.command("start")
def start(
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = MODE_OPTION,
    cycles: Optional[int] = typer.Option(None, "--cycles", help="Stop after N cycles."),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the typed LIVE confirmation."),
) -> None:
    """Start the trading engine in the configured mode (LIVE by default).

    This is the production entry point. It runs under the runtime supervisor:
    SIGTERM stops it cleanly, a dropped broker session is reconnected with
    backoff, a wedged cycle trips the watchdog, and positions are squared off at
    the configured time.
    """
    config = _load(config_path, mode)
    if config.mode is TradingMode.LIVE:
        _live_gate(config, yes=yes)
    _run(config, cycles, dashboard)


@app.command("live-start")
def live_start(
    config_path: str = CONFIG_OPTION,
    cycles: Optional[int] = typer.Option(None, "--cycles"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the typed confirmation."),
) -> None:
    """Start LIVE trading explicitly. Requires all three interlocks (REQ 43)."""
    config = _load(config_path, mode="LIVE")  # raises unless the interlocks are set
    _live_gate(config, yes=yes)
    _run(config, cycles, dashboard)


@app.command("stop")
def stop(
    config_path: str = CONFIG_OPTION,
    close_positions: bool = typer.Option(
        False, "--close-positions", help="Also flatten open positions."
    ),
    reason: str = typer.Option("operator stop", "--reason"),
) -> None:
    """Stop a running session by engaging the application kill switch.

    The running process notices at its next cycle boundary and shuts down
    cleanly. Open positions are left alone unless --close-positions is given;
    `aqtp trade close-all` does the same thing more explicitly.
    """
    config = _load(config_path)
    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    if close_positions:
        switches.emergency_shutdown(
            reason, policy=EmergencyPolicy.CLOSE_POSITIONS, engaged_by="cli"
        )
        console.print("[bold red]shutdown engaged with position closure[/]")
    else:
        switches.engage(KillScope.APPLICATION, reason=reason, engaged_by="cli")
        console.print("[yellow]application kill switch engaged; the loop will stop entering[/]")
    _journal(config).record_event(
        level="WARNING", category="lifecycle", message=f"stop requested: {reason}"
    )


@app.command("arm")
def arm(config_path: str = CONFIG_OPTION) -> None:
    """Print the shell commands that arm LIVE mode.

    They are printed rather than executed: setting them has to be a thing the
    operator did on purpose, in their own shell, where they can see it.
    """
    from ..configuration.schema import LIVE_INTERLOCKS
    import os

    console.print("[bold]Run these in your shell to arm LIVE trading:[/]\n")
    for key, value in LIVE_INTERLOCKS.items():
        already = os.environ.get(key, "") == value
        marker = "[green]# already set[/]" if already else ""
        console.print(f"  export {key}={value}   {marker}")
    console.print(
        "\n[dim]Then: aqtp doctor && aqtp start[/]\n"
        "[dim]They live only in that shell. Put them in your systemd unit or .env "
        "for an unattended run.[/]"
    )


# =========================================================================== #
# Control (REQ 44)
# =========================================================================== #
@app.command("kill")
def kill(
    config_path: str = CONFIG_OPTION,
    scope: str = typer.Option("global", "--scope", help="global|application|strategy|instrument|portfolio|broker"),
    target: str = typer.Option("", "--target", help="Strategy or instrument name."),
    reason: str = typer.Option("operator kill switch", "--reason"),
    close_positions: bool = typer.Option(False, "--close-positions", help="Also flatten positions."),
) -> None:
    """Engage a kill switch (REQ 44)."""
    config = _load(config_path)
    _banner(config)
    try:
        kill_scope = KillScope(scope.lower())
    except ValueError:
        console.print(f"[red]unknown scope {scope!r}[/] — valid: {[s.value for s in KillScope]}")
        raise typer.Exit(1)

    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")
    if kill_scope is KillScope.GLOBAL:
        policy = (
            EmergencyPolicy.CLOSE_POSITIONS if close_positions else EmergencyPolicy.CANCEL_ORDERS_ONLY
        )
        switches.emergency_shutdown(reason, policy=policy, engaged_by="cli")
        console.print(f"[bold red]GLOBAL EMERGENCY SHUTDOWN ENGAGED[/] — policy {policy.value}")
    else:
        record = switches.engage(kill_scope, target=target, reason=reason, engaged_by="cli")
        console.print(f"[red]kill switch engaged:[/] {record.describe()}")

    _journal(config).record_event(
        level="CRITICAL", category="killswitch",
        message=f"{scope} kill switch engaged: {reason}", payload={"target": target},
    )


@app.command("resume")
def resume(
    config_path: str = CONFIG_OPTION,
    scope: Optional[str] = typer.Option(None, "--scope", help="Scope to release; omit to release all."),
    target: str = typer.Option("", "--target"),
    note: str = typer.Option("", "--note", help="Why it is safe to resume."),
) -> None:
    """Release kill switches (REQ 44)."""
    config = _load(config_path)
    _banner(config)
    switches = KillSwitchManager(Path(config.paths.data_dir) / "killswitches.json")

    if not switches.active():
        console.print("[dim]no kill switches are engaged[/]")
        return

    console.print("[yellow]currently engaged:[/]")
    for record in switches.active():
        console.print(f"  {record.describe()}")

    if not typer.confirm("\nRelease?"):
        raise typer.Exit(0)

    if scope is None:
        count = switches.release_all()
        console.print(f"[green]released {count} kill switch(es)[/]")
    else:
        released = switches.release(KillScope(scope.lower()), target=target)
        console.print("[green]released[/]" if released else "[yellow]no matching switch[/]")

    _journal(config).record_event(
        level="WARNING", category="killswitch",
        message="kill switch released", payload={"scope": scope, "note": note},
    )


@app.command("explain")
def explain(
    decision_id: str = typer.Argument(..., help="Decision id to explain."),
    config_path: str = CONFIG_OPTION,
) -> None:
    """Show the full decision report for one decision (REQ 48)."""
    config = _load_disarmed(config_path)
    rows = _journal(config).query(
        "SELECT explanation, risk_checks FROM decisions WHERE decision_id = ?", (decision_id,)
    )
    if not rows:
        console.print(f"[red]no decision {decision_id!r} in the journal[/]")
        raise typer.Exit(1)
    console.print(rows[0]["explanation"] or "[dim]no explanation recorded[/]")


# =========================================================================== #
# Analysis (aqtp.analysis)
# =========================================================================== #
@app.command("analyze")
def analyze(
    underlying: str = typer.Argument(..., help="Underlying to analyze, e.g. NIFTY."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = MODE_OPTION,
    json_output: bool = typer.Option(False, "--json", help="Emit the feature dict as JSON."),
    dimensions_only: bool = typer.Option(
        False, "--dimensions", help="Show only the confluence table."
    ),
) -> None:
    """Run the full deep analysis on one underlying and print it.

    Eight independent reads — trend, momentum, volume structure, order flow,
    dealer positioning, statistical character, volatility and the cross-asset
    backdrop — combined into one conviction with the reasoning behind it.
    """
    config = _load(config_path, mode)
    _banner(config)
    underlying = underlying.strip().upper()

    orchestrator = _build(config)
    if not orchestrator.start(warm_up_days=5):
        console.print("[bold red]startup failed[/]")
        raise typer.Exit(1)

    with console.status(f"analyzing {underlying}…"):
        orchestrator.run_cycle()
    report = orchestrator.analysis.last(underlying)

    if report is None:
        console.print(
            f"[yellow]no analysis produced for {underlying}[/] — it is either outside the "
            "configured universe or had no usable market data this cycle."
        )
        available = sorted(orchestrator.universe.symbols())[:20] if orchestrator.universe else []
        if available:
            console.print(f"[dim]available: {', '.join(available)}[/]")
        raise typer.Exit(1)

    if json_output:
        console.print_json(json.dumps(report.feature_dict(), default=str))
        raise typer.Exit(0)

    table = Table("dimension", "score", "confidence", "weight", "reading", box=None)
    for entry in sorted(report.confluence.dimensions, key=lambda d: -abs(d.contribution)):
        colour = "green" if entry.score > 0.1 else "red" if entry.score < -0.1 else "dim"
        table.add_row(
            entry.name,
            f"[{colour}]{entry.score:+.2f}[/]",
            f"{entry.confidence:.2f}",
            f"{entry.weight:.2f}",
            (entry.reasons[0] if entry.reasons else "—")[:62],
        )
    console.print(table)

    verdict = report.confluence
    style = "bold red" if report.vetoed else "bold green" if verdict.net_score > 0 else "bold yellow"
    console.print(
        Panel(
            f"[{style}]{underlying} — bias {report.bias.value}[/]\n"
            f"net score {verdict.net_score:+.3f} · conviction {report.conviction:.3f} · "
            f"alignment {verdict.alignment:.0%}\n"
            f"agreeing: {', '.join(verdict.agreeing) or 'none'}\n"
            f"conflicting: {', '.join(verdict.conflicting) or 'none'}",
            subtitle=f"analysed in {report.duration_ms:.0f}ms",
            border_style="red" if report.vetoed else "cyan",
        )
    )
    if report.vetoed:
        for veto in report.vetoes:
            console.print(f"[bold red]VETO:[/] {veto}")

    if not dimensions_only:
        console.print()
        console.print(report.render())


@app.command("market")
def market(config_path: str = CONFIG_OPTION, mode: Optional[str] = MODE_OPTION) -> None:
    """Show the cross-asset backdrop: breadth, correlation, dispersion, VIX."""
    config = _load(config_path, mode)
    _banner(config)
    orchestrator = _build(config)
    if not orchestrator.start(warm_up_days=2):
        console.print("[bold red]startup failed[/]")
        raise typer.Exit(1)

    # Two cycles: the cross-asset tracker needs at least two price observations
    # per symbol before it can compute a return, let alone a correlation.
    with console.status("sampling the market…"):
        orchestrator.run_cycle()
        orchestrator.run_cycle()

    report = orchestrator.analysis.crossasset.last_report
    if report is None:
        console.print("[yellow]no cross-asset context yet[/]")
        raise typer.Exit(1)

    table = Table("metric", "value", box=None)
    table.add_row("symbols tracked", str(report.symbols_tracked))
    table.add_row("market regime", report.regime.replace("_", " "))
    if report.breadth is not None:
        table.add_row("breadth (advancing)", f"{(report.breadth + 1) / 2:.0%}")
    if report.breadth_above_vwap is not None:
        table.add_row("above VWAP", f"{report.breadth_above_vwap:.0%}")
    if report.average_correlation is not None:
        table.add_row("average correlation", f"{report.average_correlation:.2f}")
        table.add_row("diversification", f"{report.diversification_score:.2f}")
    if report.dispersion is not None:
        table.add_row("dispersion", f"{report.dispersion:.4f}")
    if report.vix is not None:
        table.add_row("India VIX", f"{report.vix:.2f}")
    console.print(table)

    if report.leaders:
        console.print(f"\n[green]leaders:[/] {', '.join(report.leaders)}")
        console.print(f"[red]laggards:[/] {', '.join(report.laggards)}")
    console.print()
    for reason in report.explain():
        console.print(f"  - {reason}")


@app.command("universe")
def universe(
    config_path: str = CONFIG_OPTION,
    filter_text: str = typer.Option("", "--filter", "-f", help="Substring filter."),
    limit: int = typer.Option(40, "--limit"),
) -> None:
    """List the discovered tradable universe."""
    config = _load(config_path)
    _banner(config)
    orchestrator = _build(config)
    orchestrator.data_broker.authenticate()
    if not orchestrator.refresh_universe() or orchestrator.universe is None:
        console.print("[red]instrument discovery failed[/]")
        raise typer.Exit(1)

    needle = filter_text.strip().upper()
    rows = [
        (name, group) for name, group in sorted(orchestrator.universe.underlyings.items())
        if not needle or needle in name
    ]
    if not rows:
        console.print(f"[yellow]nothing matches {filter_text!r}[/]")
        raise typer.Exit(1)

    table = Table("underlying", "lot", "futures", "options", "nearest expiry", box=None)
    today = orchestrator.clock.now().date()
    for name, group in rows[:limit]:
        expiries = orchestrator.discovery.tradable_expiries(name, as_of=today)
        table.add_row(
            name, str(group.lot_size), str(len(group.futures)), str(len(group.options)),
            expiries[0].isoformat() if expiries else "—",
        )
    console.print(table)
    console.print(f"[dim]showing {min(limit, len(rows))} of {len(rows)} underlyings[/]")


# Research commands live in their own module but are exposed on the same CLI, so
# `aqtp research backtest ...` sits alongside the trading controls.
from .research import research_app  # noqa: E402

app.add_typer(research_app, name="research")

# Manual order entry: `aqtp trade buy ...` sits beside the automated controls and
# goes through the same risk engine.
from .trade import trade_app  # noqa: E402

app.add_typer(trade_app, name="trade")


def main() -> None:
    try:
        app()
    except AQTPError as exc:
        console.print(f"[bold red]{type(exc).__name__}:[/] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
