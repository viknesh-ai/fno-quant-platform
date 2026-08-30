"""`aqtp trade ...` — operator-initiated order entry.

These commands place real orders when the session is LIVE. They go through the
same risk engine, sizer, execution engine and journal as the automated pipeline,
so a manual trade is managed like any other position: it has a stop, it has a
target, it is reconciled at startup and it is closed at square-off.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.types import TradingMode
from ..execution.manual import ManualTradeError, ManualTrader, ResolvedContract

console = Console()
trade_app = typer.Typer(
    name="trade",
    help="Place, modify and close orders by hand — through the full risk pipeline.",
    no_args_is_help=True,
)

CONFIG_OPTION = typer.Option("config/default.yaml", "--config", "-c", help="Path to config YAML.")


def _session(config_path: str, mode: str | None = None):
    """Build a started orchestrator plus a ManualTrader bound to it."""
    from .main import _build, _banner, _load

    config = _load(config_path, mode)
    _banner(config)
    orchestrator = _build(config)
    if not orchestrator.start(warm_up_days=0):
        console.print("[bold red]startup failed — the broker state is not safe to trade[/]")
        raise typer.Exit(1)
    return config, orchestrator, ManualTrader(orchestrator)


def _parse_expiry(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        console.print(f"[red]--expiry must be YYYY-MM-DD, got {value!r}[/]")
        raise typer.Exit(2)


def _show_contract(contract: ResolvedContract) -> None:
    quote = contract.quote
    table = Table("field", "value", box=None)
    table.add_row("instrument", contract.instrument.trading_symbol)
    table.add_row("underlying", contract.underlying or "—")
    table.add_row("lot size", str(contract.instrument.lot_size))
    if contract.instrument.strike_price:
        table.add_row("strike", f"{contract.instrument.strike_price:,.2f}")
    if contract.instrument.expiry_date:
        table.add_row("expiry", contract.instrument.expiry_date.isoformat())
    if quote is not None:
        table.add_row("last", f"{quote.last_price:,.2f}")
        if quote.bid_price and quote.ask_price:
            table.add_row("bid / ask", f"{quote.bid_price:,.2f} / {quote.ask_price:,.2f}")
        if quote.spread_pct is not None:
            table.add_row("spread", f"{quote.spread_pct:.3%}")
        if quote.volume:
            table.add_row("volume", f"{quote.volume:,.0f}")
        if quote.open_interest:
            table.add_row("open interest", f"{quote.open_interest:,.0f}")
    for note in contract.notes:
        table.add_row("[dim]note[/]", f"[dim]{note}[/]")
    console.print(table)


def _confirm_live(config, action: str, contract: ResolvedContract, quantity_note: str) -> None:
    """A LIVE order gets one explicit confirmation. Not three — one that means it."""
    if config.mode is not TradingMode.LIVE:
        return
    console.print(
        Panel(
            f"[bold white on red] LIVE ORDER — REAL MONEY [/]\n"
            f"{action} [bold]{contract.instrument.trading_symbol}[/] {quantity_note}\n"
            f"last traded price ₹{contract.last_price:,.2f}",
            border_style="red",
        )
    )
    if not typer.confirm("Send this order?", default=False):
        console.print("[green]aborted — no order was sent[/]")
        raise typer.Exit(0)


def _report(result) -> None:
    if result.success:
        console.print(f"[bold green]✓[/] {result.explain()}")
        if result.approval and result.approval.sizing:
            sizing = result.approval.sizing
            console.print(
                f"[dim]  {sizing.lots} lot(s) × {sizing.lot_size} = {sizing.quantity} units, "
                f"₹{sizing.risk_amount:,.0f} at risk"
                + (f", capped by {sizing.capped_by}" if sizing.capped_by else "")
                + "[/]"
            )
        for step in result.steps[-3:]:
            console.print(f"[dim]  {step}[/]")
    else:
        console.print(f"[bold red]✗[/] {result.explain()}")
        if result.approval:
            failed = [c for c in result.approval.checks if c.status.value == "FAIL"]
            for check in failed[:6]:
                console.print(f"[red]  {check.name}: {check.detail}[/]")


# =========================================================================== #
# Entry
# =========================================================================== #
def _entry_command(side: str):
    def command(
        symbol: str = typer.Argument(
            ...,
            help="Underlying (NIFTY) or an exact trading symbol (NIFTY26JUN24000CE).",
        ),
        config_path: str = CONFIG_OPTION,
        mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Override the trading mode."),
        option: Optional[str] = typer.Option(None, "--option", help="CE or PE."),
        strike: Optional[float] = typer.Option(None, "--strike", help="Strike; omit for ATM."),
        expiry: Optional[str] = typer.Option(None, "--expiry", help="YYYY-MM-DD; omit for nearest."),
        future: bool = typer.Option(False, "--future", help="Trade the future instead of an option."),
        lots: Optional[int] = typer.Option(None, "--lots", help="Size in lots."),
        quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="Size in units."),
        limit: Optional[float] = typer.Option(None, "--limit", help="Limit price; omit to use config."),
        stop: Optional[float] = typer.Option(None, "--stop", help="Absolute stop price."),
        target: Optional[float] = typer.Option(None, "--target", help="Absolute target price."),
        stop_pct: float = typer.Option(0.30, "--stop-pct", help="Stop as a fraction of premium."),
        target_pct: float = typer.Option(0.60, "--target-pct", help="Target as a fraction of premium."),
        strategy: str = typer.Option("manual", "--strategy", help="Tag recorded in the journal."),
        force: bool = typer.Option(
            False, "--force",
            help="Bypass risk SIZING (never kill switches). Requires --lots or --quantity.",
        ),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip the LIVE confirmation prompt."),
        dry_run: bool = typer.Option(False, "--dry-run", help="Resolve and risk-check only."),
    ) -> None:
        config, orchestrator, trader = _session(config_path, mode)
        try:
            contract = trader.resolve(
                symbol,
                option_type=option,
                strike=strike,
                expiry=_parse_expiry(expiry),
                instrument_kind="future" if future else "auto",
            )
        except ManualTradeError as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(1)

        _show_contract(contract)

        size_note = (
            f"{quantity} units" if quantity
            else f"{lots} lot(s)" if lots
            else "risk-sized"
        )

        if dry_run:
            console.print("\n[yellow]--dry-run: running the risk check without sending an order[/]")
            _dry_run(trader, contract, side, lots, quantity, stop, target, stop_pct, target_pct)
            raise typer.Exit(0)

        if not yes:
            _confirm_live(config, side, contract, size_note)

        method = trader.buy if side == "BUY" else trader.sell
        result = method(
            contract,
            lots=lots, quantity=quantity, stop_price=stop, target_price=target,
            stop_pct=stop_pct, target_pct=target_pct, limit_price=limit,
            force=force, strategy=strategy,
        )
        _report(result)
        raise typer.Exit(0 if result.success else 1)

    return command


def _dry_run(trader, contract, side, lots, quantity, stop, target, stop_pct, target_pct) -> None:
    """Show what the risk engine would decide, without sending anything."""
    from ..core.types import Decision, Direction

    orchestrator = trader.orchestrator
    now = orchestrator.clock.now()
    quote = contract.quote
    if quote is None:
        console.print("[red]no quote — cannot risk-check[/]")
        return
    entry = float(quote.last_price)
    direction = Direction.LONG if side == "BUY" else Direction.SHORT
    stop_price = stop if stop is not None else (
        max(0.05, entry * (1 - stop_pct)) if direction is Direction.LONG else entry * (1 + stop_pct)
    )
    target_price = target if target is not None else (
        entry * (1 + target_pct) if direction is Direction.LONG else max(0.05, entry * (1 - target_pct))
    )
    approval = orchestrator.risk.evaluate(
        decision=Decision.BUY if direction is Direction.LONG else Decision.SELL,
        instrument=contract.instrument,
        underlying=contract.underlying or contract.instrument.trading_symbol,
        strategy="manual-dry-run",
        entry_price=entry,
        stop_price=stop_price,
        target_price=target_price,
        portfolio_state=orchestrator._portfolio_state(now),
        margin=orchestrator._margin(),
        quote=quote,
        expected_value=None,
        underlying_price=entry,
        now=now,
    )
    table = Table("check", "status", "detail", box=None)
    for check in approval.checks:
        colour = {"PASS": "green", "FAIL": "red", "SKIP": "dim"}.get(check.status.value, "white")
        table.add_row(check.name, f"[{colour}]{check.status.value}[/]", check.detail[:70])
    console.print(table)
    if approval.approved and approval.sizing:
        console.print(
            f"\n[green]would send:[/] {side} {contract.instrument.trading_symbol} "
            f"{approval.sizing.lots} lot(s) = {approval.sizing.quantity} units "
            f"@ ~{entry:,.2f}, stop {stop_price:,.2f}, target {target_price:,.2f} "
            f"(₹{approval.sizing.risk_amount:,.0f} at risk)"
        )
    else:
        console.print(
            f"\n[yellow]would NOT send:[/] {'; '.join(approval.rejection_reasons) or 'rejected'}"
        )


trade_app.command("buy", help="Buy an option or future (long premium / long future).")(
    _entry_command("BUY")
)
trade_app.command("sell", help="Sell an option or future (short premium / short future).")(
    _entry_command("SELL")
)


# =========================================================================== #
# Exit and order management
# =========================================================================== #
@trade_app.command("exit")
def exit_position(
    position_id: str = typer.Argument(..., help="Position id from `aqtp positions`."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q", help="Partial exit size."),
    reason: str = typer.Option("manual exit", "--reason"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Close a tracked position at market, whole or in part."""
    config, orchestrator, trader = _session(config_path, mode)
    if config.mode is TradingMode.LIVE and not yes:
        if not typer.confirm(f"Close {position_id} at market?", default=True):
            raise typer.Exit(0)
    result = trader.exit_position(position_id, quantity=quantity, reason=reason)
    _report(result)
    raise typer.Exit(0 if result.success else 1)


@trade_app.command("close-all")
def close_all(
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    reason: str = typer.Option("operator close-all", "--reason"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Flatten every tracked position at market."""
    config, orchestrator, trader = _session(config_path, mode)
    open_positions = orchestrator.positions.all()
    if not open_positions:
        console.print("[dim]no tracked positions to close[/]")
        raise typer.Exit(0)

    table = Table("position", "instrument", "qty", "entry", "last", "unrealized", box=None)
    for position in open_positions:
        table.add_row(
            position.position_id, position.instrument.trading_symbol, str(position.quantity),
            f"{position.entry_price:,.2f}", f"{position.last_price:,.2f}",
            f"{position.unrealized_pnl:+,.0f}" if hasattr(position, "unrealized_pnl") else "—",
        )
    console.print(table)

    if not yes and not typer.confirm(
        f"Close all {len(open_positions)} position(s) at market?", default=False
    ):
        raise typer.Exit(0)

    results = trader.close_all(reason=reason)
    for result in results:
        _report(result)
    failures = [r for r in results if not r.success]
    if failures:
        console.print(
            f"\n[bold red]{len(failures)} position(s) did NOT close and are still open "
            "at the broker.[/]"
        )
    raise typer.Exit(1 if failures else 0)


@trade_app.command("cancel")
def cancel(
    order_id: Optional[str] = typer.Argument(None, help="Broker order id; omit with --all."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    all_orders: bool = typer.Option(False, "--all", help="Cancel every open order."),
) -> None:
    """Cancel one resting order, or all of them."""
    config, orchestrator, trader = _session(config_path, mode)
    if all_orders:
        results = trader.cancel_all()
        if not results:
            console.print("[dim]no open orders[/]")
            raise typer.Exit(0)
        for result in results:
            _report(result)
        raise typer.Exit(0)
    if not order_id:
        console.print("[red]pass a broker order id, or --all[/]")
        raise typer.Exit(2)
    result = trader.cancel(order_id)
    _report(result)
    raise typer.Exit(0 if result.success else 1)


@trade_app.command("modify")
def modify(
    order_id: str = typer.Argument(..., help="Broker order id."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    price: Optional[float] = typer.Option(None, "--price"),
    quantity: Optional[int] = typer.Option(None, "--quantity", "-q"),
    trigger_price: Optional[float] = typer.Option(None, "--trigger-price"),
) -> None:
    """Modify a resting order's price, size or trigger."""
    config, orchestrator, trader = _session(config_path, mode)
    result = trader.modify(order_id, price=price, quantity=quantity, trigger_price=trigger_price)
    _report(result)
    raise typer.Exit(0 if result.success else 1)


@trade_app.command("set-stop")
def set_stop(
    position_id: str = typer.Argument(..., help="Position id from `aqtp positions`."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    stop: Optional[float] = typer.Option(None, "--stop", help="New stop price."),
    target: Optional[float] = typer.Option(None, "--target", help="New target price."),
) -> None:
    """Move the managed stop or target on an open position."""
    config, orchestrator, trader = _session(config_path, mode)
    result = trader.set_stop(position_id, stop_price=stop, target_price=target)
    _report(result)
    raise typer.Exit(0 if result.success else 1)


@trade_app.command("quote")
def quote(
    symbol: str = typer.Argument(..., help="Underlying or exact trading symbol."),
    config_path: str = CONFIG_OPTION,
    mode: Optional[str] = typer.Option(None, "--mode", "-m"),
    option: Optional[str] = typer.Option(None, "--option", help="CE or PE."),
    strike: Optional[float] = typer.Option(None, "--strike"),
    expiry: Optional[str] = typer.Option(None, "--expiry"),
    future: bool = typer.Option(False, "--future"),
) -> None:
    """Resolve a symbol and show its live quote and book."""
    config, orchestrator, trader = _session(config_path, mode)
    try:
        contract = trader.resolve(
            symbol, option_type=option, strike=strike,
            expiry=_parse_expiry(expiry), instrument_kind="future" if future else "auto",
        )
    except ManualTradeError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(1)
    _show_contract(contract)

    if contract.quote is not None:
        from ..analysis.microstructure import analyze_quote

        micro = analyze_quote(contract.quote, test_quantity=contract.instrument.lot_size)
        console.print("\n[bold]microstructure[/]")
        for reason in micro.explain():
            console.print(f"  - {reason}")
        console.print(
            f"  net pressure {micro.pressure:+.2f}, execution quality {micro.execution_quality:.2f}"
        )
        if contract.quote.depth_buy or contract.quote.depth_sell:
            book = Table("bid qty", "bid", "ask", "ask qty", box=None)
            depth_buy, depth_sell = contract.quote.depth_buy, contract.quote.depth_sell
            for index in range(max(len(depth_buy), len(depth_sell))):
                bid = depth_buy[index] if index < len(depth_buy) else None
                ask = depth_sell[index] if index < len(depth_sell) else None
                book.add_row(
                    f"{bid.quantity:,}" if bid else "", f"{bid.price:,.2f}" if bid else "",
                    f"{ask.price:,.2f}" if ask else "", f"{ask.quantity:,}" if ask else "",
                )
            console.print(book)
