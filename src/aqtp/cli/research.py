"""Research CLI commands: backtest, walk-forward, train, experiments.

Registered onto the main `aqtp` app in `cli/main.py`. Separated here because these
commands are offline research tooling (REQ 50) rather than trading controls.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from ..configuration.loader import Credentials, load_config
from ..core.errors import AQTPError, ConfigurationError
from ..core.logging import setup_logging
from ..core.types import Timeframe

console = Console()
research_app = typer.Typer(help="Research and validation commands.", no_args_is_help=True)

CONFIG_OPTION = typer.Option("config/default.yaml", "--config", "-c")


def _load(config_path: str, mode: str = "BACKTEST"):
    try:
        return load_config(config_path, overrides={"mode": mode})
    except ConfigurationError as exc:
        console.print(f"[bold red]Configuration error[/]\n{exc}")
        raise typer.Exit(2)


def _fetch_history(config, symbol: str, timeframe: Timeframe, days: int) -> pd.DataFrame:
    """Pull historical candles through the broker adapter."""
    from ..brokers.factory import build_data_broker
    from ..data.candles import candles_to_frame

    credentials = Credentials.from_env()
    broker = build_data_broker(config, credentials)
    broker.authenticate()
    instruments = broker.fetch_instruments()

    match = next((i for i in instruments if i.trading_symbol == symbol), None)
    if match is None:
        console.print(f"[red]symbol {symbol!r} not found in the instrument master[/]")
        raise typer.Exit(1)

    end = datetime.now()
    start = end - timedelta(days=days)
    console.print(f"[dim]fetching {days}d of {timeframe.value} candles for {symbol}…[/]")
    candles = broker.get_historical_candles(match, timeframe, start, end)
    if not candles:
        console.print("[red]no historical data returned[/]")
        raise typer.Exit(1)
    console.print(f"[dim]{len(candles):,} candles[/]")
    return candles_to_frame(candles), match


@research_app.command("backtest")
def backtest(
    symbol: str = typer.Argument(..., help="Trading symbol to backtest."),
    config_path: str = CONFIG_OPTION,
    days: int = typer.Option(180, "--days"),
    timeframe: str = typer.Option("5m", "--timeframe"),
    save: bool = typer.Option(True, "--save/--no-save", help="Record as a versioned experiment."),
) -> None:
    """Run an event-driven backtest with realistic costs (REQ 38/39)."""
    config = _load(config_path)
    setup_logging(config.log_level)

    from ..backtest.engine import BacktestConfig, BacktestEngine, BacktestSignal
    from ..backtest.robustness import run_all
    from ..core.types import Decision, Regime
    from ..features.indicators import atr, ema
    from ..research.experiments import ExperimentStore
    from ..risk.costs import TransactionCostModel

    frame, instrument = _fetch_history(config, symbol, Timeframe(timeframe), days)

    # A deliberately simple reference signal source. Replace with a strategy-driven
    # source for real research; the point here is to exercise the engine end to end.
    def source(ts, history):
        if len(history) < 60:
            return None
        close = history["close"]
        fast, slow = ema(close, 9), ema(close, 21)
        a = atr(history["high"], history["low"], close, 14)
        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or pd.isna(a.iloc[-1]) or a.iloc[-1] <= 0:
            return None
        up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
        down = fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
        if not (up or down):
            return None
        price, av = float(close.iloc[-1]), float(a.iloc[-1])
        if up:
            return BacktestSignal(Decision.BUY, price, price - av, price + 2 * av,
                                  strategy="ema_cross", regime=Regime.TRENDING_UP.value)
        return BacktestSignal(Decision.SELL, price, price + av, price - 2 * av,
                              strategy="ema_cross", regime=Regime.TRENDING_DOWN.value)

    engine = BacktestEngine(
        BacktestConfig(
            initial_capital=config.backtest.initial_capital,
            fill_model=config.backtest.fill_model,
            risk_per_trade_pct=config.risk.risk_per_trade_pct,
            max_capital_per_position_pct=config.risk.max_position_size_pct,
        ),
        TransactionCostModel(config.costs),
    )
    result = engine.run(instrument=instrument, frame=frame, signal_source=source)
    console.print()
    console.print(result.summary())

    if result.trades:
        console.print("\n[bold]ATTRIBUTION[/]")
        table = Table("dimension", "key", "trades", "net P&L", "expectancy R", box=None)
        for label, mapping in (
            ("strategy", result.metrics.by_strategy),
            ("instrument", result.metrics.by_instrument),
            ("regime", result.metrics.by_regime),
        ):
            for key, stats in mapping.items():
                table.add_row(label, key, str(stats["trades"]),
                              f"{stats['net_pnl']:+,.0f}", f"{stats['expectancy_r']:+.3f}")
        console.print(table)

        console.print("\n[bold]ROBUSTNESS (REQ 40)[/]")
        report = run_all(
            result.trades,
            initial_capital=config.backtest.initial_capital,
            monte_carlo_runs=config.backtest.monte_carlo_runs,
            cost_multipliers=config.backtest.cost_sensitivity_multipliers,
            slippage_multipliers=config.backtest.slippage_sensitivity_multipliers,
            seed=config.models.random_seed,
        )
        for line in report.summary_lines():
            style = "red" if line.startswith("[FAIL") or "NOT ROBUST" in line else (
                "green" if line.startswith("[PASS") or line.startswith("ROBUST") else ""
            )
            console.print(f"[{style}]{line}[/]" if style else line)

        if save:
            store = ExperimentStore(config.paths.experiment_dir)
            experiment = store.create(
                name=f"backtest_{symbol}",
                configuration=config.model_dump(mode="json"),
                symbols=[symbol],
                random_seed=config.models.random_seed,
                backtest_start=str(result.start),
                backtest_end=str(result.end),
                stage="BACKTEST",
            )
            experiment.results = result.metrics.to_dict()
            experiment.robustness = {o.name: o.describe() for o in report.outcomes}
            experiment.verdict = report.verdict()
            experiment.passed = report.is_robust
            store.save(experiment)
            console.print(f"\n[dim]saved experiment {experiment.experiment_id}[/]")


@research_app.command("walkforward")
def walkforward(
    symbol: str = typer.Argument(...),
    config_path: str = CONFIG_OPTION,
    days: int = typer.Option(365, "--days"),
    timeframe: str = typer.Option("5m", "--timeframe"),
    train_days: int = typer.Option(45, "--train-days"),
    test_days: int = typer.Option(15, "--test-days"),
) -> None:
    """Walk-forward validate a strategy out of sample (REQ 20)."""
    config = _load(config_path)
    setup_logging(config.log_level)

    from ..backtest.engine import BacktestConfig, BacktestSignal
    from ..backtest.walkforward import WalkForwardBacktester
    from ..core.types import Decision, Regime
    from ..features.indicators import atr, ema
    from ..risk.costs import TransactionCostModel

    frame, instrument = _fetch_history(config, symbol, Timeframe(timeframe), days)

    def factory(train_frame):
        # Parameters would be fitted from `train_frame` here; the reference source
        # is fixed, which is the honest baseline to compare a fitted one against.
        def source(ts, history):
            if len(history) < 60:
                return None
            close = history["close"]
            fast, slow = ema(close, 9), ema(close, 21)
            a = atr(history["high"], history["low"], close, 14)
            if pd.isna(fast.iloc[-1]) or pd.isna(a.iloc[-1]) or a.iloc[-1] <= 0:
                return None
            up = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
            down = fast.iloc[-1] < slow.iloc[-1] and fast.iloc[-2] >= slow.iloc[-2]
            if not (up or down):
                return None
            price, av = float(close.iloc[-1]), float(a.iloc[-1])
            if up:
                return BacktestSignal(Decision.BUY, price, price - av, price + 2 * av,
                                      strategy="ema_cross", regime=Regime.TRENDING_UP.value)
            return BacktestSignal(Decision.SELL, price, price + av, price - 2 * av,
                                  strategy="ema_cross", regime=Regime.TRENDING_DOWN.value)
        return source

    backtester = WalkForwardBacktester(
        BacktestConfig(
            initial_capital=config.backtest.initial_capital,
            risk_per_trade_pct=config.risk.risk_per_trade_pct,
            max_capital_per_position_pct=config.risk.max_position_size_pct,
        ),
        TransactionCostModel(config.costs),
    )
    report = backtester.run(
        instrument=instrument, frame=frame, source_factory=factory,
        train_days=train_days, test_days=test_days,
        embargo_minutes=config.models.embargo_minutes,
    )

    windows = report.to_frame()
    if not windows.empty:
        console.print(windows.to_string(index=False))
    console.print()
    for line in report.summary_lines():
        style = "red" if "FAILED" in line else ("green" if "PASSED" in line else "")
        console.print(f"[{style}]{line}[/]" if style else line)


@research_app.command("train")
def train(
    symbol: str = typer.Argument(...),
    config_path: str = CONFIG_OPTION,
    days: int = typer.Option(365, "--days"),
    timeframe: str = typer.Option("5m", "--timeframe"),
    horizon: int = typer.Option(30, "--horizon", help="Prediction horizon in minutes."),
    register: bool = typer.Option(True, "--register/--no-register"),
) -> None:
    """Train and walk-forward validate ML models (REQ 12/16/20)."""
    config = _load(config_path)
    setup_logging(config.log_level)

    from ..data.candles import resample
    from ..features.engine import FeatureEngine
    from ..ml.dataset import DatasetBuilder
    from ..ml.labeling import LabelSpec
    from ..ml.registry import ModelRegistry
    from ..ml.train import WalkForwardTrainer

    base_frame, _ = _fetch_history(config, symbol, Timeframe("1m"), days)
    entry_tf = Timeframe(timeframe)
    frames = {
        entry_tf: resample(base_frame, entry_tf),
        Timeframe(config.timeframes.setup_timeframe): resample(
            base_frame, Timeframe(config.timeframes.setup_timeframe)
        ),
        Timeframe(config.timeframes.regime_timeframe): resample(
            base_frame, Timeframe(config.timeframes.regime_timeframe)
        ),
    }

    bars = max(1, round(horizon / entry_tf.minutes))
    spec = LabelSpec(
        horizon_bars=bars,
        target_atr_multiple=config.prediction.target_atr_multiple,
        stop_atr_multiple=config.prediction.stop_atr_multiple,
        min_movement_atr=config.prediction.min_movement_atr,
        volatility_adjusted=config.prediction.volatility_adjusted,
    )
    console.print(f"[dim]label: {spec.describe()}[/]")

    dataset = DatasetBuilder(FeatureEngine()).build(
        frames=frames, entry_timeframe=entry_tf, label_spec=spec, symbol=symbol,
        labeling=config.prediction.labeling,
    )
    console.print(f"[dim]dataset: {dataset.describe()}[/]")
    if dataset.is_empty:
        console.print("[red]no usable training rows after leakage filtering[/]")
        raise typer.Exit(1)

    result = WalkForwardTrainer(config.models).run(dataset)
    comparison = result.comparison_table()
    if not comparison.empty:
        console.print("\n[bold]MODEL COMPARISON (out-of-sample, per fold)[/]")
        console.print(comparison.round(4).to_string())

    console.print()
    style = "green" if result.passed else "red"
    console.print(f"[{style}]{result.verdict}[/]")

    if result.final_model is not None and register:
        registry = ModelRegistry(config.paths.model_dir)
        record = registry.register(
            result.final_model,
            family=result.best_family,
            dataset_description=dataset.describe(),
            label_spec=spec.name,
            walk_forward_summary=result.per_family[result.best_family].summary(),
            out_of_sample_results=result.final_metrics.to_dict() if result.final_metrics else {},
            feature_importance=result.feature_importance,
            config_snapshot=result.config_snapshot,
            random_seed=config.models.random_seed,
            verdict=result.verdict,
            passed=result.passed,
        )
        console.print(f"[dim]registered model {record.model_id} (status RESEARCH)[/]")
        if not result.passed:
            console.print(
                "[yellow]this model did not pass out-of-sample validation and cannot be "
                "promoted to ACTIVE[/]"
            )


@research_app.command("experiments")
def experiments(config_path: str = CONFIG_OPTION, limit: int = 20) -> None:
    """List recorded experiments (REQ 50)."""
    config = _load(config_path)
    from ..research.experiments import ExperimentStore

    store = ExperimentStore(config.paths.experiment_dir)
    records = store.all()[:limit]
    if not records:
        console.print("[dim]no experiments recorded[/]")
        return
    table = Table("experiment id", "name", "stage", "created", "passed", "verdict", box=None)
    for experiment in records:
        table.add_row(
            experiment.experiment_id, experiment.name, experiment.stage,
            experiment.created_at[:19],
            "[green]yes[/]" if experiment.passed else "[red]no[/]",
            experiment.verdict[:60],
        )
    console.print(table)


@research_app.command("promote")
def promote(
    model_id: str = typer.Argument(...),
    status: str = typer.Argument(..., help="VALIDATION|PAPER|ACTIVE|RETIRED"),
    config_path: str = CONFIG_OPTION,
) -> None:
    """Move a model along the deployment ladder (REQ 51)."""
    config = _load(config_path)
    from ..core.errors import ModelError
    from ..ml.registry import ModelRegistry, ModelStatus

    registry = ModelRegistry(config.paths.model_dir)
    try:
        record = registry.promote(model_id, ModelStatus(status.upper()))
    except (ModelError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    console.print(f"[green]{record.model_id} is now {record.status.value}[/]")
