"""Event-driven backtester (REQ 38).

The backtester replays historical bars through the *same* feature engine, regime
engine, strategies, ensemble and risk engine that the live orchestrator uses. This
is the whole reason those components take a `Clock` and never call `datetime.now()`
themselves: if the backtest used a parallel implementation, agreement between the
two would prove nothing.

Realism measures, all of which push results *down* rather than up:

  * Fills happen on the **next bar's open** by default, never on the signal bar's
    close. Filling at the close of the bar that generated the signal is the single
    most common way a backtest invents profit that cannot be captured.
  * Latency is simulated by delaying the fill bar.
  * Spread, slippage, brokerage and statutory charges are all deducted.
  * Stops and targets are checked intrabar against high/low, and when a bar spans
    both, the **stop is assumed to fill first**.
  * Positions are force-closed at the session end and at contract expiry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..configuration.schema import AppConfig
from ..core.clock import SimulatedClock, is_market_open, to_ist
from ..core.logging import get_logger
from ..core.types import (
    Candle,
    Decision,
    Direction,
    ExitReason,
    Instrument,
    Quote,
    Regime,
    Timeframe,
    TransactionType,
)
from ..risk.costs import TransactionCostModel
from .metrics import BacktestMetrics, ClosedTrade, compute_metrics

logger = get_logger(__name__)


@dataclass
class BacktestSignal:
    """What a signal source hands the backtester at one bar."""

    decision: Decision
    entry_price: float
    stop_price: float
    target_price: float
    strategy: str = "backtest"
    regime: str = Regime.UNCERTAIN.value
    ml_probability: float = float("nan")
    holding_minutes: int = 60
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)


# A signal source receives (timestamp, history-so-far) and returns a signal or None.
SignalSource = Callable[[datetime, pd.DataFrame], BacktestSignal | None]


@dataclass
class OpenPosition:
    trade_id: str
    instrument: Instrument
    direction: Direction
    quantity: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    target_price: float
    strategy: str
    regime: str
    ml_probability: float
    entry_costs: float
    entry_slippage: float
    spread_at_entry: float
    max_favourable: float = 0.0
    max_adverse: float = 0.0
    holding_limit_minutes: int = 60

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_price)

    def update_excursions(self, high: float, low: float) -> None:
        """Track MAE/MFE in R units as the bar develops."""
        risk = self.risk_per_unit
        if risk <= 0:
            return
        if self.direction is Direction.LONG:
            favourable = (high - self.entry_price) / risk
            adverse = (self.entry_price - low) / risk
        else:
            favourable = (self.entry_price - low) / risk
            adverse = (high - self.entry_price) / risk
        self.max_favourable = max(self.max_favourable, favourable)
        self.max_adverse = max(self.max_adverse, adverse)


@dataclass
class BacktestConfig:
    initial_capital: float = 500_000.0
    latency_bars: int = 0
    fill_model: str = "next_bar_open"
    spread_pct: float = 0.002
    entry_delay_bars: int = 0
    cost_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    square_off_at_session_end: bool = True
    max_holding_bars: int | None = None
    allow_simultaneous_positions: int = 1
    risk_per_trade_pct: float = 0.01
    max_capital_per_position_pct: float = 0.25
    # Futures are margined, not paid for in full: capital committed is roughly
    # SPAN + exposure, around 12-18% of notional for index futures. Sizing them
    # against full notional would make every index future untradable on a retail
    # account. Long options, by contrast, do cost the full premium.
    futures_margin_rate: float = 0.15


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[ClosedTrade]
    equity_curve: pd.Series
    config: BacktestConfig
    bars_processed: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None

    def summary(self) -> str:
        lines = [
            f"Backtest {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d}"
            if self.start and self.end else "Backtest",
            f"Bars processed  : {self.bars_processed:,}",
            f"Signals         : {self.signals_generated} generated, {self.signals_rejected} rejected",
        ]
        lines.extend(self.metrics.summary_lines())
        return "\n".join(lines)


class BacktestEngine:
    """Bar-by-bar replay with realistic fills."""

    def __init__(
        self,
        config: BacktestConfig,
        cost_model: TransactionCostModel,
        *,
        holidays: frozenset[date] = frozenset(),
    ) -> None:
        self.config = config
        self.costs = cost_model
        self.holidays = holidays

    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        instrument: Instrument,
        frame: pd.DataFrame,
        signal_source: SignalSource,
        expiry: date | None = None,
    ) -> BacktestResult:
        """Replay `frame` bar by bar, taking signals from `signal_source`."""
        config = self.config
        if frame.empty:
            return BacktestResult(
                metrics=compute_metrics([], pd.Series(dtype=float), initial_capital=config.initial_capital),
                trades=[], equity_curve=pd.Series(dtype=float), config=config,
            )

        frame = frame.sort_index()
        equity = config.initial_capital
        equity_points: list[tuple[datetime, float]] = []
        trades: list[ClosedTrade] = []
        open_positions: list[OpenPosition] = []
        pending: list[tuple[int, BacktestSignal]] = []  # (fill_bar_index, signal)
        rejections: dict[str, int] = {}
        signals_generated = 0
        signals_rejected = 0
        trade_counter = 0

        total_delay = config.latency_bars + config.entry_delay_bars
        index = frame.index

        for i in range(len(frame)):
            timestamp = index[i].to_pydatetime()
            bar = frame.iloc[i]
            open_, high, low, close = (
                float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
            )

            # --- 1. execute any pending entries scheduled for this bar --------
            still_pending: list[tuple[int, BacktestSignal]] = []
            for fill_index, signal in pending:
                if fill_index > i:
                    still_pending.append((fill_index, signal))
                    continue
                if len(open_positions) >= config.allow_simultaneous_positions:
                    signals_rejected += 1
                    rejections["max positions open"] = rejections.get("max positions open", 0) + 1
                    continue
                position = self._open_position(
                    signal=signal,
                    instrument=instrument,
                    bar_open=open_,
                    timestamp=timestamp,
                    equity=equity,
                    trade_id=f"BT{trade_counter:06d}",
                )
                if position is None:
                    signals_rejected += 1
                    rejections["sizing produced zero"] = rejections.get("sizing produced zero", 0) + 1
                    continue
                trade_counter += 1
                open_positions.append(position)
            pending = still_pending

            # --- 2. manage open positions against this bar --------------------
            remaining: list[OpenPosition] = []
            for position in open_positions:
                position.update_excursions(high, low)
                exit_price, reason = self._check_exit(
                    position, high=high, low=low, close=close, timestamp=timestamp,
                    bar_index=i, expiry=expiry, is_last_bar=(i == len(frame) - 1),
                )
                if exit_price is None:
                    remaining.append(position)
                    continue
                closed = self._close_position(
                    position, exit_price=exit_price, exit_time=timestamp, reason=reason
                )
                trades.append(closed)
                equity += closed.net_pnl
            open_positions = remaining

            # --- 3. ask for a new signal --------------------------------------
            # History passed to the source is strictly *up to and including* this
            # bar; the fill will happen later, so nothing sees its own future.
            if len(open_positions) < config.allow_simultaneous_positions and not pending:
                history = frame.iloc[: i + 1]
                signal = signal_source(timestamp, history)
                if signal is not None and signal.decision is not Decision.NO_TRADE:
                    signals_generated += 1
                    fill_index = i + max(1, total_delay) if config.fill_model == "next_bar_open" else i
                    if fill_index < len(frame):
                        pending.append((fill_index, signal))
                    else:
                        signals_rejected += 1
                        rejections["no bar left to fill"] = rejections.get("no bar left to fill", 0) + 1

            # --- 4. mark to market ---------------------------------------------
            unrealized = sum(
                (close - p.entry_price) * p.quantity * p.direction.sign for p in open_positions
            )
            equity_points.append((timestamp, equity + unrealized))

        # Force-close anything still open at the end of the data.
        if open_positions:
            final_timestamp = index[-1].to_pydatetime()
            final_close = float(frame["close"].iloc[-1])
            for position in open_positions:
                closed = self._close_position(
                    position, exit_price=final_close, exit_time=final_timestamp,
                    reason=ExitReason.END_OF_SESSION,
                )
                trades.append(closed)
                equity += closed.net_pnl
            equity_points.append((final_timestamp, equity))

        equity_curve = pd.Series(
            [v for _, v in equity_points],
            index=pd.DatetimeIndex([t for t, _ in equity_points]),
        )
        metrics = compute_metrics(trades, equity_curve, initial_capital=config.initial_capital)

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            config=config,
            bars_processed=len(frame),
            signals_generated=signals_generated,
            signals_rejected=signals_rejected,
            rejection_reasons=rejections,
            start=index[0].to_pydatetime(),
            end=index[-1].to_pydatetime(),
        )

    # ------------------------------------------------------------------ #
    def _open_position(
        self,
        *,
        signal: BacktestSignal,
        instrument: Instrument,
        bar_open: float,
        timestamp: datetime,
        equity: float,
        trade_id: str,
    ) -> OpenPosition | None:
        config = self.config
        direction = Direction.LONG if signal.decision is Decision.BUY else Direction.SHORT

        # Fill at the bar open, worsened by half the spread plus slippage — never at
        # the mid, and never at the signal bar's close.
        half_spread = bar_open * config.spread_pct / 2
        slippage_per_unit = (
            bar_open * self.costs.config.slippage_fixed_pct * config.slippage_multiplier
        )
        fill_price = bar_open + direction.sign * (half_spread + slippage_per_unit)
        fill_price = max(0.05, round(fill_price / instrument.tick_size) * instrument.tick_size)

        # Rescale the stop/target the signal proposed onto the actual fill, so the
        # R-multiple stays meaningful when the fill differs from the intended entry.
        risk_per_unit = abs(signal.entry_price - signal.stop_price)
        reward_per_unit = abs(signal.target_price - signal.entry_price)
        if risk_per_unit <= 0:
            return None
        stop_price = fill_price - direction.sign * risk_per_unit
        target_price = fill_price + direction.sign * reward_per_unit

        lot_size = max(1, instrument.lot_size)
        # Size the same way the live sizer does: risk budget / risk per unit,
        # rounded down to whole lots. Rounding up would breach the risk budget.
        risk_budget = equity * config.risk_per_trade_pct
        lots = int(np.floor((risk_budget / risk_per_unit) / lot_size))

        # Capital committed per lot depends on the product: margin for futures,
        # full premium for a long option.
        notional_per_lot = fill_price * lot_size
        capital_per_lot = (
            notional_per_lot * config.futures_margin_rate
            if instrument.is_future
            else notional_per_lot
        )
        capital_ceiling = equity * config.max_capital_per_position_pct
        max_lots_by_capital = (
            int(np.floor(capital_ceiling / capital_per_lot)) if capital_per_lot > 0 else 0
        )
        lots = min(lots, max_lots_by_capital)

        if instrument.freeze_quantity:
            lots = min(lots, int(instrument.freeze_quantity / lot_size))
        if lots < 1:
            return None
        quantity = lots * lot_size

        entry_side = TransactionType.BUY if direction is Direction.LONG else TransactionType.SELL
        entry_costs = self.costs.leg_cost(
            instrument=instrument,
            transaction_type=entry_side,
            quantity=quantity,
            price=fill_price,
        )
        total_entry_cost = (
            entry_costs.total_excluding_slippage * config.cost_multiplier
        )

        return OpenPosition(
            trade_id=trade_id,
            instrument=instrument,
            direction=direction,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_price=stop_price,
            target_price=target_price,
            strategy=signal.strategy,
            regime=signal.regime,
            ml_probability=signal.ml_probability,
            entry_costs=total_entry_cost,
            entry_slippage=slippage_per_unit * quantity,
            spread_at_entry=half_spread * 2,
            holding_limit_minutes=signal.holding_minutes,
        )

    def _check_exit(
        self,
        position: OpenPosition,
        *,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
        bar_index: int,
        expiry: date | None,
        is_last_bar: bool,
    ) -> tuple[float | None, ExitReason]:
        """Decide whether this bar closes the position, and at what price."""
        is_long = position.direction is Direction.LONG
        hit_stop = low <= position.stop_price if is_long else high >= position.stop_price
        hit_target = high >= position.target_price if is_long else low <= position.target_price

        # Both barriers inside one bar: assume the stop filled first. Without tick
        # data the order is unknown, and assuming the target would systematically
        # inflate every result.
        if hit_stop:
            return position.stop_price, ExitReason.STOP_REACHED
        if hit_target:
            return position.target_price, ExitReason.TARGET_REACHED

        held_minutes = (timestamp - position.entry_time).total_seconds() / 60
        if position.holding_limit_minutes and held_minutes >= position.holding_limit_minutes:
            return close, ExitReason.TIME_LIMIT

        moment = to_ist(timestamp)
        if expiry is not None and moment.date() >= expiry:
            return close, ExitReason.END_OF_SESSION

        if self.config.square_off_at_session_end:
            if moment.hour == 15 and moment.minute >= 15:
                return close, ExitReason.END_OF_SESSION

        if is_last_bar:
            return close, ExitReason.END_OF_SESSION

        return None, ExitReason.TARGET_REACHED

    def _close_position(
        self,
        position: OpenPosition,
        *,
        exit_price: float,
        exit_time: datetime,
        reason: ExitReason,
    ) -> ClosedTrade:
        config = self.config
        direction_sign = position.direction.sign

        half_spread = exit_price * config.spread_pct / 2
        slippage_per_unit = (
            exit_price * self.costs.config.slippage_fixed_pct * config.slippage_multiplier
        )
        # Exiting also crosses the spread and slips — against us on both sides.
        realized_exit = exit_price - direction_sign * (half_spread + slippage_per_unit)

        gross = (realized_exit - position.entry_price) * position.quantity * direction_sign

        exit_side = (
            TransactionType.SELL if position.direction is Direction.LONG else TransactionType.BUY
        )
        exit_costs = self.costs.leg_cost(
            instrument=position.instrument,
            transaction_type=exit_side,
            quantity=position.quantity,
            price=realized_exit,
        )
        total_costs = (
            position.entry_costs + exit_costs.total_excluding_slippage * config.cost_multiplier
        )
        total_slippage = position.entry_slippage + slippage_per_unit * position.quantity
        net = gross - total_costs - total_slippage

        risk_total = position.risk_per_unit * position.quantity
        r_multiple = net / risk_total if risk_total > 0 else 0.0

        return ClosedTrade(
            trade_id=position.trade_id,
            strategy=position.strategy,
            underlying=position.instrument.underlying_symbol or position.instrument.trading_symbol,
            symbol=position.instrument.trading_symbol,
            direction=position.direction.value,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=realized_exit,
            quantity=position.quantity,
            gross_pnl=gross,
            costs=total_costs,
            net_pnl=net,
            r_multiple=r_multiple,
            exit_reason=reason.value,
            regime=position.regime,
            mae=position.max_adverse,
            mfe=position.max_favourable,
            slippage=total_slippage,
            spread_at_entry=position.spread_at_entry,
            holding_minutes=(exit_time - position.entry_time).total_seconds() / 60,
            ml_probability=position.ml_probability,
        )
