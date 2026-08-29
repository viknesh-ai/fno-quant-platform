"""Backtest performance metrics (REQ 39).

Every metric REQ 39 lists is computed here, plus the attribution breakdowns
(by strategy, instrument and regime) that reveal whether an apparently good result
rests on one lucky instrument or one favourable month.

MAE/MFE (maximum adverse/favourable excursion) are included because they answer a
question the summary P&L cannot: were the stops nearly hit on the winners, and did
the losers ever go far in our favour? That is what tells you whether stop and target
placement is sound or merely lucky.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..core.types import Regime

# NSE trading sessions per year — used to annualize intraday-frequency returns.
TRADING_DAYS_PER_YEAR = 252


@dataclass
class ClosedTrade:
    """One completed round trip, with everything the metrics need."""

    trade_id: str
    strategy: str
    underlying: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    exit_reason: str
    regime: str = Regime.UNCERTAIN.value
    mae: float = 0.0                  # worst excursion against us, in R
    mfe: float = 0.0                  # best excursion in our favour, in R
    slippage: float = 0.0
    spread_at_entry: float = 0.0
    holding_minutes: float = 0.0
    ml_probability: float = float("nan")

    @property
    def won(self) -> bool:
        return self.net_pnl > 0


@dataclass
class BacktestMetrics:
    """The full REQ 39 report."""

    # headline
    total_return: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = float("nan")
    initial_capital: float = 0.0
    final_equity: float = 0.0

    # counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = float("nan")

    # per-trade
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    expectancy: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = float("nan")

    # risk
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    max_adverse_excursion_r: float = 0.0
    max_favourable_excursion_r: float = 0.0

    # ratios
    sharpe: float = float("nan")
    sortino: float = float("nan")
    calmar: float = float("nan")

    # behaviour
    average_holding_minutes: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # costs
    total_costs: float = 0.0
    total_slippage: float = 0.0
    cost_drag_pct: float = 0.0
    gross_pnl: float = 0.0

    # attribution
    monthly_returns: dict[str, float] = field(default_factory=dict)
    daily_returns: dict[str, float] = field(default_factory=dict)
    by_strategy: dict[str, dict] = field(default_factory=dict)
    by_instrument: dict[str, dict] = field(default_factory=dict)
    by_regime: dict[str, dict] = field(default_factory=dict)
    by_exit_reason: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"Net P&L         : {self.total_return:+,.0f} ({self.total_return_pct:+.2%})",
            f"Trades          : {self.total_trades} ({self.winning_trades}W / {self.losing_trades}L, "
            f"win rate {self.win_rate:.1%})",
            f"Expectancy      : {self.expectancy:+,.0f} per trade ({self.expectancy_r:+.3f}R)",
            f"Profit factor   : {self.profit_factor:.2f}",
            f"Max drawdown    : {self.max_drawdown:,.0f} ({self.max_drawdown_pct:.2%})",
            f"Sharpe / Sortino: {self.sharpe:.2f} / {self.sortino:.2f}",
            f"Calmar          : {self.calmar:.2f}",
            f"Costs           : {self.total_costs:,.0f} ({self.cost_drag_pct:.1%} of gross)",
            f"Avg hold        : {self.average_holding_minutes:.0f} minutes",
        ]


def compute_metrics(
    trades: list[ClosedTrade],
    equity_curve: pd.Series,
    *,
    initial_capital: float,
    risk_free_rate: float = 0.065,
) -> BacktestMetrics:
    """Build the full metric set from trades and an equity curve."""
    metrics = BacktestMetrics(initial_capital=initial_capital)

    if len(equity_curve) > 0:
        metrics.final_equity = float(equity_curve.iloc[-1])
        metrics.total_return = metrics.final_equity - initial_capital
        metrics.total_return_pct = (
            metrics.total_return / initial_capital if initial_capital > 0 else 0.0
        )

    if not trades:
        return metrics

    net = np.array([t.net_pnl for t in trades])
    gross = np.array([t.gross_pnl for t in trades])
    r_values = np.array([t.r_multiple for t in trades])
    wins, losses = net[net > 0], net[net <= 0]

    metrics.total_trades = len(trades)
    metrics.winning_trades = int(len(wins))
    metrics.losing_trades = int(len(losses))
    metrics.win_rate = len(wins) / len(trades)
    metrics.average_win = float(wins.mean()) if len(wins) else 0.0
    metrics.average_loss = float(losses.mean()) if len(losses) else 0.0
    metrics.largest_win = float(net.max())
    metrics.largest_loss = float(net.min())
    metrics.expectancy = float(net.mean())
    metrics.expectancy_r = float(np.nanmean(r_values))
    metrics.gross_pnl = float(gross.sum())

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    # Profit factor is undefined with no losses; reporting inf is more honest than
    # reporting a large finite number that looks like a measurement.
    metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    metrics.total_costs = float(sum(t.costs for t in trades))
    metrics.total_slippage = float(sum(t.slippage for t in trades))
    metrics.cost_drag_pct = (
        metrics.total_costs / abs(metrics.gross_pnl) if metrics.gross_pnl != 0 else 0.0
    )
    metrics.average_holding_minutes = float(np.mean([t.holding_minutes for t in trades]))
    metrics.max_adverse_excursion_r = float(np.nanmax([t.mae for t in trades])) if trades else 0.0
    metrics.max_favourable_excursion_r = float(np.nanmax([t.mfe for t in trades])) if trades else 0.0

    metrics.max_consecutive_wins, metrics.max_consecutive_losses = _streaks(
        [t.won for t in trades]
    )

    if len(equity_curve) > 1:
        drawdown_stats = drawdown_analysis(equity_curve)
        metrics.max_drawdown = drawdown_stats["max_drawdown"]
        metrics.max_drawdown_pct = drawdown_stats["max_drawdown_pct"]
        metrics.max_drawdown_duration_days = drawdown_stats["max_duration_days"]

        returns = equity_curve.pct_change().dropna()
        metrics.sharpe = sharpe_ratio(returns, risk_free_rate)
        metrics.sortino = sortino_ratio(returns, risk_free_rate)
        metrics.cagr = compound_annual_growth(equity_curve, initial_capital)
        metrics.calmar = (
            metrics.cagr / metrics.max_drawdown_pct
            if metrics.max_drawdown_pct > 0 and np.isfinite(metrics.cagr)
            else float("nan")
        )
        metrics.monthly_returns = _period_returns(equity_curve, "ME")
        metrics.daily_returns = _period_returns(equity_curve, "D")

    metrics.by_strategy = _attribute(trades, lambda t: t.strategy)
    metrics.by_instrument = _attribute(trades, lambda t: t.underlying)
    metrics.by_regime = _attribute(trades, lambda t: t.regime)
    counts: dict[str, int] = defaultdict(int)
    for trade in trades:
        counts[trade.exit_reason] += 1
    metrics.by_exit_reason = dict(counts)

    return metrics


# --------------------------------------------------------------------------- #
def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Annualized Sharpe. Periods-per-year is inferred from the index spacing so the
    same function works on daily and on intraday equity curves."""
    if len(returns) < 2:
        return float("nan")
    periods = _periods_per_year(returns.index)
    excess = returns - (risk_free_rate / periods)
    std = excess.std()
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    """Like Sharpe but penalizing only downside deviation."""
    if len(returns) < 2:
        return float("nan")
    periods = _periods_per_year(returns.index)
    excess = returns - (risk_free_rate / periods)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods))


def compound_annual_growth(equity_curve: pd.Series, initial_capital: float) -> float:
    if len(equity_curve) < 2 or initial_capital <= 0:
        return float("nan")
    final = float(equity_curve.iloc[-1])
    if final <= 0:
        return -1.0
    span_days = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / 86400
    if span_days < 1:
        return float("nan")
    years = span_days / 365.25
    # Annualizing a sub-annual result overstates it dramatically; refuse below a
    # quarter of a year rather than reporting a fantasy CAGR.
    if years < 0.25:
        return float("nan")
    return float((final / initial_capital) ** (1 / years) - 1)


def drawdown_analysis(equity_curve: pd.Series) -> dict[str, float]:
    peak = equity_curve.cummax()
    drawdown = peak - equity_curve
    drawdown_pct = (drawdown / peak.replace(0, np.nan)).fillna(0.0)

    max_drawdown = float(drawdown.max())
    max_drawdown_pct = float(drawdown_pct.max())

    # Longest stretch spent below a prior peak.
    underwater = equity_curve < peak
    max_duration = 0.0
    start: datetime | None = None
    for timestamp, is_under in underwater.items():
        if is_under and start is None:
            start = timestamp
        elif not is_under and start is not None:
            max_duration = max(max_duration, (timestamp - start).total_seconds() / 86400)
            start = None
    if start is not None:
        max_duration = max(
            max_duration, (equity_curve.index[-1] - start).total_seconds() / 86400
        )

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "max_duration_days": max_duration,
    }


def _periods_per_year(index: pd.Index) -> float:
    if len(index) < 2:
        return TRADING_DAYS_PER_YEAR
    try:
        median_gap = pd.Series(index).diff().median()
    except (TypeError, ValueError):
        return TRADING_DAYS_PER_YEAR
    if pd.isna(median_gap) or median_gap.total_seconds() <= 0:
        return TRADING_DAYS_PER_YEAR
    seconds = median_gap.total_seconds()
    if seconds >= 86400 * 0.9:
        return TRADING_DAYS_PER_YEAR
    # Intraday: 6.25 trading hours per session.
    return TRADING_DAYS_PER_YEAR * (6.25 * 3600 / seconds)


def _period_returns(equity_curve: pd.Series, freq: str) -> dict[str, float]:
    if len(equity_curve) < 2:
        return {}
    resampled = equity_curve.resample(freq).last().dropna()
    if len(resampled) < 2:
        return {}
    returns = resampled.pct_change().dropna()
    return {str(k.date()): round(float(v), 6) for k, v in returns.items()}


def _attribute(trades: list[ClosedTrade], key) -> dict[str, dict]:
    grouped: dict[str, list[ClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[key(trade)].append(trade)

    out: dict[str, dict] = {}
    for name, group in grouped.items():
        net = np.array([t.net_pnl for t in group])
        r_values = np.array([t.r_multiple for t in group])
        wins = net[net > 0]
        out[name] = {
            "trades": len(group),
            "net_pnl": round(float(net.sum()), 2),
            "win_rate": round(len(wins) / len(group), 4),
            "expectancy": round(float(net.mean()), 2),
            "expectancy_r": round(float(np.nanmean(r_values)), 4),
            "total_costs": round(float(sum(t.costs for t in group)), 2),
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["net_pnl"], reverse=True))


def _streaks(outcomes: list[bool]) -> tuple[int, int]:
    max_wins = max_losses = current_wins = current_losses = 0
    for won in outcomes:
        if won:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
    return max_wins, max_losses


def trades_to_frame(trades: list[ClosedTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])
