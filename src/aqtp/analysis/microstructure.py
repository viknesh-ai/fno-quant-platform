"""Order-book and tape microstructure.

An option is bought at the ask and sold at the bid, so the shape of the book is
not a detail of execution — it is part of the edge. A signal worth 0.4% that has
to cross a 0.6% spread into thinning depth is not a trade. This module measures
that directly, and separately measures whether the flow hitting the book is
one-sided enough to be worth following.

Two kinds of measurement live here:

* **Snapshot** — imbalance, micro-price, depth slope, cost to trade a given size.
  Computed from one `Quote`.
* **Sequential** — order-flow imbalance, quote-update intensity, flow toxicity,
  absorption. These need history, which is what `MicrostructureTracker` keeps.

Everything is per-instrument and bounded in memory.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

import numpy as np
import pandas as pd

from ..core.logging import get_logger
from ..core.types import Quote

logger = get_logger(__name__)


@dataclass
class MicrostructureReport:
    """The state of the book and the tape for one instrument."""

    symbol: str = ""
    mid: float | None = None
    micro_price: float | None = None
    spread_pct: float | None = None
    spread_ticks: float | None = None

    # --- book shape ---------------------------------------------------------
    touch_imbalance: float = 0.0        # [-1, 1] from L1 only
    depth_imbalance: float = 0.0        # [-1, 1] across the visible book
    book_imbalance: float = 0.0         # [-1, 1] from exchange-wide buy/sell qty
    depth_slope_bid: float | None = None
    depth_slope_ask: float | None = None
    book_levels: int = 0
    visible_depth_value: float = 0.0    # rupees resting within the visible book

    # --- cost of trading ----------------------------------------------------
    round_trip_cost_pct: float | None = None
    impact_cost_pct: float | None = None   # slippage to fill the tested size
    tested_quantity: int = 0

    # --- sequential ---------------------------------------------------------
    order_flow_imbalance: float = 0.0   # [-1, 1] over the tracked window
    quote_updates: int = 0
    price_reversals: int = 0
    flow_toxicity: float = 0.0          # [0, 1], VPIN-style
    absorption: float = 0.0             # [0, 1]: size traded with little price move
    tape_pressure: float = 0.0          # [-1, 1] from candle-derived signed volume
    cumulative_delta_slope: float = 0.0

    reasons: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def pressure(self) -> float:
        """Signed [-1, 1] net buying pressure across every measurement.

        Weighted toward measurements that are hardest to spoof: realised order
        flow and tape delta outrank resting depth, because resting depth can be
        cancelled and frequently is.
        """
        parts = [
            (self.touch_imbalance, 0.15),
            (self.depth_imbalance, 0.15),
            (self.book_imbalance, 0.10),
            (self.order_flow_imbalance, 0.35),
            (self.tape_pressure, 0.25),
        ]
        total_weight = sum(w for value, w in parts if value is not None)
        if total_weight <= 0:
            return 0.0
        return float(
            np.clip(sum(value * w for value, w in parts if value is not None) / total_weight, -1.0, 1.0)
        )

    @property
    def execution_quality(self) -> float:
        """[0, 1] — how safe it is to actually transact here."""
        parts: list[float] = []
        if self.spread_pct is not None:
            parts.append(float(np.clip(1.0 - self.spread_pct / 0.02, 0.0, 1.0)))
        if self.impact_cost_pct is not None:
            parts.append(float(np.clip(1.0 - self.impact_cost_pct / 0.01, 0.0, 1.0)))
        if self.visible_depth_value > 0:
            parts.append(float(np.clip(math.log1p(self.visible_depth_value) / math.log1p(5_000_000), 0.0, 1.0)))
        parts.append(float(np.clip(1.0 - self.flow_toxicity, 0.0, 1.0)))
        return float(np.mean(parts)) if parts else 0.0

    @property
    def is_toxic(self) -> bool:
        """Flow that is running one way through the book with no absorption.

        Entering into toxic flow means being the liquidity someone informed is
        taking. It is the single most reliable way to buy a top.
        """
        return self.flow_toxicity > 0.75 and self.absorption < 0.25

    def to_dict(self) -> dict[str, float]:
        out = {
            "micro_pressure": self.pressure,
            "micro_execution_quality": self.execution_quality,
            "micro_touch_imbalance": self.touch_imbalance,
            "micro_depth_imbalance": self.depth_imbalance,
            "micro_book_imbalance": self.book_imbalance,
            "micro_order_flow_imbalance": self.order_flow_imbalance,
            "micro_flow_toxicity": self.flow_toxicity,
            "micro_absorption": self.absorption,
            "micro_tape_pressure": self.tape_pressure,
            "micro_cum_delta_slope": self.cumulative_delta_slope,
        }
        for name in ("spread_pct", "impact_cost_pct", "round_trip_cost_pct"):
            value = getattr(self, name)
            if value is not None:
                out[f"micro_{name}"] = float(value)
        return out

    def explain(self) -> list[str]:
        return list(self.reasons)


# --------------------------------------------------------------------------- #
def _depth_slope(levels: tuple, reference: float) -> float | None:
    """How fast size thins out as you walk away from the touch.

    A steep positive slope means size is stacked behind the touch; a flat or
    negative slope means the book is a facade and a market order will walk it.
    """
    if len(levels) < 2 or reference <= 0:
        return None
    distances = []
    quantities = []
    for level in levels:
        if level.price <= 0 or level.quantity <= 0:
            continue
        distances.append(abs(level.price - reference) / reference)
        quantities.append(float(level.quantity))
    if len(distances) < 2 or np.allclose(distances, distances[0]):
        return None
    slope, _ = np.polyfit(distances, quantities, 1)
    return float(slope)


def _walk_book(levels: tuple, quantity: int, reference: float) -> float | None:
    """Volume-weighted cost of consuming `quantity` from one side of the book,
    expressed as a fraction of `reference`. `None` when the book cannot fill it."""
    if quantity <= 0 or reference <= 0 or not levels:
        return None
    remaining = quantity
    notional = 0.0
    for level in levels:
        if level.price <= 0 or level.quantity <= 0:
            continue
        take = min(remaining, int(level.quantity))
        notional += take * level.price
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        return None
    average = notional / quantity
    return abs(average - reference) / reference


def analyze_quote(quote: Quote, *, test_quantity: int = 0) -> MicrostructureReport:
    """Snapshot analysis of one quote's book."""
    report = MicrostructureReport(symbol=quote.instrument.trading_symbol)
    report.mid = quote.mid
    report.spread_pct = quote.spread_pct

    tick = quote.instrument.tick_size or 0.05
    if quote.spread is not None and tick > 0:
        report.spread_ticks = round(quote.spread / tick, 2)

    bid_qty = float(quote.bid_quantity or 0)
    ask_qty = float(quote.ask_quantity or 0)
    if bid_qty + ask_qty > 0:
        report.touch_imbalance = float((bid_qty - ask_qty) / (bid_qty + ask_qty))

    if quote.mid and quote.bid_price and quote.ask_price and (bid_qty + ask_qty) > 0:
        # Micro-price: the size-weighted fair value. When it sits above the mid,
        # the book is leaning bid and the next print is more likely at the ask.
        report.micro_price = float(
            (quote.bid_price * ask_qty + quote.ask_price * bid_qty) / (bid_qty + ask_qty)
        )

    depth_buy, depth_sell = quote.depth_buy, quote.depth_sell
    report.book_levels = max(len(depth_buy), len(depth_sell))
    if depth_buy or depth_sell:
        buy_size = sum(float(level.quantity) for level in depth_buy)
        sell_size = sum(float(level.quantity) for level in depth_sell)
        if buy_size + sell_size > 0:
            report.depth_imbalance = float((buy_size - sell_size) / (buy_size + sell_size))
        report.visible_depth_value = float(
            sum(level.price * level.quantity for level in depth_buy)
            + sum(level.price * level.quantity for level in depth_sell)
        )
        if report.mid:
            report.depth_slope_bid = _depth_slope(depth_buy, report.mid)
            report.depth_slope_ask = _depth_slope(depth_sell, report.mid)

    total_buy = float(quote.total_buy_quantity or 0)
    total_sell = float(quote.total_sell_quantity or 0)
    if total_buy + total_sell > 0:
        report.book_imbalance = float((total_buy - total_sell) / (total_buy + total_sell))

    if report.spread_pct is not None:
        report.round_trip_cost_pct = report.spread_pct  # cross in, cross out

    if test_quantity > 0 and report.mid:
        report.tested_quantity = test_quantity
        cost = _walk_book(quote.depth_sell, test_quantity, report.mid)
        report.impact_cost_pct = cost
        if cost is None and report.spread_pct is not None:
            # The visible book cannot fill the order. Assume it walks at least
            # twice the quoted spread rather than pretending the cost is unknown.
            report.impact_cost_pct = report.spread_pct * 2.0
            report.reasons.append(
                f"visible book cannot absorb {test_quantity} units — impact cost is an estimate"
            )

    # --- narrative ---------------------------------------------------------
    if report.spread_pct is not None:
        report.reasons.append(
            f"spread {report.spread_pct:.3%}"
            + (f" ({report.spread_ticks:.0f} ticks)" if report.spread_ticks else "")
        )
    if abs(report.depth_imbalance) > 0.25:
        side = "bid" if report.depth_imbalance > 0 else "ask"
        report.reasons.append(
            f"book leans {side}: depth imbalance {report.depth_imbalance:+.2f} "
            f"across {report.book_levels} level(s)"
        )
    if report.micro_price and report.mid:
        drift = (report.micro_price - report.mid) / report.mid
        if abs(drift) > 0.0005:
            report.reasons.append(
                f"micro-price {drift:+.3%} from mid — next print skews "
                f"{'up' if drift > 0 else 'down'}"
            )
    if report.impact_cost_pct is not None and report.tested_quantity:
        report.reasons.append(
            f"impact cost to lift {report.tested_quantity} units: {report.impact_cost_pct:.3%}"
        )
    return report


# --------------------------------------------------------------------------- #
def tape_analysis(frame: pd.DataFrame, *, window: int = 40) -> tuple[float, float, float]:
    """Signed-volume analysis of recent candles.

    Without a real trade feed the side of each print is unknown, so volume is
    signed by where the close sits inside the bar's range — the standard tick-rule
    approximation. It is noisy per bar and informative in aggregate.

    Returns `(tape_pressure, cumulative_delta_slope, absorption)`.
    """
    if frame is None or frame.empty or len(frame) < 10:
        return 0.0, 0.0, 0.0
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(frame.columns):
        return 0.0, 0.0, 0.0

    tail = frame.tail(window)
    high = tail["high"].astype(float).to_numpy()
    low = tail["low"].astype(float).to_numpy()
    close = tail["close"].astype(float).to_numpy()
    volume = tail["volume"].astype(float).to_numpy()

    span = np.maximum(high - low, 1e-9)
    # +1 when the bar closes on its high, -1 on its low.
    position = np.clip((2.0 * close - high - low) / span, -1.0, 1.0)
    signed = position * volume

    total_volume = float(volume.sum())
    tape_pressure = float(np.clip(signed.sum() / total_volume, -1.0, 1.0)) if total_volume > 0 else 0.0

    cumulative = np.cumsum(signed)
    slope = 0.0
    if cumulative.size >= 5 and total_volume > 0:
        x = np.arange(cumulative.size, dtype=float)
        raw_slope, _ = np.polyfit(x, cumulative, 1)
        # Normalise by average bar volume so the slope compares across symbols.
        slope = float(np.clip(raw_slope / max(total_volume / cumulative.size, 1e-9), -1.0, 1.0))

    # Absorption: heavy volume that failed to move price. High absorption at a
    # level is where trends stall, and it is invisible to any price-only feature.
    absorption = 0.0
    if total_volume > 0 and close.size >= 5:
        move = abs(close[-1] - close[0]) / max(close[0], 1e-9)
        typical_move = float(np.mean(np.abs(np.diff(close)) / np.maximum(close[:-1], 1e-9)))
        expected = typical_move * math.sqrt(close.size)
        if expected > 1e-9:
            volume_ratio = float(volume[-5:].mean() / max(volume.mean(), 1e-9))
            absorption = float(np.clip((1.0 - move / expected) * volume_ratio, 0.0, 1.0))

    return tape_pressure, slope, absorption


# --------------------------------------------------------------------------- #
@dataclass
class _QuoteSample:
    timestamp: datetime
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    last: float
    volume: float


class MicrostructureTracker:
    """Per-instrument rolling quote history for the sequential measurements.

    Bounded by construction: `history` samples per symbol and `max_symbols`
    symbols, both evicted least-recently-used. A trading session touching two
    hundred option contracts must not grow without limit.
    """

    def __init__(self, *, history: int = 120, max_symbols: int = 400) -> None:
        self._history = history
        self._max_symbols = max_symbols
        self._samples: dict[str, Deque[_QuoteSample]] = {}

    def observe(self, quote: Quote) -> None:
        symbol = quote.instrument.trading_symbol
        if symbol not in self._samples and len(self._samples) >= self._max_symbols:
            self._samples.pop(next(iter(self._samples)))
        series = self._samples.setdefault(symbol, deque(maxlen=self._history))
        series.append(
            _QuoteSample(
                timestamp=quote.timestamp,
                bid=float(quote.bid_price or 0.0),
                ask=float(quote.ask_price or 0.0),
                bid_qty=float(quote.bid_quantity or 0.0),
                ask_qty=float(quote.ask_quantity or 0.0),
                last=float(quote.last_price or 0.0),
                volume=float(quote.volume or 0.0),
            )
        )

    def samples(self, symbol: str) -> int:
        return len(self._samples.get(symbol, ()))

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        quote: Quote,
        *,
        frame: pd.DataFrame | None = None,
        test_quantity: int = 0,
        observe: bool = True,
    ) -> MicrostructureReport:
        """Full microstructure report: snapshot plus everything history allows."""
        if observe:
            self.observe(quote)
        report = analyze_quote(quote, test_quantity=test_quantity)

        series = list(self._samples.get(quote.instrument.trading_symbol, ()))
        report.quote_updates = len(series)
        if len(series) >= 3:
            ofi, reversals, toxicity = self._sequential(series)
            report.order_flow_imbalance = ofi
            report.price_reversals = reversals
            report.flow_toxicity = toxicity
            if abs(ofi) > 0.3:
                report.reasons.append(
                    f"order-flow imbalance {ofi:+.2f} over the last {len(series)} quote updates"
                )
            if toxicity > 0.6:
                report.reasons.append(
                    f"flow toxicity {toxicity:.2f} — one-sided flow is running through the book"
                )

        if frame is not None and not frame.empty:
            pressure, slope, absorption = tape_analysis(frame)
            report.tape_pressure = pressure
            report.cumulative_delta_slope = slope
            report.absorption = absorption
            if abs(pressure) > 0.15:
                report.reasons.append(
                    f"tape pressure {pressure:+.2f}, cumulative-delta slope {slope:+.2f}"
                )
            if absorption > 0.5:
                report.reasons.append(
                    f"absorption {absorption:.2f} — heavy volume, little price progress"
                )

        if report.is_toxic:
            report.reasons.append(
                "TOXIC FLOW: taking liquidity here means trading against informed flow"
            )
        return report

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sequential(series: list[_QuoteSample]) -> tuple[float, int, float]:
        """Order-flow imbalance, reversal count and VPIN-style toxicity.

        OFI follows Cont/Kukanov/Stoikov: a bid that rises or grows adds buy
        pressure, an ask that falls or grows adds sell pressure. It is the
        best-established book-derived predictor of short-horizon returns.
        """
        ofi_terms: list[float] = []
        reversals = 0
        buy_volume = 0.0
        sell_volume = 0.0

        for previous, current in zip(series, series[1:]):
            if current.bid > previous.bid:
                ofi_terms.append(current.bid_qty)
            elif current.bid < previous.bid:
                ofi_terms.append(-previous.bid_qty)
            else:
                ofi_terms.append(current.bid_qty - previous.bid_qty)

            if current.ask < previous.ask:
                ofi_terms.append(-current.ask_qty)
            elif current.ask > previous.ask:
                ofi_terms.append(previous.ask_qty)
            else:
                ofi_terms.append(previous.ask_qty - current.ask_qty)

            traded = max(0.0, current.volume - previous.volume)
            if traded > 0:
                mid_previous = (previous.bid + previous.ask) / 2.0 if previous.bid and previous.ask else previous.last
                if current.last > mid_previous:
                    buy_volume += traded
                elif current.last < mid_previous:
                    sell_volume += traded
                else:
                    buy_volume += traded / 2.0
                    sell_volume += traded / 2.0

        if len(series) >= 3:
            for a, b, c in zip(series, series[1:], series[2:]):
                if (b.last - a.last) * (c.last - b.last) < 0:
                    reversals += 1

        scale = float(np.sum(np.abs(ofi_terms))) if ofi_terms else 0.0
        ofi = float(np.clip(sum(ofi_terms) / scale, -1.0, 1.0)) if scale > 1e-9 else 0.0

        total = buy_volume + sell_volume
        toxicity = float(abs(buy_volume - sell_volume) / total) if total > 0 else 0.0
        return ofi, reversals, toxicity
