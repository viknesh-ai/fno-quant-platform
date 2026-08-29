"""Latency measurement (REQ 33).

REQ 33 names six timestamps that must be recorded and four latencies that must be
derived from them. This module owns those definitions so every component reports
them identically.

The reason this matters beyond bookkeeping: `total_latency_ms` is the gap between
the market state a decision was based on and the price actually obtained. When that
number grows, the backtest's fill assumptions stop being valid — so it is tracked
as a first-class metric, not a diagnostic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

import numpy as np


@dataclass
class LatencyTrace:
    """The six timestamps REQ 33 requires, for one order."""

    market_data_at: datetime | None = None
    signal_at: datetime | None = None
    risk_check_at: datetime | None = None
    order_submit_at: datetime | None = None
    broker_ack_at: datetime | None = None
    fill_at: datetime | None = None

    @staticmethod
    def _ms(start: datetime | None, end: datetime | None) -> float | None:
        if start is None or end is None:
            return None
        return (end - start).total_seconds() * 1000.0

    @property
    def signal_latency_ms(self) -> float | None:
        """Market data -> signal produced. How stale the data was when we decided."""
        return self._ms(self.market_data_at, self.signal_at)

    @property
    def risk_latency_ms(self) -> float | None:
        return self._ms(self.signal_at, self.risk_check_at)

    @property
    def api_latency_ms(self) -> float | None:
        """Submit -> broker acknowledgement. The broker's own round trip."""
        return self._ms(self.order_submit_at, self.broker_ack_at)

    @property
    def execution_latency_ms(self) -> float | None:
        """Acknowledgement -> fill. Time spent resting in the book."""
        return self._ms(self.broker_ack_at, self.fill_at)

    @property
    def total_latency_ms(self) -> float | None:
        """Market data -> fill. The number that governs backtest realism."""
        return self._ms(self.market_data_at, self.fill_at)

    @property
    def decision_to_submit_ms(self) -> float | None:
        return self._ms(self.signal_at, self.order_submit_at)

    def to_dict(self) -> dict[str, float | None]:
        return {
            "signal_latency_ms": _round(self.signal_latency_ms),
            "risk_latency_ms": _round(self.risk_latency_ms),
            "api_latency_ms": _round(self.api_latency_ms),
            "execution_latency_ms": _round(self.execution_latency_ms),
            "total_latency_ms": _round(self.total_latency_ms),
        }

    def timestamps(self) -> dict[str, str | None]:
        return {
            "market_data_at": _iso(self.market_data_at),
            "signal_at": _iso(self.signal_at),
            "risk_check_at": _iso(self.risk_check_at),
            "order_submit_at": _iso(self.order_submit_at),
            "broker_ack_at": _iso(self.broker_ack_at),
            "fill_at": _iso(self.fill_at),
        }

    def describe(self) -> str:
        parts = [
            f"{name}={value:.0f}ms"
            for name, value in self.to_dict().items()
            if value is not None
        ]
        return ", ".join(parts) if parts else "no latency data"


def _round(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class LatencyMonitor:
    """Rolling latency statistics used by the dashboard and health checks."""

    def __init__(self, window: int = 200) -> None:
        self._traces: Deque[LatencyTrace] = deque(maxlen=window)

    def record(self, trace: LatencyTrace) -> None:
        self._traces.append(trace)

    def _series(self, attribute: str) -> np.ndarray:
        values = [
            getattr(t, attribute) for t in self._traces if getattr(t, attribute) is not None
        ]
        return np.array(values, dtype=float)

    def statistics(self) -> dict[str, dict[str, float]]:
        """Percentiles matter more than the mean here: a p99 of two seconds is a
        real execution problem even when the mean looks fine."""
        out: dict[str, dict[str, float]] = {}
        for attribute in (
            "signal_latency_ms",
            "api_latency_ms",
            "execution_latency_ms",
            "total_latency_ms",
        ):
            values = self._series(attribute)
            if len(values) == 0:
                continue
            out[attribute] = {
                "count": int(len(values)),
                "mean": round(float(values.mean()), 2),
                "median": round(float(np.median(values)), 2),
                "p95": round(float(np.percentile(values, 95)), 2),
                "p99": round(float(np.percentile(values, 99)), 2),
                "max": round(float(values.max()), 2),
            }
        return out

    def is_degraded(self, *, total_latency_threshold_ms: float = 3000.0) -> tuple[bool, str]:
        values = self._series("total_latency_ms")
        if len(values) < 5:
            return False, ""
        p95 = float(np.percentile(values, 95))
        if p95 > total_latency_threshold_ms:
            return True, (
                f"p95 decision-to-fill latency is {p95:.0f}ms, above the "
                f"{total_latency_threshold_ms:.0f}ms threshold — fills are no longer "
                "comparable to backtest assumptions"
            )
        return False, ""

    def __len__(self) -> int:
        return len(self._traces)
