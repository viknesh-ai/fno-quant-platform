"""Trade journal (REQ 47).

Every signal, every decision, every order and every trade is persisted, with the
full field list REQ 47 requires. SQLite is used because the journal must survive a
process crash and be queryable by the research tooling (REQ 50) without a server.

Rejected trades are recorded too. The record of *why the system did not trade* is
as important as the record of what it did — it is the only way to tell an edge that
never appeared from a risk limit that was silently blocking everything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..core.logging import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id         TEXT UNIQUE NOT NULL,
    timestamp           TEXT NOT NULL,
    mode                TEXT NOT NULL,
    run_id              TEXT,
    underlying          TEXT NOT NULL,
    instrument          TEXT,
    expiry              TEXT,
    strike              REAL,
    option_type         TEXT,
    decision            TEXT NOT NULL,
    direction           TEXT,
    strategy            TEXT,
    strategies_agreeing TEXT,
    regime              TEXT,
    regime_confidence   REAL,
    ml_probability      REAL,
    ml_confidence       REAL,
    ml_uncertainty      REAL,
    ensemble_score      REAL,
    ensemble_confidence REAL,
    expected_return     REAL,
    expected_risk       REAL,
    expected_value      REAL,
    expected_value_r    REAL,
    entry_price         REAL,
    stop_price          REAL,
    target_price        REAL,
    position_size       INTEGER,
    lots                INTEGER,
    risk_amount         REAL,
    spread              REAL,
    spread_pct          REAL,
    estimated_slippage  REAL,
    approved            INTEGER NOT NULL,
    rejection_reason    TEXT,
    risk_checks         TEXT,
    portfolio_state     TEXT,
    explanation         TEXT,
    reasoning           TEXT
);

CREATE TABLE IF NOT EXISTS orders (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id       TEXT UNIQUE NOT NULL,
    decision_id           TEXT,
    broker_order_id       TEXT,
    timestamp             TEXT NOT NULL,
    mode                  TEXT NOT NULL,
    instrument            TEXT NOT NULL,
    transaction_type      TEXT NOT NULL,
    order_type            TEXT,
    product               TEXT,
    quantity              INTEGER,
    price                 REAL,
    trigger_price         REAL,
    status                TEXT,
    filled_quantity       INTEGER,
    average_fill_price    REAL,
    rejection_reason      TEXT,
    intent_recorded_at    TEXT,
    submitted_at          TEXT,
    acknowledged_at       TEXT,
    filled_at             TEXT,
    signal_latency_ms     REAL,
    api_latency_ms        REAL,
    execution_latency_ms  REAL,
    total_latency_ms      REAL,
    raw                   TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions (decision_id)
);

CREATE TABLE IF NOT EXISTS trades (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id           TEXT UNIQUE NOT NULL,
    decision_id        TEXT,
    entry_order_id     TEXT,
    exit_order_id      TEXT,
    mode               TEXT NOT NULL,
    underlying         TEXT NOT NULL,
    instrument         TEXT NOT NULL,
    expiry             TEXT,
    strike             REAL,
    option_type        TEXT,
    strategy           TEXT,
    regime             TEXT,
    direction          TEXT,
    quantity           INTEGER,
    entry_time         TEXT,
    entry_price        REAL,
    exit_time          TEXT,
    exit_price         REAL,
    stop_price         REAL,
    target_price       REAL,
    gross_pnl          REAL,
    costs              REAL,
    slippage           REAL,
    net_pnl            REAL,
    r_multiple         REAL,
    exit_reason        TEXT,
    holding_minutes    REAL,
    ml_probability     REAL,
    mae_r              REAL,
    mfe_r              REAL,
    portfolio_state    TEXT,
    explanation        TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    mode       TEXT,
    level      TEXT NOT NULL,
    category   TEXT NOT NULL,
    message    TEXT NOT NULL,
    payload    TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    mode            TEXT,
    equity          REAL NOT NULL,
    realized_pnl    REAL,
    unrealized_pnl  REAL,
    open_positions  INTEGER,
    drawdown_pct    REAL,
    drawdown_level  INTEGER,
    margin_used     REAL,
    margin_available REAL
);

CREATE INDEX IF NOT EXISTS idx_decisions_time ON decisions (timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_underlying ON decisions (underlying);
CREATE INDEX IF NOT EXISTS idx_trades_time ON trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades (strategy);
CREATE INDEX IF NOT EXISTS idx_orders_client ON orders (client_order_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_time ON equity_snapshots (timestamp);
"""


def _json(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value))


class TradeJournal:
    """Thread-safe SQLite journal."""

    def __init__(self, path: str | Path, *, mode: str = "PAPER") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # WAL keeps reads (dashboard, research tooling) from blocking the trading
        # loop's writes, and survives a crash without losing committed rows.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            finally:
                cursor.close()

    # ------------------------------------------------------------------ #
    def record_decision(self, record: Mapping[str, Any]) -> None:
        """Persist a decision — approved or rejected (REQ 47/48)."""
        payload = dict(record)
        payload.setdefault("mode", self.mode)
        payload.setdefault("timestamp", datetime.now().isoformat())
        for key in ("strategies_agreeing", "risk_checks", "portfolio_state", "reasoning"):
            if key in payload and not isinstance(payload[key], (str, type(None))):
                payload[key] = _json(payload[key])
        payload["approved"] = int(bool(payload.get("approved", False)))

        columns = [c for c in _DECISION_COLUMNS if c in payload]
        placeholders = ", ".join("?" for _ in columns)
        with self._cursor() as cursor:
            cursor.execute(
                f"INSERT OR REPLACE INTO decisions ({', '.join(columns)}) VALUES ({placeholders})",
                [payload[c] for c in columns],
            )

    def record_order_intent(self, record: Mapping[str, Any]) -> None:
        """Write the *intent* to place an order BEFORE submitting it.

        This is the crash-safety half of duplicate-order prevention (REQ 45): if the
        process dies between submit and acknowledgement, the intent row proves an
        order with this client id may exist, so restart reconciliation queries the
        broker instead of resubmitting.
        """
        payload = dict(record)
        payload.setdefault("mode", self.mode)
        payload.setdefault("timestamp", datetime.now().isoformat())
        payload.setdefault("intent_recorded_at", datetime.now().isoformat())
        payload.setdefault("status", "INTENT")
        if "raw" in payload and not isinstance(payload["raw"], (str, type(None))):
            payload["raw"] = _json(payload["raw"])

        columns = [c for c in _ORDER_COLUMNS if c in payload]
        placeholders = ", ".join("?" for _ in columns)
        with self._cursor() as cursor:
            cursor.execute(
                f"INSERT OR IGNORE INTO orders ({', '.join(columns)}) VALUES ({placeholders})",
                [payload[c] for c in columns],
            )

    def update_order(self, client_order_id: str, updates: Mapping[str, Any]) -> None:
        payload = {k: v for k, v in updates.items() if k in _ORDER_COLUMNS}
        if not payload:
            return
        if "raw" in payload and not isinstance(payload["raw"], (str, type(None))):
            payload["raw"] = _json(payload["raw"])
        assignments = ", ".join(f"{k} = ?" for k in payload)
        with self._cursor() as cursor:
            cursor.execute(
                f"UPDATE orders SET {assignments} WHERE client_order_id = ?",
                [*payload.values(), client_order_id],
            )

    def record_trade(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("mode", self.mode)
        for key in ("portfolio_state",):
            if key in payload and not isinstance(payload[key], (str, type(None))):
                payload[key] = _json(payload[key])
        columns = [c for c in _TRADE_COLUMNS if c in payload]
        placeholders = ", ".join("?" for _ in columns)
        with self._cursor() as cursor:
            cursor.execute(
                f"INSERT OR REPLACE INTO trades ({', '.join(columns)}) VALUES ({placeholders})",
                [payload[c] for c in columns],
            )

    def record_event(
        self, *, level: str, category: str, message: str, payload: Any = None
    ) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO events (timestamp, mode, level, category, message, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), self.mode, level, category, message, _json(payload)),
            )

    def record_equity(
        self,
        *,
        equity: float,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        open_positions: int = 0,
        drawdown_pct: float = 0.0,
        drawdown_level: int = 1,
        margin_used: float = 0.0,
        margin_available: float = 0.0,
    ) -> None:
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO equity_snapshots (timestamp, mode, equity, realized_pnl, "
                "unrealized_pnl, open_positions, drawdown_pct, drawdown_level, margin_used, "
                "margin_available) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now().isoformat(), self.mode, equity, realized_pnl, unrealized_pnl,
                    open_positions, drawdown_pct, drawdown_level, margin_used, margin_available,
                ),
            )

    # ------------------------------------------------------------------ #
    def pending_order_intents(self) -> list[dict[str, Any]]:
        """Orders recorded as intent but never confirmed — checked on restart (REQ 63)."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM orders WHERE status IN ('INTENT', 'SUBMITTED', 'UNKNOWN') "
                "OR broker_order_id IS NULL ORDER BY timestamp DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def recent_decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def recent_trades(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def trades_for_day(self, day: date) -> list[dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM trades WHERE date(entry_time) = ? ORDER BY entry_time", (day.isoformat(),)
            )
            return [dict(row) for row in cursor.fetchall()]

    def open_trades(self) -> list[dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute("SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time")
            return [dict(row) for row in cursor.fetchall()]

    def equity_curve(self, limit: int = 5000) -> list[dict[str, Any]]:
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT timestamp, equity, drawdown_pct FROM equity_snapshots "
                "ORDER BY id DESC LIMIT ?", (limit,)
            )
            return list(reversed([dict(row) for row in cursor.fetchall()]))

    def statistics(self) -> dict[str, Any]:
        with self._cursor() as cursor:
            stats: dict[str, Any] = {}
            for table in ("decisions", "orders", "trades", "events"):
                cursor.execute(f"SELECT COUNT(*) AS n FROM {table}")
                stats[table] = cursor.fetchone()["n"]
            cursor.execute(
                "SELECT COUNT(*) AS n, SUM(net_pnl) AS pnl, "
                "SUM(CASE WHEN net_pnl > 0 THEN 1 ELSE 0 END) AS wins "
                "FROM trades WHERE exit_time IS NOT NULL"
            )
            row = cursor.fetchone()
            closed = row["n"] or 0
            stats["closed_trades"] = closed
            stats["net_pnl"] = round(row["pnl"] or 0.0, 2)
            stats["win_rate"] = round((row["wins"] or 0) / closed, 4) if closed else None
            cursor.execute("SELECT COUNT(*) AS n FROM decisions WHERE approved = 0")
            stats["rejected_decisions"] = cursor.fetchone()["n"]
            return stats

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Read-only escape hatch for the research tooling (REQ 50)."""
        stripped = sql.strip().lower()
        if not stripped.startswith(("select", "with")):
            raise ValueError("journal.query accepts SELECT statements only")
        with self._cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._connection.close()


_DECISION_COLUMNS = [
    "decision_id", "timestamp", "mode", "run_id", "underlying", "instrument", "expiry",
    "strike", "option_type", "decision", "direction", "strategy", "strategies_agreeing",
    "regime", "regime_confidence", "ml_probability", "ml_confidence", "ml_uncertainty",
    "ensemble_score", "ensemble_confidence", "expected_return", "expected_risk",
    "expected_value", "expected_value_r", "entry_price", "stop_price", "target_price",
    "position_size", "lots", "risk_amount", "spread", "spread_pct", "estimated_slippage",
    "approved", "rejection_reason", "risk_checks", "portfolio_state", "explanation", "reasoning",
]

_ORDER_COLUMNS = [
    "client_order_id", "decision_id", "broker_order_id", "timestamp", "mode", "instrument",
    "transaction_type", "order_type", "product", "quantity", "price", "trigger_price",
    "status", "filled_quantity", "average_fill_price", "rejection_reason",
    "intent_recorded_at", "submitted_at", "acknowledged_at", "filled_at",
    "signal_latency_ms", "api_latency_ms", "execution_latency_ms", "total_latency_ms", "raw",
]

_TRADE_COLUMNS = [
    "trade_id", "decision_id", "entry_order_id", "exit_order_id", "mode", "underlying",
    "instrument", "expiry", "strike", "option_type", "strategy", "regime", "direction",
    "quantity", "entry_time", "entry_price", "exit_time", "exit_price", "stop_price",
    "target_price", "gross_pnl", "costs", "slippage", "net_pnl", "r_multiple", "exit_reason",
    "holding_minutes", "ml_probability", "mae_r", "mfe_r", "portfolio_state", "explanation",
]
