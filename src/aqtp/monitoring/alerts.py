"""Alerting (REQ 61).

Configurable alerts for the ten event types REQ 61 lists, over pluggable channels.
Two behaviours worth noting:

  * **Deduplication.** A risk limit that is breached on every loop iteration would
    otherwise emit an alert every few seconds and train the operator to ignore them.
    Repeat alerts of the same key are suppressed for a cooldown window and counted.
  * **Delivery never breaks trading.** A failing webhook must not raise into the
    trading loop, so every channel failure is caught and logged.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

from ..configuration.schema import MonitoringConfig
from ..core.logging import get_logger, redact_mapping

logger = get_logger(__name__)


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertEvent(str, Enum):
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    REJECTED_TRADE = "rejected_trade"
    RISK_LIMIT = "risk_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    BROKER_ERROR = "broker_error"
    API_DISCONNECT = "api_disconnect"
    STRATEGY_DISABLED = "strategy_disabled"
    MODEL_DRIFT = "model_drift"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    RECONCILIATION = "reconciliation"
    DATA_QUALITY = "data_quality"


_DEFAULT_LEVELS: dict[AlertEvent, AlertLevel] = {
    AlertEvent.TRADE_ENTRY: AlertLevel.INFO,
    AlertEvent.TRADE_EXIT: AlertLevel.INFO,
    AlertEvent.REJECTED_TRADE: AlertLevel.INFO,
    AlertEvent.RISK_LIMIT: AlertLevel.WARNING,
    AlertEvent.DRAWDOWN_LIMIT: AlertLevel.CRITICAL,
    AlertEvent.BROKER_ERROR: AlertLevel.WARNING,
    AlertEvent.API_DISCONNECT: AlertLevel.CRITICAL,
    AlertEvent.STRATEGY_DISABLED: AlertLevel.WARNING,
    AlertEvent.MODEL_DRIFT: AlertLevel.WARNING,
    AlertEvent.EMERGENCY_SHUTDOWN: AlertLevel.CRITICAL,
    AlertEvent.RECONCILIATION: AlertLevel.WARNING,
    AlertEvent.DATA_QUALITY: AlertLevel.WARNING,
}


@dataclass
class Alert:
    event: AlertEvent
    level: AlertLevel
    message: str
    timestamp: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    repeat_count: int = 1

    def format(self) -> str:
        suffix = f" (x{self.repeat_count})" if self.repeat_count > 1 else ""
        return f"[{self.level.value}] {self.event.value}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event": self.event.value,
            "level": self.level.value,
            "message": self.message,
            "repeat_count": self.repeat_count,
            "payload": redact_mapping(self.payload),
        }


class AlertChannel(Protocol):
    name: str

    def send(self, alert: Alert) -> None: ...


class LogChannel:
    name = "log"

    def send(self, alert: Alert) -> None:
        log = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }[alert.level]
        log("ALERT %s", alert.format())


class FileChannel:
    name = "file"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps(alert.to_dict()) + "\n")


class WebhookChannel:
    """POSTs alerts to a URL. The URL comes from the environment, never from config
    files, because it may embed a token (REQ 62)."""

    name = "webhook"

    def __init__(self, url: str | None = None, *, timeout: float = 5.0) -> None:
        self.url = url or os.environ.get("AQTP_ALERT_WEBHOOK_URL", "")
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        if not self.url:
            return
        data = json.dumps({"text": alert.format(), **alert.to_dict()}).encode()
        request = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout):
            pass


class AlertManager:
    def __init__(
        self,
        config: MonitoringConfig,
        *,
        alert_file: str | Path = "data/logs/alerts.jsonl",
        channels: list[AlertChannel] | None = None,
        dedupe_window_seconds: float = 300.0,
    ) -> None:
        self.config = config
        self.dedupe_window = timedelta(seconds=dedupe_window_seconds)
        self._lock = threading.Lock()
        self._last_sent: dict[str, tuple[datetime, int]] = {}
        self._history: list[Alert] = []
        self._suppressed: dict[str, int] = defaultdict(int)

        if channels is not None:
            self.channels = channels
        else:
            self.channels = []
            for name in config.alert_channels:
                if name == "log":
                    self.channels.append(LogChannel())
                elif name == "file":
                    self.channels.append(FileChannel(alert_file))
                elif name == "webhook":
                    self.channels.append(WebhookChannel())

        self._enabled_events = {
            e for e in AlertEvent if e.value in set(config.alert_events)
        } or set(AlertEvent)

    # ------------------------------------------------------------------ #
    def send(
        self,
        event: AlertEvent,
        message: str,
        *,
        level: AlertLevel | None = None,
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        force: bool = False,
    ) -> Alert | None:
        """Emit an alert, subject to configuration and deduplication."""
        if not self.config.alerts_enabled:
            return None
        if event not in self._enabled_events and not force:
            return None

        level = level or _DEFAULT_LEVELS.get(event, AlertLevel.INFO)
        now = datetime.now()
        key = dedupe_key or f"{event.value}:{message[:80]}"

        with self._lock:
            # Critical alerts are never suppressed — an emergency shutdown that gets
            # deduplicated into silence is the worst possible failure of this module.
            if not force and level is not AlertLevel.CRITICAL:
                previous = self._last_sent.get(key)
                if previous and now - previous[0] < self.dedupe_window:
                    self._last_sent[key] = (previous[0], previous[1] + 1)
                    self._suppressed[key] += 1
                    return None
            repeat = self._suppressed.pop(key, 0) + 1
            self._last_sent[key] = (now, 1)

        alert = Alert(
            event=event, level=level, message=message, timestamp=now,
            payload=payload or {}, repeat_count=repeat,
        )

        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as exc:
                # Never let a delivery failure propagate into the trading loop.
                logger.error("alert channel %s failed: %s", channel.name, exc)

        with self._lock:
            self._history.append(alert)
            if len(self._history) > 1000:
                del self._history[:200]
        return alert

    # -- convenience wrappers used across the codebase ------------------- #
    def trade_entry(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.TRADE_ENTRY, message, payload=payload)

    def trade_exit(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.TRADE_EXIT, message, payload=payload)

    def rejected(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.REJECTED_TRADE, message, payload=payload)

    def risk_limit(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.RISK_LIMIT, message, level=AlertLevel.WARNING, payload=payload)

    def drawdown(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.DRAWDOWN_LIMIT, message, level=AlertLevel.CRITICAL, payload=payload)

    def broker_error(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.BROKER_ERROR, message, level=AlertLevel.WARNING, payload=payload)

    def emergency(self, message: str, **payload: Any) -> None:
        self.send(
            AlertEvent.EMERGENCY_SHUTDOWN, message, level=AlertLevel.CRITICAL,
            payload=payload, force=True,
        )

    def model_drift(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.MODEL_DRIFT, message, level=AlertLevel.WARNING, payload=payload)

    def strategy_disabled(self, message: str, **payload: Any) -> None:
        self.send(AlertEvent.STRATEGY_DISABLED, message, level=AlertLevel.WARNING, payload=payload)

    def recent(self, limit: int = 50) -> list[Alert]:
        with self._lock:
            return self._history[-limit:]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for alert in self._history:
            counts[alert.event.value] += 1
        return dict(counts)
