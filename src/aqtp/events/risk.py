"""Event risk (REQ 37).

REQ 37 is explicit: event data must come through a *controlled* source and the
system must not scrape random websites inside the execution loop. So this module
defines an `EventProvider` interface and ships exactly one implementation — a local
YAML calendar the operator maintains or feeds from their own pipeline. There is no
network access anywhere in this file.

When an event is approaching, the configured action is applied by the RiskEngine:
block new positions, reduce risk, close positions, or continue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Iterable, Protocol

import yaml

from ..configuration.schema import EventRiskConfig
from ..core.clock import IST, to_ist
from ..core.logging import get_logger

logger = get_logger(__name__)


class EventImpact(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MarketEvent:
    """A scheduled event that may move the market."""

    name: str
    timestamp: datetime
    impact: EventImpact = EventImpact.MEDIUM
    category: str = "macro"          # macro | corporate | index | policy
    affected: tuple[str, ...] = ()   # empty = affects the whole market
    note: str = ""

    def affects(self, underlying: str) -> bool:
        if not self.affected:
            return True
        return underlying.upper() in {a.upper() for a in self.affected}

    def describe(self) -> str:
        scope = ", ".join(self.affected) if self.affected else "market-wide"
        return f"{self.name} ({self.impact.value} impact, {scope}) at {self.timestamp:%Y-%m-%d %H:%M}"


@dataclass
class EventWindow:
    """The assessment returned to the RiskEngine."""

    active: bool
    events: list[MarketEvent] = field(default_factory=list)
    minutes_to_next: float | None = None
    highest_impact: EventImpact | None = None
    action: str = "continue"
    reason: str = ""

    @property
    def blocks_new_positions(self) -> bool:
        return self.active and self.action in ("block_new", "close_positions")

    @property
    def requires_closing(self) -> bool:
        return self.active and self.action == "close_positions"

    @property
    def risk_multiplier_applies(self) -> bool:
        return self.active and self.action == "reduce_risk"


class EventProvider(Protocol):
    """Controlled event source. Implementations must not perform ad-hoc scraping."""

    def events(self, *, start: datetime, end: datetime) -> list[MarketEvent]: ...


class NullEventProvider:
    """Used when event awareness is disabled. Returns nothing, honestly."""

    def events(self, *, start: datetime, end: datetime) -> list[MarketEvent]:
        return []


class FileEventProvider:
    """Reads a YAML calendar maintained by the operator.

    Expected format:

        events:
          - name: RBI Monetary Policy
            timestamp: 2026-10-08 10:00
            impact: high
            category: policy
          - name: TCS Q2 Results
            timestamp: 2026-10-11 16:00
            impact: high
            category: corporate
            affected: [TCS]

    The file is re-read when its mtime changes, so an operator can update the
    calendar without restarting the bot.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._cache: list[MarketEvent] = []
        self._mtime: float | None = None

    def _load(self) -> None:
        if not self.path.exists():
            if self._mtime is not None:
                logger.warning("event calendar %s disappeared; treating as empty", self.path)
            self._cache, self._mtime = [], None
            return

        mtime = self.path.stat().st_mtime
        if self._mtime == mtime:
            return

        try:
            raw = yaml.safe_load(self.path.read_text()) or {}
        except yaml.YAMLError as exc:
            # A malformed calendar must not silently become "no events" — that would
            # let the bot trade straight through a policy announcement.
            logger.error("event calendar %s is malformed: %s", self.path, exc)
            self._mtime = mtime
            return

        events: list[MarketEvent] = []
        for entry in raw.get("events", []) or []:
            if not isinstance(entry, dict):
                continue
            try:
                timestamp = _parse_event_time(entry.get("timestamp"))
            except (TypeError, ValueError) as exc:
                logger.warning("skipping event %r: %s", entry.get("name"), exc)
                continue
            if timestamp is None:
                continue
            try:
                impact = EventImpact(str(entry.get("impact", "medium")).lower())
            except ValueError:
                impact = EventImpact.MEDIUM
            affected = entry.get("affected") or []
            events.append(
                MarketEvent(
                    name=str(entry.get("name", "unnamed event")),
                    timestamp=timestamp,
                    impact=impact,
                    category=str(entry.get("category", "macro")),
                    affected=tuple(str(a) for a in affected),
                    note=str(entry.get("note", "")),
                )
            )

        self._cache = sorted(events, key=lambda e: e.timestamp)
        self._mtime = mtime
        logger.info("loaded %d events from %s", len(self._cache), self.path)

    def events(self, *, start: datetime, end: datetime) -> list[MarketEvent]:
        self._load()
        start, end = to_ist(start), to_ist(end)
        return [e for e in self._cache if start <= e.timestamp <= end]

    def all_events(self) -> list[MarketEvent]:
        self._load()
        return list(self._cache)


def _parse_event_time(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_ist(value)
    if isinstance(value, date):
        # A date-only entry means "sometime that day"; anchor it at the open so the
        # blackout covers the session rather than starting late.
        return datetime(value.year, value.month, value.day, 9, 15, tzinfo=IST)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return to_ist(datetime.strptime(text, fmt))
        except ValueError:
            continue
    raise ValueError(f"unparseable event timestamp {value!r}")


class EventRiskEngine:
    """Applies the configured policy around scheduled events."""

    def __init__(self, config: EventRiskConfig, provider: EventProvider | None = None) -> None:
        self.config = config
        if provider is not None:
            self.provider = provider
        elif not config.enabled or config.source == "none":
            self.provider = NullEventProvider()
        else:
            self.provider = FileEventProvider(config.calendar_path)

    def assess(self, now: datetime, *, underlying: str = "") -> EventWindow:
        """Is an event window active for this underlying right now?"""
        if not self.config.enabled:
            return EventWindow(active=False, reason="event risk disabled")

        now = to_ist(now)
        before = timedelta(minutes=self.config.blackout_minutes_before)
        after = timedelta(minutes=self.config.blackout_minutes_after)

        candidates = self.provider.events(start=now - after, end=now + before)
        relevant = [e for e in candidates if not underlying or e.affects(underlying)]

        in_window = [
            e for e in relevant if (e.timestamp - before) <= now <= (e.timestamp + after)
        ]
        if not in_window:
            upcoming = [e for e in relevant if e.timestamp > now]
            minutes = (
                (min(e.timestamp for e in upcoming) - now).total_seconds() / 60 if upcoming else None
            )
            return EventWindow(active=False, minutes_to_next=minutes, reason="no active event window")

        highest = max(in_window, key=lambda e: _IMPACT_ORDER[e.impact])
        action = (
            self.config.high_impact_action
            if highest.impact is EventImpact.HIGH
            else ("reduce_risk" if highest.impact is EventImpact.MEDIUM else "continue")
        )
        minutes_to = (highest.timestamp - now).total_seconds() / 60

        return EventWindow(
            active=True,
            events=in_window,
            minutes_to_next=minutes_to,
            highest_impact=highest.impact,
            action=action,
            reason=(
                f"{highest.describe()} — {abs(minutes_to):.0f} minutes "
                f"{'away' if minutes_to > 0 else 'ago'}; action: {action}"
            ),
        )

    def risk_multiplier(self, window: EventWindow) -> float:
        if window.risk_multiplier_applies:
            return self.config.reduced_risk_multiplier
        return 1.0

    def upcoming(self, now: datetime, *, hours: int = 24) -> list[MarketEvent]:
        return self.provider.events(start=to_ist(now), end=to_ist(now) + timedelta(hours=hours))


_IMPACT_ORDER = {EventImpact.LOW: 0, EventImpact.MEDIUM: 1, EventImpact.HIGH: 2}
