"""Time, sessions and expiry arithmetic.

Every component takes its notion of "now" from a Clock instance rather than calling
`datetime.now()` directly. That is what makes the backtester able to replay history
through exactly the same code paths as the live orchestrator (REQ 38), and what
makes clock-skew survivable (REQ 63).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Protocol

IST = timezone(timedelta(hours=5, minutes=30), name="IST")

# NSE equity/F&O continuous session.
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_OPEN_START = time(9, 0)


class Clock(Protocol):
    def now(self) -> datetime: ...


class LiveClock:
    """Wall clock, always timezone-aware in IST."""

    def now(self) -> datetime:
        return datetime.now(IST)


@dataclass
class SimulatedClock:
    """Backtest clock. The engine advances it explicitly; nothing may read ahead."""

    current: datetime

    def now(self) -> datetime:
        return self.current

    def advance_to(self, moment: datetime) -> None:
        if moment < self.current:
            raise ValueError(f"simulated clock cannot move backwards: {moment} < {self.current}")
        self.current = moment


def to_ist(moment: datetime) -> datetime:
    """Normalize any datetime to IST; naive inputs are assumed to already be IST."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=IST)
    return moment.astimezone(IST)


def is_market_open(moment: datetime, *, holidays: frozenset[date] = frozenset()) -> bool:
    moment = to_ist(moment)
    if moment.weekday() >= 5 or moment.date() in holidays:
        return False
    return MARKET_OPEN <= moment.time() <= MARKET_CLOSE


def minutes_since_open(moment: datetime) -> float:
    moment = to_ist(moment)
    open_dt = moment.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    return (moment - open_dt).total_seconds() / 60.0


def minutes_until_close(moment: datetime) -> float:
    moment = to_ist(moment)
    close_dt = moment.replace(hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0)
    return (close_dt - moment).total_seconds() / 60.0


SESSION_MINUTES = (
    (datetime(2000, 1, 1, MARKET_CLOSE.hour, MARKET_CLOSE.minute)
     - datetime(2000, 1, 1, MARKET_OPEN.hour, MARKET_OPEN.minute)).total_seconds() / 60.0
)


def time_to_expiry_years(now: datetime, expiry: date, *, expiry_time: time = MARKET_CLOSE) -> float:
    """Year fraction to expiry, measured in *calendar* time.

    Calendar time (not trading time) is the right basis for Black-Scholes here
    because the greeks we compare against are quoted by the exchange on the same
    convention. Floored at a small positive value so pricing never divides by zero
    on expiry day.
    """
    now = to_ist(now)
    expiry_dt = datetime.combine(expiry, expiry_time, tzinfo=IST)
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds / (365.0 * 24 * 3600), 1e-6)


def sessions_to_expiry(now: datetime, expiry: date, *, holidays: frozenset[date] = frozenset()) -> int:
    """Number of remaining trading sessions, inclusive of expiry day."""
    now = to_ist(now)
    day = now.date()
    count = 0
    while day <= expiry:
        if day.weekday() < 5 and day not in holidays:
            count += 1
        day += timedelta(days=1)
    return count


def is_expiry_day(now: datetime, expiry: date) -> bool:
    return to_ist(now).date() == expiry
