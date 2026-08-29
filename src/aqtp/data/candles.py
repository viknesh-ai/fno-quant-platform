"""Multi-timeframe candle storage and aggregation (REQ 9).

Two rules govern this module, and they are the reason it exists as its own layer:

1. **A bar is only visible once it has closed.** `CandleSeries.closed_frame()` never
   returns the bar currently forming. Feature computation reads closed bars only,
   which is the structural defence against look-ahead bias (REQ 19).
2. **Higher timeframes are derived, not fetched twice.** A 15m bar is built by
   aggregating 1m/5m bars, so every timeframe is consistent with every other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..core.clock import IST, to_ist
from ..core.types import Candle, Timeframe

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "open_interest"]

# pandas resample rules for each timeframe.
_RESAMPLE_RULE = {
    Timeframe.M1: "1min",
    Timeframe.M3: "3min",
    Timeframe.M5: "5min",
    Timeframe.M10: "10min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "60min",
    Timeframe.H4: "240min",
    Timeframe.D1: "1D",
}


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name="timestamp"))
    frame = pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
            "open_interest": [c.open_interest for c in candles],
        },
        index=pd.DatetimeIndex([to_ist(c.timestamp) for c in candles], name="timestamp"),
    )
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def frame_to_candles(frame: pd.DataFrame) -> list[Candle]:
    return [
        Candle(
            timestamp=ts.to_pydatetime(),
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=float(row.volume),
            open_interest=float(getattr(row, "open_interest", 0.0)),
        )
        for ts, row in frame.iterrows()
    ]


def resample(frame: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate a lower-timeframe frame to a higher one.

    Bars are labelled by their *opening* time (label="left"), matching how Indian
    market data is conventionally stamped, and only complete groups are kept.
    """
    if frame.empty:
        return frame.copy()
    rule = _RESAMPLE_RULE.get(timeframe)
    if rule is None:
        raise ValueError(f"cannot resample to {timeframe.value}")

    aggregated = frame.resample(rule, label="left", closed="left", origin="start_day").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "open_interest": "last",
        }
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"])


@dataclass
class CandleSeries:
    """Bars for one instrument at one timeframe, with an explicit forming bar."""

    timeframe: Timeframe
    frame: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OHLCV_COLUMNS))
    max_bars: int = 2000

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            self.frame.index = pd.DatetimeIndex(self.frame.index, name="timestamp")

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    def __len__(self) -> int:
        return len(self.frame)

    def bar_start(self, moment: datetime) -> datetime:
        """Opening timestamp of the bar containing `moment`."""
        moment = to_ist(moment)
        if self.timeframe is Timeframe.D1:
            return moment.replace(hour=9, minute=15, second=0, microsecond=0)
        minutes = self.timeframe.minutes
        anchor = moment.replace(hour=9, minute=15, second=0, microsecond=0)
        if moment < anchor:
            anchor -= timedelta(days=1)
        elapsed = int((moment - anchor).total_seconds() // 60)
        return anchor + timedelta(minutes=(elapsed // minutes) * minutes)

    def closed_frame(self, now: datetime | None = None) -> pd.DataFrame:
        """Bars guaranteed to have closed as of `now`.

        This is the only accessor feature code should use. Passing `now=None`
        returns everything except the last bar, which is the conservative choice
        when the caller cannot supply a clock.
        """
        if self.frame.empty:
            return self.frame
        if now is None:
            return self.frame.iloc[:-1]
        current_bar = self.bar_start(now)
        return self.frame[self.frame.index < pd.Timestamp(current_bar)]

    def forming_bar(self, now: datetime | None = None) -> pd.Series | None:
        if self.frame.empty:
            return None
        if now is None:
            return self.frame.iloc[-1]
        current_bar = pd.Timestamp(self.bar_start(now))
        if self.frame.index[-1] == current_bar:
            return self.frame.iloc[-1]
        return None

    def update_from_candles(self, candles: list[Candle]) -> None:
        incoming = candles_to_frame(candles)
        if incoming.empty:
            return
        if self.frame.empty:
            self.frame = incoming
        else:
            combined = pd.concat([self.frame, incoming])
            self.frame = combined[~combined.index.duplicated(keep="last")].sort_index()
        self._trim()

    def update_from_tick(
        self, timestamp: datetime, price: float, volume: float = 0.0, open_interest: float = 0.0
    ) -> bool:
        """Fold a tick into the forming bar. Returns True if a new bar opened."""
        bar_start = pd.Timestamp(self.bar_start(timestamp))
        opened_new = False

        if self.frame.empty or bar_start not in self.frame.index:
            new_row = pd.DataFrame(
                [[price, price, price, price, volume, open_interest]],
                columns=OHLCV_COLUMNS,
                index=pd.DatetimeIndex([bar_start], name="timestamp"),
            )
            self.frame = (
                new_row if self.frame.empty else pd.concat([self.frame, new_row]).sort_index()
            )
            opened_new = not self.frame.empty and len(self.frame) > 1
        else:
            row = self.frame.loc[bar_start]
            self.frame.loc[bar_start, "high"] = max(float(row["high"]), price)
            self.frame.loc[bar_start, "low"] = min(float(row["low"]), price)
            self.frame.loc[bar_start, "close"] = price
            self.frame.loc[bar_start, "volume"] = float(row["volume"]) + volume
            if open_interest:
                self.frame.loc[bar_start, "open_interest"] = open_interest

        self._trim()
        return opened_new

    def _trim(self) -> None:
        if len(self.frame) > self.max_bars:
            self.frame = self.frame.iloc[-self.max_bars :]

    def last_close(self) -> float | None:
        if self.frame.empty:
            return None
        return float(self.frame["close"].iloc[-1])


@dataclass
class MultiTimeframeStore:
    """All timeframes for one instrument, kept mutually consistent.

    The base timeframe is stored natively; everything coarser is derived on demand
    and cached until the base series changes.
    """

    base_timeframe: Timeframe = Timeframe.M1
    timeframes: tuple[Timeframe, ...] = (
        Timeframe.M1,
        Timeframe.M5,
        Timeframe.M15,
        Timeframe.H1,
        Timeframe.D1,
    )
    max_bars: int = 2000
    _series: dict[Timeframe, CandleSeries] = field(default_factory=dict)
    _dirty: bool = True

    def __post_init__(self) -> None:
        self._series[self.base_timeframe] = CandleSeries(self.base_timeframe, max_bars=self.max_bars)

    @property
    def base(self) -> CandleSeries:
        return self._series[self.base_timeframe]

    def ingest_candles(self, candles: list[Candle], timeframe: Timeframe | None = None) -> None:
        tf = timeframe or self.base_timeframe
        series = self._series.setdefault(tf, CandleSeries(tf, max_bars=self.max_bars))
        series.update_from_candles(candles)
        if tf == self.base_timeframe:
            self._dirty = True

    def ingest_tick(
        self, timestamp: datetime, price: float, volume: float = 0.0, open_interest: float = 0.0
    ) -> None:
        self.base.update_from_tick(timestamp, price, volume, open_interest)
        self._dirty = True

    def series(self, timeframe: Timeframe) -> CandleSeries:
        if timeframe == self.base_timeframe:
            return self.base
        if self._dirty or timeframe not in self._series:
            self._rebuild(timeframe)
        return self._series[timeframe]

    def _rebuild(self, timeframe: Timeframe) -> None:
        base_frame = self.base.frame
        if base_frame.empty:
            self._series[timeframe] = CandleSeries(timeframe, max_bars=self.max_bars)
            return
        if timeframe.minutes < self.base_timeframe.minutes:
            raise ValueError(
                f"cannot derive {timeframe.value} from a {self.base_timeframe.value} base series"
            )
        self._series[timeframe] = CandleSeries(
            timeframe, frame=resample(base_frame, timeframe), max_bars=self.max_bars
        )

    def refresh_all(self) -> None:
        for tf in self.timeframes:
            if tf != self.base_timeframe:
                self._rebuild(tf)
        self._dirty = False

    def closed(self, timeframe: Timeframe, now: datetime | None = None) -> pd.DataFrame:
        """The accessor feature code uses: closed bars only, at any timeframe."""
        return self.series(timeframe).closed_frame(now)

    def bars_available(self, timeframe: Timeframe) -> int:
        return len(self.series(timeframe))


def session_vwap(frame: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, reset at each session boundary.

    Computed here rather than in the indicator module because it needs the session
    structure of the index, which is a data concern.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    volume = frame["volume"].replace(0, np.nan)
    session = frame.index.normalize()
    cum_pv = (typical * volume).groupby(session).cumsum()
    cum_vol = volume.groupby(session).cumsum()
    vwap = cum_pv / cum_vol
    # With no volume (index series), VWAP degenerates to the typical price. Say so
    # explicitly rather than emitting NaNs that silently disable VWAP strategies.
    return vwap.fillna(typical)
