"""Shared test fixtures.

Synthetic data is generated deterministically from a fixed seed so that a failing
test fails for everyone, not just on one machine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from aqtp.configuration.loader import load_config
from aqtp.configuration.schema import AppConfig
from aqtp.core.clock import IST
from aqtp.core.types import (
    Candle,
    DepthLevel,
    Exchange,
    Greeks,
    Instrument,
    InstrumentType,
    OptionChain,
    OptionContract,
    OptionType,
    Quote,
    Segment,
    Timeframe,
)
from aqtp.data.candles import candles_to_frame, resample

SESSION_START = datetime(2026, 6, 1, 9, 15, tzinfo=IST)


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #
@pytest.fixture
def index_instrument() -> Instrument:
    return Instrument(
        trading_symbol="NIFTY",
        exchange=Exchange.NSE,
        segment=Segment.CASH,
        instrument_type=InstrumentType.IDX,
        name="NIFTY 50",
        lot_size=1,
    )


@pytest.fixture
def future_instrument() -> Instrument:
    return Instrument(
        trading_symbol="NIFTY26JUNFUT",
        exchange=Exchange.NSE,
        segment=Segment.FNO,
        instrument_type=InstrumentType.FUT,
        underlying_symbol="NIFTY",
        lot_size=75,
        tick_size=0.05,
        freeze_quantity=1800,
        expiry_date=date(2026, 6, 25),
    )


@pytest.fixture
def call_instrument() -> Instrument:
    return Instrument(
        trading_symbol="NIFTY26JUN24000CE",
        exchange=Exchange.NSE,
        segment=Segment.FNO,
        instrument_type=InstrumentType.CE,
        underlying_symbol="NIFTY",
        lot_size=75,
        tick_size=0.05,
        freeze_quantity=1800,
        expiry_date=date(2026, 6, 25),
        strike_price=24000.0,
    )


@pytest.fixture
def stock_future() -> Instrument:
    return Instrument(
        trading_symbol="TESTSTK26JUNFUT",
        exchange=Exchange.NSE,
        segment=Segment.FNO,
        instrument_type=InstrumentType.FUT,
        underlying_symbol="TESTSTK",
        lot_size=250,
        tick_size=0.05,
        freeze_quantity=10000,
        expiry_date=date(2026, 6, 25),
    )


# --------------------------------------------------------------------------- #
# Price series
# --------------------------------------------------------------------------- #
def make_candles(
    *,
    bars: int = 500,
    start_price: float = 24000.0,
    drift: float = 0.0,
    volatility: float = 0.0006,
    momentum_decay: float = 0.0,
    seed: int = 42,
    start: datetime = SESSION_START,
    with_volume: bool = True,
) -> list[Candle]:
    """Deterministic synthetic 1-minute candles."""
    rng = np.random.default_rng(seed)
    price = start_price
    momentum = 0.0
    out: list[Candle] = []
    timestamp = start
    session_bar = 0

    for _ in range(bars):
        if momentum_decay:
            momentum = momentum_decay * momentum + rng.normal(0, volatility / 2)
        price *= 1 + drift + momentum + rng.normal(0, volatility)
        high = price * (1 + abs(rng.normal(0, volatility / 2)))
        low = price * (1 - abs(rng.normal(0, volatility / 2)))
        out.append(
            Candle(
                timestamp=timestamp,
                open=price,
                high=max(high, price),
                low=min(low, price),
                close=price,
                volume=float(1000 + rng.integers(0, 800)) if with_volume else 0.0,
            )
        )
        timestamp += timedelta(minutes=1)
        session_bar += 1
        if session_bar >= 375:  # roll to the next session
            session_bar = 0
            timestamp = (timestamp + timedelta(days=1)).replace(hour=9, minute=15)
            while timestamp.weekday() >= 5:
                timestamp += timedelta(days=1)
    return out


@pytest.fixture
def trending_frame() -> pd.DataFrame:
    return candles_to_frame(make_candles(bars=500, drift=0.0006, seed=1))


@pytest.fixture
def ranging_frame() -> pd.DataFrame:
    return candles_to_frame(make_candles(bars=500, drift=0.0, volatility=0.0004, seed=2))


@pytest.fixture
def downtrend_frame() -> pd.DataFrame:
    return candles_to_frame(make_candles(bars=500, drift=-0.0006, seed=3))


@pytest.fixture
def multi_timeframe_frames() -> dict[Timeframe, pd.DataFrame]:
    base = candles_to_frame(
        make_candles(bars=375 * 60, drift=0.0, momentum_decay=0.96, seed=11)
    )
    return {
        Timeframe.M5: resample(base, Timeframe.M5),
        Timeframe.M15: resample(base, Timeframe.M15),
        Timeframe.H1: resample(base, Timeframe.H1),
    }


# --------------------------------------------------------------------------- #
# Quotes and chains
# --------------------------------------------------------------------------- #
def make_quote(
    instrument: Instrument,
    *,
    price: float = 200.0,
    spread: float = 1.0,
    volume: float = 50_000,
    open_interest: float = 200_000,
    now: datetime | None = None,
    age_seconds: float = 0.0,
    with_depth: bool = True,
) -> Quote:
    now = now or datetime(2026, 6, 1, 11, 0, tzinfo=IST)
    return Quote(
        instrument=instrument,
        last_price=price,
        timestamp=now - timedelta(seconds=age_seconds),
        received_at=now,
        open=price * 0.99,
        high=price * 1.02,
        low=price * 0.97,
        close=price * 0.985,
        previous_close=price * 0.985,
        volume=volume,
        bid_price=price - spread / 2,
        ask_price=price + spread / 2,
        bid_quantity=500,
        ask_quantity=500,
        open_interest=open_interest,
        previous_open_interest=open_interest * 0.95,
        oi_day_change=open_interest * 0.05,
        implied_volatility=0.14,
        upper_circuit=price * 1.20,
        lower_circuit=price * 0.80,
        depth_buy=(
            (DepthLevel(price - spread / 2, 500), DepthLevel(price - spread, 800))
            if with_depth else ()
        ),
        depth_sell=(
            (DepthLevel(price + spread / 2, 500), DepthLevel(price + spread, 800))
            if with_depth else ()
        ),
    )


@pytest.fixture
def option_quote(call_instrument) -> Quote:
    return make_quote(call_instrument, price=200.0, spread=1.0)


def make_option_chain(
    *,
    underlying: str = "NIFTY",
    spot: float = 24000.0,
    expiry: date = date(2026, 6, 25),
    strikes: int = 11,
    step: float = 100.0,
    now: datetime | None = None,
) -> OptionChain:
    """A realistic chain priced with Black-Scholes and consistent greeks."""
    from aqtp.core.clock import time_to_expiry_years
    from aqtp.options.pricing import black_scholes_price, compute_greeks

    now = now or datetime(2026, 6, 1, 11, 0, tzinfo=IST)
    tte = time_to_expiry_years(now, expiry)
    iv = 0.14
    contracts: list[OptionContract] = []
    atm = round(spot / step) * step

    for offset in range(-(strikes // 2), strikes // 2 + 1):
        strike = atm + offset * step
        for option_type in (OptionType.CE, OptionType.PE):
            price = black_scholes_price(spot, strike, tte, iv, option_type)
            if price < 0.5:
                price = 0.5
            greeks = compute_greeks(spot, strike, tte, iv, option_type)
            instrument = Instrument(
                trading_symbol=f"{underlying}26JUN{int(strike)}{option_type.value}",
                exchange=Exchange.NSE,
                segment=Segment.FNO,
                instrument_type=InstrumentType(option_type.value),
                underlying_symbol=underlying,
                lot_size=75,
                tick_size=0.05,
                freeze_quantity=1800,
                expiry_date=expiry,
                strike_price=strike,
            )
            # Liquidity peaks at the money and falls away, as real chains do.
            liquidity = max(0.1, 1.0 - abs(offset) / (strikes / 2 + 1))
            contracts.append(
                OptionContract(
                    instrument=instrument,
                    strike=strike,
                    option_type=option_type,
                    expiry=expiry,
                    quote=make_quote(
                        instrument,
                        price=round(price, 2),
                        spread=max(0.05, price * 0.004),
                        volume=200_000 * liquidity,
                        open_interest=500_000 * liquidity,
                        now=now,
                    ),
                    greeks=greeks,
                )
            )

    return OptionChain(
        underlying_symbol=underlying,
        expiry=expiry,
        underlying_ltp=spot,
        contracts=contracts,
        fetched_at=now,
    )


@pytest.fixture
def option_chain() -> OptionChain:
    return make_option_chain()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@pytest.fixture
def config(tmp_path) -> AppConfig:
    """A valid config with all paths redirected into the test's tmp dir."""
    cfg = load_config(
        "config/default.yaml",
        overrides={
            # The shipped config defaults to LIVE. Tests must never depend on the
            # LIVE interlocks being present in the environment, so the fixture
            # pins PAPER; the mode tests below load the file unmodified.
            "mode": "PAPER",
            "paths": {
                "data_dir": str(tmp_path),
                "cache_dir": str(tmp_path / "cache"),
                "journal_db": str(tmp_path / "journal.sqlite"),
                "model_dir": str(tmp_path / "models"),
                "experiment_dir": str(tmp_path / "experiments"),
                "log_dir": str(tmp_path / "logs"),
                "state_file": str(tmp_path / "state.json"),
            },
            "monitoring": {"dashboard_enabled": False, "alert_channels": ["log"]},
            "event_risk": {"enabled": False},
        },
        use_env=False,
        dotenv_path=None,
    )
    return cfg


@pytest.fixture
def journal(config, tmp_path):
    from aqtp.journal.store import TradeJournal

    store = TradeJournal(tmp_path / "journal.sqlite", mode="PAPER")
    yield store
    store.close()


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 6, 1, 11, 0, tzinfo=IST)
