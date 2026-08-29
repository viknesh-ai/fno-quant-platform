"""Broker construction — the one place that knows which adapter a mode gets.

Keeping this decision here (rather than scattered through the orchestrator) is what
makes the mode isolation of REQ 43 auditable: there is a single function to read to
know whether a given configuration can reach a real order endpoint.
"""

from __future__ import annotations

from typing import Callable

from ..configuration.loader import Credentials
from ..configuration.schema import AppConfig
from ..core.errors import ConfigurationError
from ..core.logging import get_logger
from ..core.types import Instrument, Quote, TradingMode
from .base import BrokerAdapter
from .groww import GrowwAdapter
from .simulated import SimulatedBroker, SimulationSettings

logger = get_logger(__name__)


def build_data_broker(config: AppConfig, credentials: Credentials) -> BrokerAdapter:
    """The adapter used for *market data*.

    PAPER mode deliberately uses the real broker here — paper trading must consume
    real market data (REQ 41). Only execution is simulated.
    """
    if config.broker.name == "simulated":
        raise ConfigurationError(
            "broker.name=simulated has no market data of its own; supply a data source"
        )
    credentials.validate_for(config.mode)
    return GrowwAdapter(
        access_token=credentials.access_token,
        api_key=credentials.api_key,
        api_secret=credentials.api_secret,
        totp_secret=credentials.totp_secret,
        auth_mode=credentials.auth_mode or config.broker.auth_mode,
        base_url=config.broker.base_url,
        instruments_url=config.broker.instruments_url,
        timeout=config.broker.timeout_seconds,
        max_retries=config.broker.max_retries,
        backoff=config.broker.retry_backoff_seconds,
        api_version=config.broker.api_version,
        cache_dir=config.paths.cache_dir,
    )


def build_execution_broker(
    config: AppConfig,
    credentials: Credentials,
    *,
    quote_source: Callable[[Instrument], Quote | None],
    data_broker: BrokerAdapter | None = None,
) -> BrokerAdapter:
    """The adapter that *executes*.

    LIVE and DEMO get the real adapter. PAPER and BACKTEST get the simulator, and
    there is no configuration path that changes that — the mode alone decides.
    """
    if config.mode in (TradingMode.PAPER, TradingMode.BACKTEST):
        logger.info("mode=%s: execution routed to SimulatedBroker (no real orders)", config.mode.value)
        return SimulatedBroker(
            quote_source=quote_source,
            starting_cash=config.capital.available_capital,
            settings=SimulationSettings(
                slippage_spread_fraction=config.costs.slippage_spread_fraction,
                slippage_fixed_pct=config.costs.slippage_fixed_pct,
                latency_ms=config.backtest.latency_ms,
                seed=config.models.random_seed,
            ),
        )

    if config.mode in (TradingMode.DEMO, TradingMode.LIVE):
        logger.warning(
            "mode=%s: execution routed to the REAL broker adapter (%s)",
            config.mode.value,
            config.broker.name,
        )
        if data_broker is not None and isinstance(data_broker, GrowwAdapter):
            return data_broker  # reuse the authenticated session
        return build_data_broker(config, credentials)

    raise ConfigurationError(f"unhandled trading mode {config.mode}")
