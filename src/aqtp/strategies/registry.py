"""Strategy registry (REQ 12 — strategies are pluggable).

Adding a strategy means writing a class and registering it. Nothing else in the
system needs to change: config keys, the ensemble, health tracking and the CLI all
read from this registry.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Type

from ..configuration.schema import StrategyConfig
from ..core.logging import get_logger
from .base import Strategy
from .breakout import BreakoutStrategy
from .expiry import ExpiryStrategy
from .market_structure import MarketStructureStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .options_flow import OptionsFlowStrategy
from .trend import TrendFollowingStrategy
from .volatility_expansion import VolatilityExpansionStrategy
from .vwap import VWAPStrategy

logger = get_logger(__name__)

_REGISTRY: dict[str, Type[Strategy]] = {}


def register(strategy_class: Type[Strategy]) -> Type[Strategy]:
    name = strategy_class.name
    if name in _REGISTRY and _REGISTRY[name] is not strategy_class:
        raise ValueError(f"strategy name {name!r} is already registered")
    _REGISTRY[name] = strategy_class
    return strategy_class


for _cls in (
    TrendFollowingStrategy,
    MomentumStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    VolatilityExpansionStrategy,
    MarketStructureStrategy,
    VWAPStrategy,
    OptionsFlowStrategy,
    ExpiryStrategy,
):
    register(_cls)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def get_strategy_class(name: str) -> Type[Strategy]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available_strategies()}")
    return _REGISTRY[name]


def build_strategies(config: Mapping[str, StrategyConfig]) -> list[Strategy]:
    """Instantiate every enabled strategy from configuration.

    An unknown strategy name is a hard error rather than a warning: silently
    running fewer strategies than the operator configured would change the
    system's behaviour without telling anyone.
    """
    if not config:
        # No explicit configuration means "run everything at default weight".
        return [cls() for cls in (_REGISTRY[name] for name in available_strategies())]

    unknown = set(config) - set(_REGISTRY)
    if unknown:
        raise KeyError(
            f"configuration references unknown strategies {sorted(unknown)}; "
            f"available: {available_strategies()}"
        )

    strategies: list[Strategy] = []
    for name in sorted(config):
        entry = config[name]
        if not entry.enabled:
            logger.info("strategy %s is disabled by configuration", name)
            continue
        strategies.append(_REGISTRY[name](params=entry.params, weight=entry.weight))
    if not strategies:
        raise ValueError("no strategies are enabled")
    return strategies
