"""Configuration loading: YAML files + environment overrides + credentials.

Precedence (lowest to highest): schema defaults -> base YAML -> overlay YAML ->
`AQTP_*` environment overrides -> explicit CLI overrides.

Credentials never appear in YAML. They are read from the process environment (or a
`.env` file, which is git-ignored) into a separate `Credentials` object that is not
part of `AppConfig` and is never serialized (REQ 62).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..core.errors import ConfigurationError
from ..core.types import TradingMode
from .schema import AppConfig

_ENV_PREFIX = "AQTP_"
# Env keys that are interlocks/credentials, not config overrides.
_ENV_RESERVED = {
    "AQTP_LIVE_CONFIRM_1",
    "AQTP_LIVE_CONFIRM_2",
    "AQTP_LIVE_CONFIRM_3",
    "AQTP_ALERT_WEBHOOK_URL",
}


@dataclass(frozen=True)
class Credentials:
    """Secret material. Deliberately not a pydantic model on AppConfig so that it
    cannot be accidentally dumped by `config.model_dump()`."""

    access_token: str = ""
    api_key: str = ""
    api_secret: str = ""
    totp_secret: str = ""
    auth_mode: str = "token"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Credentials":
        env = env if env is not None else os.environ
        return cls(
            access_token=env.get("GROWW_ACCESS_TOKEN", "").strip(),
            api_key=env.get("GROWW_API_KEY", "").strip(),
            api_secret=env.get("GROWW_API_SECRET", "").strip(),
            totp_secret=env.get("GROWW_TOTP_SECRET", "").strip(),
            auth_mode=env.get("GROWW_AUTH_MODE", "token").strip().lower(),
        )

    def validate_for(self, mode: TradingMode) -> None:
        """Only modes that talk to the broker need credentials."""
        if mode in (TradingMode.BACKTEST,):
            return
        if self.auth_mode == "token":
            if not self.access_token:
                raise ConfigurationError(
                    "GROWW_ACCESS_TOKEN is not set. Export it in your environment or .env file."
                )
        elif self.auth_mode == "approval":
            if not (self.api_key and self.api_secret):
                raise ConfigurationError(
                    "auth_mode=approval requires both GROWW_API_KEY and GROWW_API_SECRET."
                )
        elif self.auth_mode == "totp":
            if not (self.api_key and self.totp_secret):
                raise ConfigurationError(
                    "auth_mode=totp requires both GROWW_API_KEY and GROWW_TOTP_SECRET."
                )
        else:
            raise ConfigurationError(
                f"unknown GROWW_AUTH_MODE={self.auth_mode!r}; expected token|approval|totp"
            )

    def __repr__(self) -> str:  # never leak secrets through a traceback
        def mask(v: str) -> str:
            return "<set>" if v else "<unset>"

        return (
            f"Credentials(auth_mode={self.auth_mode!r}, access_token={mask(self.access_token)}, "
            f"api_key={mask(self.api_key)}, api_secret={mask(self.api_secret)}, "
            f"totp_secret={mask(self.totp_secret)})"
        )


def load_dotenv(path: Path | str = ".env") -> None:
    """Minimal .env reader — avoids a dependency and never overwrites real env vars."""
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def env_overrides(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Translate ``AQTP_RISK__MAX_OPEN_POSITIONS=3`` into ``{"risk": {"max_open_positions": 3}}``.

    Double underscore separates nesting levels; single underscores are part of the
    field name, which is why field names with underscores round-trip correctly.
    """
    env = env if env is not None else os.environ
    result: dict[str, Any] = {}
    for key, value in env.items():
        if not key.startswith(_ENV_PREFIX) or key in _ENV_RESERVED:
            continue
        path = key[len(_ENV_PREFIX):].lower().split("__")
        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce_scalar(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"config file {path} must contain a YAML mapping at the top level")
    return data


def load_config(
    config_path: str | Path = "config/default.yaml",
    *,
    overlay_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    use_env: bool = True,
    dotenv_path: str | Path | None = ".env",
) -> AppConfig:
    """Build and fully validate an AppConfig. Raises ConfigurationError on any problem."""
    if dotenv_path is not None:
        load_dotenv(dotenv_path)

    data = _read_yaml(Path(config_path))
    if overlay_path is not None:
        data = _deep_merge(data, _read_yaml(Path(overlay_path)))
    if use_env:
        data = _deep_merge(data, env_overrides())
    if overrides:
        data = _deep_merge(data, overrides)

    try:
        config = AppConfig(**data)
    except Exception as exc:  # pydantic ValidationError and friends
        raise ConfigurationError(f"invalid configuration: {exc}") from exc

    _post_validate(config)
    return config


def _post_validate(config: AppConfig) -> None:
    """Cross-cutting checks that need more than one section to be visible."""
    problems: list[str] = []

    # A single trade must not be able to breach the daily loss limit.
    if config.risk.risk_per_trade_pct * config.risk.max_open_positions > config.risk.max_daily_loss_pct * 3:
        problems.append(
            f"risk_per_trade_pct ({config.risk.risk_per_trade_pct:.3f}) x max_open_positions "
            f"({config.risk.max_open_positions}) is far above max_daily_loss_pct "
            f"({config.risk.max_daily_loss_pct:.3f}); tighten one of them"
        )

    # Position sizing cannot exceed capital deployment.
    if config.risk.max_position_size_pct > config.capital.max_capital_deployment_pct:
        problems.append(
            "risk.max_position_size_pct cannot exceed capital.max_capital_deployment_pct"
        )

    # Option selection must not demand a tighter spread than risk allows and vice versa.
    if config.option_selection.max_spread_pct > config.risk.max_spread_pct:
        problems.append(
            "option_selection.max_spread_pct is looser than risk.max_spread_pct; the risk "
            "engine would reject every contract the selector proposes"
        )

    # At least one strategy must be enabled.
    if config.strategies and not any(s.enabled for s in config.strategies.values()):
        problems.append("no strategy is enabled")

    # Prediction horizons must be expressible on the entry timeframe.
    from ..core.types import Timeframe

    entry_minutes = Timeframe(config.timeframes.entry_timeframe).minutes
    for horizon in config.prediction.horizons_minutes:
        if horizon < entry_minutes:
            problems.append(
                f"prediction horizon {horizon}m is shorter than the entry timeframe "
                f"({config.timeframes.entry_timeframe}); the label could not be observed"
            )

    if config.event_risk.enabled and config.event_risk.source == "file":
        if not Path(config.event_risk.calendar_path).exists():
            # Not fatal: an absent calendar means "no known events", which the
            # EventRisk provider handles. Surface it so it isn't a silent gap.
            pass

    if problems:
        raise ConfigurationError(
            "configuration validation failed:\n  - " + "\n  - ".join(problems)
        )


def describe_config(config: AppConfig) -> dict[str, Any]:
    """Redaction-safe view for the `config` CLI command and the dashboard."""
    return config.model_dump(mode="json")
