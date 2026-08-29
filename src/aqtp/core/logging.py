"""Structured logging with credential redaction.

REQ 62 forbids logging credentials. Rather than trusting every call site to be
careful, a filter scrubs known secret-bearing patterns out of every record.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_SECRET_KEYS = (
    "token",
    "access_token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "authorization",
    "checksum",
    "totp",
)

# The optional quote after the key name matters: broker request/response bodies are
# JSON, so the secret-bearing form is `"access_token": "abc"` rather than
# `access_token=abc`. Missing that shape would log real tokens in clear text.
# The `[A-Za-z0-9_]*` prefix matters because `_` is a word character, so a bare
# `\bsecret\b` would never match inside a compound key like `api_secret`.
_SECRET_PATTERNS = [
    re.compile(rf"(?i)\b([A-Za-z0-9_]*{k})\b[\"']?\s*[:=]\s*[\"']?([^\s,\"'}}\]]+)", re.UNICODE)
    for k in _SECRET_KEYS
]
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")


def redact(text: str) -> str:
    text = _BEARER.sub("Bearer <redacted>", text)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    return text


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if any(s in key.lower() for s in _SECRET_KEYS):
            out[key] = "<redacted>"
        elif isinstance(value, dict):
            out[key] = redact_mapping(value)
        elif isinstance(value, str):
            out[key] = redact(value)
        else:
            out[key] = value
    return out


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact_mapping(record.args)
            else:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the research tooling reads these back (REQ 50)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: Path | None = None,
    json_format: bool = False,
) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler: logging.Handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)-28s %(message)s", "%H:%M:%S")
        )
    handler.addFilter(RedactionFilter())
    root.addHandler(handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "aqtp.log")
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(RedactionFilter())
        root.addHandler(file_handler)

    # Third-party chatter drowns the decision trail.
    for noisy in ("urllib3", "matplotlib", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a message with structured fields attached (picked up by JsonFormatter)."""
    logger.log(level, message, extra={"extra_fields": redact_mapping(fields)})


def scrub_env_for_display() -> dict[str, str]:
    """Environment snapshot safe to print in a diagnostics command."""
    return {
        k: ("<set>" if v else "<empty>") if any(s in k.lower() for s in _SECRET_KEYS) else v
        for k, v in os.environ.items()
        if k.startswith(("AQTP_", "GROWW_"))
    }
