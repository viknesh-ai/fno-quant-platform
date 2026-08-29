"""Deterministic identifiers and idempotency keys.

An order's client id is derived from the *intent* (instrument, side, quantity,
strategy, decision bucket), not from a random UUID. Two attempts to place the same
logical order therefore collide on the same key and the second one is refused
instead of duplicating the position (REQ 45).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime

# Groww requires order_reference_id to be 8-20 alphanumeric chars with at most 2
# hyphens. Keys are generated to satisfy that constraint at the source rather than
# being repaired inside the adapter.
_REFERENCE_RE = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+){0,2}$")
MIN_REFERENCE_LEN = 8
MAX_REFERENCE_LEN = 20


def is_valid_reference_id(value: str) -> bool:
    return (
        MIN_REFERENCE_LEN <= len(value) <= MAX_REFERENCE_LEN
        and bool(_REFERENCE_RE.match(value))
    )


def make_client_order_id(
    *,
    trading_symbol: str,
    transaction_type: str,
    quantity: int,
    strategy: str,
    bucket: str,
    prefix: str = "AQ",
) -> str:
    """Build a broker-safe idempotency key from the order's intent.

    `bucket` is the de-duplication window — typically a truncated timestamp such as
    ``2026-08-29T10:35`` — so that a retry inside the same decision cycle reuses the
    key while a genuinely new decision one cycle later gets a fresh one.
    """
    seed = "|".join([trading_symbol, transaction_type, str(quantity), strategy, bucket])
    digest = hashlib.sha256(seed.encode()).hexdigest()[:14].upper()
    ref = f"{prefix}{digest}"
    assert is_valid_reference_id(ref), ref
    return ref


def decision_bucket(moment: datetime, *, seconds: int = 60) -> str:
    """Truncate a timestamp to a de-duplication window."""
    epoch = int(moment.timestamp())
    return str(epoch - (epoch % max(1, seconds)))


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
