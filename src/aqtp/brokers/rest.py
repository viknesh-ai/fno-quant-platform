"""HTTP transport for REST brokers: rate limiting, bounded retries, idempotency.

Two invariants this module enforces, both from REQ 45:

1. **Reads may be retried; order-mutating writes may not.** A retry of a
   `POST /order/create` that already reached the exchange creates a second
   position. So `request()` retries only when the caller marks the call
   idempotent, and order submission passes `idempotent=False`.

2. **An ambiguous write is reported as ambiguous.** If a submit times out or
   returns something unparsable, we raise `OrderStateAmbiguous` — the caller must
   then resolve the truth by querying the broker.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import requests

from ..core.errors import (
    AuthenticationError,
    BrokerError,
    DuplicateOrderError,
    OrderRejected,
    OrderStateAmbiguous,
    RateLimitError,
    TransientBrokerError,
)
from ..core.logging import get_logger

logger = get_logger(__name__)


class RateCategory(str):
    """Groww applies limits per API *category*, not per endpoint."""

    AUTH = "auth"
    ORDERS = "orders"
    LIVE_DATA = "live_data"
    NON_TRADING = "non_trading"


@dataclass
class RateLimiter:
    """Token-bucket-ish sliding-window limiter, one window per category.

    Per-second and per-minute ceilings are both enforced because the documented
    limits specify both and the tighter one binds at different burst shapes.
    """

    per_second: float
    per_minute: float
    _events: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, *, now: Callable[[], float] = time.monotonic) -> float:
        """Block until a slot is free. Returns seconds actually waited."""
        waited = 0.0
        while True:
            with self._lock:
                current = now()
                while self._events and current - self._events[0] > 60.0:
                    self._events.popleft()
                in_last_second = sum(1 for t in self._events if current - t <= 1.0)
                in_last_minute = len(self._events)

                if in_last_second < self.per_second and in_last_minute < self.per_minute:
                    self._events.append(current)
                    return waited

                if in_last_second >= self.per_second:
                    oldest_in_second = next(t for t in self._events if current - t <= 1.0)
                    sleep_for = max(0.0, 1.0 - (current - oldest_in_second)) + 0.001
                else:
                    sleep_for = max(0.0, 60.0 - (current - self._events[0])) + 0.001

            time.sleep(min(sleep_for, 60.0))
            waited += sleep_for


@dataclass
class IdempotencyGuard:
    """Remembers client order ids we have already submitted in this process.

    This is the last line of defence against duplicate orders: even if calling code
    loops, the same key cannot be sent twice. The guard also survives a crash if a
    persistent store is attached (see `ExecutionEngine`, which records intents to
    the journal *before* submitting).
    """

    _seen: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    ttl_seconds: float = 86_400.0

    def reserve(self, client_order_id: str) -> None:
        if not client_order_id:
            raise ValueError("client_order_id is required for order submission")
        now = time.time()
        with self._lock:
            self._prune(now)
            if client_order_id in self._seen:
                raise DuplicateOrderError(
                    f"client_order_id {client_order_id!r} was already submitted; refusing to "
                    "resend. Query the broker for its state instead.",
                    client_order_id=client_order_id,
                )
            self._seen[client_order_id] = now

    def release(self, client_order_id: str) -> None:
        """Only called when we have *proven* the order never reached the broker."""
        with self._lock:
            self._seen.pop(client_order_id, None)

    def has(self, client_order_id: str) -> bool:
        with self._lock:
            self._prune(time.time())
            return client_order_id in self._seen

    def _prune(self, now: float) -> None:
        expired = [k for k, t in self._seen.items() if now - t > self.ttl_seconds]
        for key in expired:
            del self._seen[key]


class RestClient:
    """Thin, well-behaved HTTP client for a JSON REST broker API."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff: float = 0.5,
        limiters: Mapping[str, RateLimiter] | None = None,
        session: requests.Session | None = None,
        api_version: str = "1.0",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.api_version = api_version
        self.session = session or requests.Session()
        self._token: str = ""
        self.limiters: dict[str, RateLimiter] = dict(limiters or {})
        self.last_latency_ms: float = 0.0

    def set_token(self, token: str) -> None:
        self._token = token

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "X-API-VERSION": self.api_version,
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        category: str = RateCategory.NON_TRADING,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        idempotent: bool = True,
        expect_payload: bool = True,
    ) -> Any:
        """Perform a request and return the unwrapped ``payload``.

        `idempotent=False` disables all retries — used for order mutations.
        """
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        attempts = self.max_retries if idempotent else 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            limiter = self.limiters.get(category)
            if limiter is not None:
                limiter.acquire()

            started = time.perf_counter()
            try:
                response = self.session.request(
                    method.upper(),
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(headers),
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                self.last_latency_ms = (time.perf_counter() - started) * 1000
                if not idempotent:
                    raise OrderStateAmbiguous(
                        f"timeout on non-idempotent {method} {path}; order state is unknown "
                        "and must be resolved by querying the broker"
                    ) from exc
                last_error = TransientBrokerError(f"timeout calling {method} {path}")
            except requests.RequestException as exc:
                self.last_latency_ms = (time.perf_counter() - started) * 1000
                if not idempotent:
                    raise OrderStateAmbiguous(
                        f"network failure on non-idempotent {method} {path}: {exc}"
                    ) from exc
                last_error = TransientBrokerError(f"network error calling {method} {path}: {exc}")
            else:
                self.last_latency_ms = (time.perf_counter() - started) * 1000
                try:
                    return self._unwrap(response, expect_payload=expect_payload)
                except RateLimitError as exc:
                    if not idempotent or attempt == attempts - 1:
                        raise
                    last_error = exc
                    time.sleep(exc.retry_after)
                    continue
                except TransientBrokerError as exc:
                    if not idempotent:
                        raise OrderStateAmbiguous(
                            f"server error on non-idempotent {method} {path}: {exc}"
                        ) from exc
                    last_error = exc

            if attempt < attempts - 1:
                # Full jitter: avoids synchronized retry storms across instruments.
                delay = min(self.backoff * (2 ** attempt), 8.0)
                time.sleep(random.uniform(0, delay))

        raise last_error or BrokerError(f"{method} {path} failed after {attempts} attempts")

    def _unwrap(self, response: requests.Response, *, expect_payload: bool = True) -> Any:
        status = response.status_code

        if status == 429:
            retry_after = float(response.headers.get("Retry-After", "1") or 1)
            raise RateLimitError("rate limit exceeded", retry_after=retry_after)
        if status in (401, 403):
            raise AuthenticationError(
                f"authentication failed ({status}); token may have expired "
                "(Groww access tokens expire daily at 06:00 IST)"
            )
        if status >= 500:
            raise TransientBrokerError(f"broker server error {status}")

        try:
            body = response.json()
        except ValueError as exc:
            raise BrokerError(f"non-JSON response ({status}): {response.text[:200]}") from exc

        if not isinstance(body, dict):
            raise BrokerError(f"unexpected response shape: {type(body).__name__}")

        if body.get("status") == "FAILURE" or status >= 400:
            error = body.get("error") or {}
            code = str(error.get("code", "")) or str(status)
            message = str(error.get("message", "")) or f"HTTP {status}"
            if status == 400:
                raise OrderRejected(message, code=code)
            raise BrokerError(message, code=code, retryable=False)

        if not expect_payload:
            return body
        if "payload" not in body:
            raise BrokerError(f"response missing 'payload': {str(body)[:200]}")
        return body["payload"]

    def close(self) -> None:
        self.session.close()
