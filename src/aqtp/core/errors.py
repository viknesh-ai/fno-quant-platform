"""Exception hierarchy.

The distinction that matters operationally is *retryable* vs *not*: the REST layer
may retry a `TransientBrokerError`, but an `OrderRejected` or `DuplicateOrderError`
must never be retried blindly (REQ 45).
"""

from __future__ import annotations


class AQTPError(Exception):
    """Base for everything this platform raises."""


class ConfigurationError(AQTPError):
    """Configuration failed validation — the bot must refuse to start (REQ 58)."""


class BrokerError(AQTPError):
    def __init__(self, message: str, *, code: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class TransientBrokerError(BrokerError):
    """Timeout / 5xx / network blip — safe to retry a *read*."""

    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message, code=code, retryable=True)


class RateLimitError(TransientBrokerError):
    def __init__(self, message: str, *, retry_after: float = 1.0, code: str = "") -> None:
        super().__init__(message, code=code)
        self.retry_after = retry_after


class AuthenticationError(BrokerError):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message, code=code, retryable=False)


class OrderRejected(BrokerError):
    def __init__(self, message: str, *, code: str = "") -> None:
        super().__init__(message, code=code, retryable=False)


class DuplicateOrderError(BrokerError):
    """Raised locally when an idempotency key is reused (REQ 45)."""

    def __init__(self, message: str, *, client_order_id: str = "") -> None:
        super().__init__(message, retryable=False)
        self.client_order_id = client_order_id


class OrderStateAmbiguous(BrokerError):
    """The submit call failed in a way that leaves order existence unknown.

    The caller MUST resolve this by querying the broker, never by resubmitting.
    """

    def __init__(self, message: str, *, client_order_id: str = "") -> None:
        super().__init__(message, retryable=False)
        self.client_order_id = client_order_id


class DataQualityError(AQTPError):
    """Market data failed validation; downstream must treat this as NO_TRADE."""


class RiskViolation(AQTPError):
    """A risk control rejected a trade. Carries the failing control name."""

    def __init__(self, control: str, message: str) -> None:
        super().__init__(f"{control}: {message}")
        self.control = control


class KillSwitchActive(AQTPError):
    pass


class ModelError(AQTPError):
    pass


class LeakageError(AQTPError):
    """Raised by the dataset builder when a temporal-ordering invariant breaks."""
