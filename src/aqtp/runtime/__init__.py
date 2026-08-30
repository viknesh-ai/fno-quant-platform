"""Production process supervision.

The orchestrator knows how to make a trading decision. This package knows how to
keep a process that makes those decisions alive for six and a half hours without
supervision — reconnecting a dropped broker session, tripping a watchdog when a
cycle wedges, squaring off before the close, and shutting down cleanly when
systemd sends SIGTERM rather than leaving an unmanaged position at the broker.
"""

from .supervisor import RuntimeSupervisor, SupervisorState

__all__ = ["RuntimeSupervisor", "SupervisorState"]
