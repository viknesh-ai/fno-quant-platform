"""Kill switches (REQ 44).

Six independent scopes, as the requirement lists: application, strategy, instrument,
portfolio, broker and a global emergency shutdown. They are independent so an
operator can stop one misbehaving strategy without halting the whole system.

State is **persisted**. A kill switch that evaporates on restart is not a kill
switch — the most likely moment to restart is right after something went wrong.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..core.logging import get_logger

logger = get_logger(__name__)


class KillScope(str, Enum):
    APPLICATION = "application"
    STRATEGY = "strategy"
    INSTRUMENT = "instrument"
    PORTFOLIO = "portfolio"
    BROKER = "broker"
    GLOBAL = "global"


class EmergencyPolicy(str, Enum):
    """What an emergency shutdown does to open positions (REQ 44)."""

    HOLD_POSITIONS = "hold_positions"       # stop trading, leave positions alone
    CLOSE_POSITIONS = "close_positions"     # flatten everything at market
    CANCEL_ORDERS_ONLY = "cancel_orders_only"


@dataclass
class KillRecord:
    scope: str
    target: str = ""
    reason: str = ""
    engaged_at: str = ""
    engaged_by: str = "system"

    def describe(self) -> str:
        target = f" [{self.target}]" if self.target else ""
        return f"{self.scope}{target}: {self.reason} (since {self.engaged_at[:19]})"


class KillSwitchManager:
    def __init__(self, state_file: str | Path | None = None) -> None:
        self.state_file = Path(state_file) if state_file else None
        self._switches: dict[str, KillRecord] = {}
        self.emergency_policy = EmergencyPolicy.CANCEL_ORDERS_ONLY
        self._load()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(scope: KillScope, target: str = "") -> str:
        return f"{scope.value}:{target}" if target else scope.value

    def engage(
        self,
        scope: KillScope,
        *,
        target: str = "",
        reason: str = "",
        engaged_by: str = "operator",
    ) -> KillRecord:
        key = self._key(scope, target)
        record = KillRecord(
            scope=scope.value,
            target=target,
            reason=reason or "no reason given",
            engaged_at=datetime.now().isoformat(),
            engaged_by=engaged_by,
        )
        self._switches[key] = record
        self._save()
        logger.error("KILL SWITCH ENGAGED — %s", record.describe())
        return record

    def release(self, scope: KillScope, *, target: str = "") -> bool:
        key = self._key(scope, target)
        if key not in self._switches:
            return False
        del self._switches[key]
        self._save()
        logger.warning("kill switch released: %s", key)
        return True

    def release_all(self) -> int:
        count = len(self._switches)
        self._switches.clear()
        self._save()
        logger.warning("all %d kill switches released", count)
        return count

    # ------------------------------------------------------------------ #
    def is_engaged(self, scope: KillScope, *, target: str = "") -> bool:
        if KillScope.GLOBAL.value in self._switches:
            return True
        if self._key(scope) in self._switches:
            return True
        return self._key(scope, target) in self._switches

    def check(self, scope: KillScope, *, target: str = "") -> tuple[bool, str]:
        """Returns (allowed, reason). Global always wins."""
        global_switch = self._switches.get(KillScope.GLOBAL.value)
        if global_switch:
            return False, f"GLOBAL kill switch: {global_switch.reason}"
        scope_switch = self._switches.get(self._key(scope))
        if scope_switch:
            return False, f"{scope.value} kill switch: {scope_switch.reason}"
        targeted = self._switches.get(self._key(scope, target))
        if targeted:
            return False, f"{scope.value} kill switch on {target}: {targeted.reason}"
        return True, ""

    def can_trade(self, *, strategy: str = "", instrument: str = "", underlying: str = "") -> tuple[bool, str]:
        """Composite check the RiskEngine calls before approving anything."""
        for scope, target in (
            (KillScope.GLOBAL, ""),
            (KillScope.APPLICATION, ""),
            (KillScope.BROKER, ""),
            (KillScope.PORTFOLIO, ""),
            (KillScope.STRATEGY, strategy),
            (KillScope.INSTRUMENT, instrument),
            (KillScope.INSTRUMENT, underlying),
        ):
            if scope in (KillScope.STRATEGY, KillScope.INSTRUMENT) and not target:
                continue
            allowed, reason = self.check(scope, target=target)
            if not allowed:
                return False, reason
        return True, ""

    def emergency_shutdown(
        self, reason: str, *, policy: EmergencyPolicy | None = None, engaged_by: str = "system"
    ) -> KillRecord:
        """Global stop. The policy decides what happens to open positions."""
        if policy is not None:
            self.emergency_policy = policy
        record = self.engage(
            KillScope.GLOBAL, reason=reason, engaged_by=engaged_by
        )
        logger.critical(
            "EMERGENCY SHUTDOWN: %s — policy: %s", reason, self.emergency_policy.value
        )
        return record

    @property
    def is_shutdown(self) -> bool:
        return KillScope.GLOBAL.value in self._switches

    def active(self) -> list[KillRecord]:
        return sorted(self._switches.values(), key=lambda r: r.engaged_at)

    def summary(self) -> list[dict]:
        return [asdict(r) for r in self.active()]

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.state_file is None or not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text())
        except json.JSONDecodeError as exc:
            logger.error("kill switch state file is corrupt: %s", exc)
            return
        self._switches = {
            key: KillRecord(**value) for key, value in raw.get("switches", {}).items()
        }
        policy = raw.get("emergency_policy")
        if policy:
            try:
                self.emergency_policy = EmergencyPolicy(policy)
            except ValueError:
                pass
        if self._switches:
            logger.warning(
                "restored %d active kill switch(es) from disk: %s",
                len(self._switches),
                [r.describe() for r in self.active()],
            )

    def _save(self) -> None:
        if self.state_file is None:
            return
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "switches": {key: asdict(record) for key, record in self._switches.items()},
            "emergency_policy": self.emergency_policy.value,
            "updated_at": datetime.now().isoformat(),
        }
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.state_file)
