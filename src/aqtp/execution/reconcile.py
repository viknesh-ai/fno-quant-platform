"""State reconciliation (REQ 46/63).

The governing rule from REQ 46: **the broker is the source of truth for actual
execution state.** When local state and broker state disagree, local state is
wrong and gets corrected — never the other way around.

Reconciliation runs at startup and periodically. The startup path additionally
resolves any order intents that were written to the journal but never confirmed,
which is how a mid-flight crash is prevented from turning into a duplicate order
(REQ 45/63).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..brokers.base import BrokerAdapter
from ..core.errors import BrokerError
from ..core.logging import get_logger
from ..core.types import Direction, Instrument, Order, OrderStatus, Position, Segment
from ..journal.store import TradeJournal
from .position_manager import ManagedPosition, PositionManager

logger = get_logger(__name__)


class DiscrepancyType(str, Enum):
    MISSING_POSITION = "missing_position"          # broker has it, we don't
    UNEXPECTED_POSITION = "unexpected_position"    # we have it, broker doesn't
    QUANTITY_MISMATCH = "quantity_mismatch"
    PRICE_MISMATCH = "price_mismatch"
    MISSING_ORDER = "missing_order"
    UNEXPECTED_ORDER = "unexpected_order"
    UNRESOLVED_INTENT = "unresolved_intent"


@dataclass
class Discrepancy:
    type: DiscrepancyType
    symbol: str
    detail: str
    local_value: object = None
    broker_value: object = None
    resolved: bool = False
    resolution: str = ""

    def describe(self) -> str:
        status = "RESOLVED" if self.resolved else "OPEN"
        return f"[{status}] {self.type.value} {self.symbol}: {self.detail}"


@dataclass
class ReconciliationReport:
    timestamp: datetime
    discrepancies: list[Discrepancy] = field(default_factory=list)
    broker_positions: int = 0
    local_positions: int = 0
    broker_open_orders: int = 0
    unmanaged_positions: list[str] = field(default_factory=list)
    orphan_orders: list[str] = field(default_factory=list)
    checked: bool = True
    error: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.discrepancies and not self.error

    @property
    def unresolved(self) -> list[Discrepancy]:
        return [d for d in self.discrepancies if not d.resolved]

    @property
    def safe_to_trade(self) -> bool:
        """REQ 63: new signals are only permitted once state is reconciled.

        Unresolved intents and unexpected positions both mean we may be holding
        risk we do not know about, so trading is blocked until an operator or a
        successful re-query clears them.
        """
        blocking = {
            DiscrepancyType.UNRESOLVED_INTENT,
            DiscrepancyType.UNEXPECTED_POSITION,
            DiscrepancyType.QUANTITY_MISMATCH,
        }
        return not self.error and not any(d.type in blocking for d in self.unresolved)

    def summary(self) -> str:
        if self.error:
            return f"reconciliation failed: {self.error}"
        if self.is_clean:
            return (
                f"state reconciled: {self.broker_positions} broker positions, "
                f"{self.broker_open_orders} open orders, no discrepancies"
            )
        return (
            f"{len(self.discrepancies)} discrepancies ({len(self.unresolved)} unresolved); "
            f"safe_to_trade={self.safe_to_trade}"
        )

    def lines(self) -> list[str]:
        return [self.summary()] + [f"  {d.describe()}" for d in self.discrepancies]


class StateReconciler:
    def __init__(
        self,
        broker: BrokerAdapter,
        journal: TradeJournal,
        position_manager: PositionManager,
    ) -> None:
        self.broker = broker
        self.journal = journal
        self.positions = position_manager

    # ------------------------------------------------------------------ #
    def reconcile(self, *, now: datetime, resolve_intents: bool = False) -> ReconciliationReport:
        """Compare local state against the broker and correct local state."""
        report = ReconciliationReport(timestamp=now)

        try:
            broker_positions = self.broker.get_positions()
            broker_orders = self.broker.list_orders()
        except BrokerError as exc:
            report.error = str(exc)
            report.checked = False
            logger.error("reconciliation could not reach the broker: %s", exc)
            return report

        report.broker_positions = len([p for p in broker_positions if p.quantity != 0])
        report.local_positions = self.positions.count
        open_orders = [o for o in broker_orders if o.status.is_open]
        report.broker_open_orders = len(open_orders)

        self._reconcile_positions(broker_positions, report)
        self._reconcile_orders(open_orders, report)

        if resolve_intents:
            self._resolve_pending_intents(report)

        if report.is_clean:
            logger.info(report.summary())
        else:
            logger.warning(report.summary())
            for discrepancy in report.discrepancies:
                logger.warning("  %s", discrepancy.describe())
            self.journal.record_event(
                level="WARNING",
                category="reconciliation",
                message=report.summary(),
                payload=[d.describe() for d in report.discrepancies],
            )
        return report

    # ------------------------------------------------------------------ #
    def _reconcile_positions(
        self, broker_positions: list[Position], report: ReconciliationReport
    ) -> None:
        broker_by_symbol = {
            p.instrument.trading_symbol: p for p in broker_positions if p.quantity != 0
        }
        local_by_symbol = {p.instrument.trading_symbol: p for p in self.positions.all()}

        # Broker has a position we are not managing.
        for symbol, position in broker_by_symbol.items():
            if symbol in local_by_symbol:
                continue
            report.unmanaged_positions.append(symbol)
            report.discrepancies.append(
                Discrepancy(
                    DiscrepancyType.MISSING_POSITION,
                    symbol,
                    f"broker holds {position.quantity} units at {position.average_price:.2f} "
                    "that this process is not managing — it may be from a previous run or a "
                    "manual trade",
                    local_value=0,
                    broker_value=position.quantity,
                )
            )

        # We think we hold something the broker does not.
        for symbol, managed in local_by_symbol.items():
            broker_position = broker_by_symbol.get(symbol)
            if broker_position is None:
                discrepancy = Discrepancy(
                    DiscrepancyType.UNEXPECTED_POSITION,
                    symbol,
                    f"locally tracking {managed.quantity} units but the broker reports none; "
                    "the position was likely closed outside this process",
                    local_value=managed.quantity,
                    broker_value=0,
                )
                # The broker is authoritative: drop the local position.
                self.positions.untrack(managed.position_id)
                discrepancy.resolved = True
                discrepancy.resolution = "local position removed (broker is authoritative)"
                report.discrepancies.append(discrepancy)
                continue

            signed_local = managed.quantity * managed.direction.sign
            if signed_local != broker_position.quantity:
                discrepancy = Discrepancy(
                    DiscrepancyType.QUANTITY_MISMATCH,
                    symbol,
                    f"local {signed_local} vs broker {broker_position.quantity}",
                    local_value=signed_local,
                    broker_value=broker_position.quantity,
                )
                if broker_position.quantity == 0:
                    self.positions.untrack(managed.position_id)
                    discrepancy.resolution = "local position removed"
                else:
                    # Adopt the broker's quantity and direction.
                    managed.quantity = abs(broker_position.quantity)
                    managed.direction = (
                        Direction.LONG if broker_position.quantity > 0 else Direction.SHORT
                    )
                    discrepancy.resolution = (
                        f"local quantity corrected to {broker_position.quantity}"
                    )
                discrepancy.resolved = True
                report.discrepancies.append(discrepancy)
                continue

            if abs(managed.entry_price - broker_position.average_price) > max(
                0.05, managed.entry_price * 0.01
            ):
                report.discrepancies.append(
                    Discrepancy(
                        DiscrepancyType.PRICE_MISMATCH,
                        symbol,
                        f"local entry {managed.entry_price:.2f} vs broker average "
                        f"{broker_position.average_price:.2f}",
                        local_value=managed.entry_price,
                        broker_value=broker_position.average_price,
                        resolved=True,
                        resolution="local entry price corrected to the broker average",
                    )
                )
                managed.entry_price = broker_position.average_price

    def _reconcile_orders(self, open_orders: list[Order], report: ReconciliationReport) -> None:
        """Open broker orders that no local position or intent explains."""
        known_ids = {
            row.get("broker_order_id")
            for row in self.journal.pending_order_intents()
            if row.get("broker_order_id")
        }
        for order in open_orders:
            if order.broker_order_id in known_ids:
                continue
            report.orphan_orders.append(order.broker_order_id)
            report.discrepancies.append(
                Discrepancy(
                    DiscrepancyType.UNEXPECTED_ORDER,
                    order.instrument.trading_symbol,
                    f"open order {order.broker_order_id} ({order.status.value}, "
                    f"{order.transaction_type.value} {order.quantity}) is not tracked locally",
                    broker_value=order.broker_order_id,
                )
            )

    def _resolve_pending_intents(self, report: ReconciliationReport) -> None:
        """Startup path: settle every order intent that was never confirmed.

        This is the mechanism that makes a crash between "record intent" and
        "receive acknowledgement" safe. Each unconfirmed intent is looked up by its
        client id; if the broker has it, local state adopts it, and if not, the
        intent is closed out as never-submitted. Neither path resubmits.
        """
        for row in self.journal.pending_order_intents():
            client_order_id = row.get("client_order_id")
            if not client_order_id:
                continue
            symbol = row.get("instrument", "")
            segment = Segment.FNO
            try:
                order = self.broker.get_order_by_reference(client_order_id, segment=segment)
            except (BrokerError, NotImplementedError) as exc:
                report.discrepancies.append(
                    Discrepancy(
                        DiscrepancyType.UNRESOLVED_INTENT,
                        symbol,
                        f"order intent {client_order_id} could not be resolved ({exc}); "
                        "manual confirmation required before trading this instrument",
                        local_value=client_order_id,
                    )
                )
                continue

            if order is None:
                self.journal.update_order(
                    client_order_id,
                    {"status": "NEVER_SUBMITTED",
                     "rejection_reason": "confirmed absent at reconciliation"},
                )
                logger.info(
                    "intent %s confirmed never reached the broker; closed out", client_order_id
                )
                continue

            self.journal.update_order(
                client_order_id,
                {
                    "broker_order_id": order.broker_order_id,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity,
                    "average_fill_price": order.average_fill_price,
                },
            )
            report.discrepancies.append(
                Discrepancy(
                    DiscrepancyType.UNRESOLVED_INTENT,
                    symbol,
                    f"order intent {client_order_id} was in fact submitted and is "
                    f"{order.status.value} with {order.filled_quantity} filled",
                    broker_value=order.broker_order_id,
                    resolved=True,
                    resolution="journal updated from broker state",
                )
            )
            logger.warning(
                "recovered order %s from intent %s: %s",
                order.broker_order_id, client_order_id, order.status.value,
            )

    # ------------------------------------------------------------------ #
    def adopt_unmanaged(
        self, report: ReconciliationReport, *, strategy: str = "adopted"
    ) -> list[str]:
        """Bring broker-held positions under management (REQ 63 step 4).

        Adopted positions get a conservative synthetic stop rather than being left
        unmanaged — an unmanaged position is unbounded risk, which is worse than an
        imperfectly-placed stop.
        """
        adopted: list[str] = []
        try:
            broker_positions = {
                p.instrument.trading_symbol: p for p in self.broker.get_positions()
            }
        except BrokerError as exc:
            logger.error("cannot adopt positions: %s", exc)
            return adopted

        for symbol in report.unmanaged_positions:
            position = broker_positions.get(symbol)
            if position is None or position.quantity == 0:
                continue
            direction = Direction.LONG if position.quantity > 0 else Direction.SHORT
            entry = position.average_price or position.last_price
            if entry <= 0:
                logger.warning("cannot adopt %s: no usable entry price", symbol)
                continue
            # 10% stop / 20% target: deliberately wide and clearly synthetic, so the
            # position is bounded without pretending to know the original thesis.
            stop = entry * (0.90 if direction is Direction.LONG else 1.10)
            target = entry * (1.20 if direction is Direction.LONG else 0.80)

            managed = ManagedPosition(
                position_id=f"ADOPTED-{symbol}",
                decision_id="",
                instrument=position.instrument,
                underlying=position.instrument.underlying_symbol or symbol,
                strategy=strategy,
                direction=direction,
                quantity=abs(position.quantity),
                entry_price=entry,
                entry_time=datetime.now(),
                stop_price=stop,
                target_price=target,
                initial_stop_price=stop,
                entry_regime=__import__(
                    "aqtp.core.types", fromlist=["Regime"]
                ).Regime.UNCERTAIN,
                expected_holding_minutes=0,
                last_price=position.last_price or entry,
            )
            self.positions.track(managed)
            adopted.append(symbol)
            logger.warning(
                "adopted unmanaged position %s (%d units @ %.2f) with a synthetic %.2f stop",
                symbol, position.quantity, entry, stop,
            )
            self.journal.record_event(
                level="WARNING",
                category="reconciliation",
                message=f"adopted unmanaged position {symbol}",
                payload={"quantity": position.quantity, "entry": entry, "stop": stop},
            )
        return adopted
