"""Operator-initiated trading.

Everything the automated pipeline does is also available by hand, and — this is
the part that matters — by hand means *through the same risk engine*. A manual
order is still sized by the sizer, still checked against the drawdown ladder and
the exposure caps, still journalled with a decision id, and still tracked by the
position manager so the stop and target are managed like any other position.

`--force` exists because sometimes the operator genuinely knows better than the
model (an exit into an event, a hedge the portfolio layer has no concept of). It
bypasses the *sizing* and *scoring* checks. It does not bypass kill switches, and
it is journalled as a forced order so it is visible in any review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..core.errors import AQTPError, BrokerError
from ..core.logging import get_logger
from ..core.types import (
    Decision,
    Direction,
    ExitReason,
    Instrument,
    InstrumentType,
    ManagementAction,
    OptionType,
    Quote,
    Regime,
    TransactionType,
)
from ..execution.position_manager import ManagedPosition, ManagementDecision
from ..risk.engine import RiskApproval

logger = get_logger(__name__)


class ManualTradeError(AQTPError):
    """A manual trade could not be placed. Carries the reason verbatim."""


@dataclass
class ResolvedContract:
    """The instrument a symbol specification resolved to, with its quote."""

    instrument: Instrument
    quote: Quote | None = None
    underlying: str = ""
    candidates_considered: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def last_price(self) -> float:
        return float(self.quote.last_price) if self.quote else 0.0


@dataclass
class ManualTradeResult:
    """Outcome of one manual order, including why it was refused."""

    success: bool
    action: str
    instrument: Instrument | None = None
    quantity: int = 0
    price: float = 0.0
    decision_id: str = ""
    position_id: str = ""
    approval: RiskApproval | None = None
    reason: str = ""
    steps: list[str] = field(default_factory=list)

    def explain(self) -> str:
        if self.success:
            symbol = self.instrument.trading_symbol if self.instrument else "—"
            return (
                f"{self.action} {symbol} x{self.quantity} @ {self.price:,.2f} "
                f"[{self.decision_id or self.position_id}]"
            )
        return f"{self.action} refused: {self.reason}"


class ManualTrader:
    """Operator-facing trading actions, wired to the live orchestrator."""

    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator
        self.config = orchestrator.config
        self._counter = 0

    # ================================================================== #
    # Symbol resolution
    # ================================================================== #
    def resolve(
        self,
        symbol: str,
        *,
        option_type: str | None = None,
        strike: float | None = None,
        expiry: date | None = None,
        instrument_kind: str = "auto",
    ) -> ResolvedContract:
        """Resolve a human symbol specification to a tradable instrument.

        Accepts either an exact trading symbol (`NIFTY26JUN24000CE`) or an
        underlying plus a description (`NIFTY --option CE --strike 24000`). An
        omitted strike means ATM, an omitted expiry means the nearest tradable
        one — the two defaults an operator means nine times out of ten.
        """
        orchestrator = self.orchestrator
        if orchestrator.universe is None:
            raise ManualTradeError("the instrument universe is not loaded; run `aqtp doctor` first")

        symbol = symbol.strip().upper()

        # 1. Exact trading-symbol match — but only when the caller gave no
        #    qualifiers. "NIFTY" is itself a listed index, so `NIFTY --option CE`
        #    must describe an option on NIFTY, not resolve to the index and
        #    silently drop the rest of the specification.
        qualified = (
            option_type is not None
            or strike is not None
            or expiry is not None
            or instrument_kind != "auto"
        )
        if not qualified:
            exact = self._find_exact(symbol)
            if exact is not None:
                return ResolvedContract(
                    instrument=exact,
                    quote=self._quote(exact),
                    underlying=exact.underlying_symbol or symbol,
                    notes=[f"matched the exact trading symbol {exact.trading_symbol}"],
                )

        group = orchestrator.universe.get(symbol)
        if group is None:
            # A qualified spec whose underlying is unknown, or a symbol that is
            # neither. Fall back to an exact match before giving up.
            exact = self._find_exact(symbol)
            if exact is not None and not qualified:
                return ResolvedContract(
                    instrument=exact,
                    quote=self._quote(exact),
                    underlying=exact.underlying_symbol or symbol,
                    notes=[f"matched the exact trading symbol {exact.trading_symbol}"],
                )
            raise ManualTradeError(
                f"{symbol!r} is neither a tradable symbol nor a known underlying. "
                f"Run `aqtp universe --filter {symbol}` to see what is available."
            )

        today = orchestrator.clock.now().date()

        # 2. Futures — the default for an unqualified underlying, since an index
        #    spot is not itself tradable.
        if instrument_kind == "future" or (instrument_kind == "auto" and option_type is None
                                           and strike is None):
            future = group.nearest_future(today) if expiry is None else next(
                (f for f in group.futures if f.expiry_date == expiry), None
            )
            if future is None:
                raise ManualTradeError(f"no tradable future found for {symbol}")
            return ResolvedContract(
                instrument=future,
                quote=self._quote(future),
                underlying=symbol,
                notes=[f"nearest future {future.trading_symbol}"],
            )

        # 3. Options.
        if option_type is None:
            raise ManualTradeError(
                "specify --option CE or --option PE (or omit both --option and --strike "
                "to trade the future)"
            )
        try:
            wanted_type = OptionType(option_type.strip().upper())
        except ValueError:
            raise ManualTradeError(f"--option must be CE or PE, got {option_type!r}")

        expiries = orchestrator.discovery.tradable_expiries(symbol, as_of=today)
        if not expiries:
            raise ManualTradeError(f"no tradable expiries for {symbol}")
        target_expiry = expiry or expiries[0]
        if target_expiry not in expiries:
            raise ManualTradeError(
                f"{target_expiry} is not a tradable expiry for {symbol}; "
                f"available: {', '.join(e.isoformat() for e in expiries[:5])}"
            )

        candidates = [
            option for option in group.options
            if option.expiry_date == target_expiry
            and option.instrument_type.value == wanted_type.value
        ]
        if not candidates:
            raise ManualTradeError(
                f"no {wanted_type.value} contracts for {symbol} expiring {target_expiry}"
            )

        if strike is None:
            spot = self._spot_price(symbol, group)
            if spot <= 0:
                raise ManualTradeError(
                    f"cannot determine the spot price for {symbol}; pass --strike explicitly"
                )
            chosen = min(candidates, key=lambda c: abs((c.strike_price or 0.0) - spot))
            note = f"ATM strike {chosen.strike_price:,.0f} chosen against spot {spot:,.2f}"
        else:
            chosen = min(candidates, key=lambda c: abs((c.strike_price or 0.0) - strike))
            if abs((chosen.strike_price or 0.0) - strike) > 1e-6:
                note = (
                    f"strike {strike:,.0f} is not listed; using the nearest available "
                    f"{chosen.strike_price:,.0f}"
                )
            else:
                note = f"strike {chosen.strike_price:,.0f}"

        return ResolvedContract(
            instrument=chosen,
            quote=self._quote(chosen),
            underlying=symbol,
            candidates_considered=len(candidates),
            notes=[note, f"expiry {target_expiry.isoformat()}"],
        )

    def _find_exact(self, symbol: str) -> Instrument | None:
        """Exact trading-symbol match, restricted to instruments that can be traded.

        An index spot carries the same trading symbol as its underlying — "NIFTY"
        is both the index and the name of the F&O group — but it cannot be
        bought. Matching it would turn `aqtp trade buy NIFTY` into an order for
        an instrument no exchange will accept, so spots are skipped here and the
        caller falls through to the nearest future.
        """
        universe = self.orchestrator.universe
        if universe is None:
            return None
        for instrument in universe.all_instruments():
            if instrument.trading_symbol.upper() != symbol:
                continue
            if instrument.instrument_type in (InstrumentType.IDX, InstrumentType.EQ):
                continue
            return instrument
        return None

    def _spot_price(self, underlying: str, group) -> float:
        quote_source = group.spot or group.nearest_future(self.orchestrator.clock.now().date())
        if quote_source is None:
            return 0.0
        quote = self._quote(quote_source)
        return float(quote.last_price) if quote else 0.0

    def _quote(self, instrument: Instrument) -> Quote | None:
        try:
            return self.orchestrator.data_broker.get_quote(instrument)
        except BrokerError as exc:
            logger.warning("could not quote %s: %s", instrument.trading_symbol, exc)
            return None

    # ================================================================== #
    # Entry
    # ================================================================== #
    def buy(self, contract: ResolvedContract, **kwargs) -> ManualTradeResult:
        return self._enter(contract, TransactionType.BUY, **kwargs)

    def sell(self, contract: ResolvedContract, **kwargs) -> ManualTradeResult:
        return self._enter(contract, TransactionType.SELL, **kwargs)

    def _enter(
        self,
        contract: ResolvedContract,
        transaction_type: TransactionType,
        *,
        lots: int | None = None,
        quantity: int | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        stop_pct: float = 0.30,
        target_pct: float = 0.60,
        limit_price: float | None = None,
        force: bool = False,
        strategy: str = "manual",
        track: bool = True,
    ) -> ManualTradeResult:
        """Place a manual entry through the full risk pipeline."""
        orchestrator = self.orchestrator
        instrument = contract.instrument
        quote = contract.quote
        now = orchestrator.clock.now()
        action = f"{transaction_type.value}"

        if quote is None or quote.last_price <= 0:
            return ManualTradeResult(
                False, action, instrument=instrument,
                reason="no usable quote — refusing to trade an instrument we cannot price",
            )

        entry_price = float(limit_price or quote.last_price)
        lot_size = max(1, instrument.lot_size)
        direction = Direction.LONG if transaction_type is TransactionType.BUY else Direction.SHORT

        # Stops and targets default to premium percentages, which is how an
        # option position is actually managed when the operator names no level.
        if stop_price is None:
            stop_price = max(0.05, entry_price * (1.0 - stop_pct)) if direction is Direction.LONG \
                else entry_price * (1.0 + stop_pct)
        if target_price is None:
            target_price = entry_price * (1.0 + target_pct) if direction is Direction.LONG \
                else max(0.05, entry_price * (1.0 - target_pct))

        self._counter += 1
        decision_id = f"M{orchestrator.run_id}-{self._counter:04d}"

        portfolio_state = orchestrator._portfolio_state(now)
        margin = orchestrator._margin()
        underlying = contract.underlying or instrument.underlying_symbol or instrument.trading_symbol

        # The risk engine requires an expected value, and rightly refuses a trade
        # whose EV was never computed. A manual trade has no model probability,
        # so the analysis conviction stands in where one exists and a coin flip
        # does where it does not — which makes the operator's own stop/target the
        # thing being tested, exactly as it should be.
        win_probability = self._win_probability(underlying, direction)
        expected_value = orchestrator.expected_value.evaluate(
            instrument=instrument,
            quantity=lot_size,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            win_probability=win_probability,
            quote=quote,
        )

        approval = orchestrator.risk.evaluate(
            decision=Decision.BUY if direction is Direction.LONG else Decision.SELL,
            instrument=instrument,
            underlying=underlying,
            strategy=strategy,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            portfolio_state=portfolio_state,
            margin=margin,
            quote=quote,
            expected_value=expected_value,
            greeks=None,
            underlying_price=entry_price,
            now=now,
        )

        # A kill switch is never bypassable, forced or not.
        blocked, kill_reason = orchestrator.kill_switches.can_trade(
            strategy=strategy, instrument=instrument.trading_symbol, underlying=underlying
        )
        if not blocked:
            return ManualTradeResult(
                False, action, instrument=instrument, approval=approval,
                reason=f"kill switch engaged: {kill_reason}",
            )

        requested_quantity = None
        if quantity is not None:
            requested_quantity = int(quantity)
        elif lots is not None:
            requested_quantity = int(lots) * lot_size

        if not approval.approved and not force:
            return ManualTradeResult(
                False, action, instrument=instrument, approval=approval,
                reason="; ".join(approval.rejection_reasons) or "risk engine rejected the trade",
            )

        if force:
            if requested_quantity is None:
                return ManualTradeResult(
                    False, action, instrument=instrument, approval=approval,
                    reason="--force requires an explicit --lots or --quantity",
                )
            approval = self._forced_approval(
                approval, instrument, requested_quantity, entry_price, stop_price, now, direction
            )
            logger.warning(
                "FORCED manual order: %s %s x%d — risk checks were bypassed by the operator",
                transaction_type.value, instrument.trading_symbol, requested_quantity,
            )
            orchestrator.journal.record_event(
                level="WARNING", category="manual",
                message="forced manual order — risk sizing bypassed",
                payload={
                    "instrument": instrument.trading_symbol,
                    "quantity": requested_quantity,
                    "rejections": approval.rejection_reasons,
                },
            )
        elif requested_quantity is not None and approval.sizing is not None:
            # An explicit size is honoured only if it is no larger than what the
            # sizer approved. Asking for more than the risk budget allows is a
            # rejection, not a silent trim.
            if requested_quantity > approval.sizing.quantity:
                return ManualTradeResult(
                    False, action, instrument=instrument, approval=approval,
                    reason=(
                        f"requested {requested_quantity} units but risk sizing allows "
                        f"{approval.sizing.quantity} ({approval.sizing.capped_by or 'risk budget'}). "
                        "Reduce the size, or pass --force to override deliberately."
                    ),
                )
            approval = self._resize(approval, requested_quantity, entry_price, stop_price)

        if approval.sizing is None or approval.sizing.quantity <= 0:
            return ManualTradeResult(
                False, action, instrument=instrument, approval=approval,
                reason="position sizing returned zero units",
            )

        result = orchestrator.execution.execute(
            approval=approval,
            instrument=instrument,
            quote=quote,
            decision_id=decision_id,
            strategy=strategy,
            limit_price=limit_price,
            market_data_at=quote.timestamp,
            signal_at=now,
        )

        if not result.success:
            if result.ambiguous:
                orchestrator.trading_allowed = False
                orchestrator.alerts.emergency(
                    f"manual order for {instrument.trading_symbol} is in an ambiguous state; "
                    "entries halted pending reconciliation"
                )
            return ManualTradeResult(
                False, action, instrument=instrument, approval=approval,
                decision_id=decision_id, reason=result.failure_reason, steps=result.steps,
            )

        fill_price = result.average_fill_price or entry_price
        position_id = ""
        if track:
            position_id = self._track(
                contract=contract, direction=direction, quantity=result.filled_quantity,
                fill_price=fill_price, stop_price=stop_price, target_price=target_price,
                decision_id=decision_id, strategy=strategy, now=now,
            )

        orchestrator.journal.record_trade(
            {
                "trade_id": position_id or decision_id,
                "decision_id": decision_id,
                "entry_order_id": result.client_order_id,
                "underlying": underlying,
                "instrument": instrument.trading_symbol,
                "expiry": instrument.expiry_date.isoformat() if instrument.expiry_date else None,
                "strike": instrument.strike_price,
                "option_type": instrument.instrument_type.value,
                "strategy": strategy,
                "regime": (
                    orchestrator.regime_engine.current.dominant.value
                    if orchestrator.regime_engine.current else Regime.UNCERTAIN.value
                ),
                "direction": direction.value,
                "quantity": result.filled_quantity,
                "entry_time": now.isoformat(),
                "entry_price": fill_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "explanation": (
                    f"MANUAL {transaction_type.value} placed by the operator"
                    + (" with --force (risk sizing bypassed)" if force else "")
                    + f"\nstop {stop_price:,.2f} / target {target_price:,.2f}"
                ),
            }
        )
        orchestrator.alerts.trade_entry(
            f"MANUAL {transaction_type.value} {instrument.trading_symbol} "
            f"x{result.filled_quantity} @ {fill_price:.2f}",
            decision_id=decision_id, underlying=underlying,
        )

        return ManualTradeResult(
            True, action, instrument=instrument, quantity=result.filled_quantity,
            price=fill_price, decision_id=decision_id, position_id=position_id,
            approval=approval, steps=result.steps,
        )

    def _win_probability(self, underlying: str, direction: Direction) -> float:
        """A win probability for a hand-placed trade.

        The deep analysis is the only calibrated opinion available outside the
        pipeline, so its support for this direction nudges a coin flip. The nudge
        is deliberately small: an operator's discretionary entry is not a model
        forecast and should not be scored like one.
        """
        analysis = None
        engine = getattr(self.orchestrator, "analysis", None)
        if engine is not None:
            analysis = engine.last(underlying)
        if analysis is None:
            return 0.5
        return float(min(0.65, 0.5 + 0.15 * analysis.supports(direction)))

    # ------------------------------------------------------------------ #
    def _forced_approval(
        self, approval, instrument, quantity, entry_price, stop_price, now, direction
    ) -> RiskApproval:
        """Build an approval for a forced order, preserving the rejections."""
        from ..risk.sizing import SizingResult

        risk_per_unit = abs(entry_price - stop_price)
        sizing = SizingResult(
            quantity=quantity,
            lots=max(1, quantity // max(1, instrument.lot_size)),
            lot_size=max(1, instrument.lot_size),
            risk_amount=risk_per_unit * quantity,
            risk_per_unit=risk_per_unit,
            capital_required=entry_price * quantity,
            margin_required=entry_price * quantity,
            capped_by="operator override",
            reasons=["size set by the operator with --force; risk sizing was bypassed"],
        )
        return RiskApproval(
            approved=True,
            decision=Decision.BUY if direction is Direction.LONG else Decision.SELL,
            checks=approval.checks,
            sizing=sizing,
            risk_multiplier=1.0,
            rejection_reasons=approval.rejection_reasons,
            portfolio_before=approval.portfolio_before,
            portfolio_after=approval.portfolio_after,
            approved_at=now,
            approval_id=f"MANUAL-FORCED-{self._counter:04d}",
        )

    @staticmethod
    def _resize(approval: RiskApproval, quantity: int, entry_price: float, stop_price: float):
        sizing = approval.sizing
        if sizing is None:
            return approval
        risk_per_unit = abs(entry_price - stop_price)
        sizing.quantity = quantity
        sizing.lots = max(1, quantity // max(1, sizing.lot_size))
        sizing.risk_amount = risk_per_unit * quantity
        sizing.risk_per_unit = risk_per_unit
        sizing.capital_required = entry_price * quantity
        sizing.reasons.append(f"size reduced to the operator's request of {quantity} units")
        return approval

    def _track(
        self, *, contract, direction, quantity, fill_price, stop_price, target_price,
        decision_id, strategy, now,
    ) -> str:
        orchestrator = self.orchestrator
        orchestrator._position_counter += 1
        position_id = f"P{orchestrator.run_id}-{orchestrator._position_counter:04d}"
        managed = ManagedPosition(
            position_id=position_id,
            decision_id=decision_id,
            instrument=contract.instrument,
            underlying=contract.underlying or contract.instrument.trading_symbol,
            strategy=strategy,
            direction=direction,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=now,
            stop_price=stop_price,
            target_price=target_price,
            initial_stop_price=stop_price,
            entry_regime=(
                orchestrator.regime_engine.current.dominant
                if orchestrator.regime_engine.current else Regime.UNCERTAIN
            ),
            entry_ml_probability=0.5,
            expected_holding_minutes=60,
            underlying_entry_price=fill_price,
            last_price=fill_price,
        )
        orchestrator.positions.track(managed)
        return position_id

    # ================================================================== #
    # Exit
    # ================================================================== #
    def exit_position(
        self, position_id: str, *, quantity: int | None = None, reason: str = "manual exit"
    ) -> ManualTradeResult:
        """Close one tracked position, whole or in part."""
        orchestrator = self.orchestrator
        position = next(
            (p for p in orchestrator.positions.all() if p.position_id == position_id), None
        )
        if position is None:
            return ManualTradeResult(
                False, "EXIT",
                reason=f"no tracked position {position_id!r}; `aqtp positions` lists what is open",
            )

        close_quantity = int(quantity or position.quantity)
        if close_quantity <= 0 or close_quantity > position.quantity:
            return ManualTradeResult(
                False, "EXIT", instrument=position.instrument,
                reason=f"quantity must be between 1 and {position.quantity}",
            )

        now = orchestrator.clock.now()
        quote = self._quote(position.instrument)
        action = (
            ManagementAction.PARTIAL_EXIT if close_quantity < position.quantity
            else ManagementAction.EXIT
        )
        decision = ManagementDecision(action, ExitReason.RISK_LIMIT, reason)
        if action is ManagementAction.PARTIAL_EXIT:
            decision.exit_quantity = close_quantity

        ok = orchestrator._exit(position, decision, close_quantity, now, quote)
        if not ok:
            return ManualTradeResult(
                False, "EXIT", instrument=position.instrument,
                reason="the exit order did not fill — the position is still open",
            )
        if action is ManagementAction.PARTIAL_EXIT:
            orchestrator.positions.apply(position, decision)

        return ManualTradeResult(
            True, "EXIT", instrument=position.instrument, quantity=close_quantity,
            price=float(quote.last_price) if quote else position.last_price,
            position_id=position_id,
        )

    def close_all(self, *, reason: str = "operator close-all") -> list[ManualTradeResult]:
        """Flatten every tracked position."""
        results: list[ManualTradeResult] = []
        for position in list(self.orchestrator.positions.all()):
            results.append(self.exit_position(position.position_id, reason=reason))
        return results

    # ================================================================== #
    # Order management
    # ================================================================== #
    def cancel(self, broker_order_id: str) -> ManualTradeResult:
        """Cancel one resting order at the broker."""
        broker = self.orchestrator.execution_broker
        try:
            orders = {o.broker_order_id: o for o in broker.list_orders()}
        except BrokerError as exc:
            return ManualTradeResult(False, "CANCEL", reason=f"could not list orders: {exc}")

        order = orders.get(broker_order_id)
        if order is None:
            return ManualTradeResult(
                False, "CANCEL", reason=f"no order {broker_order_id!r} at the broker"
            )
        if not order.status.is_open:
            return ManualTradeResult(
                False, "CANCEL", instrument=order.instrument,
                reason=f"order is {order.status.value}, not open",
            )
        try:
            cancelled = broker.cancel_order(broker_order_id, segment=order.instrument.segment)
        except BrokerError as exc:
            return ManualTradeResult(
                False, "CANCEL", instrument=order.instrument, reason=str(exc)
            )
        self.orchestrator.journal.record_event(
            level="WARNING", category="manual",
            message=f"operator cancelled order {broker_order_id}",
            payload={"instrument": order.instrument.trading_symbol},
        )
        return ManualTradeResult(
            True, "CANCEL", instrument=order.instrument,
            quantity=cancelled.quantity, reason=cancelled.status.value,
        )

    def cancel_all(self) -> list[ManualTradeResult]:
        broker = self.orchestrator.execution_broker
        try:
            orders = [o for o in broker.list_orders() if o.status.is_open]
        except BrokerError as exc:
            return [ManualTradeResult(False, "CANCEL", reason=f"could not list orders: {exc}")]
        return [self.cancel(order.broker_order_id) for order in orders]

    def modify(
        self,
        broker_order_id: str,
        *,
        price: float | None = None,
        quantity: int | None = None,
        trigger_price: float | None = None,
    ) -> ManualTradeResult:
        """Modify a resting order's price, size or trigger."""
        broker = self.orchestrator.execution_broker
        try:
            orders = {o.broker_order_id: o for o in broker.list_orders()}
        except BrokerError as exc:
            return ManualTradeResult(False, "MODIFY", reason=f"could not list orders: {exc}")

        order = orders.get(broker_order_id)
        if order is None:
            return ManualTradeResult(
                False, "MODIFY", reason=f"no order {broker_order_id!r} at the broker"
            )
        if not order.status.is_open:
            return ManualTradeResult(
                False, "MODIFY", instrument=order.instrument,
                reason=f"order is {order.status.value}, not open",
            )
        if price is None and quantity is None and trigger_price is None:
            return ManualTradeResult(
                False, "MODIFY", instrument=order.instrument,
                reason="nothing to modify: pass --price, --quantity or --trigger-price",
            )
        try:
            modified = broker.modify_order(
                broker_order_id,
                segment=order.instrument.segment,
                price=price,
                quantity=quantity,
                trigger_price=trigger_price,
                order_type=order.order_type,
            )
        except BrokerError as exc:
            return ManualTradeResult(
                False, "MODIFY", instrument=order.instrument, reason=str(exc)
            )
        self.orchestrator.journal.record_event(
            level="WARNING", category="manual",
            message=f"operator modified order {broker_order_id}",
            payload={"price": price, "quantity": quantity, "trigger_price": trigger_price},
        )
        return ManualTradeResult(
            True, "MODIFY", instrument=order.instrument,
            quantity=modified.quantity, price=modified.price or 0.0,
        )

    # ================================================================== #
    def set_stop(
        self, position_id: str, *, stop_price: float | None = None, target_price: float | None = None
    ) -> ManualTradeResult:
        """Move the managed stop or target on an open position."""
        position = next(
            (p for p in self.orchestrator.positions.all() if p.position_id == position_id), None
        )
        if position is None:
            return ManualTradeResult(False, "SET-STOP", reason=f"no tracked position {position_id!r}")
        if stop_price is None and target_price is None:
            return ManualTradeResult(
                False, "SET-STOP", instrument=position.instrument,
                reason="pass --stop and/or --target",
            )
        if stop_price is not None:
            position.stop_price = float(stop_price)
        if target_price is not None:
            position.target_price = float(target_price)
        self.orchestrator.journal.record_event(
            level="INFO", category="manual",
            message=f"operator adjusted levels on {position_id}",
            payload={"stop": position.stop_price, "target": position.target_price},
        )
        return ManualTradeResult(
            True, "SET-STOP", instrument=position.instrument,
            position_id=position_id, price=position.stop_price,
        )
