"""Explainability (REQ 48).

REQ 48 requires every trade to answer eight specific questions. This module builds
a structured `DecisionReport` that answers each one by name, drawing on evidence
that was actually recorded at decision time rather than reconstructed afterwards.

The report is deliberately assembled from the same objects the pipeline produced
(ensemble result, contract scorecard, sizing result, risk checklist), so an
explanation cannot drift from what the system really did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core.types import Decision, Direction, ExitReason, Instrument
from ..options.selector import ContractScore, SelectionResult
from ..risk.engine import RiskApproval
from ..risk.sizing import SizingResult
from ..signals.ensemble import EnsembleResult
from ..signals.expected_value import ExpectedValueResult


@dataclass
class DecisionReport:
    """Structured answers to the REQ 48 questions."""

    decision_id: str
    timestamp: datetime
    decision: Decision
    underlying: str
    instrument: str = ""

    why_enter: list[str] = field(default_factory=list)
    why_this_instrument: list[str] = field(default_factory=list)
    why_this_strike: list[str] = field(default_factory=list)
    why_this_expiry: list[str] = field(default_factory=list)
    why_this_size: list[str] = field(default_factory=list)
    why_this_stop: list[str] = field(default_factory=list)
    why_this_target: list[str] = field(default_factory=list)
    why_exit: list[str] = field(default_factory=list)

    rejected_alternatives: list[str] = field(default_factory=list)
    risk_checklist: list[str] = field(default_factory=list)
    no_trade_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "decision": self.decision.value,
            "underlying": self.underlying,
            "instrument": self.instrument,
            "why_did_we_enter": self.why_enter,
            "why_this_instrument": self.why_this_instrument,
            "why_this_strike": self.why_this_strike,
            "why_this_expiry": self.why_this_expiry,
            "why_this_size": self.why_this_size,
            "why_this_stop": self.why_this_stop,
            "why_this_target": self.why_this_target,
            "why_did_we_exit": self.why_exit,
            "rejected_alternatives": self.rejected_alternatives,
            "risk_checklist": self.risk_checklist,
            "no_trade_reasons": self.no_trade_reasons,
        }

    def render(self) -> str:
        """Human-readable report for the CLI and log."""
        lines = [
            f"DECISION REPORT  {self.decision_id}  [{self.decision.value}]",
            f"  {self.timestamp:%Y-%m-%d %H:%M:%S}  {self.underlying}"
            + (f"  {self.instrument}" if self.instrument else ""),
            "",
        ]
        sections = [
            ("WHY DID WE ENTER?", self.why_enter),
            ("WHY THIS INSTRUMENT?", self.why_this_instrument),
            ("WHY THIS STRIKE?", self.why_this_strike),
            ("WHY THIS EXPIRY?", self.why_this_expiry),
            ("WHY THIS SIZE?", self.why_this_size),
            ("WHY THIS STOP?", self.why_this_stop),
            ("WHY THIS TARGET?", self.why_this_target),
            ("WHY DID WE EXIT?", self.why_exit),
        ]
        for title, entries in sections:
            if not entries:
                continue
            lines.append(title)
            lines.extend(f"  - {entry}" for entry in entries)
            lines.append("")

        if self.no_trade_reasons:
            lines.append("WHY NO TRADE?")
            lines.extend(f"  - {reason}" for reason in self.no_trade_reasons)
            lines.append("")
        if self.rejected_alternatives:
            lines.append("REJECTED ALTERNATIVES")
            lines.extend(f"  - {alt}" for alt in self.rejected_alternatives)
            lines.append("")
        if self.risk_checklist:
            lines.append("RISK CHECKLIST")
            lines.extend(f"  {entry}" for entry in self.risk_checklist)
        return "\n".join(lines)


class ExplanationBuilder:
    """Assembles a DecisionReport from the pipeline's own outputs."""

    def build(
        self,
        *,
        decision_id: str,
        timestamp: datetime,
        underlying: str,
        ensemble: EnsembleResult,
        selection: SelectionResult | None = None,
        approval: RiskApproval | None = None,
        expected_value: ExpectedValueResult | None = None,
        underlying_price: float | None = None,
    ) -> DecisionReport:
        report = DecisionReport(
            decision_id=decision_id,
            timestamp=timestamp,
            decision=ensemble.decision if approval is None or approval.approved else Decision.NO_TRADE,
            underlying=underlying,
        )

        # --- no trade -------------------------------------------------------
        if not ensemble.is_tradable:
            report.no_trade_reasons.append(ensemble.rejection_reason)
            return report
        if approval is not None and not approval.approved:
            report.no_trade_reasons.extend(approval.rejection_reasons)
            report.risk_checklist = approval.checklist()
            if selection is not None:
                report.rejected_alternatives = [
                    c.explain() for c in selection.rejected_candidates()[:5]
                ]
            return report

        # --- why enter ------------------------------------------------------
        report.why_enter.append(
            f"{len(ensemble.agreeing_strategies)} strategies agree on "
            f"{ensemble.direction.value} with {ensemble.strategy_agreement:.0%} agreement: "
            f"{', '.join(ensemble.agreeing_strategies)}"
        )
        report.why_enter.append(
            f"regime is {ensemble.regime.value} at {ensemble.regime_confidence:.0%} confidence"
        )
        if ensemble.ml_probability != 0.5:
            report.why_enter.append(
                f"ML model assigns {ensemble.ml_probability:.1%} probability that the target "
                f"is reached before the stop"
            )
        report.why_enter.append(
            f"ensemble score {ensemble.score:.3f}, confidence {ensemble.confidence:.3f}"
        )
        if expected_value is not None:
            report.why_enter.append(
                f"expected value {expected_value.expected_value_rupees:+,.0f} "
                f"({expected_value.expected_value_r:+.3f}R) against a "
                f"{expected_value.breakeven_probability:.1%} cost-adjusted breakeven"
            )
        report.why_enter.extend(ensemble.reasons[:4])
        if ensemble.dissenting_strategies:
            report.why_enter.append(
                f"dissenting: {', '.join(ensemble.dissenting_strategies)} (outweighed)"
            )

        # --- instrument / strike / expiry -------------------------------------
        if selection is not None and selection.selected is not None:
            score = selection.selected
            contract = score.contract
            report.instrument = contract.instrument.trading_symbol

            report.why_this_instrument.append(
                f"selected {contract.instrument.trading_symbol} to express a "
                f"{ensemble.direction.value} view on {underlying}"
            )
            report.why_this_instrument.append(
                f"scored {score.total:.3f} across {len(selection.candidates)} candidate contracts"
            )
            components = ", ".join(f"{k} {v:.2f}" for k, v in sorted(score.components.items()))
            report.why_this_instrument.append(f"component scores: {components}")

            metrics = score.metrics
            report.why_this_strike.append(
                f"strike {contract.strike:.0f}"
                + (
                    f" is {abs(contract.strike - underlying_price) / underlying_price:.2%} from "
                    f"spot {underlying_price:.2f}"
                    if underlying_price
                    else ""
                )
            )
            if "delta" in metrics:
                report.why_this_strike.append(
                    f"delta {metrics['delta']:.3f} sits inside the configured participation band"
                )
            if "spread_pct" in metrics:
                report.why_this_strike.append(f"spread {metrics['spread_pct']:.2%} is acceptable")
            if "open_interest" in metrics and "volume" in metrics:
                report.why_this_strike.append(
                    f"liquidity: {metrics['volume']:,.0f} volume, "
                    f"{metrics['open_interest']:,.0f} open interest"
                )
            if "expected_move_to_premium" in metrics:
                report.why_this_strike.append(
                    f"expected move is {metrics['expected_move_to_premium']:.2f}x the premium "
                    "after decay"
                )

            report.why_this_expiry.append(f"expiry {contract.expiry.isoformat()}")
            if "days_to_expiry" in metrics:
                report.why_this_expiry.append(
                    f"{metrics['days_to_expiry']:.0f} days to expiry"
                )
            if "theta_to_premium" in metrics:
                report.why_this_expiry.append(
                    f"theta would consume {metrics['theta_to_premium']:.1%} of premium over the "
                    "expected holding period, within the configured limit"
                )

            report.rejected_alternatives = [
                c.explain() for c in selection.rejected_candidates()[:5]
            ]

        # --- size ---------------------------------------------------------------
        if approval is not None and approval.sizing is not None:
            sizing = approval.sizing
            report.why_this_size.extend(sizing.reasons)
            if approval.risk_multiplier != 1.0:
                report.why_this_size.append(
                    f"risk throttled to {approval.risk_multiplier:.0%} of normal by drawdown "
                    "level, strategy health and/or event risk"
                )
            if sizing.capped_by:
                report.why_this_size.append(f"size was limited by {sizing.capped_by}")

        # --- stop / target ---------------------------------------------------------
        report.why_this_stop.append(
            f"stop at {ensemble.proposed_stop:.2f} is the tightest invalidation level proposed "
            f"by any agreeing strategy — beyond it, no strategy still holds its thesis"
        )
        risk_distance = abs(ensemble.proposed_entry - ensemble.proposed_stop)
        report.why_this_stop.append(
            f"risk of {risk_distance:.2f} per unit from an entry of {ensemble.proposed_entry:.2f}"
        )

        report.why_this_target.append(
            f"target at {ensemble.proposed_target:.2f} is the weighted consensus of the agreeing "
            f"strategies ({ensemble.expected_move_atr:.2f} ATR of expected move)"
        )
        report.why_this_target.append(f"risk/reward {ensemble.risk_reward:.2f}")

        if approval is not None:
            report.risk_checklist = approval.checklist()

        return report

    @staticmethod
    def explain_exit(
        *,
        reason: ExitReason,
        detail: str = "",
        pnl: float = 0.0,
        r_multiple: float = 0.0,
        holding_minutes: float = 0.0,
    ) -> list[str]:
        """REQ 35: every exit carries a recorded reason."""
        lines = [f"exit reason: {reason.value}"]
        if detail:
            lines.append(detail)
        lines.append(
            f"result: {pnl:+,.0f} ({r_multiple:+.2f}R) after {holding_minutes:.0f} minutes"
        )
        return lines
