"""OpportunityScanner (REQ 22).

Continuously scores the eligible universe and ranks candidates, so capital goes to
the best available opportunity rather than to whichever symbol happened to be
evaluated first.

Ranking rather than first-match matters because the position and risk budgets are
finite: with room for four positions and forty candidates, the order of evaluation
determines the portfolio. Scoring every candidate before committing is what makes
that choice deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..configuration.schema import ScannerConfig
from ..core.logging import get_logger
from ..core.types import Decision, Direction, Instrument, Quote
from ..ml.predict import Prediction
from ..regime.engine import RegimeState
from ..signals.ensemble import EnsembleResult

if TYPE_CHECKING:
    from ..analysis.engine import AnalysisReport

logger = get_logger(__name__)


@dataclass
class Opportunity:
    """One ranked candidate."""

    underlying: str
    instrument: Instrument | None
    decision: Decision
    direction: Direction
    score: float
    components: dict[str, float] = field(default_factory=dict)
    ensemble: EnsembleResult | None = None
    prediction: Prediction | None = None
    regime: RegimeState | None = None
    analysis: "AnalysisReport | None" = None
    quote: Quote | None = None
    expected_return: float = 0.0
    expected_risk: float = 0.0
    capital_required: float = 0.0
    liquidity_score: float = 0.0
    spread_score: float = 0.0
    execution_quality: float = 0.0
    timestamp: datetime | None = None
    rejection_reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.decision is not Decision.NO_TRADE and not self.rejection_reason

    @property
    def risk_reward(self) -> float:
        return self.expected_return / self.expected_risk if self.expected_risk > 0 else 0.0

    def explain(self) -> str:
        if not self.is_actionable:
            return f"{self.underlying}: NO_TRADE — {self.rejection_reason}"
        parts = ", ".join(f"{k} {v:.2f}" for k, v in sorted(self.components.items()))
        return (
            f"{self.underlying} {self.decision.value} score {self.score:.3f} "
            f"(R:R {self.risk_reward:.2f}) [{parts}]"
        )


@dataclass
class ScanResult:
    timestamp: datetime
    opportunities: list[Opportunity] = field(default_factory=list)
    rejected: list[Opportunity] = field(default_factory=list)
    scanned: int = 0
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def actionable(self) -> list[Opportunity]:
        return [o for o in self.opportunities if o.is_actionable]

    def top(self, n: int) -> list[Opportunity]:
        return self.actionable[:n]

    def summary(self) -> str:
        return (
            f"scanned {self.scanned} underlyings: {len(self.actionable)} actionable, "
            f"{len(self.rejected)} rejected, {len(self.skipped)} skipped"
        )

    def lines(self) -> list[str]:
        out = [self.summary()]
        for opportunity in self.opportunities[:10]:
            out.append(f"  {opportunity.explain()}")
        return out


class OpportunityScanner:
    def __init__(self, config: ScannerConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    def score(
        self,
        *,
        underlying: str,
        ensemble: EnsembleResult,
        prediction: Prediction | None,
        regime: RegimeState,
        quote: Quote | None,
        instrument: Instrument | None = None,
        capital_required: float = 0.0,
        available_capital: float = 0.0,
        timestamp: datetime | None = None,
        analysis: "AnalysisReport | None" = None,
    ) -> Opportunity:
        """Score one candidate across the dimensions REQ 22 lists."""
        opportunity = Opportunity(
            underlying=underlying,
            instrument=instrument,
            decision=ensemble.decision,
            direction=ensemble.direction,
            score=0.0,
            ensemble=ensemble,
            prediction=prediction,
            regime=regime,
            quote=quote,
            capital_required=capital_required,
            timestamp=timestamp,
            analysis=analysis,
        )

        if not ensemble.is_tradable:
            opportunity.rejection_reason = ensemble.rejection_reason
            return opportunity

        opportunity.expected_return = abs(ensemble.proposed_target - ensemble.proposed_entry)
        opportunity.expected_risk = abs(ensemble.proposed_entry - ensemble.proposed_stop)

        components: dict[str, float] = {}

        components["strategy"] = float(np.clip(ensemble.score, 0.0, 1.0))

        if prediction is not None and prediction.usable:
            # Map the probability's distance from a coin flip onto [0, 1]; the raw
            # probability would give a useless 0.5 floor to every candidate.
            components["ml"] = float(np.clip((prediction.probability - 0.5) * 2.5, 0.0, 1.0))
        else:
            components["ml"] = 0.3

        components["regime_fit"] = float(np.clip(regime.confidence, 0.0, 1.0))

        # The deep-analysis confluence, scored for the direction actually taken.
        # Absent analysis scores a neutral 0.4 rather than 0 so that a symbol the
        # analysis layer could not read is ranked below one it endorses, but is
        # not eliminated outright.
        if analysis is not None:
            components["analysis"] = float(np.clip(analysis.supports(ensemble.direction), 0.0, 1.0))
        else:
            components["analysis"] = 0.4

        liquidity, spread_score, execution_quality = self._execution_scores(quote)
        opportunity.liquidity_score = liquidity
        opportunity.spread_score = spread_score
        opportunity.execution_quality = execution_quality
        components["liquidity"] = liquidity
        components["execution_quality"] = execution_quality

        # Risk/reward mapped so 1.0 scores zero and 3.0 scores full — a trade with
        # equal upside and downside contributes nothing here.
        rr = opportunity.risk_reward
        components["expected_return"] = float(np.clip((rr - 1.0) / 2.0, 0.0, 1.0))

        total = sum(self.config.weights.get(name, 0.0) * value for name, value in components.items())

        # Capital efficiency is a tiebreaker, not a scoring dimension: between two
        # equally good trades, prefer the one that ties up less capital.
        if available_capital > 0 and capital_required > 0:
            efficiency = float(np.clip(1.0 - capital_required / available_capital, 0.0, 1.0))
            components["capital_efficiency"] = efficiency
            total = total * (0.95 + 0.05 * efficiency)

        opportunity.components = {k: round(v, 4) for k, v in components.items()}
        opportunity.score = float(np.clip(total, 0.0, 1.0))

        if opportunity.score < self.config.min_opportunity_score:
            opportunity.rejection_reason = (
                f"opportunity score {opportunity.score:.3f} below the "
                f"{self.config.min_opportunity_score:.3f} threshold"
            )
        return opportunity

    @staticmethod
    def _execution_scores(quote: Quote | None) -> tuple[float, float, float]:
        """Liquidity, spread and combined execution quality, all in [0, 1]."""
        if quote is None:
            return 0.0, 0.0, 0.0

        volume = float(quote.volume or 0.0)
        open_interest = float(quote.open_interest or 0.0)
        # Log scaling: the difference between 1k and 10k contracts matters far more
        # than between 100k and 109k.
        volume_score = float(np.clip(np.log1p(volume) / np.log1p(500_000), 0.0, 1.0))
        oi_score = float(np.clip(np.log1p(open_interest) / np.log1p(2_000_000), 0.0, 1.0))
        liquidity = 0.6 * volume_score + 0.4 * oi_score

        spread_pct = quote.spread_pct
        spread_score = (
            float(np.clip(1.0 - spread_pct / 0.02, 0.0, 1.0)) if spread_pct is not None else 0.2
        )

        depth_score = 0.5
        if quote.depth_buy and quote.depth_sell:
            buy_depth = sum(level.quantity for level in quote.depth_buy)
            sell_depth = sum(level.quantity for level in quote.depth_sell)
            balance = min(buy_depth, sell_depth) / max(buy_depth, sell_depth, 1)
            depth_score = float(np.clip(balance, 0.0, 1.0))

        execution_quality = 0.45 * spread_score + 0.35 * liquidity + 0.20 * depth_score
        return liquidity, spread_score, float(np.clip(execution_quality, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    def rank(self, opportunities: list[Opportunity], *, timestamp: datetime) -> ScanResult:
        """Sort candidates and split actionable from rejected."""
        actionable = sorted(
            (o for o in opportunities if o.is_actionable),
            key=lambda o: o.score,
            reverse=True,
        )
        rejected = [o for o in opportunities if not o.is_actionable]

        result = ScanResult(
            timestamp=timestamp,
            opportunities=actionable[: self.config.max_opportunities_per_cycle],
            rejected=rejected,
            scanned=len(opportunities),
        )
        if actionable:
            logger.info(
                "scan: %d/%d actionable, best %s at %.3f",
                len(actionable), len(opportunities),
                actionable[0].underlying, actionable[0].score,
            )
        return result
