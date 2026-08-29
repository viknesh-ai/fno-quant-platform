"""Expected value calculation (REQ 55).

    EV = P(win) x expected gain - P(loss) x expected loss - costs - slippage

Only trades whose EV clears the configured minimum are allowed to reach the
RiskEngine, which is exactly the ordering REQ 55 specifies.

Two modelling choices worth stating:

1. The win probability comes from the *calibrated* ML probability where available,
   blended with the ensemble confidence. Using an uncalibrated probability here
   would make every downstream number wrong in a correlated way.
2. The expected gain is not the full target distance. Trades exit early — on a
   regime change, a time stop, a trailing stop — so applying a realization factor
   avoids the systematic over-estimate that makes every backtest look better than
   live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..configuration.schema import ExpectedValueConfig
from ..core.logging import get_logger
from ..core.types import Instrument, Quote, TransactionType
from ..risk.costs import CostBreakdown, TransactionCostModel

logger = get_logger(__name__)

# Fraction of the target typically captured before an exit rule fires. Conservative
# by design; measured against the journal and updated by the research loop.
DEFAULT_TARGET_REALIZATION = 0.75
# Fraction of the stop distance typically given up, including slippage past the stop.
DEFAULT_STOP_REALIZATION = 1.05


@dataclass
class ExpectedValueResult:
    expected_value_rupees: float
    expected_value_r: float
    win_probability: float
    expected_gain: float
    expected_loss: float
    costs: CostBreakdown
    risk_reward: float
    breakeven_probability: float
    passes: bool
    reason: str = ""
    components: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        verdict = "PASS" if self.passes else "FAIL"
        return (
            f"EV {verdict}: {self.expected_value_rupees:+,.0f} ({self.expected_value_r:+.3f}R) "
            f"= p{self.win_probability:.3f} x {self.expected_gain:,.0f} - "
            f"{1 - self.win_probability:.3f} x {self.expected_loss:,.0f} - "
            f"costs {self.costs.total:,.0f}"
            + (f" — {self.reason}" if self.reason else "")
        )


class ExpectedValueCalculator:
    def __init__(self, config: ExpectedValueConfig, cost_model: TransactionCostModel) -> None:
        self.config = config
        self.costs = cost_model

    def evaluate(
        self,
        *,
        instrument: Instrument,
        quantity: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        win_probability: float,
        transaction_type: TransactionType = TransactionType.BUY,
        quote: Quote | None = None,
        target_realization: float = DEFAULT_TARGET_REALIZATION,
        stop_realization: float = DEFAULT_STOP_REALIZATION,
    ) -> ExpectedValueResult:
        """Compute EV for a fully specified candidate trade."""
        risk_per_unit = abs(entry_price - stop_price)
        reward_per_unit = abs(target_price - entry_price)

        costs = self.costs.round_trip(
            instrument=instrument,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=target_price,
            entry_side=transaction_type,
            quote=quote,
        )

        if risk_per_unit <= 0 or quantity <= 0:
            return ExpectedValueResult(
                expected_value_rupees=0.0,
                expected_value_r=0.0,
                win_probability=win_probability,
                expected_gain=0.0,
                expected_loss=0.0,
                costs=costs,
                risk_reward=0.0,
                breakeven_probability=1.0,
                passes=False,
                reason="stop distance is zero — the trade has no defined risk",
            )

        expected_gain = reward_per_unit * quantity * target_realization
        expected_loss = risk_per_unit * quantity * stop_realization
        total_risk = risk_per_unit * quantity

        p = float(np.clip(win_probability, 0.0, 1.0))
        expected_value = p * expected_gain - (1 - p) * expected_loss - costs.total
        expected_value_r = expected_value / total_risk if total_risk > 0 else 0.0
        risk_reward = reward_per_unit / risk_per_unit

        # The probability at which this trade breaks even after costs. Comparing the
        # model's probability against this is the sharpest statement of whether an
        # edge exists: if p < breakeven, the trade loses money in expectation no
        # matter how good the signal "looks".
        denominator = expected_gain + expected_loss
        breakeven = (expected_loss + costs.total) / denominator if denominator > 0 else 1.0

        passes, reason = self._verdict(
            expected_value=expected_value,
            expected_value_r=expected_value_r,
            risk_reward=risk_reward,
            win_probability=p,
            breakeven=breakeven,
        )

        return ExpectedValueResult(
            expected_value_rupees=expected_value,
            expected_value_r=expected_value_r,
            win_probability=p,
            expected_gain=expected_gain,
            expected_loss=expected_loss,
            costs=costs,
            risk_reward=risk_reward,
            breakeven_probability=float(np.clip(breakeven, 0.0, 1.0)),
            passes=passes,
            reason=reason,
            components={
                "risk_per_unit": risk_per_unit,
                "reward_per_unit": reward_per_unit,
                "total_risk": total_risk,
                "cost_r": costs.total / total_risk if total_risk > 0 else float("inf"),
                "edge_over_breakeven": p - breakeven,
            },
        )

    def _verdict(
        self,
        *,
        expected_value: float,
        expected_value_r: float,
        risk_reward: float,
        win_probability: float,
        breakeven: float,
    ) -> tuple[bool, str]:
        config = self.config
        if risk_reward < config.min_risk_reward:
            return False, (
                f"risk/reward {risk_reward:.2f} below the {config.min_risk_reward:.2f} minimum"
            )
        if win_probability <= breakeven:
            return False, (
                f"win probability {win_probability:.3f} does not exceed the "
                f"{breakeven:.3f} breakeven after costs"
            )
        if expected_value <= config.min_expected_value_rupees:
            return False, (
                f"expected value {expected_value:+,.0f} does not clear the "
                f"{config.min_expected_value_rupees:+,.0f} minimum"
            )
        if expected_value_r < config.min_expected_value_r_multiple:
            return False, (
                f"expected value {expected_value_r:+.3f}R below the "
                f"{config.min_expected_value_r_multiple:.3f}R minimum"
            )
        return True, (
            f"edge of {win_probability - breakeven:+.3f} over the {breakeven:.3f} "
            "cost-adjusted breakeven"
        )


def blended_win_probability(
    *, ml_probability: float, ensemble_confidence: float, ml_weight: float = 0.7
) -> float:
    """Combine the calibrated model probability with ensemble conviction.

    The model probability dominates because it is the only calibrated quantity;
    ensemble confidence contributes a smaller correction. The result is shrunk
    toward 0.5 slightly, because both inputs are estimated and an over-stated win
    probability inflates every EV downstream.
    """
    p = float(np.clip(ml_probability, 0.0, 1.0))
    conviction = 0.5 + 0.5 * float(np.clip(ensemble_confidence, 0.0, 1.0))
    blended = ml_weight * p + (1 - ml_weight) * conviction
    shrinkage = 0.9
    return float(np.clip(0.5 + shrinkage * (blended - 0.5), 0.0, 1.0))
