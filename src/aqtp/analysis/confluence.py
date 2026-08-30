"""Confluence: turning many measurements into one defensible conviction.

The failure mode this module exists to prevent is stacking correlated evidence.
Six momentum indicators agreeing is one opinion, not six. So the analysis is
organised into *dimensions* that measure genuinely different things — trend,
momentum, structure, order flow, dealer positioning, statistical character,
volatility and the cross-asset backdrop — and only the agreement *between*
dimensions counts as confluence.

Two properties are deliberate:

* **A veto is not a low score.** Some conditions (toxic flow, a jump that just
  invalidated the features, an unfillable book) mean *do not trade*, not "trade
  smaller". They are returned separately from the score.
* **Conflict is reported, not averaged away.** Dimensions pulling against each
  other reduce conviction rather than cancelling to a confident zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.logging import get_logger
from ..core.types import Direction

logger = get_logger(__name__)

# Default dimension weights. They need not sum to 1 — the engine normalises over
# whichever dimensions actually reported.
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 0.16,
    "momentum": 0.14,
    "structure": 0.14,
    "orderflow": 0.16,
    "positioning": 0.14,
    "statistical": 0.10,
    "volatility": 0.08,
    "crossasset": 0.08,
}


@dataclass
class Dimension:
    """One independent read on the market."""

    name: str
    score: float                  # signed [-1, 1]
    confidence: float = 1.0       # [0, 1] — how much data backed this read
    weight: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.score = float(np.clip(self.score, -1.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    @property
    def direction(self) -> int:
        if self.score > 0.10:
            return 1
        if self.score < -0.10:
            return -1
        return 0

    @property
    def contribution(self) -> float:
        return self.score * self.confidence * self.weight


@dataclass
class ConfluenceReport:
    """The combined view, with the audit trail that produced it."""

    timestamp: datetime | None = None
    underlying: str = ""
    dimensions: list[Dimension] = field(default_factory=list)
    net_score: float = 0.0        # signed [-1, 1]
    conviction: float = 0.0       # [0, 1]
    alignment: float = 0.0        # [0, 1] share of dimensions agreeing
    agreeing: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- #
    @property
    def bias(self) -> Direction:
        if self.vetoed:
            return Direction.FLAT
        if self.net_score > 0.12:
            return Direction.LONG
        if self.net_score < -0.12:
            return Direction.SHORT
        return Direction.FLAT

    @property
    def vetoed(self) -> bool:
        return bool(self.vetoes)

    def supports(self, direction: Direction) -> float:
        """[0, 1] — how strongly the analysis backs a specific direction.

        Returns 0 for a vetoed report and for a direction the analysis opposes,
        so a caller can multiply a strategy's confidence by this without
        special-casing.
        """
        if self.vetoed or direction is Direction.FLAT:
            return 0.0
        sign = 1.0 if direction is Direction.LONG else -1.0
        aligned = self.net_score * sign
        if aligned <= 0:
            return 0.0
        return float(np.clip(aligned, 0.0, 1.0) * self.conviction)

    def dimension(self, name: str) -> Dimension | None:
        for entry in self.dimensions:
            if entry.name == name:
                return entry
        return None

    def to_dict(self) -> dict[str, float]:
        out = {
            "conf_net_score": self.net_score,
            "conf_conviction": self.conviction,
            "conf_alignment": self.alignment,
            "conf_vetoed": 1.0 if self.vetoed else 0.0,
        }
        for entry in self.dimensions:
            out[f"conf_dim_{entry.name}"] = entry.score
        return out

    def explain(self) -> list[str]:
        """The full audit trail, ordered so the headline comes first."""
        lines: list[str] = []
        if self.vetoed:
            lines.append(f"VETO — {self.underlying}: {'; '.join(self.vetoes)}")
        else:
            lines.append(
                f"{self.underlying}: net {self.net_score:+.2f}, conviction {self.conviction:.2f}, "
                f"alignment {self.alignment:.0%} → bias {self.bias.value}"
            )
        if self.agreeing:
            lines.append(f"  agreeing: {', '.join(self.agreeing)}")
        if self.conflicting:
            lines.append(f"  conflicting: {', '.join(self.conflicting)}")
        if self.neutral:
            lines.append(f"  neutral: {', '.join(self.neutral)}")
        for entry in sorted(self.dimensions, key=lambda d: abs(d.contribution), reverse=True):
            if entry.confidence <= 0:
                continue
            lines.append(
                f"  [{entry.name}] {entry.score:+.2f} "
                f"(confidence {entry.confidence:.2f}, weight {entry.weight:.2f})"
            )
            for reason in entry.reasons[:4]:
                lines.append(f"      - {reason}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return lines

    def render(self) -> str:
        return "\n".join(self.explain())


# --------------------------------------------------------------------------- #
class ConfluenceEngine:
    """Combines dimensions into a `ConfluenceReport`."""

    def __init__(
        self,
        *,
        weights: dict[str, float] | None = None,
        min_dimensions: int = 4,
        conflict_penalty: float = 0.5,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.min_dimensions = min_dimensions
        self.conflict_penalty = conflict_penalty

    # ------------------------------------------------------------------ #
    def combine(
        self,
        dimensions: list[Dimension],
        *,
        underlying: str = "",
        timestamp: datetime | None = None,
        vetoes: list[str] | None = None,
        warnings: list[str] | None = None,
    ) -> ConfluenceReport:
        report = ConfluenceReport(
            timestamp=timestamp,
            underlying=underlying,
            vetoes=list(vetoes or []),
            warnings=list(warnings or []),
        )

        for entry in dimensions:
            if not entry.weight:
                entry.weight = self.weights.get(entry.name, 0.05)
        report.dimensions = dimensions

        reporting = [d for d in dimensions if d.confidence > 0.05]
        if len(reporting) < self.min_dimensions:
            report.warnings.append(
                f"only {len(reporting)} of {len(dimensions)} analysis dimensions had enough "
                f"data (minimum {self.min_dimensions}); conviction is capped"
            )

        effective_weight = sum(d.weight * d.confidence for d in reporting)
        if effective_weight <= 0:
            report.warnings.append("no analysis dimension produced a usable reading")
            return report

        report.net_score = float(
            np.clip(sum(d.contribution for d in reporting) / effective_weight, -1.0, 1.0)
        )

        sign = 1 if report.net_score > 0 else -1 if report.net_score < 0 else 0
        for entry in reporting:
            if entry.direction == 0:
                report.neutral.append(entry.name)
            elif sign != 0 and entry.direction == sign:
                report.agreeing.append(entry.name)
            else:
                report.conflicting.append(entry.name)

        directional = len(report.agreeing) + len(report.conflicting)
        report.alignment = (
            float(len(report.agreeing) / directional) if directional else 0.0
        )

        # Conviction: magnitude, scaled by how much of the analysis actually
        # reported, how much of it agreed, and penalised for direct conflict.
        coverage = float(len(reporting) / max(len(dimensions), 1))
        data_quality = float(np.mean([d.confidence for d in reporting]))
        conflict = float(len(report.conflicting) / directional) if directional else 0.0

        conviction = abs(report.net_score)
        conviction *= 0.4 + 0.6 * report.alignment
        conviction *= 0.5 + 0.5 * coverage
        conviction *= 0.5 + 0.5 * data_quality
        conviction *= max(0.0, 1.0 - self.conflict_penalty * conflict)
        if len(reporting) < self.min_dimensions:
            conviction *= 0.6
        report.conviction = float(np.clip(conviction, 0.0, 1.0))

        if report.vetoed:
            report.conviction = 0.0
        return report
