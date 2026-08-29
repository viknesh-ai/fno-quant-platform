"""Experiment versioning and the research loop (REQ 50).

REQ 50 requires that every experiment be versioned with its dataset, feature,
strategy and model versions, its configuration, its random seed, its backtest
period and its results. This module records exactly that, as JSON on disk, so
Claude Code (or any other tool) can read results back without the trading
application running (REQ 49).

The `experiment_id` is a content hash of the inputs, so re-running with identical
inputs collides rather than silently creating a "new" experiment — which is what
makes results reproducible instead of merely re-derived.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.ids import content_hash, new_run_id
from ..core.logging import get_logger
from ..features.engine import FEATURE_VERSION

logger = get_logger(__name__)

STRATEGY_VERSION = "1.0.0"
PIPELINE_STAGES = (
    "DATA", "FEATURES", "TRAIN", "BACKTEST", "WALK-FORWARD",
    "PAPER TRADE", "ANALYZE", "EXPERIMENT", "RETRAIN", "VALIDATE",
)


@dataclass
class Experiment:
    """One versioned experiment record (REQ 50)."""

    experiment_id: str
    name: str
    created_at: str
    stage: str = "EXPERIMENT"

    # versions
    dataset_version: str = ""
    feature_version: str = FEATURE_VERSION
    strategy_version: str = STRATEGY_VERSION
    model_version: str = ""

    # inputs
    configuration: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    backtest_start: str = ""
    backtest_end: str = ""
    symbols: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    # outputs
    results: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    passed: bool = False

    # provenance
    git_commit: str = ""
    python_version: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "created_at": self.created_at[:19],
            "stage": self.stage,
            "passed": self.passed,
            "verdict": self.verdict[:100],
        }


class ExperimentStore:
    """Filesystem-backed experiment registry."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "experiments.json"

    # ------------------------------------------------------------------ #
    def create(
        self,
        *,
        name: str,
        configuration: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        symbols: list[str] | None = None,
        dataset_version: str = "",
        model_version: str = "",
        random_seed: int = 42,
        backtest_start: str = "",
        backtest_end: str = "",
        stage: str = "EXPERIMENT",
        notes: str = "",
    ) -> Experiment:
        seed_payload = json.dumps(
            {
                "name": name,
                "config": configuration or {},
                "params": parameters or {},
                "symbols": sorted(symbols or []),
                "dataset": dataset_version,
                "features": FEATURE_VERSION,
                "strategy": STRATEGY_VERSION,
                "model": model_version,
                "seed": random_seed,
                "start": backtest_start,
                "end": backtest_end,
            },
            sort_keys=True, default=str,
        )
        experiment_id = f"{name[:20].replace(' ', '_')}_{content_hash(seed_payload)}"

        return Experiment(
            experiment_id=experiment_id,
            name=name,
            created_at=datetime.now().isoformat(),
            stage=stage,
            dataset_version=dataset_version,
            model_version=model_version,
            configuration=configuration or {},
            parameters=parameters or {},
            symbols=symbols or [],
            random_seed=random_seed,
            backtest_start=backtest_start,
            backtest_end=backtest_end,
            git_commit=_git_commit(),
            python_version=platform.python_version(),
            notes=notes,
        )

    def save(self, experiment: Experiment) -> Path:
        path = self.directory / f"{experiment.experiment_id}.json"
        path.write_text(json.dumps(experiment.to_dict(), indent=2, default=str))
        self._reindex()
        logger.info("saved experiment %s (%s)", experiment.experiment_id, experiment.verdict[:60])
        return path

    def load(self, experiment_id: str) -> Experiment:
        path = self.directory / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"no experiment {experiment_id!r}")
        return Experiment(**json.loads(path.read_text()))

    def all(self) -> list[Experiment]:
        out: list[Experiment] = []
        for path in sorted(self.directory.glob("*.json")):
            if path.name == "experiments.json":
                continue
            try:
                out.append(Experiment(**json.loads(path.read_text())))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping unreadable experiment %s: %s", path.name, exc)
        return sorted(out, key=lambda e: e.created_at, reverse=True)

    def _reindex(self) -> None:
        index = [e.summary() for e in self.all()]
        self.index_path.write_text(json.dumps(index, indent=2, default=str))

    def compare(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        """Side-by-side comparison for the research loop."""
        rows: list[dict[str, Any]] = []
        for experiment_id in experiment_ids:
            try:
                experiment = self.load(experiment_id)
            except FileNotFoundError:
                continue
            row: dict[str, Any] = {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "passed": experiment.passed,
                "seed": experiment.random_seed,
                "feature_version": experiment.feature_version,
            }
            row.update({f"result.{k}": v for k, v in experiment.results.items()
                        if isinstance(v, (int, float, str, bool))})
            rows.append(row)
        return rows

    def __len__(self) -> int:
        return len(list(self.directory.glob("*.json"))) - (1 if self.index_path.exists() else 0)


def _git_commit() -> str:
    """Record the commit so a result can be traced back to the code that made it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
