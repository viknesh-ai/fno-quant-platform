"""Model registry (REQ 51).

Every trained model gets a unique ID, a recorded lineage and a deployment status.
The status ladder RESEARCH -> VALIDATION -> PAPER -> ACTIVE -> RETIRED is enforced:
a model cannot jump straight from RESEARCH to ACTIVE, because the whole point of the
ladder is that a model earns live capital by first surviving paper trading.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import joblib

from ..core.errors import ModelError
from ..core.ids import content_hash
from ..core.logging import get_logger
from ..features.engine import FEATURE_VERSION

logger = get_logger(__name__)


class ModelStatus(str, Enum):
    RESEARCH = "RESEARCH"
    VALIDATION = "VALIDATION"
    PAPER = "PAPER"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


# A model may only move along this ladder one rung at a time (or be retired).
_ALLOWED_TRANSITIONS: dict[ModelStatus, set[ModelStatus]] = {
    ModelStatus.RESEARCH: {ModelStatus.VALIDATION, ModelStatus.RETIRED},
    ModelStatus.VALIDATION: {ModelStatus.PAPER, ModelStatus.RESEARCH, ModelStatus.RETIRED},
    ModelStatus.PAPER: {ModelStatus.ACTIVE, ModelStatus.VALIDATION, ModelStatus.RETIRED},
    ModelStatus.ACTIVE: {ModelStatus.PAPER, ModelStatus.RETIRED},
    ModelStatus.RETIRED: {ModelStatus.RESEARCH},
}


@dataclass
class ModelRecord:
    """The metadata REQ 51 requires for every trained model."""

    model_id: str
    family: str
    status: ModelStatus = ModelStatus.RESEARCH
    training_date: str = ""
    feature_version: str = FEATURE_VERSION
    dataset_description: dict[str, Any] = field(default_factory=dict)
    label_spec: str = ""
    validation_results: dict[str, Any] = field(default_factory=dict)
    out_of_sample_results: dict[str, Any] = field(default_factory=dict)
    calibration_metrics: dict[str, Any] = field(default_factory=dict)
    trading_metrics: dict[str, Any] = field(default_factory=dict)
    walk_forward_summary: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    random_seed: int = 42
    verdict: str = ""
    passed: bool = False
    artifact_path: str = ""
    promoted_at: str = ""
    retired_reason: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        data = dict(data)
        data["status"] = ModelStatus(data.get("status", "RESEARCH"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class ModelRegistry:
    """File-backed registry. Deliberately simple and inspectable — the index is a
    JSON file an operator can read without the application running."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "registry.json"
        self._records: dict[str, ModelRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            raw = json.loads(self.index_path.read_text())
        except json.JSONDecodeError as exc:
            logger.error("model registry index is corrupt: %s", exc)
            return
        self._records = {
            model_id: ModelRecord.from_dict(data) for model_id, data in raw.items()
        }

    def _save(self) -> None:
        payload = {model_id: record.to_dict() for model_id, record in self._records.items()}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.index_path)  # atomic: a crash mid-write cannot corrupt the index

    # ------------------------------------------------------------------ #
    def register(
        self,
        model: Any,
        *,
        family: str,
        dataset_description: dict[str, Any],
        label_spec: str = "",
        walk_forward_summary: dict[str, Any] | None = None,
        out_of_sample_results: dict[str, Any] | None = None,
        calibration_metrics: dict[str, Any] | None = None,
        trading_metrics: dict[str, Any] | None = None,
        feature_importance: dict[str, float] | None = None,
        config_snapshot: dict[str, Any] | None = None,
        random_seed: int = 42,
        verdict: str = "",
        passed: bool = False,
        notes: str = "",
    ) -> ModelRecord:
        timestamp = datetime.now()
        # The ID encodes what the model *is*, so two runs on the same data and config
        # collide rather than silently proliferating near-identical models.
        seed_string = json.dumps(
            {"family": family, "dataset": dataset_description, "label": label_spec, "seed": random_seed},
            sort_keys=True, default=str,
        )
        model_id = f"{family}_{timestamp:%Y%m%d_%H%M%S}_{content_hash(seed_string)[:8]}"

        artifact_path = self.directory / f"{model_id}.joblib"
        joblib.dump(model, artifact_path)

        record = ModelRecord(
            model_id=model_id,
            family=family,
            status=ModelStatus.RESEARCH,
            training_date=timestamp.isoformat(),
            feature_version=FEATURE_VERSION,
            dataset_description=dataset_description,
            label_spec=label_spec,
            walk_forward_summary=walk_forward_summary or {},
            out_of_sample_results=out_of_sample_results or {},
            calibration_metrics=calibration_metrics or {},
            trading_metrics=trading_metrics or {},
            feature_importance=feature_importance or {},
            config_snapshot=config_snapshot or {},
            random_seed=random_seed,
            verdict=verdict,
            passed=passed,
            artifact_path=str(artifact_path),
            notes=notes,
        )
        self._records[model_id] = record
        self._save()
        logger.info("registered model %s (%s)", model_id, verdict or "no verdict")
        return record

    # ------------------------------------------------------------------ #
    def promote(self, model_id: str, status: ModelStatus, *, force: bool = False) -> ModelRecord:
        """Move a model along the deployment ladder, enforcing the transition rules."""
        record = self.get(model_id)
        current = record.status
        if status is current:
            return record
        if not force and status not in _ALLOWED_TRANSITIONS[current]:
            raise ModelError(
                f"cannot move model {model_id} from {current.value} to {status.value}; "
                f"allowed next states are {sorted(s.value for s in _ALLOWED_TRANSITIONS[current])}. "
                "A model must survive paper trading before it can go ACTIVE."
            )
        if status is ModelStatus.ACTIVE and not record.passed and not force:
            raise ModelError(
                f"refusing to activate model {model_id}: its walk-forward verdict was "
                f"{record.verdict!r}. REQ 67.17 forbids claiming an edge without "
                "out-of-sample evidence."
            )

        # Only one ACTIVE model per family — otherwise it is ambiguous which one
        # produced a given live decision.
        if status is ModelStatus.ACTIVE:
            for other in self._records.values():
                if (
                    other.model_id != model_id
                    and other.family == record.family
                    and other.status is ModelStatus.ACTIVE
                ):
                    other.status = ModelStatus.RETIRED
                    other.retired_reason = f"superseded by {model_id}"
                    logger.info("retired %s in favour of %s", other.model_id, model_id)

        record.status = status
        record.promoted_at = datetime.now().isoformat()
        self._save()
        logger.warning("model %s: %s -> %s", model_id, current.value, status.value)
        return record

    def retire(self, model_id: str, reason: str = "") -> ModelRecord:
        record = self.get(model_id)
        record.status = ModelStatus.RETIRED
        record.retired_reason = reason
        record.promoted_at = datetime.now().isoformat()
        self._save()
        return record

    # ------------------------------------------------------------------ #
    def get(self, model_id: str) -> ModelRecord:
        if model_id not in self._records:
            raise ModelError(f"unknown model id {model_id!r}")
        return self._records[model_id]

    def load(self, model_id: str) -> Any:
        record = self.get(model_id)
        path = Path(record.artifact_path)
        if not path.exists():
            raise ModelError(f"model artifact missing for {model_id}: {path}")
        model = joblib.load(path)
        # A model trained on a different feature version would silently receive
        # columns it never saw. Refuse rather than serve nonsense (REQ 18).
        if record.feature_version != FEATURE_VERSION:
            raise ModelError(
                f"model {model_id} was trained on feature version {record.feature_version} "
                f"but the running system produces {FEATURE_VERSION}; retrain before use"
            )
        return model

    def active(self, family: str | None = None) -> ModelRecord | None:
        candidates = [
            r for r in self._records.values()
            if r.status is ModelStatus.ACTIVE and (family is None or r.family == family)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.training_date)

    def by_status(self, status: ModelStatus) -> list[ModelRecord]:
        return sorted(
            (r for r in self._records.values() if r.status is status),
            key=lambda r: r.training_date,
            reverse=True,
        )

    def all(self) -> list[ModelRecord]:
        return sorted(self._records.values(), key=lambda r: r.training_date, reverse=True)

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": r.model_id,
                "family": r.family,
                "status": r.status.value,
                "trained": r.training_date[:19],
                "passed": r.passed,
                "verdict": r.verdict[:80],
            }
            for r in self.all()
        ]

    def __len__(self) -> int:
        return len(self._records)
