"""Model families and the calibrated wrapper (REQ 16/17).

REQ 16 says to start with strong tabular models before reaching for deep learning,
so the default roster is linear + trees + boosting. Optional families (LightGBM,
XGBoost, CatBoost) register themselves only if installed, so the platform runs on a
bare scikit-learn environment without pretending a model exists that does not.

Every model is wrapped in `CalibratedModel`, which fits the estimator on the
training fold and the calibrator on the *validation* fold. Calibrating on training
data would produce a calibrator fit to memorized predictions — a subtle leak that
makes probabilities look excellent in-sample and fail live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..core.errors import ModelError
from ..core.logging import get_logger

logger = get_logger(__name__)

ModelFactory = Callable[[int], Any]
_FAMILIES: dict[str, ModelFactory] = {}


def register_family(name: str, factory: ModelFactory) -> None:
    _FAMILIES[name] = factory


def available_families() -> list[str]:
    return sorted(_FAMILIES)


# --- always available (scikit-learn) --------------------------------------- #
def _logistic(seed: int) -> Pipeline:
    # Scaling matters for the linear model and is harmless for the others; it lives
    # inside the pipeline so it is refit per fold and never leaks across a split.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, C=0.5, random_state=seed)),
        ]
    )


def _random_forest(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,          # shallow: financial data overfits deep trees
                    min_samples_leaf=50,  # large leaves resist noise fitting
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state=seed,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def _gradient_boosting(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    min_samples_leaf=40,
                    random_state=seed,
                ),
            ),
        ]
    )


register_family("logistic_regression", _logistic)
register_family("random_forest", _random_forest)
register_family("gradient_boosting", _gradient_boosting)


# --- optional families ------------------------------------------------------ #
try:
    import lightgbm as lgb

    def _lightgbm(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    lgb.LGBMClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=4,
                        min_child_samples=50,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.7,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    register_family("lightgbm", _lightgbm)
except ImportError:  # pragma: no cover - environment dependent
    logger.debug("lightgbm not installed; family unavailable")

try:
    import xgboost as xgb

    def _xgboost(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    xgb.XGBClassifier(
                        n_estimators=400,
                        learning_rate=0.03,
                        max_depth=4,
                        min_child_weight=20,
                        subsample=0.8,
                        colsample_bytree=0.7,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=-1,
                        eval_metric="logloss",
                        tree_method="hist",
                    ),
                ),
            ]
        )

    register_family("xgboost", _xgboost)
except ImportError:  # pragma: no cover
    logger.debug("xgboost not installed; family unavailable")

try:
    from catboost import CatBoostClassifier

    def _catboost(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=400, learning_rate=0.03, depth=4,
                        random_seed=seed, verbose=False, allow_writing_files=False,
                    ),
                ),
            ]
        )

    register_family("catboost", _catboost)
except ImportError:  # pragma: no cover
    logger.debug("catboost not installed; family unavailable")


def build_model(family: str, seed: int = 42) -> Any:
    if family not in _FAMILIES:
        raise ModelError(
            f"unknown or unavailable model family {family!r}; available: {available_families()}"
        )
    return _FAMILIES[family](seed)


# --------------------------------------------------------------------------- #
class CalibratedModel:
    """An estimator plus a probability calibrator fit on held-out data.

    Calibration is not cosmetic here. The risk engine sizes positions from the
    probability and the expected-value gate compares it against costs, so a
    probability that is 15 points optimistic silently doubles real risk.
    """

    def __init__(self, estimator: Any, *, method: str = "isotonic", family: str = "") -> None:
        self.estimator = estimator
        self.method = method
        self.family = family
        self._calibrator: Any = None
        self.feature_names: list[str] = []
        self.is_fitted = False
        self.training_rows = 0
        self.positive_rate = float("nan")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        X_validation: pd.DataFrame | None = None,
        y_validation: pd.Series | None = None,
    ) -> "CalibratedModel":
        if X_train.empty:
            raise ModelError("cannot fit on an empty training set")
        if len(np.unique(y_train)) < 2:
            raise ModelError(
                f"training labels contain a single class ({np.unique(y_train)}); "
                "the label specification or the period needs revisiting"
            )

        self.feature_names = list(X_train.columns)
        self.training_rows = len(X_train)
        self.positive_rate = float(np.mean(y_train))
        self.estimator.fit(X_train.values, y_train.values)

        if (
            self.method != "none"
            and X_validation is not None
            and not X_validation.empty
            and y_validation is not None
            and len(np.unique(y_validation)) > 1
        ):
            raw = self._raw_proba(X_validation)
            if self.method == "isotonic":
                self._calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                self._calibrator.fit(raw, y_validation.values)
            elif self.method == "sigmoid":
                calibrator = LogisticRegression(max_iter=1000)
                logit = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
                calibrator.fit(logit.reshape(-1, 1), y_validation.values)
                self._calibrator = calibrator
        elif self.method != "none":
            logger.warning(
                "%s: no usable validation set; serving uncalibrated probabilities",
                self.family or "model",
            )

        self.is_fitted = True
        return self

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.feature_names].values if self.feature_names else X.values
        proba = self.estimator.predict_proba(values)
        return np.asarray(proba)[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ModelError("model is not fitted")
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            # Serving a model features it was not trained on is exactly the failure
            # the feature_version check exists to prevent; fail loudly.
            raise ModelError(f"input is missing {len(missing)} trained features: {sorted(missing)[:5]}")

        raw = self._raw_proba(X)
        if self._calibrator is None:
            return np.clip(raw, 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(self._calibrator.predict(raw), 0.0, 1.0)
        clipped = np.clip(raw, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped))
        return np.clip(self._calibrator.predict_proba(logit.reshape(-1, 1))[:, 1], 0.0, 1.0)

    def predict_one(self, row: pd.Series) -> float:
        frame = pd.DataFrame([row])
        return float(self.predict_proba(frame)[0])

    def feature_importance(self) -> pd.Series:
        """Importance where the estimator exposes it; empty otherwise."""
        model = self.estimator
        if isinstance(model, Pipeline):
            model = model.named_steps.get("model", model)
        values = None
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            values = np.abs(np.asarray(model.coef_).ravel())
        if values is None or len(values) != len(self.feature_names):
            return pd.Series(dtype=float)
        return pd.Series(values, index=self.feature_names).sort_values(ascending=False)

    def __repr__(self) -> str:
        return (
            f"CalibratedModel(family={self.family!r}, method={self.method!r}, "
            f"fitted={self.is_fitted}, rows={self.training_rows})"
        )


# --------------------------------------------------------------------------- #
@dataclass
class ModelEnsemble:
    """Meta-layer combining several calibrated models (REQ 17).

    Weights come from validation performance rather than being hand-set, and are
    shrunk toward equal weighting for the same reason strategy allocation is:
    validation folds are short, and trusting them fully is overfitting.
    """

    models: dict[str, CalibratedModel] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    shrinkage: float = 0.4

    def add(self, name: str, model: CalibratedModel, weight: float = 1.0) -> None:
        self.models[name] = model
        self.weights[name] = max(0.0, weight)

    def set_weights_from_scores(self, scores: Mapping[str, float]) -> None:
        """Higher score = better. Scores below zero are excluded outright."""
        positive = {k: v for k, v in scores.items() if v > 0 and k in self.models}
        if not positive:
            equal = 1.0 / max(1, len(self.models))
            self.weights = {k: equal for k in self.models}
            return
        total = sum(positive.values())
        raw = {k: v / total for k, v in positive.items()}
        equal = 1.0 / len(raw)
        self.weights = {
            k: self.shrinkage * equal + (1 - self.shrinkage) * v for k, v in raw.items()
        }
        for name in self.models:
            self.weights.setdefault(name, 0.0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        active = {k: w for k, w in self.weights.items() if w > 0 and k in self.models}
        if not active:
            raise ModelError("ensemble has no active models")
        total = sum(active.values())
        stacked = np.zeros(len(X))
        for name, weight in active.items():
            stacked += (weight / total) * self.models[name].predict_proba(X)
        return np.clip(stacked, 0.0, 1.0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (probability, disagreement).

        Disagreement is the weighted standard deviation across member models. It is
        the platform's measure of *epistemic* uncertainty — models agreeing at 0.62
        is a very different situation from models split between 0.35 and 0.85, and
        REQ 17 requires uncertainty in the output.
        """
        active = {k: w for k, w in self.weights.items() if w > 0 and k in self.models}
        if not active:
            raise ModelError("ensemble has no active models")
        predictions = np.vstack([self.models[name].predict_proba(X) for name in active])
        weights = np.array([active[name] for name in active], dtype=float)
        weights = weights / weights.sum()
        mean = np.average(predictions, axis=0, weights=weights)
        variance = np.average((predictions - mean) ** 2, axis=0, weights=weights)
        return np.clip(mean, 0.0, 1.0), np.sqrt(variance)

    @property
    def is_ready(self) -> bool:
        return any(w > 0 and self.models[n].is_fitted for n, w in self.weights.items() if n in self.models)

    def describe(self) -> dict[str, float]:
        return {name: round(weight, 4) for name, weight in sorted(self.weights.items())}
