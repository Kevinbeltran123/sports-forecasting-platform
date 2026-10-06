"""MarketOverUnder — binary classifier wrapper for Over/Under N.5 goals.

Phase 1 default is OU 2.5 (spec §4.8). The wrapper accepts a `line`
parameter so OU 1.5 / 3.5 etc. can plug in identically — the y label is
just an indicator over the line.

Binary classes:
  0 = under N.5
  1 = over N.5
"""

from __future__ import annotations

import numpy as np
import structlog
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression  # noqa: F401 — kept for parity with 1x2 stack
from xgboost import XGBClassifier

from bip.train.base_models import CB_PARAMS, LGBM_PARAMS, XGB_PARAMS
from bip.train.calibration import select_calibrator, _to_sklearn_method

log = structlog.get_logger(__name__)


def _binary_base_params(params: dict, key: str) -> dict:
    """Strip multiclass-specific keys for binary fitting."""
    out = dict(params)
    if key == "xgb":
        out["objective"] = "binary:logistic"
        out["eval_metric"] = "logloss"
        out.pop("num_class", None)
    elif key == "cb":
        out["loss_function"] = "Logloss"
    elif key == "lgbm":
        out["objective"] = "binary"
        out["metric"] = "binary_logloss"
        out.pop("num_class", None)
    return out


def _build_binary_base_models() -> list:
    return [
        XGBClassifier(**_binary_base_params(XGB_PARAMS, "xgb")),
        CatBoostClassifier(**_binary_base_params(CB_PARAMS, "cb")),
        LGBMClassifier(**_binary_base_params(LGBM_PARAMS, "lgbm")),
    ]


class MarketOverUnder:
    """Binary OU classifier with stacked ensemble + Platt/Isotonic calibration."""

    market_id_prefix: str = "ou_"

    def __init__(self, line: float = 2.5) -> None:
        if line <= 0 or line >= 10:
            raise ValueError(f"Implausible OU line: {line}")
        self.line = float(line)
        self.market_id = f"ou_{line:g}"
        self._base_models: list = []
        self._calibrator: CalibratedClassifierCV | None = None
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y_binary: np.ndarray) -> "MarketOverUnder":
        """Fit on binary y where 1 = over the line, 0 = under."""
        X = np.asarray(X)
        y = np.asarray(y_binary).astype(int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}")
        if set(np.unique(y).tolist()) - {0, 1}:
            raise ValueError("y_binary must be in {0, 1}")

        self._base_models = _build_binary_base_models()
        for m in self._base_models:
            m.fit(X, y)

        adapter = _BinaryAdapter(self._base_models)
        method = _to_sklearn_method(select_calibrator(len(y)))
        cal = CalibratedClassifierCV(
            estimator=FrozenEstimator(adapter),
            method=method,
            cv=None,
        )
        cal.fit(X, y)
        self._calibrator = cal
        self._fitted = True
        log.info(
            "market_ou_fitted",
            line=self.line,
            n_samples=len(X),
            n_features=X.shape[1],
            cal_method=method,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (n, 2) — column 0 = P(under), 1 = P(over)."""
        if not self._fitted:
            raise RuntimeError("MarketOverUnder must be fit() before predict_proba")
        return self._calibrator.predict_proba(np.asarray(X))

    def predict_for_fixture(self, x: np.ndarray) -> dict[str, float]:
        """Return {'under': p_under, 'over': p_over}."""
        proba = self.predict_proba(np.asarray(x).reshape(1, -1))[0]
        return {"under": float(proba[0]), "over": float(proba[1])}


class _BinaryAdapter(ClassifierMixin, BaseEstimator):
    """sklearn-compatible adapter that averages base predict_proba (binary).

    Inherits BaseEstimator + ClassifierMixin so sklearn 1.8 can read
    __sklearn_tags__ via CalibratedClassifierCV's validation path.
    """

    def __init__(self, base_models: list | None = None) -> None:
        self._base = base_models if base_models is not None else []
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas: list[np.ndarray] = []
        for m in self._base:
            raw = np.asarray(m.predict_proba(X))
            if raw.ndim == 3:
                raw = raw.reshape(raw.shape[0], -1)
            # Align to [0, 1] columns; pad zero column if model only saw one class
            classes = list(getattr(m, "classes_", [0, 1]))
            aligned = np.zeros((raw.shape[0], 2))
            for i, c in enumerate(classes):
                aligned[:, int(c)] = raw[:, i]
            probas.append(aligned)
        return np.mean(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn 1.8 needs predict() even with FrozenEstimator
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BinaryAdapter":  # noqa: ARG002
        return self


__all__ = ["MarketOverUnder"]
