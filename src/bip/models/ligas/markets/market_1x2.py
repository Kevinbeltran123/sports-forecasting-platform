"""Market1x2 — wrapper around StackedEnsemble for the 1x2 market.

CLASSES (legacy convention from train/stacking.py):
  0 = home win
  1 = draw
  2 = away win

The wrapper presents a clean .fit(X, y, dates) + .predict_proba(X) → dict
interface, hiding the per-fold OOF machinery. Walk-forward driver (Ola C)
calls fit_fold directly when it needs leakage-safe OOF — this wrapper
exposes the simpler full-fit path used by LigasModel.predict() at inference.
"""

from __future__ import annotations

import numpy as np
import structlog
from sklearn.base import BaseEstimator, ClassifierMixin

from bip.train.calibration import calibrate
from bip.train.stacking import CLASSES, StackedEnsemble, _build_base_models

log = structlog.get_logger(__name__)

# Selection key for each class index
_CLASS_TO_SELECTION: dict[int, str] = {0: "home", 1: "draw", 2: "away"}


class Market1x2:
    """1x2 ensemble: XGB + CB + LGBM stacked + Platt/Isotonic calibrator."""

    market_id: str = "1x2"

    def __init__(self) -> None:
        self._ensemble = StackedEnsemble()
        self._calibrator = None  # set after fit()
        self._fitted: bool = False
        # We keep a copy of base models for the simple full-fit path so
        # predict_proba doesn't depend on the OOF state of fit_fold.
        self._base_models: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Market1x2":
        """Fit base ensemble + calibrator on the full training window.

        For walk-forward CV use StackedEnsemble.fit_fold directly via
        the backtest harness (Ola C). This `fit` is the offline/global
        training entrypoint used by LigasModel.train().
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}")

        self._base_models = _build_base_models()
        for m in self._base_models:
            m.fit(X, y)

        # Average base-model probas → calibrator input
        proba_inputs = self._mean_base_proba(X)
        # Build a frozen "ensemble" we can calibrate. We use a tiny adapter
        # so calibrate() can call .predict_proba on us.
        adapter = _StackedProbaAdapter(self._base_models)
        self._calibrator = calibrate(adapter, X, y)
        self._fitted = True
        log.info(
            "market_1x2_fitted",
            n_samples=len(X),
            n_features=X.shape[1],
            classes=CLASSES,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities, shape (n, 3)."""
        if not self._fitted:
            raise RuntimeError("Market1x2 must be fit() before predict_proba")
        return self._calibrator.predict_proba(np.asarray(X))

    def predict_for_fixture(self, x: np.ndarray) -> dict[str, float]:
        """Return {selection: p} for one fixture, summing to ~1.0."""
        x = np.asarray(x).reshape(1, -1)
        proba = self.predict_proba(x)[0]
        return {_CLASS_TO_SELECTION[i]: float(proba[i]) for i in range(len(CLASSES))}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mean_base_proba(self, X: np.ndarray) -> np.ndarray:
        from bip.train.stacking import _align_proba

        return np.mean([_align_proba(m, X) for m in self._base_models], axis=0)


class _StackedProbaAdapter(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-compatible adapter exposing predict_proba + classes_.

    Lets us pass the (fitted) mean-of-base-models into calibrate() without
    requiring StackedEnsemble.fit_fold to have run. ClassifierMixin +
    BaseEstimator inheritance gives sklearn 1.8 the __sklearn_tags__
    metadata it needs to route this through CalibratedClassifierCV.
    """

    def __init__(self, base_models: list | None = None) -> None:
        self._base = base_models if base_models is not None else []
        self.classes_ = np.array(CLASSES)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from bip.train.stacking import _align_proba

        return np.mean([_align_proba(m, X) for m in self._base], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn 1.8 CalibratedClassifierCV requires predict() on the
        # underlying estimator even when wrapped in FrozenEstimator.
        return self.predict_proba(X).argmax(axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StackedProbaAdapter":  # noqa: ARG002
        # No-op — base models are already fit. sklearn's CalibratedClassifierCV
        # with FrozenEstimator does not call fit, but the protocol expects it.
        return self


__all__ = ["Market1x2"]
