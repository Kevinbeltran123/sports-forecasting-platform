"""Nested OOF stacking -- ML-01 (D-03a) + D-03b (temporal integrity).

Hand-rolled rather than using sklearn.StackingClassifier because:
  1. StackingClassifier's default cv=5 is stratified KFold -- temporally unsafe.
  2. Even with cv=TimeSeriesSplit, the base-model refit semantics after OOF
     are not transparent; hand-rolling makes every temporal boundary explicit.

Per D-03b: OOF predictions are generated WITHIN each outer fold's train window.
Never generate OOF globally across the full dataset -- that leaks future fixtures.
"""

from __future__ import annotations

import numpy as np
import structlog
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from bip.train.base_models import CB_PARAMS, LGBM_PARAMS, XGB_PARAMS

logger = structlog.get_logger(__name__)

CLASSES: list[int] = [0, 1, 2]  # 0=home win, 1=draw, 2=away win


def _build_base_models() -> list:
    """Instantiate XGB + CB + LGB with Phase 2 default hyperparams (D-03a)."""
    return [
        XGBClassifier(**XGB_PARAMS),
        CatBoostClassifier(**CB_PARAMS),
        LGBMClassifier(**LGBM_PARAMS),
    ]


def _align_proba(model, X: np.ndarray) -> np.ndarray:
    """predict_proba re-aligned to CLASSES = [0,1,2] (pads zero columns if missing).

    Small-fold training windows may miss a class; we fill the missing column
    with zeros rather than crashing on a shape mismatch.
    """
    raw = np.asarray(model.predict_proba(X))
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    # CatBoost can return shape (n, 1, k) -- squeeze to (n, k)
    if raw.ndim == 3:
        raw = raw.reshape(raw.shape[0], -1)
    classes = list(getattr(model, "classes_", CLASSES))
    out = np.zeros((raw.shape[0], len(CLASSES)), dtype=float)
    for i, c in enumerate(classes):
        c_int = int(c)
        if c_int in CLASSES:
            out[:, CLASSES.index(c_int)] = raw[:, i]
    return out


class StackedEnsemble:
    """XGB + CB + LGB base models with LogisticRegression meta-learner.

    Exposes fit_fold() for walk-forward; OOF is generated inside the fold's
    train window via TimeSeriesSplit (D-03b).
    """

    def __init__(self) -> None:
        self._final_base: list = []
        self._meta: LogisticRegression | None = None

    def fit_fold(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_te: np.ndarray,
        dates_tr: np.ndarray,
        n_inner: int = 5,
    ) -> np.ndarray:
        """Return test-fold probability predictions, shape (len(X_te), 3).

        Flow:
          1. Inner TimeSeriesSplit on X_tr produces per-row OOF predictions.
          2. Meta-learner (LogReg) is trained on (OOF, y_tr).
          3. Base models are retrained on the FULL X_tr.
          4. Test predictions = meta.predict_proba(mean base-model probs on X_te).
        """
        oof = np.zeros((len(X_tr), len(CLASSES)))
        inner = TimeSeriesSplit(n_splits=n_inner)
        for inner_train, inner_val in inner.split(X_tr):
            # D-03b: every inner training date strictly < inner val date
            assert dates_tr[inner_train].max() < dates_tr[inner_val].min(), (
                "Inner OOF temporal leakage"
            )
            base = _build_base_models()
            fold_probs = []
            for m in base:
                m.fit(X_tr[inner_train], y_tr[inner_train])
                fold_probs.append(_align_proba(m, X_tr[inner_val]))
            oof[inner_val] = np.mean(fold_probs, axis=0)

        # Train meta-learner on OOF. Discard the first inner-test gap (rows with all zeros).
        mask = oof.sum(axis=1) > 0
        self._meta = LogisticRegression(max_iter=2000)
        self._meta.fit(oof[mask], y_tr[mask])

        # Retrain base models on full train window
        self._final_base = _build_base_models()
        for m in self._final_base:
            m.fit(X_tr, y_tr)

        # Predict test fold
        test_probs = np.mean(
            [_align_proba(m, X_te) for m in self._final_base], axis=0
        )
        test_preds = self._meta.predict_proba(test_probs)

        # Align to CLASSES again (meta may drop a class on tiny folds)
        meta_classes = list(self._meta.classes_)
        aligned = np.zeros((test_preds.shape[0], len(CLASSES)))
        for i, c in enumerate(meta_classes):
            c_int = int(c)
            if c_int in CLASSES:
                aligned[:, CLASSES.index(c_int)] = test_preds[:, i]
        # Renormalize so each row sums to 1 (protects against zero-column edge cases)
        row_sums = aligned.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return aligned / row_sums

    def base_model_packages(self) -> dict[str, str]:
        """Package versions for metadata.json.base_model_packages (ML-04)."""
        import catboost
        import lightgbm
        import sklearn
        import xgboost
        return {
            "xgboost": xgboost.__version__,
            "catboost": catboost.__version__,
            "lightgbm": lightgbm.__version__,
            "sklearn": sklearn.__version__,
        }
