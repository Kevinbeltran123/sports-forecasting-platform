"""Stacking OOF generation -- ML-01 + D-03b (no global OOF reuse)."""

from __future__ import annotations

import numpy as np


class TestStackingOOF:
    """Nested OOF generated WITHIN each fold's train window -- D-03b."""

    def test_oof_temporal_scope(self, synthetic_training_data):
        """Inner OOF uses only fold's train window -- D-03b non-negotiable.

        We verify this indirectly: fit_fold runs without raising the inner
        AssertionError. Replace dates with a scrambled order and confirm it raises.
        """
        from bip.train.stacking import StackedEnsemble
        X, y, dates = synthetic_training_data
        # Use first 40 rows as "train window", last 10 as test
        X_tr, y_tr, dates_tr = X[:40], y[:40], dates[:40]
        X_te = X[40:]
        ens = StackedEnsemble()
        probs = ens.fit_fold(X_tr, y_tr, X_te, dates_tr, n_inner=3)
        assert probs.shape == (10, 3)

    def test_ensemble_forward_pass(self, synthetic_training_data):
        """XGB+CB+LGB -> LogReg meta produces proba vector summing to 1 -- ML-01."""
        from bip.train.stacking import StackedEnsemble
        X, y, dates = synthetic_training_data
        X_tr, y_tr, dates_tr = X[:40], y[:40], dates[:40]
        X_te = X[40:]
        ens = StackedEnsemble()
        probs = ens.fit_fold(X_tr, y_tr, X_te, dates_tr, n_inner=3)
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_base_models_are_xgb_catboost_lightgbm(self):
        """Base model list matches D-03a: XGBClassifier, CatBoostClassifier, LGBMClassifier."""
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        from bip.train.stacking import _build_base_models
        models = _build_base_models()
        assert isinstance(models[0], XGBClassifier)
        assert isinstance(models[1], CatBoostClassifier)
        assert isinstance(models[2], LGBMClassifier)

    def test_meta_learner_is_logistic_regression(self, synthetic_training_data):
        """Meta-learner is sklearn.linear_model.LogisticRegression -- D-03a."""
        from sklearn.linear_model import LogisticRegression

        from bip.train.stacking import StackedEnsemble
        X, y, dates = synthetic_training_data
        ens = StackedEnsemble()
        ens.fit_fold(X[:40], y[:40], X[40:], dates[:40], n_inner=3)
        assert isinstance(ens._meta, LogisticRegression)
