"""Base-model hyperparameter defaults for the Phase 2 ensemble.

Sources:
  - XGBoost 3.x: tree_method='hist', device='cpu' (CPU-only by design)
  - CatBoost 1.2.x: task_type='CPU'
  - LightGBM 4.6: verbose=-1 suppresses noisy stdout

Tune only if walk-forward CLV is below baseline (RESEARCH.md).
"""

from __future__ import annotations

XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}

CB_PARAMS: dict = {
    "iterations": 500,
    "depth": 4,
    "learning_rate": 0.05,
    "loss_function": "MultiClass",
    "task_type": "CPU",
    "verbose": 0,
    "random_seed": 42,
}

LGBM_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "verbose": -1,
    "random_state": 42,
}
