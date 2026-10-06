"""bip.train -- offline ML training pipeline for football ensembles.

Phase 2: walk-forward stacking with XGBoost/CatBoost/LightGBM
+ LogisticRegression meta-learner (D-03a). Calibration via sklearn 1.8
FrozenEstimator. CLI entry point: python -m bip.train fit|backtest|promote.
"""

from bip.train.backtest import SLIPPAGE_PCT, apply_slippage, compute_clv
from bip.train.calibration import calibrate, select_calibrator
from bip.train.loader import ModelLoader
from bip.train.metadata import ModelMetadata, feature_set_hash
from bip.train.registry import ModelRegistry
from bip.train.stacking import StackedEnsemble
from bip.train.walkforward import WalkForwardSplitter

__all__ = [
    "SLIPPAGE_PCT",
    "apply_slippage",
    "compute_clv",
    "calibrate",
    "select_calibrator",
    "ModelLoader",
    "ModelMetadata",
    "feature_set_hash",
    "ModelRegistry",
    "StackedEnsemble",
    "WalkForwardSplitter",
]
