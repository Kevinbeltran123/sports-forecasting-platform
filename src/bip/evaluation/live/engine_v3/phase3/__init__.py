"""Live Engine v3 — Phase 3 expansion.

Predictors:
- ``cards_predictor.CardsPredictor``: Poisson conditional on ref + state.
- ``next_goal_predictor.NextGoalPredictor``: hazard-point with state covariates.
- ``props_predictor.PlayerPropsPredictor``: shrinkage + adjusters, shadow-mode.

Calibration stack (sec 5):
- ``regimes.bucket_gsv``: 25-30 archetypal regime keys.
- ``calibration.TimeBucketedIsotonic``: per-(regime, family, minute_bucket) isotonic.
- ``calibration.transfer_calibration``: cross-family cold-start.
- ``drift_detector.CalibrationDriftDetector``: KS rolling 200 + suspension.

Promotion layer (sec 10):
- ``confidence_modulator.modulate``: post-detection Kelly modulator.
- ``stake_policy.StakeRotationPolicy``: concentration cap + new-market ramp.
- ``tier_promoter.TierDPromoter``: synthesises operator-actionable Tier-D picks.

See internal research notes sec 10 Phase 3.
"""
from bip.evaluation.live.engine_v3.phase3.calibration import (
    IsotonicCell,
    TimeBucketedIsotonic,
    transfer_calibration,
)
from bip.evaluation.live.engine_v3.phase3.cards_predictor import CardsPredictor
from bip.evaluation.live.engine_v3.phase3.confidence_modulator import (
    ConfidenceBreakdown,
    modulate,
)
from bip.evaluation.live.engine_v3.phase3.drift_detector import (
    CalibrationDriftDetector,
    DriftReport,
)
from bip.evaluation.live.engine_v3.phase3.next_goal_predictor import (
    NextGoalPredictor,
)
from bip.evaluation.live.engine_v3.phase3.props_predictor import (
    PlayerPrior,
    PlayerPropsPredictor,
)
from bip.evaluation.live.engine_v3.phase3.regimes import (
    RegimeKey,
    all_minute_buckets,
    bucket_gsv,
    minute_bucket,
)
from bip.evaluation.live.engine_v3.phase3.stake_policy import (
    StakeDecision,
    StakeRotationPolicy,
)
from bip.evaluation.live.engine_v3.phase3.tier_promoter import (
    PromotedPick,
    Stage,
    TierDPromoter,
    record_outcome,
)

__all__ = [
    "CalibrationDriftDetector",
    "CardsPredictor",
    "ConfidenceBreakdown",
    "DriftReport",
    "IsotonicCell",
    "NextGoalPredictor",
    "PlayerPrior",
    "PlayerPropsPredictor",
    "PromotedPick",
    "RegimeKey",
    "Stage",
    "StakeDecision",
    "StakeRotationPolicy",
    "TierDPromoter",
    "TimeBucketedIsotonic",
    "all_minute_buckets",
    "bucket_gsv",
    "minute_bucket",
    "modulate",
    "record_outcome",
    "transfer_calibration",
]
