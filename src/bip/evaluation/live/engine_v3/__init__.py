"""Live Engine v3 — causal-reasoning pipeline.

See internal research notes for the architectural blueprint.

End-to-end surface: GSV primitives, hypothesis layer (12 archetypes),
MES routing, family-specific conditional predictors, 8-rule No-Bet
Gate, line recorder, shadow logger, and the composing ``V3Pipeline``.

Usage::

    from bip.evaluation.live.engine_v3 import V3Pipeline, PreMatchPriors

    pipeline = V3Pipeline()
    out = pipeline.run(live_state, priors=priors, markets=markets)
    for pick in out.allowed_picks:
        log.info("v3 shadow pick: %s", pick)
"""

from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.calibrator import (
    CalibrationSample,
    IsotonicCalibrator,
    map_v2_market_to_family,
    minute_bucket,
)
from bip.evaluation.live.engine_v3.conditional_predictor import (
    BTTSPredictor,
    ConditionalPredictor,
    CornersPredictor,
    Goals2HPredictor,
    PredictionPoint,
)
from bip.evaluation.live.engine_v3.drift_monitor import (
    CalibrationDriftMonitor,
    DriftStatus,
)
from bip.evaluation.live.engine_v3.next_goal_predictor import (
    NextGoalPredictor,
)
from bip.evaluation.live.engine_v3.gsv import (
    CardsState,
    CornerState,
    CriticalEvent,
    FlowState,
    GameStateVector,
    MarketLine,
    MarketSnapshot,
    NumericalState,
    PreMatchPriors,
    RosterState,
    ScoreState,
    TacticalState,
    TimeState,
    XGState,
)
from bip.evaluation.live.engine_v3.gsv_builder import GSVBuilder
from bip.evaluation.live.engine_v3.kill_criteria import (
    KILL_CRITERIA_VERSION,
    CohortMetrics,
    CohortStage,
    KillVerdict,
    evaluate_cohort,
)
from bip.evaluation.live.engine_v3.line_recorder import LineRecorder
from bip.evaluation.live.engine_v3.market_selector import (
    MarketCandidate,
    family_for_market_id,
    select_markets,
)
from bip.evaluation.live.engine_v3.mes import MESResult, compute_mes
from bip.evaluation.live.engine_v3.mispricing_window import (
    MispricingWindowConfig,
    WindowLabel,
    WindowResult,
)
from bip.evaluation.live.engine_v3.mispricing_window import (
    classify as classify_window,
)
from bip.evaluation.live.engine_v3.mispricing_window import (
    classify_gsv as classify_window_gsv,
)
from bip.evaluation.live.engine_v3.no_bet_gate import (
    GateResult,
    NoBetVerdict,
    allowed_candidates,
    run_gate,
)
from bip.evaluation.live.engine_v3.ood_detector import OODDetector, vectorize_gsv
from bip.evaluation.live.engine_v3.pattern_layer import (
    PatternLayer,
    PatternLayerConfig,
    PatternRecord,
    generate_theses_hybrid,
    merge_theses,
)
from bip.evaluation.live.engine_v3.pipeline import (
    PipelineOutput,
    ShadowPick,
    V3Pipeline,
)
from bip.evaluation.live.engine_v3.shadow_logger import (
    LoadFailure,
    ShadowLogger,
    load_shadow_gsvs,
)
from bip.evaluation.live.engine_v3.thesis import (
    CausalChain,
    CausalStep,
    ConditionalShift,
    GSVPredicate,
    InvalidationTrigger,
    MarketFamily,
    Thesis,
    ThesisArchetype,
    ThesisSource,
    TimeWindow,
)

__all__ = [
    "BTTSPredictor",
    "CalibrationDriftMonitor",
    "CalibrationSample",
    "CardsState",
    "CausalChain",
    "CausalStep",
    "CohortMetrics",
    "CohortStage",
    "ConditionalPredictor",
    "ConditionalShift",
    "CornerState",
    "CornersPredictor",
    "CriticalEvent",
    "DriftStatus",
    "IsotonicCalibrator",
    "NextGoalPredictor",
    "FlowState",
    "GSVBuilder",
    "GSVPredicate",
    "GameStateVector",
    "GateResult",
    "Goals2HPredictor",
    "InvalidationTrigger",
    "KILL_CRITERIA_VERSION",
    "KillVerdict",
    "LineRecorder",
    "LoadFailure",
    "MESResult",
    "MarketCandidate",
    "MarketFamily",
    "MarketLine",
    "MarketSnapshot",
    "MispricingWindowConfig",
    "NoBetVerdict",
    "NumericalState",
    "OODDetector",
    "PatternLayer",
    "PatternLayerConfig",
    "PatternRecord",
    "PipelineOutput",
    "PreMatchPriors",
    "PredictionPoint",
    "RosterState",
    "ScoreState",
    "ShadowLogger",
    "ShadowPick",
    "TacticalState",
    "Thesis",
    "ThesisArchetype",
    "ThesisSource",
    "TimeState",
    "TimeWindow",
    "V3Pipeline",
    "WindowLabel",
    "WindowResult",
    "XGState",
    "allowed_candidates",
    "classify_window",
    "classify_window_gsv",
    "compute_mes",
    "evaluate_cohort",
    "family_for_market_id",
    "generate_theses",
    "generate_theses_hybrid",
    "load_shadow_gsvs",
    "map_v2_market_to_family",
    "merge_theses",
    "minute_bucket",
    "run_gate",
    "select_markets",
    "vectorize_gsv",
]
