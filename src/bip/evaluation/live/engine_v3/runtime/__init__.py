"""V3 runtime adapters — wiring into the production loop.

Phase-4 deliverables that live alongside the engine but are NOT part
of the pure pipeline: the dual-write runtime (T1.3) that bridges
watch.py to V3Pipeline + ShadowLogger without leaking any side effect
into v2's Telegram path.
"""

from bip.evaluation.live.engine_v3.runtime.cohort_accountant import (
    AlertSink,
    CohortEvalResult,
    DEFAULT_REPORT_ROOT,
    assign_cohort_stage,
    build_cohort_metrics,
    engage_kill_switch,
    join_picks_with_outcomes,
    load_v3_denials,
    load_v3_outcomes,
    load_v3_picks,
    notify_operator,
    run_cohort_eval,
    write_cohort_report,
)
from bip.evaluation.live.engine_v3.runtime.cohort_scheduler import (
    CohortJob,
    register_cohort_job,
)
from bip.evaluation.live.engine_v3.runtime.dual_write import (
    DEFAULT_KILL_SWITCH_PATH,
    DEFAULT_V3_TIMEOUT_SEC,
    DualWriteRuntime,
    is_v3_kill_switch_engaged,
    is_v3_shadow_enabled,
)
from bip.evaluation.live.engine_v3.runtime.live_observer import (
    FileObserver,
    LivePickObserver,
    NullObserver,
    StdoutObserver,
    build_observer_from_env,
    format_v3_pick,
)
from bip.evaluation.live.engine_v3.runtime.refit_scheduler import (
    RefitJob,
    register_refit_job,
)
from bip.evaluation.live.engine_v3.runtime.training_data import (
    JoinReport,
    RealPatternPair,
    build_real_pattern_pairs,
    pairs_to_fit_input,
)

__all__ = [
    "AlertSink",
    "CohortEvalResult",
    "CohortJob",
    "DEFAULT_KILL_SWITCH_PATH",
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_V3_TIMEOUT_SEC",
    "DualWriteRuntime",
    "FileObserver",
    "JoinReport",
    "LivePickObserver",
    "NullObserver",
    "RealPatternPair",
    "RefitJob",
    "StdoutObserver",
    "assign_cohort_stage",
    "build_observer_from_env",
    "build_real_pattern_pairs",
    "format_v3_pick",
    "build_cohort_metrics",
    "engage_kill_switch",
    "is_v3_kill_switch_engaged",
    "is_v3_shadow_enabled",
    "join_picks_with_outcomes",
    "load_v3_denials",
    "load_v3_outcomes",
    "load_v3_picks",
    "notify_operator",
    "pairs_to_fit_input",
    "register_cohort_job",
    "register_refit_job",
    "run_cohort_eval",
    "write_cohort_report",
]
