"""Phase-4 shadow kill-criteria — runtime gate for v3.

The full rationale (cohort sizing, statistical defense, RACI, anti-patterns)
lives in internal research notes. This module is the
**executable contract**: numeric thresholds frozen at shadow start and
checked by ``shadow_metrics_report.py`` (weekly) plus the watch.py auto-suspend
filesystem flag.

Three secuential cohortes (A→B→C) measured over settled picks. Each cohort
has hard-stop conditions (auto-suspend) and soft warnings (telegram alert,
no suspend). Greenlight to Tier-D real money requires Cohort C completed
with all §4 criteria met simultaneously.

Public API::

    from bip.evaluation.live.engine_v3.kill_criteria import (
        CohortStage,
        CohortMetrics,
        KillVerdict,
        evaluate_cohort,
    )

    verdict = evaluate_cohort(metrics)
    if verdict.is_hard_stop:
        kill_switch.engage(verdict.reason)

The thresholds here are **frozen at shadow start**. Changing them requires
the operator to commit a refreeze (``docs(engine_v3): refreeze kill criteria,
v<N>``) AND reset cohort accounting from zero. See Papers doc §8.

Versioning: ``KILL_CRITERIA_VERSION`` is bumped on every refreeze. Reports
include this version so analysis windows are attributable to specific
threshold sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

KILL_CRITERIA_VERSION: Final[str] = "1.0"


class CohortStage(StrEnum):
    """Sequential shadow phase cohorts. A → B → C."""

    A = "cohort_a"  # picks 1-100: early-warning
    B = "cohort_b"  # picks 101-300: decision point
    C = "cohort_c"  # picks 301+: greenlight gate


# ──────────────────────────────────────────────────────────────────────
# Frozen thresholds — DO NOT EDIT without refreeze commit + cohort reset.
# See internal research notes §§2-4 for justification of each.
# ──────────────────────────────────────────────────────────────────────

# Cohort sizing (minimum settled picks before advancing to next cohort).
COHORT_A_MIN_SETTLED: Final[int] = 100
COHORT_B_MIN_SETTLED: Final[int] = 300
COHORT_C_MIN_SETTLED: Final[int] = 400  # 100 picks past 300 acumulado

# Hard-stop performance thresholds (per cohort).
HARD_STOP_THRESHOLDS: Final[dict[CohortStage, dict[str, float]]] = {
    CohortStage.A: {
        "wr_floor": 0.40,
        # Cohort A has no ROI hard stop — too noisy at n=100.
    },
    CohortStage.B: {
        "wr_floor": 0.45,
        "roi_flat_floor": -0.10,
    },
    CohortStage.C: {
        "wr_floor": 0.48,
        "roi_flat_floor": -0.05,
    },
}

# Soft-warning thresholds (alert operator, no auto-suspend).
SOFT_WARN_THRESHOLDS: Final[dict[CohortStage, dict[str, tuple[float, float]]]] = {
    CohortStage.A: {
        "wr_band": (0.40, 0.45),
        "roi_flat_band": (-0.10, -0.05),
    },
    CohortStage.B: {
        "wr_band": (0.45, 0.48),
        "roi_flat_band": (-0.10, -0.05),
    },
    CohortStage.C: {
        "wr_band": (0.48, 0.50),
        "roi_flat_band": (-0.05, 0.00),
    },
}

# System-level hard stops (apply at any cohort, any time).
SYSTEM_OOD_DENIAL_RATE_CEILING: Final[float] = 0.25
SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW: Final[int] = 50  # consecutive picks
SYSTEM_DRIFT_SUSTAINED_WINDOW: Final[int] = 50
SYSTEM_PIPELINE_ERROR_RATE_CEILING: Final[int] = 3  # errors per 100 consecutive picks
SYSTEM_SHADOW_LOG_FAILURE_RATE_CEILING: Final[float] = 0.10
SYSTEM_MARKET_CONCENTRATION_CEILING: Final[float] = 0.70

# Greenlight criteria (§4).
GREENLIGHT_CLV_FLOOR: Final[float] = 0.0  # CLV vs Pinnacle > 0%
GREENLIGHT_ROI_FLOOR: Final[float] = 0.0  # ROI flat > 0%
GREENLIGHT_MIN_CONTRIBUTING_MARKETS: Final[int] = 3
GREENLIGHT_MIN_PICKS_PER_CONTRIBUTING_MARKET: Final[int] = 30
GREENLIGHT_DRIFT_QUIET_WINDOW: Final[int] = 200
GREENLIGHT_PATTERN_HIT_BAND: Final[tuple[float, float]] = (0.10, 0.35)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CohortMetrics:
    """Rolling stats for the current shadow cohort.

    All fields are over picks **settled** (outcome known). Pending picks
    do not count toward N.
    """

    stage: CohortStage
    n_settled: int
    n_won: int
    n_lost: int
    n_void: int  # excluded from WR/ROI denominator
    roi_flat: float  # signed fraction, e.g., -0.024 for -2.4%
    roi_kelly: float
    # System-level signals (rolling windows).
    ood_denial_rate_50: float  # last-50 picks rolling
    drift_active_50: bool  # KS test fired in last 50
    pipeline_error_count_100: int  # in last 100
    shadow_log_failure_rate_100: float  # in last 100
    market_concentration_top1_100: float  # share of most-frequent market in last 100
    # Greenlight inputs (only checked when stage == C and N ≥ COHORT_C_MIN_SETTLED).
    avg_clv: float = 0.0
    contributing_markets: list[str] = field(default_factory=list)
    drift_quiet_picks: int = 0  # consecutive picks without drift trigger
    pattern_layer_hit_rate: float = 0.0

    @property
    def wr(self) -> float:
        """Win rate over settled non-void."""
        denom = self.n_won + self.n_lost
        return self.n_won / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class KillVerdict:
    """Outcome of evaluating a cohort against the kill criteria."""

    is_hard_stop: bool
    is_soft_warning: bool
    is_greenlight_ready: bool
    rule_triggered: str = ""
    reason: str = ""

    @classmethod
    def ok(cls) -> KillVerdict:
        return cls(False, False, False)

    @classmethod
    def hard_stop(cls, rule: str, reason: str) -> KillVerdict:
        return cls(True, False, False, rule, reason)

    @classmethod
    def soft_warn(cls, rule: str, reason: str) -> KillVerdict:
        return cls(False, True, False, rule, reason)

    @classmethod
    def greenlight(cls) -> KillVerdict:
        return cls(False, False, True, "greenlight", "All §4 criteria met")


# ──────────────────────────────────────────────────────────────────────
# Evaluation logic
# ──────────────────────────────────────────────────────────────────────


def evaluate_cohort(metrics: CohortMetrics) -> KillVerdict:
    """Evaluate a cohort against frozen kill criteria.

    Returns the **first** matching verdict in priority order: system hard
    stops → performance hard stops → greenlight → soft warnings → ok.

    System hard stops take precedence because they indicate the data itself
    is unreliable (OOD shift, drift, pipeline errors). Performance hard stops
    come next because they assume the data is reliable but the v3 is losing.
    """
    sys_verdict = _check_system_hard_stops(metrics)
    if sys_verdict.is_hard_stop:
        return sys_verdict

    perf_verdict = _check_performance_hard_stops(metrics)
    if perf_verdict.is_hard_stop:
        return perf_verdict

    if metrics.stage == CohortStage.C and metrics.n_settled >= COHORT_C_MIN_SETTLED:
        green = _check_greenlight(metrics)
        if green.is_greenlight_ready:
            return green

    soft = _check_soft_warnings(metrics)
    if soft.is_soft_warning:
        return soft

    return KillVerdict.ok()


def _check_system_hard_stops(metrics: CohortMetrics) -> KillVerdict:
    if metrics.ood_denial_rate_50 > SYSTEM_OOD_DENIAL_RATE_CEILING:
        return KillVerdict.hard_stop(
            "system_ood_denial_rate",
            f"OOD denial rate {metrics.ood_denial_rate_50:.1%} exceeds "
            f"{SYSTEM_OOD_DENIAL_RATE_CEILING:.0%} ceiling over last "
            f"{SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW} picks",
        )
    if metrics.drift_active_50:
        return KillVerdict.hard_stop(
            "system_drift_sustained",
            f"KS drift active in last {SYSTEM_DRIFT_SUSTAINED_WINDOW} picks",
        )
    if metrics.pipeline_error_count_100 >= SYSTEM_PIPELINE_ERROR_RATE_CEILING:
        return KillVerdict.hard_stop(
            "system_pipeline_errors",
            f"Pipeline errors {metrics.pipeline_error_count_100} >= "
            f"{SYSTEM_PIPELINE_ERROR_RATE_CEILING} in last 100 picks",
        )
    if metrics.shadow_log_failure_rate_100 > SYSTEM_SHADOW_LOG_FAILURE_RATE_CEILING:
        return KillVerdict.hard_stop(
            "system_shadow_log_failures",
            f"Shadow log failure rate {metrics.shadow_log_failure_rate_100:.1%} "
            f"exceeds {SYSTEM_SHADOW_LOG_FAILURE_RATE_CEILING:.0%}",
        )
    if metrics.market_concentration_top1_100 > SYSTEM_MARKET_CONCENTRATION_CEILING:
        return KillVerdict.hard_stop(
            "system_market_concentration",
            f"Single market share {metrics.market_concentration_top1_100:.1%} "
            f"exceeds {SYSTEM_MARKET_CONCENTRATION_CEILING:.0%} over last 100 picks",
        )
    return KillVerdict.ok()


def _check_performance_hard_stops(metrics: CohortMetrics) -> KillVerdict:
    threshold = HARD_STOP_THRESHOLDS[metrics.stage]
    min_n = {
        CohortStage.A: COHORT_A_MIN_SETTLED,
        CohortStage.B: COHORT_B_MIN_SETTLED,
        CohortStage.C: COHORT_C_MIN_SETTLED,
    }[metrics.stage]
    if metrics.n_settled < min_n:
        return KillVerdict.ok()

    if metrics.wr < threshold["wr_floor"]:
        return KillVerdict.hard_stop(
            f"{metrics.stage.value}_wr_floor",
            f"WR {metrics.wr:.3f} below floor {threshold['wr_floor']:.2f} "
            f"at n_settled={metrics.n_settled}",
        )
    if "roi_flat_floor" in threshold and metrics.roi_flat < threshold["roi_flat_floor"]:
        return KillVerdict.hard_stop(
            f"{metrics.stage.value}_roi_floor",
            f"ROI flat {metrics.roi_flat:.3f} below floor "
            f"{threshold['roi_flat_floor']:.2f} at n_settled={metrics.n_settled}",
        )
    return KillVerdict.ok()


def _check_soft_warnings(metrics: CohortMetrics) -> KillVerdict:
    bands = SOFT_WARN_THRESHOLDS[metrics.stage]
    min_n = {
        CohortStage.A: COHORT_A_MIN_SETTLED,
        CohortStage.B: COHORT_B_MIN_SETTLED,
        CohortStage.C: COHORT_C_MIN_SETTLED,
    }[metrics.stage]
    if metrics.n_settled < min_n:
        return KillVerdict.ok()

    wr_lo, wr_hi = bands["wr_band"]
    if wr_lo <= metrics.wr < wr_hi:
        return KillVerdict.soft_warn(
            f"{metrics.stage.value}_wr_band",
            f"WR {metrics.wr:.3f} in soft-warn band [{wr_lo:.2f}, {wr_hi:.2f})",
        )

    roi_lo, roi_hi = bands["roi_flat_band"]
    if roi_lo <= metrics.roi_flat < roi_hi:
        return KillVerdict.soft_warn(
            f"{metrics.stage.value}_roi_band",
            f"ROI flat {metrics.roi_flat:.3f} in soft-warn band [{roi_lo:.2f}, {roi_hi:.2f})",
        )
    return KillVerdict.ok()


def _check_greenlight(metrics: CohortMetrics) -> KillVerdict:
    if metrics.roi_flat <= GREENLIGHT_ROI_FLOOR:
        return KillVerdict.ok()
    if metrics.avg_clv <= GREENLIGHT_CLV_FLOOR:
        return KillVerdict.ok()
    if len(metrics.contributing_markets) < GREENLIGHT_MIN_CONTRIBUTING_MARKETS:
        return KillVerdict.ok()
    if metrics.drift_quiet_picks < GREENLIGHT_DRIFT_QUIET_WINDOW:
        return KillVerdict.ok()
    lo, hi = GREENLIGHT_PATTERN_HIT_BAND
    if not (lo <= metrics.pattern_layer_hit_rate <= hi):
        return KillVerdict.ok()
    return KillVerdict.greenlight()
