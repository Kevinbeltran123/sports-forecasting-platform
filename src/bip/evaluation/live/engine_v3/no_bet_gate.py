"""No-Bet Policy Gate — the 8 hard rules from section 6.

Each rule is a pure ``(candidate, gsv) → NoBetVerdict`` function. The
gate runs them in order; the first failure aborts the candidate with a
reason string. Reasons are persisted to the audit log per sec 7.2.

The most load-bearing rule is **#2 — the Napoli rule**. It is the one
that codifies the structural insight from the design doc: a
``dominant_losing=True`` state is INCOMPATIBLE with Under-direction
theses except via explicit allow-list (``cruise_mode``). This is the
asymmetric guard the current system lacks.

Rules are AND-composed — any failure → no pick. Each failure is logged
with the rule number, the reason, and the (thesis, market) it killed.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.market_selector import MarketCandidate
from bip.evaluation.live.engine_v3.mispricing_window import (
    MispricingWindowConfig,
    WindowResult,
    classify_gsv,
)
from bip.evaluation.live.engine_v3.ood_detector import OODDetector
from bip.evaluation.live.engine_v3.mes import _squashed_goals_cvar
from bip.evaluation.live.engine_v3.thesis import (
    MarketFamily,
    Thesis,
    ThesisArchetype,
    UNDER_DIRECTION_ALLOWED_ARCHETYPES,
)

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.shadow_logger import ShadowLogger

# Sanity bound: total goals outside [0, 15] indicate a genuinely broken
# GSV (parser error, fixture ID collision, etc.) and are ENFORCED denials
# even in the shadow-only OOD path.
_TOTAL_GOALS_SANITY_MAX = 15


# Family-specific line freshness thresholds (sec).
# Justified empirically by Day-3 line age percentiles per family (see
# internal research notes — line age p75 is 1028s globally but varies
# dramatically by family: BTTS lines refresh in <30s, corners markets in
# 1100-4600s as their counts only change with the next corner event).
# A line that hasn't been touched for 30 minutes is NOT stale if the
# underlying state hasn't changed in those 30 minutes — Sportmonks just
# doesn't tick for stable cells. Forcing 60s/300s rejects those picks
# unnecessarily.
_LINE_MAX_AGE_BY_FAMILY: dict[MarketFamily, float] = {
    MarketFamily.CORNERS: 1800.0,        # corners count only changes on a new corner
    MarketFamily.NEXT_CORNER: 1800.0,
    MarketFamily.GOALS: 1500.0,          # goals count only changes on a new goal
    MarketFamily.BTTS: 300.0,            # BTTS lines refresh quickly (p90=30s)
    MarketFamily.NEXT_GOAL: 600.0,
    MarketFamily.CARDS: 600.0,
    MarketFamily.PROPS: 600.0,
    MarketFamily.ASIAN_HANDICAP: 600.0,
    MarketFamily.DOUBLE_CHANCE: 600.0,
    MarketFamily.RESULT_1X2: 600.0,
    MarketFamily.RACE_TO_X: 600.0,
}


def line_max_age_for_family(family: MarketFamily, default: float = 600.0) -> float:
    """Per-family stale-line threshold. See _LINE_MAX_AGE_BY_FAMILY."""
    return _LINE_MAX_AGE_BY_FAMILY.get(family, default)

if TYPE_CHECKING:
    from bip.evaluation.live.engine_v3.drift_monitor import (
        CalibrationDriftMonitor,
    )

UNDER_DIRECTIONS = frozenset({"under", "no"})


@dataclass(frozen=True)
class NoBetVerdict:
    """Result of the gate. ``allowed=False`` means the candidate is killed."""

    allowed: bool
    rule_number: int | None = None
    reason: str = ""

    @classmethod
    def ok(cls) -> NoBetVerdict:
        return cls(allowed=True)

    @classmethod
    def deny(cls, rule_number: int, reason: str) -> NoBetVerdict:
        return cls(allowed=False, rule_number=rule_number, reason=reason)


# ──────────────────────────────────────────────────────────────────────
# Rule implementations
# ──────────────────────────────────────────────────────────────────────


def rule_1_thesis_present(theses: list[Thesis]) -> NoBetVerdict:
    """#1 — No-thesis-no-pick: if the rule layer (and pattern/commentary
    layers) produced zero theses, abort. The EV is not evaluated."""
    if not theses:
        return NoBetVerdict.deny(1, "no thesis activated by hypothesis generator")
    return NoBetVerdict.ok()


def rule_2_score_state_inversion(candidate: MarketCandidate, gsv: GameStateVector) -> NoBetVerdict:
    """#2 — Score-state inversion rule (the Napoli rule).

    If ``dominant_losing=True`` AND the candidate direction is in
    ``{under, no}``, abort UNLESS the thesis archetype is in the
    allow-list (cruise_mode, late_collapse_underdog).
    """
    if not gsv.score.dominant_losing:
        return NoBetVerdict.ok()
    if candidate.thesis.prediction.direction not in UNDER_DIRECTIONS:
        return NoBetVerdict.ok()
    if candidate.thesis.archetype in UNDER_DIRECTION_ALLOWED_ARCHETYPES:
        return NoBetVerdict.ok()
    return NoBetVerdict.deny(
        2,
        f"dominant_losing=True + under-direction thesis "
        f"{candidate.thesis.archetype.value} not in allow-list",
    )


def rule_3_critical_event_freshness(gsv: GameStateVector) -> NoBetVerdict:
    """#3 — If the last critical event was <90s ago, abort. State has
    not yet stabilized."""
    age = gsv.last_critical_event_age_sec
    if age is None:
        return NoBetVerdict.ok()
    if age < 90.0:
        return NoBetVerdict.deny(3, f"last critical event {age:.0f}s ago < 90s")
    return NoBetVerdict.ok()


def rule_4_line_freshness(
    candidate: MarketCandidate,
    gsv: GameStateVector,
    max_age_sec: float | None = None,
) -> NoBetVerdict:
    """#4 — If the line in question hasn't moved in too long, abort.

    The ``last_update_utc`` is read against the GSV timestamp. The
    threshold defaults to a family-specific value
    (``line_max_age_for_family``) when ``max_age_sec`` is None;
    callers can override with a single number to restore the legacy
    single-threshold behaviour.

    Day-3 empirical analysis showed: Sportmonks line update cadence
    varies by family (BTTS lines tick in <30s, corners in 1000-4600s).
    A 60s universal cutoff rejected 76% of pre-gate candidates with
    perfectly valid underlying state.
    """
    line = gsv.markets.lines.get(candidate.market_id)
    if line is None or line.last_update_utc is None:
        return NoBetVerdict.deny(4, "no line snapshot")
    age = (gsv.timestamp_utc - line.last_update_utc).total_seconds()
    threshold = (
        max_age_sec
        if max_age_sec is not None
        else line_max_age_for_family(candidate.family)
    )
    if age > threshold:
        return NoBetVerdict.deny(4, f"line stale ({age:.0f}s > {threshold:.0f}s)")
    return NoBetVerdict.ok()


def rule_5_thesis_market_mismatch(
    candidate: MarketCandidate,
    threshold: float = 0.6,
    *,
    calibrated_floor: float = 0.35,
) -> NoBetVerdict:
    """#5 — If MES < threshold, abort. Means valid thesis + no market
    expresses it well. Default to no-bet, NOT to the 3 safe markets.

    When ``mes.calibrated_winprob`` is present (a fitted MES→win-prob
    calibrator was available at compute_mes time), an additional floor
    check is applied: if the calibrated win probability is below
    ``calibrated_floor``, the pick is denied even if the raw MES
    passes. This catches cases where the raw score looks acceptable
    but the calibrated evidence shows the underlying win probability
    is too low to justify the pick.

    When ``calibrated_winprob`` is None the behavior is byte-identical
    to the pre-Wave-3 gate (only raw score checked).
    """
    if candidate.mes.score < threshold:
        return NoBetVerdict.deny(
            5, f"MES {candidate.mes.score:.3f} < {threshold:.2f}"
        )
    # Calibrated win-prob floor (only when calibrator was provided)
    cal = candidate.mes.calibrated_winprob
    if cal is not None and cal < calibrated_floor:
        return NoBetVerdict.deny(
            5,
            f"calibrated_winprob {cal:.3f} < floor {calibrated_floor:.2f} "
            f"(raw MES {candidate.mes.score:.3f} passed threshold)",
        )
    return NoBetVerdict.ok()


def rule_6_commentary_lag(candidate: MarketCandidate, gsv: GameStateVector,
                          commentary_required: bool = False) -> NoBetVerdict:
    """#6 — If the candidate is derived from a commentary-layer thesis
    AND the commentary parse hasn't completed for the triggering event,
    abort.

    For rule-layer theses, this trivially passes. For Phase 1 the
    commentary layer is not yet wired in production, so ``commentary_required``
    defaults to False.
    """
    if candidate.thesis.source.layer != "commentary":
        return NoBetVerdict.ok()
    if commentary_required and gsv.last_critical_event is None:
        return NoBetVerdict.deny(6, "commentary-derived thesis lacks event anchor")
    return NoBetVerdict.ok()


def rule_7_liquidity_gate(candidate: MarketCandidate) -> NoBetVerdict:
    """#7 — If liquidity_score = 0 (stake cap < 50% of target), abort.

    Hard floor (not multiplicative) so a small-stake market never gets
    routing — they signal stale or illiquid lines."""
    if candidate.mes.liquidity_score <= 0.0:
        return NoBetVerdict.deny(7, "liquidity_score=0 — stake cap below half target")
    return NoBetVerdict.ok()


def rule_8_predictive_uncertainty(
    candidate: MarketCandidate,
    uncertainty_band: float = 0.08,
    *,
    gsv: GameStateVector | None = None,
    shadow_logger: "ShadowLogger | None" = None,
    fixture_id: int | None = None,
    ts=None,
) -> NoBetVerdict:
    """#8 — If the conditional predictor reports a confidence interval
    wider than book_implied ± 8%, abort.

    The width is currently embedded in the ``mes.conditional_variance``
    factor. We compare against a normalised threshold: a variance
    larger than ``uncertainty_band * 10`` is considered too wide.

    SHADOW-ONLY exception: when the raw cvar WOULD deny AND the family is
    GOALS AND the current minute is >= 40 AND the squashed cvar (raw/(1+raw))
    would PASS the band → candidate passes, shadow denial recorded.

    In all other cases rule_8 enforces exactly as before. This shadow path
    was added because open_game/GOALS candidates at half-time (minute ~45)
    are structurally un-passable: λ_total × 40/90 ≈ 1.11, squashed ≈ 0.53,
    which is well within the band. Collecting shadow data for 1-2 weeks will
    confirm whether squashing is a sound policy before enforcing it.
    """
    raw_cvar = candidate.mes.conditional_variance
    band = uncertainty_band * 10
    if raw_cvar > band:
        # Check shadow path: GOALS, minute >= 40, squashed would pass
        if (
            gsv is not None
            and candidate.family == MarketFamily.GOALS
            and gsv.time.minute >= 40
            and shadow_logger is not None
            and fixture_id is not None
            and ts is not None
        ):
            squashed = _squashed_goals_cvar(raw_cvar)
            if squashed <= band:
                # Shadow-only: record denial but let candidate pass.
                shadow_deny = NoBetVerdict.deny(
                    8,
                    f"rule_8 shadow: raw cvar {raw_cvar:.2f} > band {band:.2f} "
                    f"but squashed {squashed:.2f} <= {band:.2f} "
                    f"(GOALS, minute={gsv.time.minute})",
                )
                shadow_logger.record_shadow_denial(
                    GateResult(candidate=candidate, verdict=shadow_deny),
                    fixture_id=fixture_id,
                    ts=ts,
                )
                return NoBetVerdict.ok()
        return NoBetVerdict.deny(
            8,
            f"predictive variance {raw_cvar:.2f} "
            f"exceeds band {band:.2f}",
        )
    return NoBetVerdict.ok()


def rule_10_mispricing_window(
    candidate: MarketCandidate,
    window: WindowResult,
    cfg: MispricingWindowConfig,
) -> NoBetVerdict:
    """#10 — Mispricing window enforcement (principle #3 of the design doc).

    When the window is COLD (>600s since last critical event) AND the
    base edge is below ``cfg.cold_edge_threshold``, abort. The book has
    had ample time to adjust; persistent small edge in this window is
    model noise, not real mispricing.

    HOT, OPTIMAL, WARM, INDEFINITE windows pass — only COLD with weak
    edge dies here.
    """
    if not window.is_cold:
        return NoBetVerdict.ok()
    edge = candidate.mes.base_edge
    if edge < cfg.cold_edge_threshold:
        return NoBetVerdict.deny(
            10,
            f"COLD window (age {window.age_sec:.0f}s) + edge "
            f"{edge:.3f} < cold-threshold {cfg.cold_edge_threshold:.3f}",
        )
    return NoBetVerdict.ok()


def rule_11_calibration_drift(
    candidate: MarketCandidate,
    gsv: GameStateVector,
    monitor: "CalibrationDriftMonitor | None",
) -> NoBetVerdict:
    """#11 — Calibration/P&L drift gate.

    When the ``CalibrationDriftMonitor`` reports the candidate's
    ``(market_family, minute_bucket)`` cell as drifted, abort the
    candidate. Drift is defined as EITHER:

    - Calibration drift: the reliability gap |predicted_avg - empirical_wr|
      exceeds ``reliability_gap_threshold`` (default 0.15). This catches
      systematic over- or under-confidence that KS-on-WR would also
      catch, but is family- and archetype-agnostic (predicted 0.2 vs
      actual 0.2 → gap = 0, no drift, regardless of WR vs a fixed prior).
    - P&L drift: the rolling sum of ``profit_units`` per observation
      drops below ``pnl_floor_per_obs`` (default -0.10 / obs). This
      catches profitable-calibration / unprofitable-bookmaker scenarios.

    Motivation: the Day-1→Day-2 contrafactual analysis showed cards
    miscalibration of 59 percentage points. The principled reliability
    gap catches that catastrophic failure without needing a hand-tuned
    WR prior per archetype.

    Why no napoli exemption:
    - The old rule_11 compared WR against a 0.67 prior — wrong for
      long-shot archetypes. DOMINANT_LOSING_NAPOLI with predicted=0.20,
      realized~=0.20 → reliability gap ≈ 0, NOT drifted. If P&L is
      positive the P&L gate also passes. No special-case needed.
    - Historical napoli exemption (added 2026-05-xx, +96u/14 Day-3)
      is SUPERSEDED by this principled gate. If napoli P&L turns
      negative, it will now be caught by the P&L floor — that is the
      correct response (pause and recalibrate), not a permanent bypass.

    Fail-safe:
    - ``monitor=None`` → pass (Phase-1 deployments without the monitor).
    - cell not warm (n < min_observations) → pass.
    - cell warm and either condition drifted → deny.
    """
    if monitor is None:
        return NoBetVerdict.ok()
    family = candidate.family
    status = monitor.status(family, gsv.time.minute)
    if not status.is_warm:
        return NoBetVerdict.ok()
    if status.is_drifted:
        return NoBetVerdict.deny(
            11,
            f"calibration/P&L drifted for {status.family}@{status.minute_bucket}: "
            f"reliability_gap={status.reliability_gap:.3f} "
            f"(expected_wr={status.expected_win_rate:.2f} vs "
            f"empirical_wr={status.empirical_win_rate:.2f}), "
            f"rolling_pnl={status.rolling_pnl:.2f} n={status.n} "
            f"[{status.drift_reason}]",
        )
    return NoBetVerdict.ok()


def rule_9_ood_detector(
    gsv: GameStateVector,
    detector: OODDetector | None,
    *,
    threshold: float | None = None,
) -> NoBetVerdict:
    """#9 — Out-of-distribution game state (SHADOW-ONLY).

    This rule is now SHADOW-ONLY: if the legacy detector would deny,
    a shadow denial row is recorded (is_shadow=True) and the candidate
    PASSES. The shadow data accrues so a new detector can be refit with
    the trimmed 14-feature schema (an offline research script (not published)).

    EXCEPTION: total_goals outside [0, 15] is a REAL enforced deny —
    that indicates a genuinely broken GSV (parser error, fixture ID
    collision). No detector is needed for this sanity check.

    When ``detector`` is unfitted or ``None`` the rule passes (fail-safe).

    The shadow recording happens in run_gate (which has access to the
    shadow_logger). This function returns the would-be verdict so
    run_gate can decide: REAL deny (sanity fail) or shadow-and-pass.
    """
    total_goals = gsv.score.home_goals + gsv.score.away_goals
    if total_goals < 0 or total_goals > _TOTAL_GOALS_SANITY_MAX:
        return NoBetVerdict.deny(
            9,
            f"broken GSV: total_goals={total_goals} outside [0, {_TOTAL_GOALS_SANITY_MAX}]",
        )

    if detector is None or not detector.is_fitted:
        return NoBetVerdict.ok()
    score = detector.score(gsv)
    cutoff = detector.threshold if threshold is None else threshold
    if score > cutoff:
        # Shadow-only: return a special deny so run_gate can record the
        # shadow row and then pass the candidate through.
        return NoBetVerdict.deny(
            9,
            f"OOD game state (shadow) — Mahalanobis {score:.2f} > threshold {cutoff:.2f} "
            f"(trained on n={detector.n_train})",
        )
    return NoBetVerdict.ok()


def rule_12_mes_dead_zone(
    candidate: MarketCandidate,
    *,
    calibrated_cruise_floor: float = 0.45,
) -> NoBetVerdict:
    """#12 — cruise_mode/GOALS dead-zone suppression (ENFORCED).

    Replicated-loss evidence (2026-05-14 forensic):
    - Day-3 (n=31 cruise_mode/GOALS picks in bin [3,4)): WR 38.7%, -13.78u
    - Day-4 (n=5 in same bin): WR 40%, -2.50u

    The [2.5, 4.0) MES band is a structural dead zone for cruise_mode/GOALS:
    the thesis fires but the market expression score is too uncertain to
    justify a pick.

    Calibrated-winprob mode (when ``mes.calibrated_winprob`` is present):
    The raw-band `[2.5, 4.0)` check is re-expressed in calibrated win-prob
    units: if calibrated_winprob < ``calibrated_cruise_floor`` (default
    0.45), deny. This is the principled version — the bin-3 patch becomes
    "calibrated probability below a family-specific floor" rather than a
    hard-coded MES bin.

    Fallback (``calibrated_winprob=None``): raw-band behavior is
    byte-identical to pre-Wave-3 (``2.5 <= score < 4.0`` check).

    Scope in raw-band mode: ONLY cruise_mode + GOALS in [2.5, 4.0).
    Scope in calibrated mode: cruise_mode + GOALS with low calibrated prob.
    Other archetypes and families are unaffected in both modes.
    """
    if (
        candidate.thesis.archetype != ThesisArchetype.CRUISE_MODE
        or candidate.family != MarketFamily.GOALS
    ):
        return NoBetVerdict.ok()

    cal = candidate.mes.calibrated_winprob
    if cal is not None:
        # Calibrated mode: re-express dead-zone in win-prob units
        if cal < calibrated_cruise_floor:
            return NoBetVerdict.deny(
                12,
                f"cruise_mode/goals calibrated_winprob {cal:.3f} "
                f"< floor {calibrated_cruise_floor:.2f} "
                f"(raw MES={candidate.mes.score:.3f})",
            )
        return NoBetVerdict.ok()

    # Raw-band fallback (no calibrator): byte-identical pre-Wave-3 behavior
    if 2.5 <= candidate.mes.score < 4.0:
        return NoBetVerdict.deny(
            12,
            f"cruise_mode/goals MES dead-zone [2.5,4.0): score={candidate.mes.score:.3f}",
        )
    return NoBetVerdict.ok()


# ──────────────────────────────────────────────────────────────────────
# Compose
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GateResult:
    candidate: MarketCandidate
    verdict: NoBetVerdict


def run_gate(
    theses: list[Thesis],
    candidates: list[MarketCandidate],
    gsv: GameStateVector,
    *,
    mes_threshold: float = 0.6,
    line_max_age_sec: float | None = None,
    commentary_required: bool = False,
    uncertainty_band: float = 0.08,
    ood_detector: OODDetector | None = None,
    mispricing_window_cfg: MispricingWindowConfig | None = None,
    drift_monitor: "CalibrationDriftMonitor | None" = None,
    shadow_logger: "ShadowLogger | None" = None,
) -> list[GateResult]:
    """Run the rules against each candidate.

    Returns one ``GateResult`` per candidate; callers filter by
    ``r.verdict.allowed``. Even denied results are returned (with their
    rule number + reason) so the audit log can capture them — sec 7.2
    requires that "every rejected pick is logged with reason".

    Rule 9 (OOD) is global to the GSV — it has two sub-paths:
      - total_goals sanity fail → REAL enforced deny (still short-circuits)
      - Mahalanobis OOD → SHADOW-ONLY: candidate passes, shadow row logged

    Rule 10 (mispricing window) is per-candidate (it inspects the
    candidate's base_edge against a window-dependent threshold) but
    the window classification itself is GSV-level, computed once.

    ``shadow_logger``: when provided, shadow-only verdicts from rule_9
    and rule_8 are recorded via ``shadow_logger.record_shadow_denial``.
    Required to capture shadow denials without affecting the candidate's
    allowed/denied outcome.
    """
    # Rule 1 is global (applies once). If no theses, every candidate is denied
    # against rule 1; we short-circuit to one verdict per candidate.
    if not theses:
        return [
            GateResult(candidate=c, verdict=NoBetVerdict.deny(1, "no theses"))
            for c in candidates
        ]
    # Rule 9 — OOD check (state-global).
    ood_verdict = rule_9_ood_detector(gsv, ood_detector)
    if not ood_verdict.allowed:
        # Sanity fail (total_goals out of range): real enforced deny.
        # Mahalanobis OOD: shadow-only — pass candidates through but log.
        is_sanity_fail = "broken GSV" in ood_verdict.reason
        if is_sanity_fail:
            return [GateResult(candidate=c, verdict=ood_verdict) for c in candidates]
        # Shadow: log and allow candidates to continue through remaining rules.
        if shadow_logger is not None:
            for c in candidates:
                shadow_logger.record_shadow_denial(
                    GateResult(candidate=c, verdict=ood_verdict),
                    fixture_id=gsv.fixture_id,
                    ts=gsv.timestamp_utc,
                )
        # Fall through — candidates are NOT blocked by shadow OOD.

    win_cfg = mispricing_window_cfg or MispricingWindowConfig()
    window = classify_gsv(gsv, win_cfg)
    out: list[GateResult] = []
    for c in candidates:
        for verdict in (
            rule_2_score_state_inversion(c, gsv),
            rule_3_critical_event_freshness(gsv),
            rule_4_line_freshness(c, gsv, line_max_age_sec),
            rule_5_thesis_market_mismatch(c, mes_threshold),
            rule_12_mes_dead_zone(c),
            rule_6_commentary_lag(c, gsv, commentary_required),
            rule_7_liquidity_gate(c),
            rule_8_predictive_uncertainty(
                c, uncertainty_band,
                gsv=gsv,
                shadow_logger=shadow_logger,
                fixture_id=gsv.fixture_id,
                ts=gsv.timestamp_utc,
            ),
            rule_10_mispricing_window(c, window, win_cfg),
            rule_11_calibration_drift(c, gsv, drift_monitor),
        ):
            if not verdict.allowed:
                out.append(GateResult(candidate=c, verdict=verdict))
                break
        else:
            out.append(GateResult(candidate=c, verdict=NoBetVerdict.ok()))
    return out


def allowed_candidates(results: list[GateResult]) -> list[MarketCandidate]:
    return [r.candidate for r in results if r.verdict.allowed]


__all__ = [
    "GateResult",
    "NoBetVerdict",
    "allowed_candidates",
    "line_max_age_for_family",
    "rule_1_thesis_present",
    "rule_2_score_state_inversion",
    "rule_3_critical_event_freshness",
    "rule_4_line_freshness",
    "rule_5_thesis_market_mismatch",
    "rule_6_commentary_lag",
    "rule_7_liquidity_gate",
    "rule_8_predictive_uncertainty",
    "rule_9_ood_detector",
    "rule_10_mispricing_window",
    "rule_11_calibration_drift",
    "rule_12_mes_dead_zone",
    "run_gate",
]
