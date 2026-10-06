"""Tests for the Phase-2 additions: NextGoalPredictor, A2 BTTS
companion, CalibrationDriftMonitor."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from bip.evaluation.live.engine_v3.archetypes import (
    detect_dominant_losing_napoli,
    detect_napoli_btts_companion,
    generate_theses,
)
from bip.evaluation.live.engine_v3.drift_monitor import (
    CalibrationDriftMonitor,
)
from bip.evaluation.live.engine_v3.gsv import (
    CardsState,
    CornerState,
    FlowState,
    GameStateVector,
    MarketSnapshot,
    NumericalState,
    PreMatchPriors,
    RosterState,
    ScoreState,
    TacticalState,
    TimeState,
    XGState,
)
from bip.evaluation.live.engine_v3.next_goal_predictor import (
    NextGoalPredictor,
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
    build_horizon,
)


def _gsv_napoli(*, minute: int = 35, home_goals: int = 0, away_goals: int = 1) -> GameStateVector:
    """A Napoli-style state: dominant home trailing, xG outshooting away."""
    return GameStateVector(
        fixture_id=1, state_version=1,
        timestamp_utc=datetime.now(timezone.utc),
        home_team_id=100, away_team_id=200,
        home_team_name="Home FC", away_team_name="Away FC",
        score=ScoreState(
            home_goals=home_goals, away_goals=away_goals,
            goal_diff=home_goals - away_goals,
            dominant_team_id=100,
            dominant_losing=(home_goals < away_goals),
            minutes_since_last_goal=10.0,
        ),
        time=TimeState(
            minute=minute, period="1H" if minute <= 45 else "2H",
            time_remaining_half=max(0, 45 - minute) if minute <= 45 else max(0, 90 - minute),
            time_remaining_match=max(0, 90 - minute),
        ),
        numerical=NumericalState(),
        xg=XGState(
            home_xg_total=1.8, away_xg_total=0.4, xg_diff=1.4,
            xg_per_min_home_last_15=0.08, xg_per_min_away_last_15=0.01,
            xg_vs_score_divergence=0.5,
            shots_total=(14, 4), shots_on_target=(7, 1),
            shots_in_box=(5, 1), big_chances=(3, 0),
        ),
        flow=FlowState(possession_home_5min=65.0, possession_home_match=62.0),
        corners=CornerState(corners_home=6, corners_away=2),
        cards=CardsState(),
        roster=RosterState(formation_home="4-3-3", formation_away="5-4-1"),
        tactical=TacticalState(home_phase="controlling", away_phase="parking_bus",
                               game_phase="open_attacking"),
        priors=PreMatchPriors(
            lambda_home_prematch=1.9, lambda_away_prematch=0.9,
            expected_corners_total=10.4, expected_cards_total=3.9, elo_diff=200.0,
        ),
        markets=MarketSnapshot(),
    )


def _thesis(
    *,
    family: MarketFamily,
    direction: str,
    magnitude: float = 0.10,
) -> Thesis:
    return Thesis(
        id=f"T@m35", archetype=ThesisArchetype.DOMINANT_LOSING_NAPOLI,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(family=family, direction=direction,
                                    magnitude_pp=magnitude,
                                    horizon=build_horizon("rest_of_match", 35)),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=35,
    )


# ──────────────────────────────────────────────────────────────────────
# NextGoalPredictor
# ──────────────────────────────────────────────────────────────────────


def test_next_goal_predictor_rejects_non_next_goal_thesis():
    pred = NextGoalPredictor()
    gsv = _gsv_napoli()
    out = pred.predict(
        _thesis(family=MarketFamily.GOALS, direction="over"),
        "team_to_score_first_home",
        gsv,
    )
    assert out is None


def test_next_goal_predictor_rejects_non_next_goal_market():
    pred = NextGoalPredictor()
    gsv = _gsv_napoli()
    out = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "match_goals_over_2.5",
        gsv,
    )
    assert out is None


def test_next_goal_predictor_assigns_higher_p_to_home_when_outshooting():
    """Napoli regime: home is outshooting away by xG diff +1.4. The
    hazard model must give home a higher P(next goal) than away."""
    pred = NextGoalPredictor()
    gsv = _gsv_napoli()
    p_home = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "first_goal_home",
        gsv,
    )
    # Symmetric query for the away side requires changing direction; we
    # build it by hand to compare fairly.
    p_away_thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="away")
    p_away = pred.predict(p_away_thesis, "first_goal_away", gsv)
    assert p_home is not None
    assert p_away is not None
    assert p_home.p > p_away.p, (
        f"expected home>away on the Napoli state, got {p_home.p:.3f} vs {p_away.p:.3f}"
    )


def test_next_goal_predictor_returns_none_on_side_mismatch():
    pred = NextGoalPredictor()
    gsv = _gsv_napoli()
    # Thesis says home, but the market expresses the away side.
    out = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "first_goal_away",
        gsv,
    )
    assert out is None


def test_next_goal_predictor_horizon_uses_remaining_half_for_first_half_markets():
    pred = NextGoalPredictor()
    gsv = _gsv_napoli(minute=35)  # 10 min remaining in half
    # first_half market: τ = 10 min
    p_fh = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "first_half_team_to_score_first_home",
        gsv,
    )
    # full match market: τ = 55 min
    p_full = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "team_to_score_first_home",
        gsv,
    )
    assert p_fh is not None and p_full is not None
    # Larger τ → higher P(any goal) → higher fair_prob.
    assert p_full.p > p_fh.p


# ──────────────────────────────────────────────────────────────────────
# A2 BTTS companion archetype
# ──────────────────────────────────────────────────────────────────────


def test_napoli_btts_companion_fires_when_predicate_holds():
    gsv = _gsv_napoli(minute=35, home_goals=0, away_goals=1)
    t = detect_napoli_btts_companion(gsv)
    assert t is not None
    assert t.prediction.family == MarketFamily.BTTS
    assert t.prediction.direction == "yes"


def test_napoli_btts_companion_does_not_fire_before_minute_30():
    gsv = _gsv_napoli(minute=28, home_goals=0, away_goals=1)
    t = detect_napoli_btts_companion(gsv)
    assert t is None, (
        "v2 outcome analysis: BTTS picks at minute 15-30 had ROI -40%. "
        "The companion must defer until minute ≥ 30."
    )


def test_napoli_btts_companion_does_not_fire_when_dominant_already_scored():
    gsv = _gsv_napoli(minute=35, home_goals=1, away_goals=2)
    # If dominant already scored, BTTS Yes may already be settled.
    # dominant_losing is True (1 < 2) but dom_goals != 0 → no companion.
    t = detect_napoli_btts_companion(gsv)
    assert t is None


def test_napoli_btts_companion_does_not_fire_when_underdog_has_not_scored():
    gsv = _gsv_napoli(minute=35, home_goals=0, away_goals=0)
    # No goals at all → dominant_losing is False (tied) → companion skipped.
    t = detect_napoli_btts_companion(gsv)
    assert t is None


def test_generate_theses_emits_both_a2_and_a2b_in_napoli_state():
    """When the Napoli predicate AND the companion preconditions both
    hold, generate_theses must emit two theses: the NEXT_GOAL one (A2)
    and the BTTS companion (A2b)."""
    gsv = _gsv_napoli(minute=35, home_goals=0, away_goals=1)
    theses = generate_theses(gsv)
    families = [t.prediction.family for t in theses]
    assert MarketFamily.NEXT_GOAL in families, (
        f"expected A2 NEXT_GOAL thesis, got families={families}"
    )
    assert MarketFamily.BTTS in families, (
        f"expected A2b BTTS thesis, got families={families}"
    )


# ──────────────────────────────────────────────────────────────────────
# Anti-Napoli regression — companion must NOT emit under-direction theses
# ──────────────────────────────────────────────────────────────────────


def test_napoli_btts_companion_never_emits_under_direction():
    """The companion's direction must be "yes" (an over-style), never
    "no" — that would replicate the Napoli failure mode in BTTS form."""
    gsv = _gsv_napoli(minute=35, home_goals=0, away_goals=1)
    t = detect_napoli_btts_companion(gsv)
    assert t is not None
    assert t.prediction.direction == "yes"


# ──────────────────────────────────────────────────────────────────────
# CalibrationDriftMonitor
# ──────────────────────────────────────────────────────────────────────


def test_drift_monitor_cold_start_is_not_drifted():
    monitor = CalibrationDriftMonitor()
    status = monitor.status(MarketFamily.GOALS, 45)
    assert status.n == 0
    assert status.is_drifted is False
    assert status.is_warm is False


def test_drift_monitor_warm_status_aligned_with_observations():
    """200 picks where predicted ≈ actual → no drift."""
    monitor = CalibrationDriftMonitor(window_size=200, min_observations=30)
    rng_outcomes = [1 if i % 2 == 0 else 0 for i in range(40)]
    for o in rng_outcomes:
        monitor.observe(MarketFamily.GOALS, minute=45, predicted_p=0.5, outcome=o)
    status = monitor.status(MarketFamily.GOALS, 45)
    assert status.is_warm is True
    assert status.n == 40
    assert status.empirical_win_rate == pytest.approx(0.5, abs=0.02)
    assert status.expected_win_rate == pytest.approx(0.5, abs=0.02)


def test_drift_monitor_flags_drifted_when_predicted_far_from_actual():
    """Predicted 0.9, actual ~0.3 → reliability gap 0.6 > threshold 0.15.

    The monitor now uses reliability gap (|predicted_avg - empirical_wr|)
    rather than a KS test. The gap of 0.6 far exceeds the default threshold
    of 0.15, so the cell must be flagged as drifted.
    """
    monitor = CalibrationDriftMonitor(window_size=200, min_observations=30)
    # 60 observations where predicted ≈ 0.9 (high confidence) but outcome
    # is mostly 0 (the model is wrong)
    for i in range(60):
        outcome = 1 if i < 18 else 0  # ~30% actual
        monitor.observe(MarketFamily.BTTS, minute=45, predicted_p=0.9, outcome=outcome)
    status = monitor.status(MarketFamily.BTTS, 45)
    assert status.is_warm is True
    assert status.is_drifted is True, (
        f"expected drift flagged when predicted=0.9 actual≈0.3, got "
        f"empirical={status.empirical_win_rate:.2f}, expected={status.expected_win_rate:.2f}, "
        f"reliability_gap={status.reliability_gap:.4f}"
    )
    # Reliability gap should be ~0.6 (0.9 - 0.3)
    assert status.reliability_gap > 0.15, (
        f"reliability_gap {status.reliability_gap:.3f} should exceed threshold 0.15"
    )


def test_drift_monitor_rolling_window_evicts_old_observations():
    """With window_size=10 and 30 inserts, only the last 10 are kept."""
    monitor = CalibrationDriftMonitor(window_size=10, min_observations=5)
    for i in range(30):
        monitor.observe(MarketFamily.GOALS, minute=45, predicted_p=0.5, outcome=i % 2)
    status = monitor.status(MarketFamily.GOALS, 45)
    assert status.n == 10


def test_drift_monitor_buckets_independently():
    """Two cells in the same family but different minute buckets are
    tracked separately."""
    monitor = CalibrationDriftMonitor()
    for _ in range(40):
        monitor.observe(MarketFamily.GOALS, minute=20, predicted_p=0.7, outcome=1)
        monitor.observe(MarketFamily.GOALS, minute=80, predicted_p=0.3, outcome=0)
    s_early = monitor.status(MarketFamily.GOALS, 20)
    s_late = monitor.status(MarketFamily.GOALS, 80)
    assert s_early.minute_bucket == "15-30"
    assert s_late.minute_bucket == "75-90"
    assert s_early.empirical_win_rate == pytest.approx(1.0)
    assert s_late.empirical_win_rate == pytest.approx(0.0)


def test_drift_monitor_observe_validates_outcome():
    monitor = CalibrationDriftMonitor()
    with pytest.raises(ValueError):
        monitor.observe(MarketFamily.GOALS, 45, 0.5, outcome=2)  # not 0/1


# ──────────────────────────────────────────────────────────────────────
# Rule #11 (calibration drift gate) integration
# ──────────────────────────────────────────────────────────────────────


def test_rule_11_passes_when_monitor_absent():
    """Phase-1 deployments without a monitor must not be affected."""
    from bip.evaluation.live.engine_v3.no_bet_gate import (
        rule_11_calibration_drift,
    )

    # Construct a minimal candidate via a synthetic thesis
    from bip.evaluation.live.engine_v3.mes import MESResult
    from bip.evaluation.live.engine_v3.market_selector import MarketCandidate

    thesis = Thesis(
        id="T@m45",
        archetype=ThesisArchetype.CRUISE_MODE,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(family=MarketFamily.GOALS, direction="under",
                                    magnitude_pp=0.06,
                                    horizon=build_horizon("rest_of_match", 45)),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=45,
    )
    candidate = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5", family=MarketFamily.GOALS,
        fair_prob=0.7, mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_under_2.5",
            family=MarketFamily.GOALS, base_edge=0.05, signal_clarity=0.9,
            book_slowness=1.0, liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    gsv = _gsv_napoli(minute=45)
    v = rule_11_calibration_drift(candidate, gsv, monitor=None)
    assert v.allowed is True


def test_rule_11_denies_when_cell_drifted():
    """If the monitor flags the cell drifted, the gate must deny for non-napoli archetypes.

    Note: DOMINANT_LOSING_NAPOLI is exempt from rule_11 (see rule_11_calibration_drift
    docstring). This test uses BTTS_LATE_COMPRESSION to cover the general drift path.
    """
    from bip.evaluation.live.engine_v3.no_bet_gate import (
        rule_11_calibration_drift,
    )
    from bip.evaluation.live.engine_v3.mes import MESResult
    from bip.evaluation.live.engine_v3.market_selector import MarketCandidate

    monitor = CalibrationDriftMonitor(window_size=200, min_observations=30)
    # Force a drift: predicted 0.9, actual ~0.3 in btts 30-45 cell.
    for i in range(60):
        monitor.observe(MarketFamily.BTTS, minute=37, predicted_p=0.9,
                        outcome=1 if i < 18 else 0)
    status = monitor.status(MarketFamily.BTTS, 37)
    assert status.is_drifted is True

    thesis = Thesis(
        id="T2@m37",
        archetype=ThesisArchetype.OPEN_GAME_FORMATIONS,  # non-napoli archetype
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(family=MarketFamily.BTTS, direction="yes",
                                    magnitude_pp=0.08,
                                    horizon=build_horizon("rest_of_match", 37)),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=37,
    )
    candidate = MarketCandidate(
        thesis=thesis, market_id="btts_yes", family=MarketFamily.BTTS,
        fair_prob=0.5, mes=MESResult(
            thesis_id=thesis.id, market_id="btts_yes",
            family=MarketFamily.BTTS, base_edge=0.05, signal_clarity=0.9,
            book_slowness=1.0, liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    gsv = _gsv_napoli(minute=37)
    v = rule_11_calibration_drift(candidate, gsv, monitor=monitor)
    assert v.allowed is False
    assert v.rule_number == 11
    assert "drifted" in v.reason.lower()


def test_rule_11_passes_when_cell_cold():
    """A cell with too few observations is COLD — rule passes (no false
    positive). Better to defer to other rules until the cell warms up."""
    from bip.evaluation.live.engine_v3.no_bet_gate import (
        rule_11_calibration_drift,
    )
    from bip.evaluation.live.engine_v3.mes import MESResult
    from bip.evaluation.live.engine_v3.market_selector import MarketCandidate

    monitor = CalibrationDriftMonitor(window_size=200, min_observations=30)
    # Only 5 observations — cell is COLD
    for _ in range(5):
        monitor.observe(MarketFamily.GOALS, 45, 0.5, outcome=1)
    thesis = Thesis(
        id="T@m45",
        archetype=ThesisArchetype.CRUISE_MODE,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(family=MarketFamily.GOALS, direction="under",
                                    magnitude_pp=0.06,
                                    horizon=build_horizon("rest_of_match", 45)),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=45,
    )
    candidate = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5", family=MarketFamily.GOALS,
        fair_prob=0.7, mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_under_2.5",
            family=MarketFamily.GOALS, base_edge=0.05, signal_clarity=0.9,
            book_slowness=1.0, liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    gsv = _gsv_napoli(minute=45)
    v = rule_11_calibration_drift(candidate, gsv, monitor=monitor)
    assert v.allowed is True


def test_drift_monitor_warm_up_from_v2_history_detects_day1_day2_drift(tmp_path):
    """Honest contrafactual: after warming with v2 picks_graded, the
    cards family should be visibly miscalibrated (predicted_avg far
    from empirical), reproducing the Day-1→Day-2 finding."""
    import polars as pl

    src = Path("reports/sportmonks_live/exports/picks_graded.parquet")
    if not src.exists():
        pytest.skip("picks_graded.parquet not present in this checkout")

    monitor = CalibrationDriftMonitor(window_size=400, min_observations=30)
    n = monitor.warm_up_from_v2_history(src)
    assert n > 0, "no v2 picks ingested"
    # btts 45-60' has >50 picks in the v2 history and is one of the cells
    # the diagnosis flagged as miscalibrated. After warming, the cell
    # should be warm and the predicted vs empirical numbers should be sane.
    status = monitor.status(MarketFamily.BTTS, 50)
    assert status.is_warm, (
        f"btts 45-60 cell did not warm up — got n={status.n}, "
        f"warm_threshold={monitor.min_observations}"
    )
    assert 0.0 <= status.expected_win_rate <= 1.0
    assert 0.0 <= status.empirical_win_rate <= 1.0
    # The btts overconfidence delta from the diagnosis: predicted ≈ 0.82,
    # actual ≈ 0.59. The gap must be at least 5 percentage points.
    assert (status.expected_win_rate - status.empirical_win_rate) > 0.05, (
        f"expected btts overconfidence delta >5pp, got "
        f"predicted={status.expected_win_rate:.3f} vs "
        f"empirical={status.empirical_win_rate:.3f}"
    )
