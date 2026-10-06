"""Tests for the Mahalanobis OOD detector + No-Bet rule #9.

The detector closes the Risk #2 gap in the design doc: predictors
"default to global mean" on out-of-distribution game states. Without
this gate, a fitted regime calibrator can produce high-confidence
predictions on states it never saw.

Tested invariants:
- A clearly off-distribution state (huge minute, large goal_diff) is
  flagged when the detector is fitted on a tight training distribution.
- An unfitted detector NEVER blocks (fail-safe), but emits a warning.
- The 50 anti-Napoli regression states do NOT trigger the OOD gate
  when the detector is trained on a realistic spread of GSVs that
  includes those states. This prevents rule #9 from silently
  neutralising the rule #2 invariant.
- Singular covariance (n < d) is handled via shrinkage — fit does
  not crash and scoring is finite.
- save() + load() round-trip preserves both scores AND threshold.
- The pipeline routes OOD short-circuit through rule #9.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from bip.evaluation.live.engine_v3 import (
    GSVBuilder,
    OODDetector,
    PreMatchPriors,
    V3Pipeline,
    vectorize_gsv,
)
from bip.evaluation.live.engine_v3.no_bet_gate import (
    rule_9_ood_detector,
    run_gate,
)
from bip.evaluation.live.engine_v3.ood_detector import N_FEATURES
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import HOME_ID, make_state
from tests.evaluation.live.engine_v3.test_anti_napoli import (
    _build_minimal_markets,
    _synthesize_states,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _build_realistic_training_gsvs(n: int = 200) -> list:
    """Generate n diverse GSVs that span realistic live-football states.

    Variation axes: minute ∈ [5, 88], score states ∈ {0-0, 1-0, 0-1,
    1-1, 2-0, 0-2}, xG signals scaled to minute. The mix is intentionally
    representative — these are the states the production system "sees"
    most weekends.
    """
    rng = np.random.default_rng(42)
    builder = GSVBuilder()
    priors = PreMatchPriors(
        lambda_home_prematch=1.6, lambda_away_prematch=1.1,
        expected_corners_total=10.0, expected_cards_total=4.0,
    )
    markets = _build_minimal_markets()
    out = []
    score_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2)]
    for _ in range(n):
        minute = int(rng.integers(5, 88))
        h, a = score_grid[int(rng.integers(0, len(score_grid)))]
        # Scale stats to minute so older minutes have more accumulation.
        scale = max(0.4, minute / 60.0)
        home_stats = {
            StatType.SHOTS_TOTAL: int(rng.integers(2, 7) * scale),
            StatType.SHOTS_ON_TARGET: int(rng.integers(0, 4) * scale),
            StatType.SHOTS_INSIDEBOX: int(rng.integers(0, 4) * scale),
            StatType.BIG_CHANCES_CREATED: int(rng.integers(0, 2)),
            StatType.CORNERS: int(rng.integers(0, 5) * scale),
            StatType.BALL_POSSESSION: float(rng.normal(52.0, 6.0)),
            StatType.DANGEROUS_ATTACKS: int(rng.integers(8, 40) * scale),
            StatType.KEY_PASSES: int(rng.integers(1, 8) * scale),
        }
        away_stats = {
            StatType.SHOTS_TOTAL: int(rng.integers(1, 6) * scale),
            StatType.SHOTS_ON_TARGET: int(rng.integers(0, 3) * scale),
            StatType.SHOTS_INSIDEBOX: int(rng.integers(0, 3) * scale),
            StatType.BIG_CHANCES_CREATED: int(rng.integers(0, 2)),
            StatType.CORNERS: int(rng.integers(0, 4) * scale),
            StatType.BALL_POSSESSION: float(100.0 - home_stats[StatType.BALL_POSSESSION]),
            StatType.DANGEROUS_ATTACKS: int(rng.integers(6, 30) * scale),
            StatType.KEY_PASSES: int(rng.integers(1, 6) * scale),
        }
        state = make_state(
            home_goals=h, away_goals=a, minute=minute,
            home_stats=home_stats, away_stats=away_stats,
        )
        out.append(builder.build(state, priors=priors, markets=markets))
    return out


# ──────────────────────────────────────────────────────────────────────
# Vectorization
# ──────────────────────────────────────────────────────────────────────


def test_vectorize_gsv_returns_fixed_width(priors, market_snapshot):
    state = make_state()
    gsv = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    v = vectorize_gsv(gsv)
    assert v.shape == (N_FEATURES,)
    assert np.all(np.isfinite(v))


# ──────────────────────────────────────────────────────────────────────
# Detector behaviour
# ──────────────────────────────────────────────────────────────────────


def test_unfitted_detector_never_flags(priors, market_snapshot):
    """Fail-safe contract: never block when not fitted."""
    state = make_state()
    gsv = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    det = OODDetector()
    assert det.is_fitted is False
    assert det.is_ood(gsv) is False
    assert det.score(gsv) == 0.0


def test_fit_with_zero_gsvs_raises():
    det = OODDetector()
    with pytest.raises(ValueError):
        det.fit([])


def test_fit_with_few_samples_does_not_crash(priors, market_snapshot):
    """Singular covariance case: n < d. Shrinkage should handle it."""
    builder = GSVBuilder()
    # Just 5 GSVs against 17 features — Σ is rank-deficient.
    small = [
        builder.build(make_state(minute=m), priors=priors, markets=market_snapshot)
        for m in (15, 30, 45, 60, 75)
    ]
    det = OODDetector().fit(small, regularization=1e-1)
    assert det.is_fitted is True
    assert np.isfinite(det.threshold)
    # Score on a training sample should be finite and ≤ threshold most of the time.
    s = det.score(small[0])
    assert np.isfinite(s)
    assert s >= 0.0


def test_fitted_detector_flags_clear_outlier(priors, market_snapshot):
    """An unrealistic state (minute=120, 5-0 score) is past the 99th
    percentile of a realistic training set."""
    training = _build_realistic_training_gsvs(n=200)
    det = OODDetector().fit(training)
    # Build a wildly outlying GSV: 5-0 at minute 120 with extreme stats.
    extreme_state = make_state(
        home_goals=5, away_goals=0, minute=120,
        home_stats={
            StatType.SHOTS_TOTAL: 35, StatType.SHOTS_ON_TARGET: 18,
            StatType.SHOTS_INSIDEBOX: 22, StatType.BIG_CHANCES_CREATED: 12,
            StatType.CORNERS: 18, StatType.BALL_POSSESSION: 78.0,
            StatType.DANGEROUS_ATTACKS: 95, StatType.KEY_PASSES: 30,
        },
        away_stats={
            StatType.SHOTS_TOTAL: 1, StatType.SHOTS_ON_TARGET: 0,
            StatType.SHOTS_INSIDEBOX: 0, StatType.BIG_CHANCES_CREATED: 0,
            StatType.CORNERS: 0, StatType.BALL_POSSESSION: 22.0,
            StatType.DANGEROUS_ATTACKS: 3, StatType.KEY_PASSES: 1,
        },
    )
    extreme_gsv = GSVBuilder().build(extreme_state, priors=priors, markets=market_snapshot)
    assert det.is_ood(extreme_gsv) is True
    assert det.score(extreme_gsv) > det.threshold


def test_fitted_detector_passes_typical_state(priors, market_snapshot):
    """Score on a sample drawn from the training distribution falls
    within the bulk."""
    training = _build_realistic_training_gsvs(n=200)
    det = OODDetector().fit(training)
    # Any training sample by construction passes 99% of the time.
    flagged = sum(int(det.is_ood(g)) for g in training)
    # 99th percentile threshold → exactly 1% (= 2 samples out of 200) flagged.
    # Allow drift of ±1 due to rounding when ties land on the boundary.
    assert flagged <= 3


# ──────────────────────────────────────────────────────────────────────
# Anti-Napoli regression — the load-bearing constraint
# ──────────────────────────────────────────────────────────────────────


def test_anti_napoli_states_are_not_flagged_as_ood():
    """The 50 anti-Napoli synthetic states are realistic — they MUST
    NOT be classified OOD, else rule #9 short-circuits rule #2 and the
    Napoli regression invariant evaporates silently.

    Training is the realistic spread plus the anti-Napoli states
    themselves (since they're in-distribution for "minute 25-45,
    dominant trailing"). Threshold = 99th percentile means at most 1%
    of the combined training set sits beyond it — that is, at most a
    handful of Napoli states may flag, NOT the bulk.
    """
    napoli_priors = PreMatchPriors(
        lambda_home_prematch=2.30, lambda_away_prematch=0.95,
        expected_corners_total=10.8, expected_cards_total=4.10,
        elo_diff=140.0,
    )
    builder = GSVBuilder()
    markets = _build_minimal_markets()
    napoli_gsvs = [
        builder.build(s, priors=napoli_priors, markets=markets, dominant_team_id=HOME_ID)
        for s in _synthesize_states()
    ]
    # Train on realistic spread + the napoli states (they ARE realistic
    # examples of "dominant trailing" — in-distribution).
    realistic = _build_realistic_training_gsvs(n=200)
    det = OODDetector().fit(realistic + napoli_gsvs)
    # Rule #9 must not flag the bulk of Napoli regression states.
    flagged = sum(int(det.is_ood(g)) for g in napoli_gsvs)
    assert flagged / len(napoli_gsvs) < 0.05, (
        f"OOD detector flagged {flagged}/{len(napoli_gsvs)} anti-Napoli states "
        f"as OOD — rule #9 would short-circuit rule #2 on these"
    )


# ──────────────────────────────────────────────────────────────────────
# Rule #9 integration
# ──────────────────────────────────────────────────────────────────────


def test_rule_9_passes_when_detector_is_none(priors, market_snapshot):
    state = make_state()
    gsv = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    verdict = rule_9_ood_detector(gsv, None)
    assert verdict.allowed is True


def test_rule_9_passes_when_detector_unfitted(priors, market_snapshot):
    state = make_state()
    gsv = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    det = OODDetector()
    verdict = rule_9_ood_detector(gsv, det)
    assert verdict.allowed is True


def test_rule_9_denies_with_score_in_reason(priors, market_snapshot):
    """When fitted on a tight distribution and the GSV is far away,
    rule_9 must deny with the score embedded for audit."""
    builder = GSVBuilder()
    # Tight training: all minute=30, 0-0 baseline.
    base_states = [make_state(minute=30) for _ in range(10)]
    base_gsvs = [builder.build(s, priors=priors, markets=market_snapshot)
                 for s in base_states]
    det = OODDetector().fit(base_gsvs)
    # An off-distribution state: minute=85, 4-1.
    out_state = make_state(home_goals=4, away_goals=1, minute=85,
                           home_stats={StatType.SHOTS_TOTAL: 25,
                                       StatType.SHOTS_INSIDEBOX: 15,
                                       StatType.BALL_POSSESSION: 70.0,
                                       StatType.DANGEROUS_ATTACKS: 80,
                                       StatType.CORNERS: 12})
    out_gsv = builder.build(out_state, priors=priors, markets=market_snapshot)
    verdict = rule_9_ood_detector(out_gsv, det)
    assert verdict.allowed is False
    assert verdict.rule_number == 9
    assert "Mahalanobis" in verdict.reason


def test_run_gate_ood_is_shadow_only(priors, tmp_path):
    """Rule 9 OOD (Mahalanobis) is now SHADOW-ONLY: a candidate in an OOD
    state PASSES the gate, but a shadow denial row is recorded with
    is_shadow=True in the shadow_logger.

    Previously the gate short-circuited on rule 9. Now the candidate
    falls through to other rules. If all other rules pass, the candidate
    is ALLOWED. If another rule denies (e.g. rule_8), the denial is
    recorded under that rule's number.

    Sanity exception: total_goals outside [0,15] is still a REAL deny.
    """
    from bip.evaluation.live.engine_v3 import MarketCandidate, MarketFamily, MESResult, Thesis
    from bip.evaluation.live.engine_v3.thesis import (
        CausalChain, CausalStep, ConditionalShift, GSVPredicate,
        InvalidationTrigger, ThesisArchetype, ThesisSource, build_horizon,
    )
    from bip.evaluation.live.engine_v3.shadow_logger import ShadowLogger

    # Use an isolated narrow training set so a normal state goes OOD.
    builder = GSVBuilder()
    now = datetime.now(timezone.utc)
    from bip.evaluation.live.engine_v3 import MarketLine, MarketSnapshot
    markets = MarketSnapshot(lines={
        "match_goals_over_2.5": MarketLine(
            market_id="match_goals_over_2.5",
            side_a_decimal=1.95, side_b_decimal=1.95,
            line_value=2.5, max_stake_cap=500.0,
            last_update_utc=now - timedelta(seconds=20),
        ),
    })
    # Training: only late-game states.
    training_gsvs = [
        builder.build(make_state(minute=85), priors=priors, markets=markets)
        for _ in range(15)
    ]
    det = OODDetector().fit(training_gsvs)
    # Test GSV: minute=15 — well outside the training distribution.
    test_gsv = builder.build(
        make_state(minute=15), priors=priors, markets=markets,
    )

    # Build a candidate that passes all other rules (score=1.0 > 0.6, cvar=1.0 < 0.8*10=0.8? No).
    # Let's make a candidate that passes rule_8 too by keeping cvar low.
    thesis = Thesis(
        id="T@m15",
        archetype=ThesisArchetype.OPEN_GAME_FORMATIONS,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(
            family=MarketFamily.GOALS, direction="over", magnitude_pp=0.05,
            horizon=build_horizon("rest_of_match", test_gsv.time.minute),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.5,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=test_gsv.time.minute,
    )
    candidate = MarketCandidate(
        thesis=thesis,
        market_id="match_goals_over_2.5",
        family=MarketFamily.GOALS,
        fair_prob=0.55,
        mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_over_2.5",
            family=MarketFamily.GOALS,
            base_edge=0.05, signal_clarity=1.0, book_slowness=1.0,
            liquidity_score=1.0,
            conditional_variance=0.5,  # low enough to pass rule_8 (0.5 < 0.08*10=0.8)
            score=1.0,
        ),
    )
    shadow_log = ShadowLogger(output_root=tmp_path)
    results = run_gate([thesis], [candidate], test_gsv, ood_detector=det, shadow_logger=shadow_log)
    assert len(results) == 1
    # Candidate PASSES (shadow-only OOD doesn't block it)
    assert results[0].verdict.allowed is True, (
        f"Expected allowed=True (shadow-only OOD), got rule_number={results[0].verdict.rule_number}"
    )
    # Shadow denial was recorded
    assert shadow_log.buffer_size()[1] == 1, "Expected 1 shadow denial in buffer"


def test_pipeline_with_ood_detector_passes_normal_state(priors, market_snapshot):
    """A pipeline with an OOD detector trained on realistic states does
    NOT regress on the normal flow."""
    training = _build_realistic_training_gsvs(n=200)
    det = OODDetector().fit(training)
    pipeline = V3Pipeline(ood_detector=det)
    state = make_state(minute=30)
    out = pipeline.run(state, priors=priors, markets=market_snapshot)
    # Should not have an OOD denial — minute=30 baseline is in-distribution.
    ood_denials = [r for r in out.gate_results if r.verdict.rule_number == 9]
    assert len(ood_denials) == 0


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


def test_save_load_roundtrip_preserves_scores(tmp_path, priors, market_snapshot):
    training = _build_realistic_training_gsvs(n=100)
    det = OODDetector().fit(training)
    path = det.save(tmp_path / "det.pkl")
    loaded = OODDetector.load(path)
    assert loaded.is_fitted is True
    assert loaded.n_train == det.n_train
    assert loaded.threshold == det.threshold
    # Score consistency on a fresh sample.
    builder = GSVBuilder()
    probe = builder.build(make_state(minute=42), priors=priors, markets=market_snapshot)
    assert abs(det.score(probe) - loaded.score(probe)) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Task-3 regression tests: shadow-only rule_9 + sanity bound
# ──────────────────────────────────────────────────────────────────────


def test_rule_9_shadow_high_scoring_gsv_allowed_shadow_recorded(priors, market_snapshot, tmp_path):
    """A GSV that the detector flags as OOD: candidate PASSES the gate
    (shadow-only) and a shadow denial row is recorded.

    Strategy: train on late-game states (minute=85), test with early-game
    (minute=5). The minute dimension creates the OOD signal without
    relying on total_goals (which was removed from FEATURE_NAMES).
    """
    from bip.evaluation.live.engine_v3 import MarketCandidate, MarketFamily, MESResult, Thesis
    from bip.evaluation.live.engine_v3.thesis import (
        CausalChain, CausalStep, ConditionalShift, GSVPredicate,
        InvalidationTrigger, ThesisArchetype, ThesisSource, build_horizon,
    )
    from bip.evaluation.live.engine_v3.shadow_logger import ShadowLogger

    # Fit detector on late-game states only (minute ~85)
    builder = GSVBuilder()
    now = datetime.now(timezone.utc)
    from bip.evaluation.live.engine_v3 import MarketLine, MarketSnapshot
    tight_markets = MarketSnapshot(lines={
        "match_goals_over_2.5": MarketLine(
            market_id="match_goals_over_2.5",
            side_a_decimal=1.95, side_b_decimal=1.95,
            line_value=2.5, max_stake_cap=500.0,
            last_update_utc=now - timedelta(seconds=20),
        ),
    })
    training_gsvs = [
        builder.build(make_state(minute=85), priors=priors, markets=tight_markets)
        for _ in range(20)
    ]
    # Aggressive threshold (50th %ile) to ensure early-game states flag OOD.
    det = OODDetector().fit(training_gsvs, threshold_percentile=50.0)

    # Build a GSV at minute=5 — well outside training distribution (minute=85)
    gsv = builder.build(make_state(minute=5), priors=priors, markets=tight_markets)

    # Verify detector actually flags it as OOD.
    assert det.is_ood(gsv), "Precondition: early-game GSV must be OOD vs late-game training set"

    thesis = Thesis(
        id="SHADOW_TEST",
        archetype=ThesisArchetype.OPEN_GAME_FORMATIONS,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(
            family=MarketFamily.GOALS, direction="over", magnitude_pp=0.05,
            horizon=build_horizon("rest_of_match", 20),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.5,
        source=ThesisSource(layer="rule", identifier="SHADOW_TEST"),
        activated_at_minute=70,
    )
    candidate = MarketCandidate(
        thesis=thesis,
        market_id="match_goals_over_2.5",
        family=MarketFamily.GOALS,
        fair_prob=0.6,
        mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_over_2.5",
            family=MarketFamily.GOALS,
            base_edge=0.05, signal_clarity=1.0, book_slowness=1.0,
            liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    shadow_log = ShadowLogger(output_root=tmp_path)
    results = run_gate([thesis], [candidate], gsv, ood_detector=det, shadow_logger=shadow_log)

    # Candidate must PASS (shadow-only: OOD doesn't block)
    assert len(results) == 1
    assert results[0].verdict.allowed is True
    # Shadow denial must be recorded
    assert shadow_log.buffer_size()[1] >= 1, "Shadow denial row must be in the denial buffer"


def test_rule_9_sanity_total_goals_20_real_deny(priors, market_snapshot):
    """total_goals > 15 is a REAL enforced deny (genuinely broken GSV),
    NOT a shadow denial. Candidate is blocked regardless of is_shadow."""
    from bip.evaluation.live.engine_v3 import MarketCandidate, MarketFamily, MESResult, Thesis
    from bip.evaluation.live.engine_v3.thesis import (
        CausalChain, CausalStep, ConditionalShift, GSVPredicate,
        InvalidationTrigger, ThesisArchetype, ThesisSource, build_horizon,
    )

    # Build a GSV with 20 total goals (clearly broken)
    builder = GSVBuilder()
    broken_state = make_state(home_goals=12, away_goals=8, minute=70)
    gsv = builder.build(broken_state, priors=priors, markets=market_snapshot)

    thesis = Thesis(
        id="SANITY_TEST",
        archetype=ThesisArchetype.OPEN_GAME_FORMATIONS,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(
            family=MarketFamily.GOALS, direction="over", magnitude_pp=0.05,
            horizon=build_horizon("rest_of_match", 20),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.5,
        source=ThesisSource(layer="rule", identifier="SANITY_TEST"),
        activated_at_minute=70,
    )
    candidate = MarketCandidate(
        thesis=thesis,
        market_id="match_goals_over_2.5",
        family=MarketFamily.GOALS,
        fair_prob=0.6,
        mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_over_2.5",
            family=MarketFamily.GOALS,
            base_edge=0.05, signal_clarity=1.0, book_slowness=1.0,
            liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    # No detector needed for sanity check
    results = run_gate([thesis], [candidate], gsv, ood_detector=None)
    assert len(results) == 1
    assert results[0].verdict.allowed is False
    assert results[0].verdict.rule_number == 9
    assert "broken GSV" in results[0].verdict.reason


_V2_REAL_PKL = Path("data/cache/ood_detector_v2_real.pkl")


@pytest.mark.skipif(
    not _V2_REAL_PKL.exists(),
    reason="ood_detector_v2_real.pkl not found in data/cache/ — skip on CI without real pkl",
)
def test_real_pkl_loads_and_scores_without_exception(priors, market_snapshot):
    """Loading the real 17-feature ood_detector_v2_real.pkl and scoring a
    GSV must NOT raise a shape-mismatch error.

    This is known_correctness_constraint #1: the pkl was fit on the legacy
    17-feature schema. OODDetector.score() now auto-selects vectorize_gsv_legacy
    (17-dim) for detectors with the legacy feature_names, so the loaded pkl
    scores correctly without a dimension mismatch.
    """
    from bip.evaluation.live.engine_v3.ood_detector import _FEATURE_NAMES_LEGACY

    det = OODDetector.load(_V2_REAL_PKL)
    assert det.is_fitted
    # Confirm it was fit on the legacy 17-feature schema
    assert det.feature_names == _FEATURE_NAMES_LEGACY or len(det.feature_names) == 17

    builder = GSVBuilder()
    gsv = builder.build(make_state(minute=45), priors=priors, markets=market_snapshot)
    # Must not raise — shape mismatch was the bug this test guards
    score = det.score(gsv)
    assert score >= 0.0  # Mahalanobis distance is non-negative
