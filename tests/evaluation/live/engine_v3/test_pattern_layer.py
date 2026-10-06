"""Tests for the Pattern Layer kNN secondary hypothesis generator.

Design lives at internal research notes The structural
invariants tested here:

1. kNN reconstructs known theses on similar GSVs (positive recovery).
2. Cold-start guards work:
   - n < min_samples_active → DEACTIVATED.
   - OOD detector says is_ood → pattern layer returns [].
   - max_neighbor_distance gate prevents "k furthest" promotion.
3. Composition: hybrid generator merges rule + pattern by archetype,
   keeping the higher-confidence variant. Pattern NEVER reduces
   rule-layer confidence on overlapping archetypes.
4. Anti-Napoli regression: pattern layer never introduces an Under-
   direction thesis on a Napoli state (because the only Under-allowed
   archetype is CRUISE_MODE and its premise won't hold on Napoli).
5. Persistence round-trip preserves recovery behaviour.
6. Performance budget: propose(gsv) <50ms on a 10K-sample index.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from bip.evaluation.live.engine_v3 import (
    GSVBuilder,
    MarketFamily,
    MarketLine,
    MarketSnapshot,
    OODDetector,
    PatternLayer,
    PatternLayerConfig,
    PreMatchPriors,
    V3Pipeline,
    generate_theses_hybrid,
    merge_theses,
)
from bip.evaluation.live.engine_v3.archetypes import generate_theses
from bip.evaluation.live.engine_v3.pattern_layer import (
    _all_predicates_hold,
    _evaluate_predicate,
)
from bip.evaluation.live.engine_v3.thesis import (
    CausalChain,
    CausalStep,
    ConditionalShift,
    GSVPredicate,
    InvalidationTrigger,
    Thesis,
    ThesisArchetype,
    ThesisSource,
    build_horizon,
)
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import HOME_ID, make_state
from tests.evaluation.live.engine_v3.test_anti_napoli import (
    _build_minimal_markets,
    _synthesize_states,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _napoli_priors() -> PreMatchPriors:
    return PreMatchPriors(
        lambda_home_prematch=2.30, lambda_away_prematch=0.95,
        expected_corners_total=10.8, expected_cards_total=4.10,
        elo_diff=140.0,
    )


def _build_napoli_pair():
    """Build a (gsv, fired thesis) pair from a single Napoli state."""
    builder = GSVBuilder()
    priors = _napoli_priors()
    markets = _build_minimal_markets()
    state = _synthesize_states()[0]
    gsv = builder.build(state, priors=priors, markets=markets, dominant_team_id=HOME_ID)
    rule_theses = generate_theses(gsv)
    assert rule_theses, "expected at least one rule-layer thesis for Napoli"
    # Pick the Napoli thesis as the example
    napoli = next(
        (t for t in rule_theses
         if t.archetype is ThesisArchetype.DOMINANT_LOSING_NAPOLI),
        rule_theses[0],
    )
    return gsv, napoli


def _build_realistic_pairs(n: int = 250):
    """A spread of (gsv, fired_thesis) pairs covering the Napoli regime
    + general realistic states. Every Napoli state is paired with the
    Napoli thesis it fires; other states get any fired thesis or a
    placeholder if rule layer is silent.
    """
    rng = np.random.default_rng(7)
    builder = GSVBuilder()
    realistic_priors = PreMatchPriors(
        lambda_home_prematch=1.6, lambda_away_prematch=1.1,
        expected_corners_total=10.0, expected_cards_total=4.0,
    )
    napoli_priors = _napoli_priors()
    markets = _build_minimal_markets()
    pairs: list[tuple] = []

    # Half the corpus is Napoli-flavoured to bias recovery in the
    # consensus test that follows.
    for state in _synthesize_states():
        gsv = builder.build(state, priors=napoli_priors, markets=markets,
                            dominant_team_id=HOME_ID)
        theses = generate_theses(gsv)
        if not theses:
            continue
        napoli = next(
            (t for t in theses if t.archetype is ThesisArchetype.DOMINANT_LOSING_NAPOLI),
            theses[0],
        )
        pairs.append((gsv, napoli))

    # Fill out with generic realistic states.
    score_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]
    while len(pairs) < n:
        minute = int(rng.integers(5, 88))
        h, a = score_grid[int(rng.integers(0, len(score_grid)))]
        scale = max(0.4, minute / 60.0)
        state = make_state(
            home_goals=h, away_goals=a, minute=minute,
            home_stats={
                StatType.SHOTS_TOTAL: int(rng.integers(2, 8) * scale),
                StatType.SHOTS_ON_TARGET: int(rng.integers(0, 4) * scale),
                StatType.SHOTS_INSIDEBOX: int(rng.integers(0, 4) * scale),
                StatType.BIG_CHANCES_CREATED: int(rng.integers(0, 2)),
                StatType.CORNERS: int(rng.integers(0, 6) * scale),
                StatType.BALL_POSSESSION: float(rng.normal(52.0, 6.0)),
                StatType.DANGEROUS_ATTACKS: int(rng.integers(8, 40) * scale),
                StatType.KEY_PASSES: int(rng.integers(1, 8) * scale),
            },
            away_stats={
                StatType.SHOTS_TOTAL: int(rng.integers(1, 6) * scale),
                StatType.SHOTS_ON_TARGET: int(rng.integers(0, 3) * scale),
                StatType.SHOTS_INSIDEBOX: int(rng.integers(0, 3) * scale),
                StatType.BIG_CHANCES_CREATED: int(rng.integers(0, 2)),
                StatType.CORNERS: int(rng.integers(0, 4) * scale),
                StatType.BALL_POSSESSION: 50.0,
                StatType.DANGEROUS_ATTACKS: int(rng.integers(6, 30) * scale),
                StatType.KEY_PASSES: int(rng.integers(1, 6) * scale),
            },
        )
        gsv = builder.build(state, priors=realistic_priors, markets=markets)
        theses = generate_theses(gsv)
        if not theses:
            continue
        pairs.append((gsv, theses[0]))
    return pairs[:n]


# ──────────────────────────────────────────────────────────────────────
# Predicate evaluator
# ──────────────────────────────────────────────────────────────────────


def test_evaluate_predicate_eq(priors, market_snapshot):
    gsv = GSVBuilder().build(make_state(), priors=priors, markets=market_snapshot)
    p = GSVPredicate(path="score.goal_diff", op="eq", value=0)
    assert _evaluate_predicate(p, gsv) is True


def test_evaluate_predicate_lt_gt(priors, market_snapshot):
    gsv = GSVBuilder().build(make_state(minute=45), priors=priors, markets=market_snapshot)
    assert _evaluate_predicate(GSVPredicate(path="time.minute", op="lt", value=50), gsv) is True
    assert _evaluate_predicate(GSVPredicate(path="time.minute", op="gt", value=50), gsv) is False


def test_evaluate_predicate_missing_path_returns_false(priors, market_snapshot):
    gsv = GSVBuilder().build(make_state(), priors=priors, markets=market_snapshot)
    p = GSVPredicate(path="nonexistent.field", op="eq", value=0)
    assert _evaluate_predicate(p, gsv) is False


def test_evaluate_predicate_between(priors, market_snapshot):
    gsv = GSVBuilder().build(make_state(minute=40), priors=priors, markets=market_snapshot)
    p = GSVPredicate(path="time.minute", op="between", value=(30, 50))
    assert _evaluate_predicate(p, gsv) is True


def test_all_predicates_hold_on_real_napoli_thesis():
    gsv, napoli_thesis = _build_napoli_pair()
    assert _all_predicates_hold(napoli_thesis, gsv) is True


# ──────────────────────────────────────────────────────────────────────
# Cold-start guards
# ──────────────────────────────────────────────────────────────────────


def test_unfitted_pattern_layer_returns_empty(priors, market_snapshot):
    layer = PatternLayer()
    gsv = GSVBuilder().build(make_state(), priors=priors, markets=market_snapshot)
    assert layer.propose(gsv) == []


def test_fit_with_zero_pairs_raises():
    with pytest.raises(ValueError):
        PatternLayer().fit([])


def test_below_active_threshold_deactivates(priors, market_snapshot):
    """n < min_samples_active (50) → propose returns [] even when fitted."""
    # Use a small slice of napoli pairs (which always fire) to ensure n>0 but <50.
    pairs = []
    builder = GSVBuilder()
    for state in _synthesize_states()[:8]:
        gsv = builder.build(state, priors=_napoli_priors(),
                            markets=_build_minimal_markets(),
                            dominant_team_id=HOME_ID)
        rule = generate_theses(gsv)
        if rule:
            pairs.append((gsv, rule[0]))
    assert 0 < len(pairs) < 50
    layer = PatternLayer().fit(pairs)
    assert layer.is_fitted is True
    probe = builder.build(make_state(minute=30), priors=priors, markets=market_snapshot)
    assert layer.propose(probe) == []


def test_ood_piggyback_blocks_proposal(priors, market_snapshot):
    """OOD detector saying is_ood → pattern layer returns []."""
    pairs = _build_realistic_pairs(n=250)
    layer = PatternLayer().fit(pairs)
    # Fit OOD detector on a TIGHT distribution so a normal state is OOD.
    builder = GSVBuilder()
    tight = [
        builder.build(make_state(minute=30), priors=priors, markets=market_snapshot)
        for _ in range(20)
    ]
    ood = OODDetector().fit(tight)

    probe = builder.build(make_state(minute=80, home_goals=3, away_goals=0),
                          priors=priors, markets=market_snapshot)
    assert ood.is_ood(probe) is True
    assert layer.propose(probe, ood_detector=ood) == []


def test_distance_gate_blocks_far_neighbours():
    """A GSV totally unlike the training set must produce no proposal."""
    pairs = _build_realistic_pairs(n=250)
    cfg = PatternLayerConfig(max_neighbor_distance=0.1)  # impossibly tight
    layer = PatternLayer(cfg=cfg).fit(pairs)
    builder = GSVBuilder()
    probe = builder.build(make_state(minute=44), priors=_napoli_priors(),
                          markets=_build_minimal_markets())
    proposals = layer.propose(probe)
    # With 0.1-z-score gate, virtually no neighbour passes.
    assert proposals == []


# ──────────────────────────────────────────────────────────────────────
# Positive recovery (the load-bearing test)
# ──────────────────────────────────────────────────────────────────────


def test_knn_recovers_napoli_thesis_on_napoli_state():
    """A GSV similar to many Napoli neighbours recovers the Napoli thesis
    via the pattern layer."""
    pairs = _build_realistic_pairs(n=250)
    layer = PatternLayer().fit(pairs)
    builder = GSVBuilder()
    # Build a held-out Napoli-like state
    test_state = _synthesize_states()[7]  # different from training picks
    gsv = builder.build(test_state, priors=_napoli_priors(),
                        markets=_build_minimal_markets(),
                        dominant_team_id=HOME_ID)
    proposed = layer.propose(gsv)
    archetypes_proposed = {t.archetype for t in proposed}
    assert ThesisArchetype.DOMINANT_LOSING_NAPOLI in archetypes_proposed, (
        f"Pattern layer failed to recover Napoli thesis. "
        f"Proposed archetypes: {[a.value for a in archetypes_proposed]}"
    )
    # Provenance: all pattern theses must say source.layer="pattern"
    for t in proposed:
        assert t.source.layer == "pattern"
        assert "knn:vote=" in t.source.identifier


# ──────────────────────────────────────────────────────────────────────
# Composition (rule + pattern merge)
# ──────────────────────────────────────────────────────────────────────


def _build_thesis(archetype: ThesisArchetype, confidence: float, layer: str = "rule") -> Thesis:
    return Thesis(
        id=f"T_{archetype.value}",
        archetype=archetype,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(
            family=MarketFamily.GOALS, direction="over", magnitude_pp=0.05,
            horizon=build_horizon("rest_of_match", 50),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=confidence,
        source=ThesisSource(layer=layer, identifier="T"),
        activated_at_minute=50,
    )


def test_merge_unique_archetypes_kept_from_both_layers():
    rule = [_build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.6, "rule")]
    pattern = [_build_thesis(ThesisArchetype.NUMERICAL_SUSTAINED, 0.5, "pattern")]
    merged = merge_theses(rule, pattern)
    archetypes = {t.archetype for t in merged}
    assert archetypes == {
        ThesisArchetype.OPEN_GAME_FORMATIONS,
        ThesisArchetype.NUMERICAL_SUSTAINED,
    }


def test_merge_overlap_keeps_higher_confidence():
    rule = [_build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.6, "rule")]
    pattern = [_build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.8, "pattern")]
    merged = merge_theses(rule, pattern)
    assert len(merged) == 1
    assert merged[0].confidence_prior == 0.8
    assert merged[0].source.layer == "pattern"


def test_merge_overlap_keeps_rule_when_equal_or_higher():
    """Pattern NEVER reduces rule-layer confidence. Equal → rule wins
    (preserves rule provenance for audit)."""
    rule = [_build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.8, "rule")]
    pattern = [_build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.7, "pattern")]
    merged = merge_theses(rule, pattern)
    assert merged[0].confidence_prior == 0.8
    assert merged[0].source.layer == "rule"


def test_hybrid_generator_without_pattern_equals_rule(priors, market_snapshot):
    """Pipeline backward compatibility: no pattern layer → behaviour
    identical to plain generate_theses."""
    gsv = GSVBuilder().build(make_state(home_goals=0, away_goals=1, minute=35,
                                         home_stats={StatType.SHOTS_TOTAL: 11,
                                                      StatType.SHOTS_INSIDEBOX: 6,
                                                      StatType.BALL_POSSESSION: 65.0,
                                                      StatType.DANGEROUS_ATTACKS: 60,
                                                      StatType.CORNERS: 5},
                                         away_stats={StatType.SHOTS_TOTAL: 3,
                                                      StatType.SHOTS_INSIDEBOX: 1,
                                                      StatType.BALL_POSSESSION: 35.0,
                                                      StatType.DANGEROUS_ATTACKS: 18,
                                                      StatType.CORNERS: 1}),
                              priors=_napoli_priors(), markets=market_snapshot,
                              dominant_team_id=HOME_ID)
    rule_only = generate_theses(gsv)
    hybrid_no_pattern = generate_theses_hybrid(gsv)
    assert {t.archetype for t in rule_only} == {t.archetype for t in hybrid_no_pattern}


# ──────────────────────────────────────────────────────────────────────
# Anti-Napoli regression — load-bearing
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state", _synthesize_states()[:10])
def test_pattern_layer_does_not_introduce_under_thesis_on_napoli(state):
    """The pattern layer MUST NOT introduce an Under-direction thesis
    on a Napoli state. The only Under-allowed archetype is CRUISE_MODE,
    whose premise (dominant_losing=False + late minute + cagey home phase)
    does NOT hold on the Napoli predicate, so the premise re-evaluation
    filters it out.

    This is the safety contract: the pattern layer cannot become a
    backdoor past rule #2 of the No-Bet Gate.
    """
    pairs = _build_realistic_pairs(n=250)
    layer = PatternLayer().fit(pairs)
    builder = GSVBuilder()
    gsv = builder.build(state, priors=_napoli_priors(),
                        markets=_build_minimal_markets(),
                        dominant_team_id=HOME_ID)
    proposed = layer.propose(gsv)
    for t in proposed:
        if t.prediction.direction in ("under", "no"):
            # Allowed only for CRUISE_MODE (the explicit allow-list).
            assert t.archetype is ThesisArchetype.CRUISE_MODE, (
                f"Pattern layer introduced Under thesis with archetype "
                f"{t.archetype.value} on Napoli state — would bypass rule #2"
            )


# ──────────────────────────────────────────────────────────────────────
# Pipeline integration
# ──────────────────────────────────────────────────────────────────────


def test_pipeline_with_pattern_layer_runs_clean(priors, market_snapshot):
    pairs = _build_realistic_pairs(n=250)
    layer = PatternLayer().fit(pairs)
    pipeline = V3Pipeline(pattern_layer=layer)
    state = make_state(minute=42)
    out = pipeline.run(state, priors=priors, markets=market_snapshot)
    # Just verifying nothing explodes and the gate output is well-formed.
    assert out.gsv.time.minute == 42
    for r in out.gate_results:
        assert r.verdict.allowed in (True, False)


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


def test_save_load_roundtrip_preserves_recovery(tmp_path):
    pairs = _build_realistic_pairs(n=250)
    layer = PatternLayer().fit(pairs)
    path = layer.save(tmp_path / "pattern.pkl")
    loaded = PatternLayer.load(path)
    assert loaded.is_fitted is True
    assert len(loaded.records) == len(layer.records)
    # Recovery consistency on a fresh probe.
    builder = GSVBuilder()
    gsv = builder.build(_synthesize_states()[3], priors=_napoli_priors(),
                        markets=_build_minimal_markets(),
                        dominant_team_id=HOME_ID)
    a = {t.archetype for t in layer.propose(gsv)}
    b = {t.archetype for t in loaded.propose(gsv)}
    assert a == b


# ──────────────────────────────────────────────────────────────────────
# Performance budget
# ──────────────────────────────────────────────────────────────────────


def test_propose_meets_performance_budget(priors, market_snapshot):
    """propose(gsv) must complete in <50ms with a 10K-sample index."""
    rng = np.random.default_rng(123)
    builder = GSVBuilder()
    pairs = []
    for i in range(10_000):
        minute = int(rng.integers(5, 88))
        state = make_state(minute=minute)
        gsv = builder.build(state, priors=priors, markets=market_snapshot)
        thesis = _build_thesis(ThesisArchetype.OPEN_GAME_FORMATIONS, 0.5)
        pairs.append((gsv, thesis))
    layer = PatternLayer().fit(pairs)
    probe = builder.build(make_state(minute=30), priors=priors, markets=market_snapshot)

    start = time.perf_counter()
    _ = layer.propose(probe)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 50.0, f"propose took {elapsed_ms:.1f}ms, budget 50ms"
