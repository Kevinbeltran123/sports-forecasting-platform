"""End-to-end pipeline smoke tests.

Verifies that for a few representative states the pipeline:
- builds a GSV
- generates ≥0 theses
- runs candidates through the selector + gate
- returns a coherent ``PipelineOutput`` even when picks are zero
"""
from __future__ import annotations

from bip.evaluation.live.engine_v3 import V3Pipeline
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import AWAY_ID, HOME_ID, make_state


def test_pipeline_runs_on_napoli_scenario(priors, market_snapshot, state_napoli_scenario):
    pipeline = V3Pipeline()
    out = pipeline.run(
        state_napoli_scenario,
        priors=priors,
        markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert out.gsv.score.dominant_losing is True
    # The Napoli archetype should at least be detected (whether it
    # passes routing is a separate question)
    arch_ids = {t.archetype.value for t in out.theses}
    assert "dominant_losing_napoli" in arch_ids


def test_pipeline_on_numerical_advantage(priors, market_snapshot):
    """Late game with sustained 11v10 — NUMERICAL_SUSTAINED is suppressed
    from the _ARCHETYPE_DETECTORS registry (net-negative both sample days:
    D3 -5.79u wr=0.50 n=42; D4 -1.00u).  The detector function still exists
    but must NOT appear in pipeline output."""
    state = make_state(
        home_goals=1, away_goals=0, minute=70,
        red_card_events=[(25, AWAY_ID)],
        home_stats={
            StatType.SHOTS_TOTAL: 13, StatType.SHOTS_INSIDEBOX: 6,
            StatType.BIG_CHANCES_CREATED: 2, StatType.SHOTS_ON_TARGET: 5,
            StatType.CORNERS: 7, StatType.BALL_POSSESSION: 65.0,
            StatType.DANGEROUS_ATTACKS: 50, StatType.KEY_PASSES: 8,
        },
        away_stats={
            StatType.SHOTS_TOTAL: 4, StatType.SHOTS_INSIDEBOX: 1,
            StatType.CORNERS: 1, StatType.BALL_POSSESSION: 35.0,
            StatType.DANGEROUS_ATTACKS: 12, StatType.KEY_PASSES: 1,
        },
    )
    pipeline = V3Pipeline()
    out = pipeline.run(state, priors=priors, markets=market_snapshot,
                       dominant_team_id=HOME_ID)
    arch_ids = {t.archetype.value for t in out.theses}
    # Suppressed: detector removed from _ARCHETYPE_DETECTORS.
    assert "numerical_sustained" not in arch_ids


def test_pipeline_zero_picks_when_no_thesis(priors, market_snapshot):
    """A bland mid-game state should activate no archetypes → no picks."""
    state = make_state(minute=50)
    pipeline = V3Pipeline()
    out = pipeline.run(state, priors=priors, markets=market_snapshot)
    if not out.theses:
        assert out.candidates == []
        assert out.allowed_picks == []


def test_pipeline_output_contains_gate_results(priors, market_snapshot, state_napoli_scenario):
    """Every candidate must be recorded in gate_results (allowed or
    denied). This is the audit-trail contract (sec 7.2)."""
    pipeline = V3Pipeline()
    out = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert len(out.gate_results) == len(out.candidates)


def test_pipeline_dedupes_repeat_emission_within_fixture(
    priors, market_snapshot, state_napoli_scenario
):
    """Same (fixture, archetype, market_id) re-firing across consecutive
    frames must emit exactly once. Day-4 (2026-05-13) observed 271 raw
    picks vs 45 unique trios = 6x avg re-emission inflation. This
    invariant kills that inflation at the pipeline boundary."""
    pipeline = V3Pipeline()

    # Frame 1: napoli archetype fires; some picks get through.
    out1 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    first_emission_count = len(out1.allowed_picks)
    # Pre-condition: this scenario produces at least one allowed pick
    # otherwise the test is vacuous.
    assert first_emission_count > 0, "scenario must emit ≥1 pick"
    first_trios = {(p.candidate.thesis.archetype.value, p.candidate.market_id)
                   for p in out1.allowed_picks}
    assert out1.deduped_count == 0

    # Frame 2: identical state → no NEW emissions, deduped_count == frame1 emissions.
    out2 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert out2.allowed_picks == []
    assert out2.deduped_count == first_emission_count

    # Frame 3: still no emissions; theses + candidates still generate
    # (audit trail must persist) but no picks leak through.
    out3 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert out3.allowed_picks == []
    assert out3.theses, "theses must still generate for audit"


def test_pipeline_dedupe_disabled_emits_each_frame(
    priors, market_snapshot, state_napoli_scenario
):
    """Opt-out flag for backward compatibility / debugging — when
    ``dedupe_emissions=False`` every frame re-emits as before."""
    pipeline = V3Pipeline(dedupe_emissions=False)

    out1 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    out2 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    # Same scenario → same allowed picks each frame.
    assert len(out2.allowed_picks) == len(out1.allowed_picks)
    assert out2.deduped_count == 0


def test_pipeline_dedupe_isolated_per_fixture(priors, market_snapshot, state_napoli_scenario):
    """Dedup sets are keyed per fixture_id — a trio emitted in fixture A
    must NOT block the same trio in fixture B."""
    pipeline = V3Pipeline()
    # Frame 1 in fixture 1234 (the make_state default)
    out_a = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert len(out_a.allowed_picks) > 0

    # Frame 1 in fixture 5678 — new fixture should emit fresh picks.
    from dataclasses import replace
    state_b = replace(state_napoli_scenario, fixture_id=5678)
    out_b = pipeline.run(
        state_b, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert len(out_b.allowed_picks) > 0, "different fixture must not be deduped"


def test_pipeline_reset_dedupe_re_enables_emission(
    priors, market_snapshot, state_napoli_scenario
):
    """``reset_dedupe`` is the test-only escape hatch. It clears the
    per-fixture set so the same trio fires again on the next frame."""
    pipeline = V3Pipeline()
    out1 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    out2 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert out2.allowed_picks == []

    pipeline.reset_dedupe(state_napoli_scenario.fixture_id)
    out3 = pipeline.run(
        state_napoli_scenario, priors=priors, markets=market_snapshot,
        dominant_team_id=HOME_ID,
    )
    assert len(out3.allowed_picks) == len(out1.allowed_picks)
