"""Tests for the mispricing window classifier + No-Bet rule #10.

Window thresholds (defaults, all configurable):
    HOT       0-60s    multiplier 1.10
    OPTIMAL   60-180s  multiplier 1.20
    WARM     180-600s  multiplier 0.70
    COLD     >600s     multiplier 0.30 (vetoed by rule #10 if edge < 0.08)
    INDEFINITE         multiplier 0.85 (no event observed, conservative)

Tested invariants:
- Each window class is hit at the right age boundary, with the right
  multiplier returned.
- INDEFINITE (no event) gets the conservative multiplier and passes
  rule #10 (we don't veto on absence of event).
- COLD window denies via rule #10 ONLY when edge below the cold edge
  threshold; high-edge COLD picks still pass (rare but legitimate).
- Anti-Napoli regression suite is NOT regressed by this rule: those
  states have no critical event (their state.minute=35 is "ongoing
  play", no recent goal/red flag) → INDEFINITE → no rule-10 veto.
- Pipeline output exposes the WindowResult.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bip.evaluation.live.engine_v3 import (
    GSVBuilder,
    MarketCandidate,
    MarketFamily,
    MarketLine,
    MarketSnapshot,
    MESResult,
    MispricingWindowConfig,
    PreMatchPriors,
    V3Pipeline,
    WindowLabel,
    classify_window,
)
from bip.evaluation.live.engine_v3.gsv import CriticalEvent
from bip.evaluation.live.engine_v3.no_bet_gate import (
    rule_10_mispricing_window,
    run_gate,
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
from tests.evaluation.live.engine_v3.conftest import make_state


# ──────────────────────────────────────────────────────────────────────
# Pure classifier
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "age,expected_label,expected_mult",
    [
        (None, WindowLabel.INDEFINITE, 0.85),
        (0.0, WindowLabel.HOT, 1.10),
        (30.0, WindowLabel.HOT, 1.10),
        (59.9, WindowLabel.HOT, 1.10),
        (60.0, WindowLabel.OPTIMAL, 1.20),
        (120.0, WindowLabel.OPTIMAL, 1.20),
        (180.0, WindowLabel.OPTIMAL, 1.20),
        (180.1, WindowLabel.WARM, 0.70),
        (400.0, WindowLabel.WARM, 0.70),
        (600.0, WindowLabel.WARM, 0.70),
        (600.1, WindowLabel.COLD, 0.30),
        (800.0, WindowLabel.COLD, 0.30),
        (3600.0, WindowLabel.COLD, 0.30),
    ],
)
def test_classify_boundary_values(age, expected_label, expected_mult):
    """Boundary parametric — caught the off-by-one bug from operator
    memory; same pattern applied here."""
    res = classify_window(age)
    assert res.label is expected_label
    assert res.multiplier == pytest.approx(expected_mult)
    assert res.age_sec == age


def test_classify_config_override():
    cfg = MispricingWindowConfig(
        hot_end_sec=30.0, optimal_end_sec=90.0, warm_end_sec=300.0,
    )
    assert classify_window(45.0, cfg).label is WindowLabel.OPTIMAL
    assert classify_window(200.0, cfg).label is WindowLabel.WARM
    assert classify_window(500.0, cfg).label is WindowLabel.COLD


# ──────────────────────────────────────────────────────────────────────
# Rule #10 behaviour
# ──────────────────────────────────────────────────────────────────────


def _make_candidate(edge: float) -> MarketCandidate:
    thesis = Thesis(
        id="T@m70",
        archetype=ThesisArchetype.OPEN_GAME_FORMATIONS,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(
            family=MarketFamily.GOALS, direction="over",
            magnitude_pp=0.05, horizon=build_horizon("rest_of_match", 70),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.5,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=70,
    )
    return MarketCandidate(
        thesis=thesis,
        market_id="match_goals_over_2.5",
        family=MarketFamily.GOALS,
        fair_prob=0.55,
        mes=MESResult(
            thesis_id=thesis.id, market_id="match_goals_over_2.5",
            family=MarketFamily.GOALS,
            base_edge=edge, signal_clarity=1.0, book_slowness=1.0,
            liquidity_score=1.0, conditional_variance=1.0, score=1.0,
        ),
    )


def test_rule_10_passes_when_window_hot():
    cfg = MispricingWindowConfig()
    window = classify_window(30.0, cfg)
    cand = _make_candidate(edge=0.03)  # low edge
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is True


def test_rule_10_passes_when_window_optimal():
    cfg = MispricingWindowConfig()
    window = classify_window(120.0, cfg)
    cand = _make_candidate(edge=0.03)
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is True


def test_rule_10_passes_when_window_warm():
    cfg = MispricingWindowConfig()
    window = classify_window(400.0, cfg)
    cand = _make_candidate(edge=0.03)
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is True


def test_rule_10_passes_when_window_indefinite():
    """No critical event observed → INDEFINITE → policy is to pass
    rule 10. The conservative multiplier (0.85) applies downstream
    to stake sizing, NOT to gating."""
    cfg = MispricingWindowConfig()
    window = classify_window(None, cfg)
    cand = _make_candidate(edge=0.03)
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is True


def test_rule_10_denies_cold_window_with_low_edge():
    cfg = MispricingWindowConfig()
    window = classify_window(800.0, cfg)
    cand = _make_candidate(edge=0.03)  # below 0.08 threshold
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is False
    assert verdict.rule_number == 10
    assert "COLD" in verdict.reason


def test_rule_10_passes_cold_window_when_edge_is_massive():
    """A COLD window with truly large edge still passes — the system
    should not veto when the signal is unambiguous, even if the book
    "had time" to adjust. Rare but legitimate (e.g. liquidity-driven
    line lag on an obscure prop)."""
    cfg = MispricingWindowConfig()
    window = classify_window(800.0, cfg)
    cand = _make_candidate(edge=0.20)  # 20%, well above 8%
    verdict = rule_10_mispricing_window(cand, window, cfg)
    assert verdict.allowed is True


# ──────────────────────────────────────────────────────────────────────
# Pipeline integration
# ──────────────────────────────────────────────────────────────────────


def test_pipeline_output_exposes_window(priors, market_snapshot):
    pipeline = V3Pipeline()
    state = make_state()
    out = pipeline.run(state, priors=priors, markets=market_snapshot)
    assert out.mispricing_window is not None
    # No critical event in this state → INDEFINITE
    assert out.mispricing_window.label is WindowLabel.INDEFINITE


def test_pipeline_window_respects_critical_event_age(priors, market_snapshot):
    """When the pipeline is called with a known critical event age,
    the window classifier surfaces it correctly on the output."""
    pipeline = V3Pipeline()
    state = make_state(minute=70)
    now = datetime.now(timezone.utc)
    out = pipeline.run(
        state, priors=priors, markets=market_snapshot,
        last_critical_event=CriticalEvent(
            kind="goal", minute=68, team_id=100,
            timestamp_utc=now - timedelta(seconds=120),
        ),
        last_critical_event_age_sec=120.0,
    )
    assert out.mispricing_window is not None
    assert out.mispricing_window.label is WindowLabel.OPTIMAL


def test_run_gate_routes_cold_low_edge_to_rule_10(priors):
    """End-to-end on the gate: cold + low edge denies under rule 10."""
    builder = GSVBuilder()
    # Build a GSV with last_critical_event_age_sec = 800 (cold)
    now = datetime.now(timezone.utc)
    state = make_state(minute=80)
    markets = MarketSnapshot(lines={
        "match_goals_over_2.5": MarketLine(
            market_id="match_goals_over_2.5",
            side_a_decimal=1.95, side_b_decimal=1.95,
            line_value=2.5, max_stake_cap=500.0,
            last_update_utc=now - timedelta(seconds=20),
        ),
    })
    gsv = builder.build(
        state, priors=priors, markets=markets,
        last_critical_event=CriticalEvent(
            kind="goal", minute=60, team_id=100,
            timestamp_utc=now - timedelta(seconds=800),
        ),
        last_critical_event_age_sec=800.0,
    )
    cand = _make_candidate(edge=0.03)
    # Bypass rules 5 (MES≥0.6), 7 (liquidity>0), 8 (variance < 0.8).
    cand_high_mes = MarketCandidate(
        thesis=cand.thesis,
        market_id=cand.market_id,
        family=cand.family,
        fair_prob=cand.fair_prob,
        mes=MESResult(
            thesis_id=cand.thesis.id, market_id=cand.market_id,
            family=cand.family,
            base_edge=0.03, signal_clarity=1.0, book_slowness=1.0,
            liquidity_score=1.0, conditional_variance=0.5, score=1.0,
        ),
    )
    results = run_gate([cand.thesis], [cand_high_mes], gsv)
    assert len(results) == 1
    assert results[0].verdict.allowed is False
    assert results[0].verdict.rule_number == 10


# ──────────────────────────────────────────────────────────────────────
# Anti-Napoli regression — the load-bearing invariant
# ──────────────────────────────────────────────────────────────────────


def test_anti_napoli_states_are_indefinite_window(priors):
    """The 50 anti-Napoli synthetic states have no critical event
    anchored. Their window must be INDEFINITE, not COLD — otherwise
    rule #10 could accidentally accept where rule #2 should deny.

    This is the symmetric invariant to the OOD detector test: rule
    #10's "cold" path must not accidentally rescue a Napoli candidate
    that rule #2 was about to kill.
    """
    from tests.evaluation.live.engine_v3.test_anti_napoli import _synthesize_states

    builder = GSVBuilder()
    napoli_priors = PreMatchPriors(
        lambda_home_prematch=2.30, lambda_away_prematch=0.95,
        expected_corners_total=10.8, expected_cards_total=4.10,
        elo_diff=140.0,
    )
    markets = MarketSnapshot(lines={
        "match_goals_under_2.5": MarketLine(
            market_id="match_goals_under_2.5",
            side_a_decimal=1.85, side_b_decimal=2.10,
            line_value=2.5, max_stake_cap=400.0,
            last_update_utc=datetime.now(timezone.utc),
        ),
    })
    for state in _synthesize_states()[:10]:
        gsv = builder.build(state, priors=napoli_priors, markets=markets)
        # By construction these states have no anchored critical event.
        assert gsv.last_critical_event_age_sec is None
        window = classify_window(gsv.last_critical_event_age_sec)
        assert window.label is WindowLabel.INDEFINITE
