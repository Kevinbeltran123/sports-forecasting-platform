"""Confidence Modulator unit tests.

Coverage:
- The modulator only multiplies, never flips direction.
- Aligned evidence → multiplier > 1.0; opposed evidence → < 1.0.
- Soft-veto floor (< 0.30) → 0.0.
"""
from __future__ import annotations

from bip.evaluation.live.engine_v3.phase3 import modulate
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


def _thesis(*, family: MarketFamily, direction: str,
            archetype: ThesisArchetype = ThesisArchetype.NUMERICAL_SUSTAINED) -> Thesis:
    return Thesis(
        id="T@m60", archetype=archetype,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="c", effect="e", mechanism="m")]),
        prediction=ConditionalShift(
            family=family, direction=direction, magnitude_pp=0.05,
            horizon=build_horizon("rest_of_match", 60),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="g")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=60,
    )


def test_modulator_returns_neutral_when_no_evidence(base_gsv):
    """Base GSV has xg_diff = 0.4, slight home edge; for a directionless
    over-thesis evidence is mostly neutral."""
    thesis = _thesis(family=MarketFamily.GOALS, direction="over")
    out = modulate(base_gsv, thesis)
    assert 0.7 <= out.multiplier <= 1.3


def test_modulator_aligned_home_thesis_amplifies(napoli_gsv):
    """In the Napoli scenario, home has +1.85 divergence + hot xG +
    65% possession → all aligned with a home-direction thesis."""
    thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="home")
    out = modulate(napoli_gsv, thesis)
    assert out.xg_alignment > 0
    assert out.shot_trend_alignment > 0
    assert out.possession_alignment > 0
    assert out.multiplier > 1.0


def test_modulator_opposed_thesis_dampens(napoli_gsv):
    """Same Napoli state but thesis points to away — every signal opposes."""
    thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="away")
    out = modulate(napoli_gsv, thesis)
    assert out.xg_alignment < 0
    assert out.possession_alignment < 0
    # Multiplier should be substantially below 1.0
    assert out.multiplier < 1.0


def test_modulator_soft_veto_at_low_confidence():
    """Construct a synthetic GSV with all three signals strongly opposed
    to a home-direction thesis → modulator should hit the soft-veto floor."""
    from datetime import datetime, timezone

    from bip.evaluation.live.engine_v3 import (
        GameStateVector, MarketSnapshot, PreMatchPriors,
    )
    from bip.evaluation.live.engine_v3.gsv import (
        CardsState, CornerState, FlowState, NumericalState, RosterState,
        ScoreState, TacticalState, TimeState, XGState,
    )

    gsv = GameStateVector(
        fixture_id=1, state_version=1, timestamp_utc=datetime.now(timezone.utc),
        home_team_id=100, away_team_id=200,
        score=ScoreState(home_goals=0, away_goals=1, goal_diff=-1,
                         dominant_team_id=100, dominant_losing=True),
        time=TimeState(minute=70, period="2H", time_remaining_match=20.0),
        numerical=NumericalState(),
        # All evidence points to AWAY:
        xg=XGState(home_xg_total=0.3, away_xg_total=2.0, xg_diff=-1.7,
                   xg_vs_score_divergence=-0.9,  # away is over-performing
                   xg_per_min_home_last_15=0.001,
                   xg_per_min_away_last_15=0.05),
        flow=FlowState(possession_home_5min=25.0),
        corners=CornerState(),
        cards=CardsState(),
        roster=RosterState(),
        tactical=TacticalState(),
        priors=PreMatchPriors(lambda_home_prematch=1.4, lambda_away_prematch=1.4),
        markets=MarketSnapshot(),
    )
    # Thesis claims HOME — every signal opposes it.
    thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="home")
    out = modulate(gsv, thesis)
    assert out.multiplier == 0.0


def test_modulator_directionless_thesis_uses_abs_divergence(napoli_gsv):
    """Over/under theses get a boost from |xg_vs_score_divergence|."""
    thesis = _thesis(family=MarketFamily.GOALS, direction="over")
    out = modulate(napoli_gsv, thesis)
    assert out.xg_alignment > 0  # large |divergence| favours over


def test_modulator_breakdown_audit_fields(napoli_gsv):
    """The breakdown dataclass must expose each signal — required for audit."""
    thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="home")
    out = modulate(napoli_gsv, thesis)
    assert hasattr(out, "xg_alignment")
    assert hasattr(out, "shot_trend_alignment")
    assert hasattr(out, "possession_alignment")
    assert hasattr(out, "multiplier")
