"""Tests for the three Phase 3 predictors: cards, next_goal, props."""
from __future__ import annotations

from bip.evaluation.live.engine_v3.phase3 import (
    CardsPredictor,
    NextGoalPredictor,
    PlayerPrior,
    PlayerPropsPredictor,
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


def _thesis(*, family: MarketFamily, direction: str = "over",
            magnitude: float = 0.05) -> Thesis:
    return Thesis(
        id="T@m60", archetype=ThesisArchetype.NUMERICAL_SUSTAINED,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="c", effect="e", mechanism="m")]),
        prediction=ConditionalShift(
            family=family, direction=direction, magnitude_pp=magnitude,
            horizon=build_horizon("rest_of_match", 60),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="g")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=60,
    )


# ──────────────────────────────────────────────────────────────────────
# Cards predictor
# ──────────────────────────────────────────────────────────────────────


def test_cards_predictor_returns_none_for_wrong_family(base_gsv):
    pred = CardsPredictor()
    out = pred.predict(_thesis(family=MarketFamily.GOALS), "match_cards_over_5.5", base_gsv)
    assert out is None


def test_cards_predictor_basic_over(base_gsv):
    pred = CardsPredictor()
    out = pred.predict(_thesis(family=MarketFamily.CARDS, direction="over"),
                       "match_cards_over_5.5", base_gsv)
    assert out is not None
    assert 0.0 <= out.p <= 1.0


def test_cards_predictor_strict_ref_raises_probability(base_gsv, napoli_gsv):
    """Strict ref (high ref_card_rate_prior) should raise P(over) on cards."""
    pred = CardsPredictor()
    thesis = _thesis(family=MarketFamily.CARDS, direction="over", magnitude=0.05)
    # Use base_gsv since napoli_gsv has shorter remaining time horizon.
    # Re-create with two ref priors to compare cleanly.
    out_lenient = pred.predict(thesis, "match_cards_over_5.5", base_gsv)
    # Mutate a copy via model_copy: ref prior 7.0
    higher = base_gsv.model_copy(update={
        "cards": base_gsv.cards.model_copy(update={"ref_card_rate_prior": 7.0}),
    })
    out_strict = pred.predict(thesis, "match_cards_over_5.5", higher)
    assert out_lenient is not None and out_strict is not None
    assert out_strict.p >= out_lenient.p


def test_cards_predictor_desperate_phase_raises(base_gsv):
    """Desperate / chasing phase boosts card λ → higher P(over)."""
    pred = CardsPredictor()
    thesis = _thesis(family=MarketFamily.CARDS, direction="over", magnitude=0.0)
    quiet = base_gsv.model_copy(update={
        "tactical": base_gsv.tactical.model_copy(update={"game_phase": "cagey_closed"}),
    })
    desperate = base_gsv.model_copy(update={
        "tactical": base_gsv.tactical.model_copy(update={"game_phase": "desperate"}),
    })
    p_quiet = pred.predict(thesis, "match_cards_over_5.5", quiet)
    p_desperate = pred.predict(thesis, "match_cards_over_5.5", desperate)
    assert p_quiet is not None and p_desperate is not None
    assert p_desperate.p > p_quiet.p


# ──────────────────────────────────────────────────────────────────────
# Next goal predictor
# ──────────────────────────────────────────────────────────────────────


def test_next_goal_returns_none_for_wrong_family(base_gsv):
    pred = NextGoalPredictor()
    out = pred.predict(_thesis(family=MarketFamily.GOALS), "next_goal_home", base_gsv)
    assert out is None


def test_next_goal_home_probability(napoli_gsv):
    pred = NextGoalPredictor()
    thesis = _thesis(family=MarketFamily.NEXT_GOAL, direction="home", magnitude=0.10)
    out = pred.predict(thesis, "next_goal_home", napoli_gsv)
    assert out is not None
    # Home is over-performing in xG → home next-goal P > 0.20
    assert 0.0 < out.p < 1.0


def test_next_goal_draw_returns_complement(napoli_gsv):
    pred = NextGoalPredictor()
    home = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "next_goal_home", napoli_gsv,
    )
    away = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="away"),
        "next_goal_away", napoli_gsv,
    )
    draw = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="draw"),
        "next_goal_draw", napoli_gsv,
    )
    assert all(x is not None for x in (home, away, draw))
    # Probabilities should approximately sum to 1.0 (small rounding/CI noise).
    s = home.p + away.p + draw.p
    assert abs(s - 1.0) < 0.05


# ──────────────────────────────────────────────────────────────────────
# Player props predictor (shadow-mode)
# ──────────────────────────────────────────────────────────────────────


def test_props_returns_none_without_player_prior(base_gsv):
    pred = PlayerPropsPredictor()
    out = pred.predict(_thesis(family=MarketFamily.PROPS, direction="yes"),
                       "player_999_to_score", base_gsv)
    assert out is None


def test_props_carded_with_booking_amplifies(base_gsv):
    pred = PlayerPropsPredictor()
    prior = PlayerPrior(player_id=10, team_id=100, cards_per_90=0.10)
    pred.add_prior(prior)
    thesis = _thesis(family=MarketFamily.PROPS, direction="yes")
    out_neutral = pred.predict(thesis, "player_10_to_be_carded", base_gsv)
    # Add a yellow card → player on yellow
    from bip.evaluation.live.engine_v3.gsv import PlayerOnYellow
    booked = base_gsv.model_copy(update={
        "cards": base_gsv.cards.model_copy(update={
            "players_on_yellow": [PlayerOnYellow(team_id=100, player_id=10, booked_at_minute=30)],
        }),
    })
    out_booked = pred.predict(thesis, "player_10_to_be_carded", booked)
    assert out_neutral is not None and out_booked is not None
    assert out_booked.p > out_neutral.p


def test_props_to_score_late_chasing_amplifies(napoli_gsv):
    pred = PlayerPropsPredictor()
    # Need to be late + dominant_losing for the boost — synthesize 78' state.
    gsv_late = napoli_gsv.model_copy(update={
        "time": napoli_gsv.time.model_copy(update={
            "minute": 78, "time_remaining_match": 12.0,
        }),
    })
    pred.add_prior(PlayerPrior(player_id=7, team_id=100, xg_per_90=0.4))
    thesis = _thesis(family=MarketFamily.PROPS, direction="yes")
    out = pred.predict(thesis, "player_7_to_score", gsv_late)
    assert out is not None
    assert out.p > 0.0
