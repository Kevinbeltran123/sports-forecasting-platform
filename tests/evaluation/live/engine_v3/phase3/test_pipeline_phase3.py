"""End-to-end pipeline tests with Phase-3 promoter wired in.

These verify the existing V3Pipeline output stays backward-compatible
when no promoter is supplied, AND that wiring one populates the
``promoted_picks`` field with the right cardinality.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bip.evaluation.live.engine_v3 import (
    ConditionalPredictor,
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
    V3Pipeline,
)
from bip.evaluation.live.engine_v3.phase3 import (
    CalibrationDriftDetector,
    CardsPredictor,
    NextGoalPredictor,
    PlayerPropsPredictor,
    StakeRotationPolicy,
    TierDPromoter,
)
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import AWAY_ID, HOME_ID, make_state


def _markets():
    now = datetime.now(timezone.utc)
    return MarketSnapshot(lines={
        "match_corners_over_10.5": MarketLine(
            market_id="match_corners_over_10.5", side_a_decimal=2.10,
            line_value=10.5, max_stake_cap=400.0, last_update_utc=now,
        ),
        "match_goals_under_2.5": MarketLine(
            market_id="match_goals_under_2.5", side_a_decimal=2.20,
            line_value=2.5, max_stake_cap=300.0, last_update_utc=now,
        ),
        "next_goal_home": MarketLine(
            market_id="next_goal_home", side_a_decimal=2.40,
            max_stake_cap=250.0, last_update_utc=now,
        ),
    })


def test_pipeline_without_promoter_phase1_compat():
    """Without a tier_promoter, promoted_picks must be empty (Phase 1/2 compat)."""
    state = make_state(
        home_goals=1, away_goals=0, minute=70,
        red_card_events=[(25, AWAY_ID)],
        home_stats={
            StatType.SHOTS_TOTAL: 14, StatType.SHOTS_INSIDEBOX: 6,
            StatType.BIG_CHANCES_CREATED: 2, StatType.CORNERS: 7,
            StatType.BALL_POSSESSION: 65.0,
            StatType.DANGEROUS_ATTACKS: 50, StatType.KEY_PASSES: 9,
        },
    )
    pipeline = V3Pipeline()
    out = pipeline.run(
        state, priors=PreMatchPriors(lambda_home_prematch=1.8, lambda_away_prematch=1.0),
        markets=_markets(), dominant_team_id=HOME_ID,
    )
    assert out.promoted_picks == []


def test_pipeline_with_promoter_emits_promoted_picks():
    """When tier_promoter is wired, every allowed pick gets promoted."""
    state = make_state(
        home_goals=1, away_goals=0, minute=70,
        red_card_events=[(25, AWAY_ID)],
        home_stats={
            StatType.SHOTS_TOTAL: 14, StatType.SHOTS_INSIDEBOX: 6,
            StatType.BIG_CHANCES_CREATED: 2, StatType.CORNERS: 7,
            StatType.BALL_POSSESSION: 65.0,
            StatType.DANGEROUS_ATTACKS: 50, StatType.KEY_PASSES: 9,
        },
    )
    predictor = ConditionalPredictor.default()
    predictor.register(CardsPredictor())
    predictor.register(NextGoalPredictor())
    predictor.register(PlayerPropsPredictor())
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=CalibrationDriftDetector(),
        base_unit=1.0,
    )
    pipeline = V3Pipeline(
        conditional_predictor=predictor,
        tier_promoter=promoter,
    )
    out = pipeline.run(
        state, priors=PreMatchPriors(lambda_home_prematch=1.8, lambda_away_prematch=1.0),
        markets=_markets(), dominant_team_id=HOME_ID,
    )
    # The number of promoted picks must equal allowed picks (one-to-one).
    assert len(out.promoted_picks) == len(out.allowed_picks)


def test_promoter_audit_includes_regime():
    """Every promoted pick's audit dict must carry the regime key."""
    state = make_state(home_goals=1, away_goals=0, minute=70,
                       red_card_events=[(25, AWAY_ID)])
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=CalibrationDriftDetector(),
    )
    pipeline = V3Pipeline(tier_promoter=promoter)
    out = pipeline.run(
        state, priors=PreMatchPriors(lambda_home_prematch=1.8, lambda_away_prematch=1.0),
        markets=_markets(), dominant_team_id=HOME_ID,
    )
    for pick in out.promoted_picks:
        assert "regime" in pick.audit
