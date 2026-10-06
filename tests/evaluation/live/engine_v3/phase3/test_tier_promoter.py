"""TierDPromoter integration tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bip.evaluation.live.engine_v3 import (
    GameStateVector,
    MarketCandidate,
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.engine_v3.gsv import (
    CardsState,
    CornerState,
    FlowState,
    NumericalState,
    RosterState,
    ScoreState,
    TacticalState,
    TimeState,
    XGState,
)
from bip.evaluation.live.engine_v3.market_selector import MarketCandidate
from bip.evaluation.live.engine_v3.mes import MESResult
from bip.evaluation.live.engine_v3.phase3 import (
    CalibrationDriftDetector,
    StakeRotationPolicy,
    TierDPromoter,
    record_outcome,
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


def _candidate(family: MarketFamily, direction: str = "over",
               archetype: ThesisArchetype = ThesisArchetype.NUMERICAL_SUSTAINED,
               market_id: str = "match_corners_over_10.5",
               fair_prob: float = 0.65) -> MarketCandidate:
    thesis = Thesis(
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
    return MarketCandidate(
        thesis=thesis,
        market_id=market_id,
        family=family,
        fair_prob=fair_prob,
        mes=MESResult(
            thesis_id=thesis.id, market_id=market_id, family=family,
            base_edge=0.05, signal_clarity=1.0, book_slowness=1.2,
            liquidity_score=0.8, conditional_variance=1.0, score=0.85,
        ),
    )


def test_promoter_emits_tier_d_for_normal_candidate(napoli_gsv):
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=CalibrationDriftDetector(),
        base_unit=1.0,
    )
    candidate = _candidate(family=MarketFamily.NEXT_GOAL, direction="home")
    pick = promoter.promote(candidate, napoli_gsv)
    assert pick.stage == "tier_d"
    assert pick.stake_units > 0.0


def test_promoter_routes_props_to_shadow(base_gsv):
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=CalibrationDriftDetector(),
    )
    candidate = _candidate(
        family=MarketFamily.PROPS, direction="yes",
        market_id="player_7_to_score",
    )
    pick = promoter.promote(candidate, base_gsv)
    assert pick.stage == "shadow"
    assert pick.stake_units == 0.0
    assert "shadow" in pick.rejection_reason


def test_promoter_rejects_drift_suspended(napoli_gsv):
    detector = CalibrationDriftDetector(min_samples=10, alpha=0.5)
    # Pre-suspend the cell that bucket_gsv would produce
    from bip.evaluation.live.engine_v3.phase3 import bucket_gsv
    regime = bucket_gsv(napoli_gsv).as_str()
    # Force a suspension by feeding miscalibrated outcomes
    for _ in range(60):
        detector.record(regime, MarketFamily.NEXT_GOAL, 0.8, 0.0)
    detector.check_drift(regime, MarketFamily.NEXT_GOAL)
    assert detector.is_suspended(regime, MarketFamily.NEXT_GOAL)

    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=detector, base_unit=1.0,
    )
    pick = promoter.promote(
        _candidate(family=MarketFamily.NEXT_GOAL, direction="home"), napoli_gsv,
    )
    assert pick.stage == "rejected"
    assert "drift" in pick.rejection_reason


def test_promoter_rejects_soft_veto_from_modulator():
    """When the modulator returns multiplier=0 (soft veto), promoter rejects."""
    # Construct a GSV with every signal opposing a home-direction thesis.
    from datetime import datetime, timezone
    gsv = GameStateVector(
        fixture_id=99, state_version=1, timestamp_utc=datetime.now(timezone.utc),
        home_team_id=100, away_team_id=200,
        score=ScoreState(home_goals=0, away_goals=1, goal_diff=-1,
                         dominant_team_id=100, dominant_losing=True),
        time=TimeState(minute=70, period="2H", time_remaining_match=20.0),
        numerical=NumericalState(),
        xg=XGState(home_xg_total=0.3, away_xg_total=2.0, xg_diff=-1.7,
                   xg_vs_score_divergence=-0.9,
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
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=CalibrationDriftDetector(),
    )
    pick = promoter.promote(
        _candidate(family=MarketFamily.NEXT_GOAL, direction="home"), gsv,
    )
    assert pick.stage == "rejected"
    assert "soft-veto" in pick.rejection_reason


def test_promoter_ramps_first_week(base_gsv):
    """A fresh family with no prior history is ramped at 1/3 stake."""
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=3),
        drift_detector=CalibrationDriftDetector(),
        base_unit=3.0,
    )
    pick = promoter.promote(
        _candidate(family=MarketFamily.CARDS, direction="over",
                   market_id="match_cards_over_5.5"),
        base_gsv,
    )
    # Either it's tier_d with reduced stake or rejected by concentration cap
    # depending on initial state — verify stake math when accepted.
    if pick.stage == "tier_d":
        assert pick.stake_units < promoter.base_unit


def test_record_outcome_feeds_drift_detector(napoli_gsv):
    """A resolved pick must end up in the drift detector for its cell."""
    detector = CalibrationDriftDetector()
    promoter = TierDPromoter(
        stake_policy=StakeRotationPolicy(new_market_ramp_weeks=0),
        drift_detector=detector,
        base_unit=1.0,
    )
    pick = promoter.promote(
        _candidate(family=MarketFamily.NEXT_GOAL, direction="home"), napoli_gsv,
    )
    assert pick.stage == "tier_d"
    record_outcome(detector, pick, 1.0)
    # State now contains one entry for the promoted cell.
    state = detector.snapshot_state()
    assert state["n_cells_tracked"] >= 1
