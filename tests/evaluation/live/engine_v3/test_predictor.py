"""Conditional Predictor sanity tests for the Phase-1 corners+goals scaffold."""
from __future__ import annotations

import pytest

from bip.evaluation.live.engine_v3 import (
    ConditionalPredictor,
    CornersPredictor,
    GameStateVector,
    Goals2HPredictor,
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
from datetime import datetime, timezone


def _gsv(*, minute: int = 60, corners_home: int = 4, corners_away: int = 2,
         game_phase: str = "open_attacking") -> GameStateVector:
    return GameStateVector(
        fixture_id=1, state_version=1,
        timestamp_utc=datetime.now(timezone.utc),
        home_team_id=100, away_team_id=200,
        score=ScoreState(home_goals=1, away_goals=1, goal_diff=0,
                         dominant_team_id=100, dominant_losing=False),
        time=TimeState(minute=minute, period="2H",
                       time_remaining_match=max(0, 90 - minute)),
        numerical=NumericalState(),
        xg=XGState(),
        flow=FlowState(),
        corners=CornerState(corners_home=corners_home, corners_away=corners_away),
        cards=CardsState(),
        roster=RosterState(),
        tactical=TacticalState(game_phase=game_phase),  # type: ignore[arg-type]
        priors=PreMatchPriors(expected_corners_total=10.4),
        markets=MarketSnapshot(),
    )


def _thesis(family: MarketFamily = MarketFamily.CORNERS,
            direction: str = "over",
            magnitude: float = 0.07) -> Thesis:
    return Thesis(
        id="T@m60", archetype=ThesisArchetype.NUMERICAL_SUSTAINED,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(cause="x", effect="y", mechanism="z")]),
        prediction=ConditionalShift(family=family, direction=direction,
                                    magnitude_pp=magnitude,
                                    horizon=build_horizon("rest_of_match", 60)),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="T"),
        activated_at_minute=60,
    )


def test_corners_predictor_basic_over_line():
    pred = CornersPredictor()
    gsv = _gsv(minute=60, corners_home=4, corners_away=3)  # 7 already
    out = pred.predict(_thesis(direction="over"), "match_corners_over_10.5", gsv)
    assert out is not None
    # 7 already + Poisson expected ≈ 3.5 more → P(>=4 more) is non-trivial.
    assert 0.0 < out.p < 1.0


def test_corners_predictor_handles_no_line():
    pred = CornersPredictor()
    gsv = _gsv()
    out = pred.predict(_thesis(), "match_corners_over_bogus", gsv)
    assert out is None  # no parseable line


def test_corners_predictor_returns_none_for_wrong_family():
    pred = CornersPredictor()
    gsv = _gsv()
    out = pred.predict(_thesis(family=MarketFamily.GOALS), "match_corners_over_10.5", gsv)
    assert out is None


def test_goals_predictor_lower_p_in_cagey_phase():
    pred = Goals2HPredictor()
    cagey = _gsv(minute=80, game_phase="cagey_closed")
    open_g = _gsv(minute=80, game_phase="open_attacking")
    cagey_p = pred.predict(_thesis(family=MarketFamily.GOALS, direction="over"),
                           "match_goals_over_2.5", cagey)
    open_p = pred.predict(_thesis(family=MarketFamily.GOALS, direction="over"),
                          "match_goals_over_2.5", open_g)
    assert cagey_p is not None and open_p is not None
    # Cagey phase has 0.55× multiplier vs open → lower P(over)
    assert cagey_p.p < open_p.p


def test_composite_predictor_dispatches_by_family():
    cp = ConditionalPredictor.default()
    gsv = _gsv()
    corners = cp.predict(_thesis(family=MarketFamily.CORNERS),
                          "match_corners_over_10.5", gsv)
    goals = cp.predict(_thesis(family=MarketFamily.GOALS),
                        "match_goals_over_2.5", gsv)
    assert corners is not None
    assert goals is not None
    assert corners.family == MarketFamily.CORNERS
    assert goals.family == MarketFamily.GOALS


# ──────────────────────────────────────────────────────────────────────
# Routing-bug regressions (Day-3 silence postmortem)
# ──────────────────────────────────────────────────────────────────────


def test_corners_predictor_rejects_goals_market_id():
    """The CornersPredictor must not score a market whose id maps to a
    non-corners family — even if a parseable number can be extracted
    from the id. This was the Day-3 bug where "5.5" in
    "alternative_match_goals_over_5.5" was treated as a corners line
    and produced an inflated fair_prob ≈ 1.0.
    """
    pred = CornersPredictor()
    gsv = _gsv(minute=67, corners_home=4, corners_away=4)  # 8 corners already
    # The market id matches the GOALS family ("match_goals_") despite the
    # parseable "5.5" suffix. The defensive check must return None.
    out = pred.predict(
        _thesis(family=MarketFamily.CORNERS, direction="over"),
        "alternative_match_goals_over_5.5",
        gsv,
    )
    assert out is None


def test_goals_predictor_rejects_btts_thesis():
    """The Goals2H predictor must NOT serve BTTS theses. The thesis
    "both teams score" is semantically distinct from "over X.5 goals" —
    a Phase-3 BTTSPredictor will own that path. Until then, BTTS
    theses with no registered BTTS predictor must return None, not be
    silently absorbed by Goals2H."""
    pred = Goals2HPredictor()
    gsv = _gsv(minute=70)
    out = pred.predict(
        _thesis(family=MarketFamily.BTTS, direction="yes"),
        "btts_yes",
        gsv,
    )
    assert out is None


def test_goals_predictor_next_goal_thesis_uses_line_based_p_over():
    """A NEXT_GOAL thesis falling back to a GOALS market must compute
    P(over line) using the line-based Poisson flow, NOT the old
    P(any future goal) shortcut. The shortcut inflated fair_prob to
    near 1.0 in many minute-67+ states where the actual P(over X.5)
    is much smaller.
    """
    pred = Goals2HPredictor()
    gsv = _gsv(minute=70)  # tied 1-1 at 70', 20 min remaining
    out = pred.predict(
        _thesis(family=MarketFamily.NEXT_GOAL, direction="home"),
        "match_goals_over_4.5",
        gsv,
    )
    assert out is not None
    # Already 2 goals; need 3 more in ~20 min for over 4.5 → low.
    # The buggy shortcut returned ~ 1.0 − exp(−λ) ≈ 0.55+; the correct
    # line-based P(>=3 more goals) is well under 0.30.
    assert out.p < 0.30


def test_composite_predictor_routes_btts_to_btts_predictor():
    """Phase 2 wires BTTSPredictor. A BTTS thesis on a btts_yes market
    must now produce a fair_prob (not None) — and the family on the
    returned point must be BTTS so the calibrator looks up the right
    cell."""
    cp = ConditionalPredictor.default()
    gsv = _gsv()
    out = cp.predict(
        _thesis(family=MarketFamily.BTTS, direction="yes"),
        "btts_yes",
        gsv,
    )
    assert out is not None
    assert out.family == MarketFamily.BTTS
    assert 0.0 <= out.p <= 1.0


def test_composite_predictor_returns_none_when_no_predictor_for_family():
    """A CARDS thesis with no CardsPredictor registered still returns
    None — confirms the dispatcher's missing-predictor branch is intact."""
    cp = ConditionalPredictor.default()  # no CARDS predictor registered
    gsv = _gsv()
    out = cp.predict(
        _thesis(family=MarketFamily.CARDS, direction="over"),
        "number_of_cards_over_4.5",
        gsv,
    )
    assert out is None


# ──────────────────────────────────────────────────────────────────────
# BTTSPredictor (Phase 2)
# ──────────────────────────────────────────────────────────────────────


def test_btts_predictor_yes_already_both_scored_returns_one():
    """If both teams have already scored, BTTS yes is certain."""
    from bip.evaluation.live.engine_v3 import BTTSPredictor

    pred = BTTSPredictor()
    gsv = _gsv()
    # _gsv defaults to home_goals=1, away_goals=1 → both scored
    out = pred.predict(
        _thesis(family=MarketFamily.BTTS, direction="yes"),
        "btts_yes",
        gsv,
    )
    assert out is not None
    assert out.p == pytest.approx(1.0)


def test_btts_predictor_no_already_both_scored_returns_zero():
    from bip.evaluation.live.engine_v3 import BTTSPredictor

    pred = BTTSPredictor()
    gsv = _gsv()  # both scored
    out = pred.predict(
        _thesis(family=MarketFamily.BTTS, direction="no"),
        "btts_no",
        gsv,
    )
    assert out is not None
    assert out.p == pytest.approx(0.0)


def test_btts_predictor_rejects_thesis_side_market_mismatch():
    """A BTTS-yes thesis on a btts_no market would have its fair_prob
    refer to side_b — the predictor must return None to keep the MES
    aligned with side_a."""
    from bip.evaluation.live.engine_v3 import BTTSPredictor

    pred = BTTSPredictor()
    gsv = _gsv()
    out = pred.predict(
        _thesis(family=MarketFamily.BTTS, direction="yes"),
        "btts_no",
        gsv,
    )
    assert out is None


def test_btts_predictor_rejects_non_btts_market_id():
    from bip.evaluation.live.engine_v3 import BTTSPredictor

    pred = BTTSPredictor()
    gsv = _gsv()
    out = pred.predict(
        _thesis(family=MarketFamily.BTTS, direction="yes"),
        "match_goals_over_2.5",
        gsv,
    )
    assert out is None


# ──────────────────────────────────────────────────────────────────────
# Calibrator integration in ConditionalPredictor
# ──────────────────────────────────────────────────────────────────────


def test_calibrator_applied_to_predicted_probability():
    """When a fitted calibrator is wired into ConditionalPredictor, the
    .p of the returned PredictionPoint is the calibrated value (not the
    raw model output)."""
    from bip.evaluation.live.engine_v3 import (
        CalibrationSample,
        ConditionalPredictor,
        IsotonicCalibrator,
    )

    # Build a calibrator that maps high p to low p (extreme inversion
    # for testability — picks "predicted 0.9, outcome 0 always").
    samples = [
        CalibrationSample(
            family=MarketFamily.GOALS,
            minute=45,
            predicted_p=0.9,
            outcome=0,
        )
        for _ in range(40)
    ]
    cal = IsotonicCalibrator().fit(samples)
    cp = ConditionalPredictor.default(calibrator=cal)
    gsv = _gsv(minute=45)
    out = cp.predict(
        _thesis(family=MarketFamily.GOALS, direction="over"),
        "match_goals_over_2.5",
        gsv,
    )
    assert out is not None
    # The raw predictor would emit some p>0 for over_2.5 at 45'; the
    # calibrator maps everything in the training set's predicted region
    # to ≈0 (outcome was always 0).
    assert out.p < 0.2


def test_calibrator_absent_means_raw_passthrough():
    from bip.evaluation.live.engine_v3 import ConditionalPredictor

    cp_no_cal = ConditionalPredictor.default(calibrator=None)
    cp_default = ConditionalPredictor.default()
    gsv = _gsv()
    p_no_cal = cp_no_cal.predict(
        _thesis(family=MarketFamily.GOALS, direction="over"),
        "match_goals_over_2.5",
        gsv,
    )
    p_default = cp_default.predict(
        _thesis(family=MarketFamily.GOALS, direction="over"),
        "match_goals_over_2.5",
        gsv,
    )
    assert p_no_cal is not None and p_default is not None
    assert p_no_cal.p == pytest.approx(p_default.p)
