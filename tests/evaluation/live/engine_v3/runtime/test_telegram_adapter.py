"""Tests for the v3 ShadowPick → LivePick adapter."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bip.evaluation.live.engine_v3 import (
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.evaluation.live.engine_v3.gsv_builder import GSVBuilder
from bip.evaluation.live.engine_v3.market_selector import MarketCandidate
from bip.evaluation.live.engine_v3.mes import MESResult
from bip.evaluation.live.engine_v3.pipeline import ShadowPick
from bip.evaluation.live.engine_v3.runtime.telegram_adapter import (
    shadow_pick_to_live_pick,
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
from tests.evaluation.live.engine_v3.conftest import HOME_ID, make_state


def _build_thesis(
    *, family: MarketFamily, direction: str, archetype: ThesisArchetype,
    minute: int = 80,
) -> Thesis:
    return Thesis(
        id=f"TEST@m{minute}",
        archetype=archetype,
        premise=[GSVPredicate(path="time.minute", op="ge", value=0)],
        mechanism=CausalChain(steps=[CausalStep(
            cause="t", effect="t", mechanism="t",
        )]),
        prediction=ConditionalShift(
            family=family, direction=direction,
            magnitude_pp=0.05, horizon=build_horizon("rest_of_match", minute),
        ),
        invalidation_triggers=[InvalidationTrigger(kind="any_goal", description="x")],
        confidence_prior=0.6,
        source=ThesisSource(layer="rule", identifier="TEST"),
        activated_at_minute=minute,
    )


def _build_mes_result(*, score: float, base_edge: float) -> MESResult:
    return MESResult(
        thesis_id="TEST",
        market_id="test_market",
        family=MarketFamily.GOALS,
        base_edge=base_edge,
        signal_clarity=1.0,
        book_slowness=1.0,
        liquidity_score=1.0,
        conditional_variance=1.0,
        score=score,
    )


def _build_gsv_with_market(
    *, market_id: str, side_a_decimal: float,
    side_b_decimal: float | None = None,
    line_value: float | None = None,
    minute: int = 80,
):
    now = datetime.now(timezone.utc)
    market_line = MarketLine(
        market_id=market_id,
        side_a_decimal=side_a_decimal,
        side_b_decimal=side_b_decimal,
        line_value=line_value,
        max_stake_cap=200.0,
        last_update_utc=now,
    )
    state = make_state(minute=minute, home_goals=1, away_goals=0)
    priors = PreMatchPriors(
        lambda_home_prematch=1.8, lambda_away_prematch=0.9,
    )
    markets = MarketSnapshot(lines={market_id: market_line})
    return GSVBuilder().build(state, priors=priors, markets=markets)


def test_adapter_converts_under_pick_to_live_pick():
    """The canonical Day-4 cruise_mode case: under goals @ min 80,
    1-0 lead. Tier classification should route to tier 2 or 1 depending
    on the resulting edge / logical_score."""
    gsv = _build_gsv_with_market(
        market_id="match_goals_under_2.5",
        side_a_decimal=1.95, side_b_decimal=1.95, line_value=2.5,
    )
    thesis = _build_thesis(
        family=MarketFamily.GOALS, direction="under",
        archetype=ThesisArchetype.CRUISE_MODE,
    )
    cand = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5",
        family=MarketFamily.GOALS, fair_prob=0.65,
        mes=_build_mes_result(score=3.5, base_edge=0.15),
    )
    pick = ShadowPick(
        fixture_id=gsv.fixture_id, timestamp_utc=gsv.timestamp_utc,
        candidate=cand, full_thesis=thesis,
    )
    out = shadow_pick_to_live_pick(pick, gsv)
    assert out is not None
    assert out.fixture_id == gsv.fixture_id
    assert out.minute == 80
    assert out.market == "goals"
    assert out.selection == "under 2.5"
    assert out.bookmaker_odd == pytest.approx(1.95)
    assert out.our_probability == pytest.approx(0.65)
    # edge_pct = (1.95 × 0.65 − 1) × 100 = 26.75
    assert out.edge_pct == pytest.approx(26.75, abs=0.01)
    assert out.logical_score == pytest.approx(0.7)  # 3.5 / 5.0
    assert out.flagged_reason is None


def test_adapter_skips_when_market_line_absent():
    """Gate approved a pick but the line is missing at the GSV's
    market_snapshot — common transient odds-feed gap. Adapter returns
    None rather than emitting a malformed alert."""
    gsv = _build_gsv_with_market(
        market_id="match_goals_under_2.5", side_a_decimal=1.95,
    )
    thesis = _build_thesis(
        family=MarketFamily.GOALS, direction="under",
        archetype=ThesisArchetype.CRUISE_MODE,
    )
    cand = MarketCandidate(
        thesis=thesis, market_id="some_other_market_id_not_in_snapshot",
        family=MarketFamily.GOALS, fair_prob=0.65,
        mes=_build_mes_result(score=3.5, base_edge=0.15),
    )
    pick = ShadowPick(
        fixture_id=gsv.fixture_id, timestamp_utc=gsv.timestamp_utc,
        candidate=cand, full_thesis=thesis,
    )
    assert shadow_pick_to_live_pick(pick, gsv) is None


def test_adapter_skips_when_edge_is_zero_or_negative():
    """MES-composite gates can pass a pick with thin nominal edge. The
    v2 alert templates require positive edge_pct — refuse rather than
    emit a misleading 'value bet' with zero edge."""
    # 1.50 odd × 0.60 prob = 0.90 < 1 → edge_pct = -10
    gsv = _build_gsv_with_market(
        market_id="match_goals_under_2.5",
        side_a_decimal=1.50, side_b_decimal=1.50, line_value=2.5,
    )
    thesis = _build_thesis(
        family=MarketFamily.GOALS, direction="under",
        archetype=ThesisArchetype.CRUISE_MODE,
    )
    cand = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5",
        family=MarketFamily.GOALS, fair_prob=0.60,
        mes=_build_mes_result(score=3.0, base_edge=0.10),
    )
    pick = ShadowPick(
        fixture_id=gsv.fixture_id, timestamp_utc=gsv.timestamp_utc,
        candidate=cand, full_thesis=thesis,
    )
    assert shadow_pick_to_live_pick(pick, gsv) is None


def test_adapter_kelly_uses_quarter_fraction():
    """Sizing convention: max 1/4 Kelly. Adapter must
    pre-multiply the full Kelly by 0.25 in ``suggested_stake_pct``,
    while ``kelly_fraction_full`` preserves the unscaled value."""
    gsv = _build_gsv_with_market(
        market_id="match_goals_under_2.5",
        side_a_decimal=2.00, side_b_decimal=2.00, line_value=2.5,
    )
    thesis = _build_thesis(
        family=MarketFamily.GOALS, direction="under",
        archetype=ThesisArchetype.CRUISE_MODE,
    )
    cand = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5",
        family=MarketFamily.GOALS, fair_prob=0.70,
        mes=_build_mes_result(score=3.0, base_edge=0.15),
    )
    pick = ShadowPick(
        fixture_id=gsv.fixture_id, timestamp_utc=gsv.timestamp_utc,
        candidate=cand, full_thesis=thesis,
    )
    out = shadow_pick_to_live_pick(pick, gsv)
    assert out is not None
    # Kelly: p=0.7, b=1.0 → f* = (0.7×1 − 0.3) / 1 = 0.40
    assert out.kelly_fraction_full == pytest.approx(0.40)
    # 1/4 Kelly: 0.40 × 0.25 × 100 = 10%
    assert out.suggested_stake_pct == pytest.approx(10.0)


def test_adapter_uses_side_b_for_under_direction_when_present():
    """``under`` / ``no`` / ``away`` should route to side_b when the
    market line has both sides priced. Mirrors ShadowLogger logic."""
    gsv = _build_gsv_with_market(
        market_id="match_goals_under_2.5",
        side_a_decimal=2.10, side_b_decimal=1.72, line_value=2.5,
    )
    thesis = _build_thesis(
        family=MarketFamily.GOALS, direction="under",
        archetype=ThesisArchetype.CRUISE_MODE,
    )
    cand = MarketCandidate(
        thesis=thesis, market_id="match_goals_under_2.5",
        family=MarketFamily.GOALS, fair_prob=0.70,
        mes=_build_mes_result(score=3.0, base_edge=0.15),
    )
    pick = ShadowPick(
        fixture_id=gsv.fixture_id, timestamp_utc=gsv.timestamp_utc,
        candidate=cand, full_thesis=thesis,
    )
    out = shadow_pick_to_live_pick(pick, gsv)
    assert out is not None
    assert out.bookmaker_odd == pytest.approx(1.72)
