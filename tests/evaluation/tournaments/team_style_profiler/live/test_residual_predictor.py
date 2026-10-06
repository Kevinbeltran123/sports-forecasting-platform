"""Tests for live.residual_predictor."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    MarketPredictions,
    MarketProb,
)
from bip.evaluation.tournaments.team_style_profiler.live.match_state import (
    MatchState,
)
from bip.evaluation.tournaments.team_style_profiler.live.residual_predictor import (
    predict_live_markets,
)


def _mp(market: str, prob: float, conf: str = "HIGH") -> MarketProb:
    return MarketProb(market=market, probability=prob, confidence=conf, rationale="test")


def _pre_match(lam_h: float = 1.5, lam_a: float = 1.2) -> MarketPredictions:
    return MarketPredictions(
        home_team="H", away_team="A", home_source="own_tsv", away_source="own_tsv",
        over_2_5=_mp("O2.5", 0.55), over_3_5=_mp("O3.5", 0.30),
        under_2_5=_mp("U2.5", 0.45),
        btts_yes=_mp("BTTS_yes", 0.55), btts_no=_mp("BTTS_no", 0.45),
        corners_over_8_5=None, corners_over_9_5=None, corners_over_10_5=None,
        cards_over_3_5=None, cards_over_4_5=None, cards_over_5_5=None,
        ah_home_minus_0_5=_mp("AH_home_-0.5", 0.45),
        ah_home_minus_1_5=_mp("AH_home_-1.5", 0.25),
        ah_away_minus_0_5=_mp("AH_away_-0.5", 0.32),
        ah_away_minus_1_5=_mp("AH_away_-1.5", 0.15),
        lambda_home=lam_h, lambda_away=lam_a,
    )


def _state(
    minute: int = 45, status: str = "HT",
    h_goals: int = 0, a_goals: int = 0,
    h_yc: int = 0, a_yc: int = 0,
) -> MatchState:
    return MatchState(
        fixture_id=999, status=status, minute=minute,  # type: ignore[arg-type]
        home_team_id=1, away_team_id=2,
        home_team="H", away_team="A",
        home_goals=h_goals, away_goals=a_goals,
        home_yellow_cards=h_yc, away_yellow_cards=a_yc,
        home_red_cards=0, away_red_cards=0,
    )


class TestPredictLiveMarkets:
    def test_pre_match_matches_baseline(self) -> None:
        # Pre-match state -> residual_frac = 1.0 -> same as pre-match prediction
        pre = _pre_match(lam_h=1.5, lam_a=1.5)
        state = _state(minute=0, status="NS")
        live = predict_live_markets(pre, state)
        # O/U at 2.5 should be near pre-match level (slight diff due to no empirical anchor)
        assert 0.40 < live.over_2_5.probability < 0.65

    def test_ht_no_goals_drops_over_25(self) -> None:
        # At HT 0-0, lambdas halve. P(over 2.5 FT) drops vs pre-match.
        pre = _pre_match(lam_h=1.5, lam_a=1.5)
        state = _state(minute=45, status="HT", h_goals=0, a_goals=0)
        live = predict_live_markets(pre, state)
        assert live.over_2_5.probability < 0.30  # cut down meaningfully

    def test_already_over_locks_to_one(self) -> None:
        # Score 3-1 at min 60 — already over 2.5 and over 3.5
        pre = _pre_match()
        state = _state(minute=60, h_goals=3, a_goals=1)
        live = predict_live_markets(pre, state)
        assert live.over_2_5.probability == 1.0
        assert live.over_3_5.probability == 1.0

    def test_btts_already_locks(self) -> None:
        # Already 1-1 -> BTTS already true
        pre = _pre_match()
        state = _state(minute=40, h_goals=1, a_goals=1)
        live = predict_live_markets(pre, state)
        assert live.btts_yes.probability == 1.0
        assert live.btts_no.probability == 0.0

    def test_btts_needs_away_to_score(self) -> None:
        # Home leads 1-0, BTTS = P(away scores in residual)
        pre = _pre_match(lam_h=1.5, lam_a=1.5)
        state = _state(minute=60, h_goals=1, a_goals=0)
        live = predict_live_markets(pre, state)
        # residual_frac = 30/90 = 0.333; λ_a_res = 0.5
        # P(away >=1) = 1 - exp(-0.5) = 0.393
        assert 0.30 < live.btts_yes.probability < 0.50

    def test_match_finished_degenerates(self) -> None:
        pre = _pre_match()
        state = _state(minute=92, status="FT", h_goals=2, a_goals=1)
        live = predict_live_markets(pre, state)
        # FT 2-1: total=3, btts=yes, AH home -0.5 home wins
        assert live.over_2_5.probability == 1.0  # 3 > 2.5
        assert live.over_3_5.probability == 0.0  # 3 < 3.5
        assert live.btts_yes.probability == 1.0
        assert live.ah_home_minus_0_5.probability == 1.0  # diff=1 > 0.5
        assert live.ah_home_minus_1_5.probability == 0.0  # diff=1 < 1.5

    def test_ah_already_covered(self) -> None:
        # Home 3-0 at min 70 -> AH -1.5 already covered
        pre = _pre_match()
        state = _state(minute=70, h_goals=3, a_goals=0)
        live = predict_live_markets(pre, state)
        # FT prob home-1.5 cover = P(FT diff > 1.5) — already diff=3, just need not to lose 2+
        # P should be very high
        assert live.ah_home_minus_1_5.probability > 0.85

    def test_cards_uses_residual_lambda(self) -> None:
        pre = _pre_match()
        state = _state(minute=60, h_yc=2, a_yc=1)
        live = predict_live_markets(pre, state, pre_match_cards_lambda_total=5.0)
        # current=3 cards, residual_frac=30/90=0.333, lam_res=1.67
        # P(total > 3.5) = P(res > 0.5) = 1 - P(res=0) = 1 - exp(-1.67) ~ 0.81
        assert live.cards_over_3_5 is not None
        assert 0.70 < live.cards_over_3_5.probability < 0.95
        # Already at 3, P(total > 4.5) = P(res > 1.5) = P(res >= 2) ~ less
        assert live.cards_over_4_5 is not None

    def test_cards_skipped_without_lambda(self) -> None:
        pre = _pre_match()
        state = _state(minute=60, h_yc=2, a_yc=1)
        live = predict_live_markets(pre, state)  # no cards lambda
        assert live.cards_over_3_5 is None
