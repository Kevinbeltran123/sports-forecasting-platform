"""Tests for C-series avoid rules."""
from __future__ import annotations

import pytest

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_avoids import (
    LineMoveContext,
    avoid_c1_lock_soft_ah,
    avoid_c2_tm_extreme_oversized_ah,
    avoid_c3_cinderella_ml,
    avoid_c4_sentiment_line_move,
    is_blocked,
    run_avoids,
)
from bip.evaluation.live.engine_v3.wc2026_bias_flags import LockPrediction
from bip.evaluation.live.engine_v3.wc2026_patterns import (
    MatchContext,
    OddsSnapshot,
)


def _ctx(
    home="Belgium", away="Egypt",
    home_mv=534_200_000.0, away_mv=136_230_000.0,
):
    return MatchContext(
        match_id="m1",
        tournament_slug="world_cup_2026",
        home_team=home,
        away_team=away,
        phase="group",
        home_market_value_eur=home_mv,
        away_market_value_eur=away_mv,
    )


def _ah_odds(selection="ah_-1.5_home", odds_value=1.85):
    return OddsSnapshot(
        stage="pre_match",
        market_family=MarketFamily.ASIAN_HANDICAP,
        selection=selection,
        decimal_odds=odds_value,
    )


def _ml_odds(selection="home", odds_value=2.50):
    return OddsSnapshot(
        stage="pre_match",
        market_family=MarketFamily.RESULT_1X2,
        selection=selection,
        decimal_odds=odds_value,
    )


def _lock(p_home=0.42, p_draw=0.26, p_away=0.32):
    return LockPrediction(p_home_win=p_home, p_draw=p_draw, p_away_win=p_away)


# ── C1 ──────────────────────────────────────────────────────────────────


class TestC1LockSoftAH:
    def test_blocks_belgium_ah_at_lock_042(self):
        """Belgium lock H=0.42, AH -1.5 home → C1 blocks."""
        out = avoid_c1_lock_soft_ah(_ctx(), _ah_odds("ah_-1.5_home"), _lock())
        assert out.blocked is True
        assert "lock favorite_prob" in out.reason

    def test_no_block_when_lock_strong(self):
        """If lock H=0.55, no soft-favorite worry."""
        out = avoid_c1_lock_soft_ah(_ctx(), _ah_odds("ah_-1.5_home"), _lock(p_home=0.55))
        assert out.blocked is False

    def test_no_block_for_symmetric_ah(self):
        """AH -0.5 is not asymmetric per C1's threshold."""
        out = avoid_c1_lock_soft_ah(_ctx(), _ah_odds("ah_-0.5_home"), _lock())
        assert out.blocked is False

    def test_no_block_when_ah_on_underdog(self):
        """+1.5 on underdog Egypt is not the trap."""
        out = avoid_c1_lock_soft_ah(_ctx(), _ah_odds("ah_-1.5_away"), _lock())
        assert out.blocked is False
        assert "AH is on the underdog side" in out.reason

    def test_no_block_for_non_ah_market(self):
        odds = OddsSnapshot(
            stage="pre_match", market_family=MarketFamily.GOALS,
            selection="under_2_5", decimal_odds=1.85,
        )
        out = avoid_c1_lock_soft_ah(_ctx(), odds, _lock())
        assert out.blocked is False
        assert "not asian_handicap" in out.reason


# ── C2 ──────────────────────────────────────────────────────────────────


class TestC2TMExtremeAH:
    def test_blocks_spain_minus_3_ah(self):
        """Spain vs Cape Verde TM 23x; AH -3 home is a trap."""
        ctx = _ctx(
            home="Spain", away="Cape Verde",
            home_mv=1_310_000_000.0, away_mv=56_380_000.0,
        )
        out = avoid_c2_tm_extreme_oversized_ah(ctx, _ah_odds("ah_-3.0_home"))
        assert out.blocked is True
        assert "public trap" in out.reason

    def test_no_block_below_extreme_threshold(self):
        """TM ratio 3.9x — well below 15x."""
        out = avoid_c2_tm_extreme_oversized_ah(_ctx(), _ah_odds("ah_-3.0_home"))
        assert out.blocked is False
        assert "< 15.0" in out.reason

    def test_no_block_smaller_ah_value(self):
        """AH -1.5 is OK even at extreme TM."""
        ctx = _ctx(
            home="Spain", away="Cape Verde",
            home_mv=1_310_000_000.0, away_mv=56_380_000.0,
        )
        out = avoid_c2_tm_extreme_oversized_ah(ctx, _ah_odds("ah_-1.5_home"))
        assert out.blocked is False
        assert "not an AH" in out.reason


# ── C3 ──────────────────────────────────────────────────────────────────


class TestC3CinderellaML:
    def test_blocks_nzl_ml_at_short_price(self):
        ctx = _ctx(
            home="New Zealand", away="Egypt",
            home_mv=31_700_000.0, away_mv=136_230_000.0,
        )
        out = avoid_c3_cinderella_ml(ctx, _ml_odds("home", odds_value=2.40))
        assert out.blocked is True

    def test_no_block_at_honest_long_price(self):
        ctx = _ctx(
            home="New Zealand", away="Egypt",
            home_mv=31_700_000.0, away_mv=136_230_000.0,
        )
        out = avoid_c3_cinderella_ml(ctx, _ml_odds("home", odds_value=3.50))
        assert out.blocked is False
        assert "price is honest" in out.reason

    def test_no_block_for_non_cinderella(self):
        """Backing Brazil at ML is fine (Brazil not in debutant list)."""
        ctx = _ctx(
            home="Brazil", away="Haiti",
            home_mv=905_700_000.0, away_mv=55_730_000.0,
        )
        out = avoid_c3_cinderella_ml(ctx, _ml_odds("home", odds_value=1.30))
        assert out.blocked is False
        assert "not Cinderella" in out.reason

    def test_blocks_curacao_at_home_ml(self):
        ctx = _ctx(
            home="Curaçao", away="Germany",
            home_mv=25_400_000.0, away_mv=1_010_000_000.0,
        )
        out = avoid_c3_cinderella_ml(ctx, _ml_odds("home", odds_value=2.00))
        assert out.blocked is True

    def test_no_block_for_draw_selection(self):
        out = avoid_c3_cinderella_ml(_ctx(), _ml_odds("draw"))
        assert out.blocked is False


# ── C4 ──────────────────────────────────────────────────────────────────


class TestC4Sentiment:
    def test_blocks_5pct_sentiment_move(self):
        out = avoid_c4_sentiment_line_move(
            LineMoveContext(pct_move_since_open=-6.5, sentiment_event_tagged=True)
        )
        assert out.blocked is True

    def test_no_block_below_threshold(self):
        out = avoid_c4_sentiment_line_move(
            LineMoveContext(pct_move_since_open=-3.0, sentiment_event_tagged=True)
        )
        assert out.blocked is False

    def test_no_block_when_not_sentiment(self):
        out = avoid_c4_sentiment_line_move(
            LineMoveContext(pct_move_since_open=-10.0, sentiment_event_tagged=False)
        )
        assert out.blocked is False

    def test_no_block_when_feed_missing(self):
        out = avoid_c4_sentiment_line_move(None)
        assert out.blocked is False
        assert "feed not wired" in out.reason


# ── Orchestrator ────────────────────────────────────────────────────────


class TestRunAvoids:
    def test_runs_all_four_rules(self):
        verdicts = run_avoids(
            _ctx(), _ah_odds("ah_-1.5_home"), _lock(),
        )
        assert {v.rule_id for v in verdicts} == {"C1", "C2", "C3", "C4"}

    def test_is_blocked_true_when_any_blocks(self):
        verdicts = run_avoids(
            _ctx(), _ah_odds("ah_-1.5_home"), _lock(),
        )
        assert is_blocked(verdicts) is True  # C1 blocks

    def test_is_blocked_false_when_clean(self):
        """All four rules return non-block for a vanilla pre-match odds
        snapshot."""
        ctx = _ctx(home="Brazil", away="Haiti",
                   home_mv=905_700_000.0, away_mv=55_730_000.0)
        clean = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.GOALS,
            selection="over_2_5",
            decimal_odds=1.90,
        )
        verdicts = run_avoids(ctx, clean, _lock(p_home=0.55))
        assert is_blocked(verdicts) is False

    def test_no_lock_still_runs_c2_c3_c4(self):
        verdicts = run_avoids(
            _ctx(), _ah_odds("ah_-1.5_home"), lock=None,
        )
        # C1 returns blocked=False with "no lock" reason, but the other
        # three still execute.
        c1 = next(v for v in verdicts if v.rule_id == "C1")
        assert c1.blocked is False
        assert "no lock prediction" in c1.reason
