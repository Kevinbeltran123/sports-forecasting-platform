"""Tests for the WC2026 statistically-validated pattern rules.

Coverage:
- A1 trigger / non-trigger paths (stage, market, HT score, odds threshold)
- A2 trigger / non-trigger paths (phase, AFCON regime, TM ratio, market family)
- A3 same shape + favorite-team-total selection match
- A4 AFCON regime detection
- Helpers: _favorite_side, enrich_context, load_market_values
"""
from __future__ import annotations

import pytest

from bip.evaluation.live.engine_v3.thesis import MarketFamily
from bip.evaluation.live.engine_v3.wc2026_patterns import (
    AFCON_SLUGS,
    BREAKEVEN_ODDS_HT00_U25,
    MatchContext,
    OddsSnapshot,
    enrich_context,
    rule_a1_ht00_live_u25,
    rule_a2_favorite_corner_tilt,
    rule_a3_favorite_team_total,
    rule_a4_afcon_parity,
)


# ── Fixtures ────────────────────────────────────────────────────────────


def _ctx(
    *,
    tournament="world_cup_2026",
    phase="group",
    home="Mexico",
    away="South Africa",
    home_mv=101_800_000.0,
    away_mv=46_400_000.0,
):
    return MatchContext(
        match_id="wc2026_grp_000",
        tournament_slug=tournament,
        home_team=home,
        away_team=away,
        phase=phase,
        home_market_value_eur=home_mv,
        away_market_value_eur=away_mv,
    )


# ── A1 ──────────────────────────────────────────────────────────────────


class TestA1HT00LiveU25:
    def test_fires_at_exact_min_threshold_130(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.30,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        out = rule_a1_ht00_live_u25(ctx, odds)
        assert out.triggered is True
        assert out.market_family == MarketFamily.GOALS
        assert out.direction == "under"
        assert out.max_kelly == 0.25
        assert out.breakeven_odds == pytest.approx(BREAKEVEN_ODDS_HT00_U25, rel=1e-6)

    @pytest.mark.parametrize("odds_v", [1.35, 1.50, 1.80, 2.20])
    def test_fires_for_higher_odds(self, odds_v):
        """High odds = more EV; rule must fire (no upper ceiling)."""
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=odds_v,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        assert rule_a1_ht00_live_u25(ctx, odds).triggered is True

    def test_no_trigger_below_min_odds(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.28,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        out = rule_a1_ht00_live_u25(ctx, odds)
        assert out.triggered is False
        assert "insufficient EV cushion" in out.reason

    @pytest.mark.parametrize("h,a", [(1, 0), (0, 1), (1, 1), (2, 0)])
    def test_no_trigger_when_ht_not_00(self, h, a):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.20,
            home_goals_ht=h,
            away_goals_ht=a,
        )
        assert rule_a1_ht00_live_u25(ctx, odds).triggered is False

    def test_no_trigger_pre_match_stage(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.20,
        )
        out = rule_a1_ht00_live_u25(ctx, odds)
        assert out.triggered is False
        assert "stage != live" in out.reason

    def test_no_trigger_wrong_market(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.CORNERS,
            selection="under_2_5",
            decimal_odds=1.20,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        assert rule_a1_ht00_live_u25(ctx, odds).triggered is False

    def test_no_trigger_wrong_selection(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="over_2_5",
            decimal_odds=1.20,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        assert rule_a1_ht00_live_u25(ctx, odds).triggered is False

    def test_no_trigger_when_ht_score_missing(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.20,
        )
        out = rule_a1_ht00_live_u25(ctx, odds)
        assert out.triggered is False
        assert "HT score not in snapshot" in out.reason


# ── A2 ──────────────────────────────────────────────────────────────────


class TestA2FavoriteCornerTilt:
    def _corner_odds(self):
        return OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.CORNERS,
            selection="ah_-2_home",
            decimal_odds=1.95,
        )

    def test_fires_with_clear_favorite_home(self):
        ctx = _ctx()  # Mexico €101.8m vs RSA €46.4m → 2.19x
        out = rule_a2_favorite_corner_tilt(ctx, self._corner_odds())
        assert out.triggered is True
        assert out.direction == "home"
        assert out.market_family == MarketFamily.CORNERS

    def test_fires_with_clear_favorite_away(self):
        ctx = _ctx(home="South Africa", away="Mexico", home_mv=46_400_000.0, away_mv=101_800_000.0)
        out = rule_a2_favorite_corner_tilt(ctx, self._corner_odds())
        assert out.triggered is True
        assert out.direction == "away"

    def test_no_trigger_below_tm_threshold(self):
        # ratio = 101.8 / 80 = 1.27x < 1.5x
        ctx = _ctx(away_mv=80_000_000.0)
        out = rule_a2_favorite_corner_tilt(ctx, self._corner_odds())
        assert out.triggered is False
        assert "TM ratio" in out.reason

    def test_no_trigger_in_knockout(self):
        ctx = _ctx(phase="knockout")
        assert rule_a2_favorite_corner_tilt(ctx, self._corner_odds()).triggered is False

    def test_no_trigger_in_afcon(self):
        ctx = _ctx(tournament="afcon_2025")
        out = rule_a2_favorite_corner_tilt(ctx, self._corner_odds())
        assert out.triggered is False
        assert "A4 AFCON" in out.reason

    def test_no_trigger_when_market_values_missing(self):
        ctx = _ctx(home_mv=None, away_mv=None)
        out = rule_a2_favorite_corner_tilt(ctx, self._corner_odds())
        assert out.triggered is False
        assert "no market-value data" in out.reason

    def test_no_trigger_wrong_market_family(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.BTTS,
            selection="yes",
            decimal_odds=1.90,
        )
        assert rule_a2_favorite_corner_tilt(ctx, odds).triggered is False

    def test_accepts_asian_handicap_family(self):
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.ASIAN_HANDICAP,
            selection="ah_-2_home",
            decimal_odds=1.95,
        )
        assert rule_a2_favorite_corner_tilt(ctx, odds).triggered is True


# ── A3 ──────────────────────────────────────────────────────────────────


class TestA3FavoriteTeamTotal:
    def _tt_odds(self, selection="home_team_total_over_1_5"):
        return OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.GOALS,
            selection=selection,
            decimal_odds=1.85,
        )

    def test_fires_for_home_favorite(self):
        ctx = _ctx()  # Mexico home favorite
        out = rule_a3_favorite_team_total(ctx, self._tt_odds())
        assert out.triggered is True
        assert out.direction == "home"

    def test_fires_for_away_favorite_with_matching_selection(self):
        ctx = _ctx(home="South Africa", away="Mexico", home_mv=46_400_000.0, away_mv=101_800_000.0)
        out = rule_a3_favorite_team_total(ctx, self._tt_odds("away_team_total_over_1_5"))
        assert out.triggered is True
        assert out.direction == "away"

    def test_no_trigger_when_selection_does_not_match_favorite(self):
        ctx = _ctx()  # home is favorite
        out = rule_a3_favorite_team_total(ctx, self._tt_odds("away_team_total_over_1_5"))
        assert out.triggered is False
        assert "does not match favorite-team-total" in out.reason

    def test_no_trigger_in_knockout(self):
        ctx = _ctx(phase="knockout")
        assert rule_a3_favorite_team_total(ctx, self._tt_odds()).triggered is False

    def test_no_trigger_in_afcon(self):
        ctx = _ctx(tournament="afcon_2023")
        out = rule_a3_favorite_team_total(ctx, self._tt_odds())
        assert out.triggered is False

    def test_no_trigger_for_1x2_market(self):
        """Critical guard: A3 must never propose 1X2 tilts (Elo prior absorbs)."""
        ctx = _ctx()
        odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.RESULT_1X2,
            selection="home",
            decimal_odds=2.10,
        )
        out = rule_a3_favorite_team_total(ctx, odds)
        assert out.triggered is False
        assert "not goals" in out.reason


# ── A4 ──────────────────────────────────────────────────────────────────


class TestA4AFCONParity:
    @pytest.mark.parametrize("slug", sorted(AFCON_SLUGS))
    def test_fires_for_afcon_tournaments(self, slug):
        ctx = _ctx(tournament=slug)
        out = rule_a4_afcon_parity(ctx)
        assert out.triggered is True
        assert "AFCON regime" in out.reason

    @pytest.mark.parametrize(
        "slug",
        ["world_cup_2026", "euro_2024", "copa_2024", "wc_2022"],
    )
    def test_no_trigger_for_non_afcon(self, slug):
        ctx = _ctx(tournament=slug)
        assert rule_a4_afcon_parity(ctx).triggered is False


# ── Helpers ─────────────────────────────────────────────────────────────


class TestEnrichContext:
    def test_pulls_market_values_from_dict(self):
        mvs = {"Spain": 1_310_000_000.0, "Cape Verde": 56_380_000.0}
        ctx = enrich_context(
            {
                "match_id": "wc2026_grp_015",
                "tournament_slug": "world_cup_2026",
                "home_team": "Spain",
                "away_team": "Cape Verde",
                "phase": "group",
            },
            mvs,
        )
        assert ctx.home_market_value_eur == 1_310_000_000.0
        assert ctx.away_market_value_eur == 56_380_000.0

    def test_explicit_values_not_overwritten(self):
        mvs = {"Spain": 9_999.0, "Cape Verde": 1_111.0}
        ctx = enrich_context(
            {
                "match_id": "wc2026_grp_015",
                "tournament_slug": "world_cup_2026",
                "home_team": "Spain",
                "away_team": "Cape Verde",
                "phase": "group",
                "home_market_value_eur": 1.0,
                "away_market_value_eur": 2.0,
            },
            mvs,
        )
        assert ctx.home_market_value_eur == 1.0
        assert ctx.away_market_value_eur == 2.0


class TestIntegration:
    """End-to-end: realistic mid-tournament scenario."""

    def test_canada_vs_qatar_md2_p4a_fires(self):
        ctx = _ctx(
            home="Canada",
            away="Qatar",
            home_mv=140_000_000.0,
            away_mv=17_900_000.0,
        )  # ratio 7.8x
        tt_odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.GOALS,
            selection="home_team_total_over_1_5",
            decimal_odds=1.70,
        )
        corner_odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.CORNERS,
            selection="ah_-2_home",
            decimal_odds=1.95,
        )
        live_u25 = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.35,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        assert rule_a3_favorite_team_total(ctx, tt_odds).triggered is True
        assert rule_a2_favorite_corner_tilt(ctx, corner_odds).triggered is True
        assert rule_a1_ht00_live_u25(ctx, live_u25).triggered is True

    def test_afcon_match_kills_a2_a3_but_a1_still_works(self):
        ctx = _ctx(tournament="afcon_2025")
        tt_odds = OddsSnapshot(
            stage="pre_match",
            market_family=MarketFamily.GOALS,
            selection="home_team_total_over_1_5",
            decimal_odds=1.70,
        )
        assert rule_a3_favorite_team_total(ctx, tt_odds).triggered is False
        assert rule_a2_favorite_corner_tilt(
            ctx,
            OddsSnapshot(
                stage="pre_match",
                market_family=MarketFamily.CORNERS,
                selection="ah_-2_home",
                decimal_odds=1.95,
            ),
        ).triggered is False
        # A1 still fires in AFCON (it's a universal live gate)
        live = OddsSnapshot(
            stage="live",
            market_family=MarketFamily.GOALS,
            selection="under_2_5",
            decimal_odds=1.35,
            home_goals_ht=0,
            away_goals_ht=0,
        )
        assert rule_a1_ht00_live_u25(ctx, live).triggered is True
