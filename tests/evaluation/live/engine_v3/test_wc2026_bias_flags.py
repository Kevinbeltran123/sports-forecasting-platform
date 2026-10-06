"""Tests for B-series bias flaggers."""
from __future__ import annotations

import pytest

from bip.evaluation.live.engine_v3.wc2026_bias_flags import (
    LockPrediction,
    POST_LOCK_EVENTS,
    PostLockEvent,
    TEAM_CONFEDERATION,
    WC_DEBUTANT_OR_WEAK,
    flag_b1_strong_favorite_vs_debutant,
    flag_b2_lock_tilts_against_elite,
    flag_b3_post_lock_events,
    post_lock_events_for,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import MatchContext


def _ctx(
    home="Brazil", away="Haiti",
    home_mv=905_700_000.0, away_mv=55_730_000.0,
    tournament="world_cup_2026", phase="group",
):
    return MatchContext(
        match_id="m1",
        tournament_slug=tournament,
        home_team=home,
        away_team=away,
        phase=phase,
        home_market_value_eur=home_mv,
        away_market_value_eur=away_mv,
    )


def _lock(p_home=0.52, p_draw=0.25, p_away=0.23):
    return LockPrediction(p_home_win=p_home, p_draw=p_draw, p_away_win=p_away)


# ── B1 ──────────────────────────────────────────────────────────────────


class TestB1StrongFavoriteVsDebutant:
    def test_fires_for_bra_hai(self):
        """Brazil-Haiti: TM 16.2x, Haiti is debutant. Should fire HIGH."""
        out = flag_b1_strong_favorite_vs_debutant(_ctx(), _lock())
        assert out.triggered is True
        assert out.severity == "high"
        assert out.affected_team == "Brazil"
        assert "derivatives" in out.market_guidance.lower()

    def test_fires_for_esp_cpv(self):
        """Spain-Cape Verde: TM 23.2x, Cape Verde debutant."""
        out = flag_b1_strong_favorite_vs_debutant(
            _ctx(home="Spain", away="Cape Verde",
                 home_mv=1_310_000_000.0, away_mv=56_380_000.0),
            _lock(p_home=0.47, p_draw=0.25, p_away=0.28),
        )
        assert out.triggered is True
        assert out.affected_team == "Spain"

    def test_no_trigger_below_10x(self):
        """USA vs Paraguay: TM 3.3x — not B1 territory."""
        out = flag_b1_strong_favorite_vs_debutant(
            _ctx(home="United States", away="Paraguay",
                 home_mv=280_000_000.0, away_mv=85_000_000.0),
            _lock(p_home=0.42),
        )
        assert out.triggered is False
        assert "< 10.0" in out.reason

    def test_no_trigger_underdog_not_debutant(self):
        """Brazil vs Argentina at hypothetical 10x — Argentina not in weak list."""
        out = flag_b1_strong_favorite_vs_debutant(
            _ctx(home="Brazil", away="Argentina",
                 home_mv=900_000_000.0, away_mv=80_000_000.0),
            _lock(),
        )
        assert out.triggered is False
        assert "not in debutant" in out.reason

    def test_no_trigger_missing_mv(self):
        out = flag_b1_strong_favorite_vs_debutant(
            _ctx(home_mv=None), _lock(),
        )
        assert out.triggered is False
        assert "no market-value data" in out.reason

    def test_away_favorite_branch(self):
        """Same ratio but flipped sides — should still fire."""
        out = flag_b1_strong_favorite_vs_debutant(
            _ctx(home="Curaçao", away="Germany",
                 home_mv=25_400_000.0, away_mv=1_010_000_000.0),
            _lock(p_home=0.29, p_draw=0.25, p_away=0.46),
        )
        assert out.triggered is True
        assert out.affected_team == "Germany"


# ── B2 ──────────────────────────────────────────────────────────────────


class TestB2LockTiltsAgainstElite:
    def test_fires_for_aus_tur(self):
        """Australia-Turkey: TM 10.2x TUR, lock has AUS 0.38. Fire."""
        out = flag_b2_lock_tilts_against_elite(
            _ctx(home="Australia", away="Turkey",
                 home_mv=51_330_000.0, away_mv=525_200_000.0),
            _lock(p_home=0.38, p_draw=0.26, p_away=0.36),
        )
        assert out.triggered is True
        assert out.affected_team == "Turkey"
        assert "ML at implied" in out.market_guidance

    def test_fires_for_aut_jor(self):
        """Austria-Jordan: TM 15.9x AUT, lock has JOR 0.38. Fire."""
        out = flag_b2_lock_tilts_against_elite(
            _ctx(home="Austria", away="Jordan",
                 home_mv=258_300_000.0, away_mv=16_230_000.0),
            _lock(p_home=0.36, p_draw=0.26, p_away=0.38),
        )
        assert out.triggered is True
        assert out.affected_team == "Austria"

    def test_no_trigger_when_lock_already_favors_elite(self):
        """If lock has fav prob ≥ 0.45, no fade signal."""
        out = flag_b2_lock_tilts_against_elite(
            _ctx(home="Australia", away="Turkey",
                 home_mv=51_330_000.0, away_mv=525_200_000.0),
            _lock(p_home=0.20, p_draw=0.25, p_away=0.55),
        )
        assert out.triggered is False
        assert "no fade signal" in out.reason

    def test_no_trigger_outside_band(self):
        """TM 76x is OUTSIDE the 6-16 band (B1 territory)."""
        out = flag_b2_lock_tilts_against_elite(
            _ctx(home="France", away="Iraq",
                 home_mv=1_470_000_000.0, away_mv=19_280_000.0),
            _lock(p_home=0.42, p_draw=0.26, p_away=0.32),
        )
        assert out.triggered is False
        assert "outside" in out.reason

    def test_no_trigger_when_confeds_dont_match(self):
        """UEFA vs UEFA — B2 only fires inter-confederation."""
        out = flag_b2_lock_tilts_against_elite(
            _ctx(home="Germany", away="Bosnia and Herzegovina",
                 home_mv=1_010_000_000.0, away_mv=105_000_000.0),
            _lock(p_home=0.30, p_draw=0.27, p_away=0.43),
        )
        assert out.triggered is False
        assert "underdog confederation" in out.reason or "favorite confederation" in out.reason

    def test_team_confederation_coverage(self):
        """48 WC2026 teams should all be in TEAM_CONFEDERATION."""
        # Spot-check a few from each confederation
        assert TEAM_CONFEDERATION["France"] == "UEFA"
        assert TEAM_CONFEDERATION["Brazil"] == "CONMEBOL"
        assert TEAM_CONFEDERATION["Morocco"] == "CAF"
        assert TEAM_CONFEDERATION["Japan"] == "AFC"
        assert TEAM_CONFEDERATION["Mexico"] == "CONCACAF"
        assert TEAM_CONFEDERATION["New Zealand"] == "OFC"
        # Count check — 16 + 6 + 10 + 9 + 6 + 1 = 48
        assert len(TEAM_CONFEDERATION) == 48


# ── B3 ──────────────────────────────────────────────────────────────────


class TestB3PostLockEvents:
    def test_fires_for_spain_match_yamal(self):
        out = flag_b3_post_lock_events(_ctx(home="Spain", away="Cape Verde"))
        assert out.triggered is True
        assert out.severity == "high"
        assert "Yamal" in out.reason

    def test_fires_for_japan_match_mitoma(self):
        out = flag_b3_post_lock_events(_ctx(home="Netherlands", away="Japan"))
        assert out.triggered is True
        # Both NED (Simons + Depay) and JPN (Mitoma) hit
        assert out.severity == "high"

    def test_no_trigger_when_neither_team_has_events(self):
        out = flag_b3_post_lock_events(
            _ctx(home="Switzerland", away="Qatar")
        )
        # Neither team in POST_LOCK_EVENTS
        assert out.triggered is False

    def test_post_lock_events_helper(self):
        spain_events = post_lock_events_for("Spain")
        assert len(spain_events) == 1
        assert spain_events[0].event_type == "injury_out"
        assert "Yamal" in spain_events[0].description

    def test_all_events_dated_after_lock_emit(self):
        """Sanity: every registered event must post-date 2025-10 reasonable
        margin (lock was emitted 2026-05-09; we include coach changes
        from late 2025 as 'lock didn't fully absorb them')."""
        for ev in POST_LOCK_EVENTS:
            assert ev.event_date.year >= 2025
            assert isinstance(ev.severity, str)
            assert ev.severity in {"high", "medium", "low"}


# ── Helpers ─────────────────────────────────────────────────────────────


class TestDebutantList:
    def test_curacao_is_debutant(self):
        assert "Curaçao" in WC_DEBUTANT_OR_WEAK

    def test_brazil_not_debutant(self):
        assert "Brazil" not in WC_DEBUTANT_OR_WEAK
