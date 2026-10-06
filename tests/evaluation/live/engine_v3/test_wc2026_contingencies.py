"""Tests for D-series contingency rules."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bip.evaluation.live.engine_v3.wc2026_contingencies import (
    FriendlyResult,
    KEY_PLAYERS_REDUCE_ON_OMIT,
    LineupSnapshot,
    MD1Result,
    StakeAdjustment,
    condition_d2_md1,
    gate_d1_lineup,
    run_contingencies,
    upweight_d3_friendly,
    worst_adjustment,
)
from bip.evaluation.live.engine_v3.wc2026_patterns import MatchContext


def _ctx(home="United States", away="Paraguay"):
    return MatchContext(
        match_id="m1",
        tournament_slug="world_cup_2026",
        home_team=home,
        away_team=away,
        phase="group",
        home_market_value_eur=280_000_000.0,
        away_market_value_eur=85_000_000.0,
    )


def _lineup(team: str, players: list[str]) -> LineupSnapshot:
    return LineupSnapshot(
        team=team,
        starting_xi=players,
        captured_at_utc=datetime(2026, 6, 12, 0, 0, tzinfo=timezone.utc),
    )


def _md1(team: str, *, won=False, drew=False, lost=False, gf=1, ga=1) -> MD1Result:
    return MD1Result(team=team, won=won, drew=drew, lost=lost,
                     goals_for=gf, goals_against=ga)


# ── D1 ──────────────────────────────────────────────────────────────────


class TestD1Lineup:
    def test_reduces_when_pulisic_missing(self):
        ctx = _ctx()
        home_xi = _lineup("United States", ["Berhalter", "Tyler Adams", "Yunus Musah"])
        out = gate_d1_lineup(ctx, home_xi, None)
        assert out.triggered is True
        assert out.adjustment == StakeAdjustment.REDUCE_50
        assert "Pulisic" in out.reason

    def test_holds_when_pulisic_present(self):
        ctx = _ctx()
        home_xi = _lineup("United States", ["Christian Pulisic", "Tyler Adams"])
        out = gate_d1_lineup(ctx, home_xi, None)
        assert out.triggered is False
        assert out.adjustment == StakeAdjustment.HOLD

    def test_cancels_when_both_teams_have_misses(self):
        ctx = _ctx(home="United States", away="England")
        # Pulisic AND Bellingham missing from their respective XIs
        home_xi = _lineup("United States", ["No Pulisic"])
        away_xi = _lineup("England", ["Harry Kane"])
        out = gate_d1_lineup(ctx, home_xi, away_xi)
        assert out.triggered is True
        assert out.adjustment == StakeAdjustment.CANCEL

    def test_no_lineups_returns_no_trigger(self):
        out = gate_d1_lineup(_ctx(), None, None)
        assert out.triggered is False

    def test_team_without_known_stars_is_no_op(self):
        """Some teams have no entries in KEY_PLAYERS_REDUCE_ON_OMIT."""
        ctx = _ctx(home="Qatar", away="Switzerland")
        # Qatar has no listed stars; Switzerland has no listed stars
        home_xi = _lineup("Qatar", ["random"])
        away_xi = _lineup("Switzerland", ["random"])
        out = gate_d1_lineup(ctx, home_xi, away_xi)
        assert out.triggered is False

    def test_key_players_dict_well_formed(self):
        for team, players in KEY_PLAYERS_REDUCE_ON_OMIT.items():
            assert isinstance(team, str)
            assert isinstance(players, frozenset)
            assert len(players) >= 1


# ── D2 ──────────────────────────────────────────────────────────────────


class TestD2MD1Conditioner:
    def test_triggers_when_both_won(self):
        ctx = _ctx()
        out = condition_d2_md1(
            ctx,
            _md1("United States", won=True, gf=2, ga=0),
            _md1("Paraguay", won=True, gf=1, ga=0),
        )
        assert out.triggered is True
        assert out.adjustment == StakeAdjustment.REDUCE_50
        assert "rotation" in out.reason

    def test_triggers_when_home_lost(self):
        ctx = _ctx()
        out = condition_d2_md1(
            ctx,
            _md1("United States", lost=True, gf=0, ga=1),
            _md1("Paraguay", drew=True, gf=1, ga=1),
        )
        assert out.triggered is True
        assert "must-win" in out.reason
        assert "United States" in out.reason

    def test_no_trigger_when_both_drew(self):
        ctx = _ctx()
        out = condition_d2_md1(
            ctx,
            _md1("United States", drew=True, gf=1, ga=1),
            _md1("Paraguay", drew=True, gf=0, ga=0),
        )
        assert out.triggered is False

    def test_no_trigger_when_md1_missing(self):
        out = condition_d2_md1(_ctx(), None, None)
        assert out.triggered is False


# ── D3 ──────────────────────────────────────────────────────────────────


class TestD3Friendly:
    def test_fires_for_nzl_beats_england(self):
        ctx = _ctx(home="New Zealand", away="Egypt")
        nzl_friendly = FriendlyResult(
            date_str="2026-06-06",
            team="New Zealand", opponent="England",
            goals_for=4, goals_against=1,
            opponent_tier="elite_uefa",
        )
        out = upweight_d3_friendly(ctx, nzl_friendly, None)
        assert out.triggered is True
        assert out.adjustment == StakeAdjustment.REDUCE_50
        assert "New Zealand" in out.reason

    def test_fires_for_a_draw_against_elite(self):
        ctx = _ctx(home="Tunisia", away="Japan")
        tun_friendly = FriendlyResult(
            date_str="2026-06-01",
            team="Tunisia", opponent="Spain",
            goals_for=1, goals_against=1,
            opponent_tier="elite_uefa",
        )
        out = upweight_d3_friendly(ctx, tun_friendly, None)
        assert out.triggered is True

    def test_no_fire_for_loss(self):
        ctx = _ctx(home="Iran", away="New Zealand")
        irn_friendly = FriendlyResult(
            date_str="2026-06-04",
            team="Iran", opponent="Brazil",
            goals_for=0, goals_against=3,
            opponent_tier="elite_conmebol",
        )
        out = upweight_d3_friendly(ctx, irn_friendly, None)
        assert out.triggered is False

    def test_no_fire_for_mid_tier_opponent(self):
        ctx = _ctx(home="Iran", away="New Zealand")
        irn_friendly = FriendlyResult(
            date_str="2026-06-04",
            team="Iran", opponent="Costa Rica",
            goals_for=5, goals_against=0,
            opponent_tier="mid",
        )
        out = upweight_d3_friendly(ctx, irn_friendly, None)
        assert out.triggered is False


# ── Orchestrator ────────────────────────────────────────────────────────


class TestRunContingencies:
    def test_runs_all_three_d_rules(self):
        verdicts = run_contingencies(_ctx())
        assert {v.rule_id for v in verdicts} == {"D1", "D2", "D3"}

    def test_worst_adjustment_priority(self):
        v1_hold = run_contingencies(_ctx())  # all HOLD
        assert worst_adjustment(v1_hold) == StakeAdjustment.HOLD

        # Build a CANCEL-via-D1 scenario
        ctx = _ctx(home="United States", away="England")
        home_xi = _lineup("United States", [])
        away_xi = _lineup("England", [])
        v_cancel = run_contingencies(
            ctx, home_lineup=home_xi, away_lineup=away_xi
        )
        assert worst_adjustment(v_cancel) == StakeAdjustment.CANCEL
