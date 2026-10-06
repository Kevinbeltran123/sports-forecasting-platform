"""Tests for the ESPN player-stats parser + source=espn profile path."""
from __future__ import annotations

import math

from bip.evaluation.tournaments.team_style_profiler.espn_ingest import (
    ESPN_STAT_MAP,
    match_player_counts,
)
from bip.evaluation.tournaments.team_style_profiler.player_props import (
    build_player_profiles,
    prop_board,
)


def _stats(**kw):
    return [{"name": k, "value": v} for k, v in kw.items()]


def _summary():
    return {"rosters": [
        {"team": {"id": "464", "displayName": "Norway"}, "roster": [
            {"starter": True, "position": {"abbreviation": "F"},
             "athlete": {"id": "1", "displayName": "Striker"},
             "stats": _stats(appearances=1, totalShots=4, shotsOnTarget=2, totalGoals=1,
                             foulsCommitted=3, yellowCards=1, goalAssists=0)},
            {"starter": False, "subbedIn": False, "position": {"abbreviation": "D"},
             "athlete": {"id": "2", "displayName": "Benchwarmer"},
             "stats": _stats(appearances=0)},
        ]},
        {"team": {"id": "999", "displayName": "Other"}, "roster": [
            {"starter": True, "athlete": {"id": "3", "displayName": "Opponent"},
             "stats": _stats(appearances=1, foulsCommitted=9)},
        ]},
    ]}


def test_match_player_counts_maps_metrics():
    counts, meta, minutes = match_player_counts(_summary(), "464")
    assert set(counts) == {1}                      # only the player who appeared, Norway only
    c = counts[1]
    assert c["shots"] == 4 and c["sot"] == 2 and c["goals"] == 1
    assert c["fouls"] == 3 and c["yellows"] == 1
    assert c["xg"] == 0.0 and c["key_passes"] == 0.0 and c["pens"] == 0.0
    assert meta[1] == ("Striker", "Norway", "F")
    assert minutes[1] == 90.0


def test_non_appearing_player_skipped():
    counts, _, _ = match_player_counts(_summary(), "464")
    assert 2 not in counts                          # appearances=0, not starter/sub


def test_other_team_ignored():
    counts, _, _ = match_player_counts(_summary(), "464")
    assert 3 not in counts                          # belongs to team 999


def test_team_id_int_or_str():
    counts, _, _ = match_player_counts(_summary(), 464)  # int also matches
    assert 1 in counts


def test_stat_map_covers_prop_metrics():
    assert set(ESPN_STAT_MAP.values()) >= {"shots", "sot", "goals", "fouls", "yellows", "assists"}


def test_espn_profile_source_and_goals_fallback():
    cm = match_player_counts(_summary(), "464")
    profiles = build_player_profiles([cm] * 8, source="espn")
    striker = {p.player_id: p for p in profiles}[1]
    assert striker.source == "espn"
    assert striker.xg_per90.mean == 0.0            # ESPN has no xG
    assert math.isclose(striker.goals_per90.mean, 1.0, rel_tol=1e-6)
    # p_anytime_scorer falls back to goals rate when xG absent
    assert math.isclose(striker.p_anytime_scorer, 1 - math.exp(-1.0), rel_tol=1e-6)


def test_espn_board_anytime_uses_goals_not_xg():
    cm = match_player_counts(_summary(), "464")
    profiles = build_player_profiles([cm] * 8, source="espn")
    board = prop_board(profiles)
    scorer = [c for c in board if c.market == "Anytime scorer"]
    assert any(c.player_name == "Striker" for c in scorer)
    # stat string shows goals rate, not xG, for ESPN profiles
    assert any("goles/90" in c.stat for c in scorer)
    assert all("xG" not in c.stat for c in scorer)
