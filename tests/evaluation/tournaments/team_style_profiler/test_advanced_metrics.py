"""Tests for advanced team metrics (field tilt, GK shot-stopping, game-state xG)."""
from __future__ import annotations

import math

from bip.evaluation.tournaments.team_style_profiler.advanced_metrics import (
    _goal_timeline,
    _state,
    aggregate_team,
    build_profiles,
    extract_match,
)


def _shot(team, minute, xg, outcome):
    return {"type": {"name": "Shot"}, "team": {"name": team}, "minute": minute,
            "shot": {"statsbomb_xg": xg, "outcome": {"name": outcome}}}


def _pass(team, x):
    return {"type": {"name": "Pass"}, "team": {"name": team}, "location": [x, 40]}


def _match_events():
    return [
        _shot("A", 10, 0.3, "Off T"),      # A level
        _shot("A", 20, 0.5, "Goal"),       # A level when taken → then 1-0
        _shot("A", 30, 0.4, "Saved"),      # A leading
        _shot("B", 40, 0.6, "Goal"),       # B trailing when taken → then 1-1
        _pass("A", 95), _pass("A", 82), _pass("A", 88),  # 3 final-third (A)
        _pass("B", 90),                                   # 1 final-third (B)
        _pass("A", 40),                                   # midfield, ignored
    ]


def test_goal_timeline_and_state():
    goals = _goal_timeline(_match_events())
    assert goals == [(20, "A"), (40, "B")]
    assert _state("A", "B", 25, goals) == "leading"   # A 1-0 before min 25
    assert _state("B", "A", 25, goals) == "trailing"
    assert _state("A", "B", 15, goals) == "level"     # nothing before min 15


def test_extract_match_field_tilt_and_gk():
    a, b = sorted(extract_match(_match_events(), "A", "B"), key=lambda m: m.team)
    assert a.team == "A"
    assert a.ft_for == 3 and a.ft_against == 1
    assert b.ft_for == 1 and b.ft_against == 3
    # A's keeper faces B's on-target shots: only the min40 goal (xg 0.6); concedes 1
    assert math.isclose(a.xgot_faced, 0.6, rel_tol=1e-6)
    assert a.goals_conceded == 1
    # B's keeper faces A's on-target: min20 Goal (0.5) + min30 Saved (0.4) = 0.9; concedes 1
    assert math.isclose(b.xgot_faced, 0.9, rel_tol=1e-6)
    assert b.goals_conceded == 1


def test_extract_match_game_state_xg():
    a = next(m for m in extract_match(_match_events(), "A", "B") if m.team == "A")
    # A: min10 level(0.3) + min20 level(0.5) = 0.8 level; min30 leading(0.4); trailing 0
    assert math.isclose(a.xg_level, 0.8, rel_tol=1e-6)
    assert math.isclose(a.xg_leading, 0.4, rel_tol=1e-6)
    assert a.xg_trailing == 0.0
    b = next(m for m in extract_match(_match_events(), "A", "B") if m.team == "B")
    assert math.isclose(b.xg_trailing, 0.6, rel_tol=1e-6)   # B shot while behind


def test_aggregate_shares_sum_to_one_and_tilt():
    matches = [m for m in extract_match(_match_events(), "A", "B") if m.team == "A"]
    prof = aggregate_team("A", matches)
    assert math.isclose(
        prof.xg_share_leading + prof.xg_share_level + prof.xg_share_trailing, 1.0, rel_tol=1e-6)
    # A field tilt = 3/(3+1) = 0.75
    assert math.isclose(prof.field_tilt.mean, 0.75, rel_tol=1e-6)


def test_line_height_and_directness():
    events = [
        # A defends high (x 70, 80 via interception + tackle), B deep (x 20, 30)
        {"type": {"name": "Interception"}, "team": {"name": "A"}, "location": [70, 40]},
        {"type": {"name": "Duel"}, "team": {"name": "A"}, "location": [80, 40],
         "duel": {"type": {"name": "Tackle"}}},
        {"type": {"name": "Interception"}, "team": {"name": "B"}, "location": [20, 40]},
        {"type": {"name": "Clearance"}, "team": {"name": "B"}, "location": [30, 40]},
        # A short pass (Δ5), B long pass (Δ30)
        {"type": {"name": "Pass"}, "team": {"name": "A"}, "location": [50, 40],
         "pass": {"end_location": [55, 40]}},
        {"type": {"name": "Pass"}, "team": {"name": "B"}, "location": [50, 40],
         "pass": {"end_location": [80, 40]}},
    ]
    a, b = sorted(extract_match(events, "A", "B"), key=lambda m: m.team)
    assert a.line_height == 75.0       # mean(70, 80) — high line
    assert b.line_height == 25.0       # mean(20, 30) — deep block
    assert a.directness == 0.0         # 0/1 passes long
    assert b.directness == 1.0         # 1/1 passes long


def test_build_profiles_groups_both_teams():
    profs = build_profiles([extract_match(_match_events(), "A", "B")])
    assert set(profs) == {"A", "B"}
    assert profs["A"].n_matches == 1
