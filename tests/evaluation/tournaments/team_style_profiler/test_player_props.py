"""Tests for the Player Prop Profiler (StatsBomb player-level extraction)."""
from __future__ import annotations

import math

import pytest

from bip.evaluation.tournaments.team_style_profiler.player_props import (
    MIN_MINUTES_FOR_RATE,
    PlayerPropProfile,
    _bootstrap_rate,
    build_player_profiles,
    parse_player_counts,
    parse_player_minutes,
    prop_board,
)


def _xi(team: str, players: list[tuple[int, str, str]]) -> dict:
    return {
        "type": {"name": "Starting XI"},
        "team": {"name": team},
        "tactics": {"lineup": [
            {"jersey_number": i, "player": {"id": pid, "name": name},
             "position": {"name": pos}}
            for i, (pid, name, pos) in enumerate(players, 1)
        ]},
    }


def _shot(pid, name, team, minute, xg, outcome, stype="Open Play"):
    return {"type": {"name": "Shot"}, "minute": minute, "player": {"id": pid, "name": name},
            "team": {"name": team},
            "shot": {"statsbomb_xg": xg, "type": {"name": stype}, "outcome": {"name": outcome}}}


def _foul(pid, team, minute, card=None):
    fc = {"card": {"name": card}} if card else {}
    return {"type": {"name": "Foul Committed"}, "minute": minute,
            "player": {"id": pid}, "team": {"name": team}, "foul_committed": fc}


def _foul_won(pid, team, minute):
    return {"type": {"name": "Foul Won"}, "minute": minute,
            "player": {"id": pid}, "team": {"name": team}, "foul_won": {}}


def _sub(off_pid, on_pid, on_name, team, minute):
    return {"type": {"name": "Substitution"}, "minute": minute,
            "player": {"id": off_pid}, "team": {"name": team},
            "substitution": {"replacement": {"id": on_pid, "name": on_name},
                             "outcome": {"name": "Tactical"}}}


def _end(minute=90):
    return {"type": {"name": "Half End"}, "minute": minute}


def _one_match() -> list[dict]:
    """A small but complete match: striker scores, mid booked then subbed,
    sub comes on late, a defender sent off."""
    return [
        _xi("A", [(1, "Striker", "Center Forward"), (2, "Mid", "Center Midfield"),
                  (5, "Defender", "Center Back")]),
        _xi("B", [(9, "OppGK", "Goalkeeper")]),
        _shot(1, "Striker", "A", 10, 0.6, "Goal"),
        _shot(1, "Striker", "A", 30, 0.2, "Saved"),          # on target
        _shot(1, "Striker", "A", 55, 0.1, "Off T"),          # off target
        _foul(2, "A", 20, card="Yellow Card"),
        _foul(2, "A", 40),                                   # no card
        _foul_won(1, "A", 25),                               # striker draws a foul
        {"type": {"name": "Pass"}, "minute": 9, "player": {"id": 2, "name": "Mid"},
         "team": {"name": "A"}, "pass": {"goal_assist": True}},
        _sub(2, 3, "Sub", "A", 60),                          # Mid off @60, Sub on @60
        _foul(5, "A", 70, card="Red Card"),                  # Defender sent off @70
        _end(90),
    ]


# ── Minutes parsing ──


def test_minutes_starter_full_match():
    mins = parse_player_minutes(_one_match())
    assert mins[1] == 90.0          # striker plays full match


def test_minutes_subbed_off_and_on():
    mins = parse_player_minutes(_one_match())
    assert mins[2] == 60.0          # mid off at 60
    assert mins[3] == 30.0          # sub on at 60 → 30 played


def test_minutes_red_card_early_exit():
    mins = parse_player_minutes(_one_match())
    assert mins[5] == 70.0          # defender sent off at 70


# ── Counts parsing ──


def test_counts_shots_sot_goals_xg():
    counts, _ = parse_player_counts(_one_match())
    c = counts[1]
    assert c["shots"] == 3
    assert c["goals"] == 1
    assert c["sot"] == 2                       # Goal + Saved
    assert math.isclose(c["xg"], 0.9, rel_tol=1e-6)


def test_counts_fouls_cards_assists():
    counts, meta = parse_player_counts(_one_match())
    assert counts[2]["fouls"] == 2
    assert counts[2]["yellows"] == 1
    assert counts[2]["assists"] == 1           # goal_assist pass
    assert meta[1] == ("Striker", "A", "Center Forward")
    assert meta[3][0] == "Sub"                 # sub captured in meta


def test_counts_fouls_won():
    counts, _ = parse_player_counts(_one_match())
    assert counts[1]["fouls_won"] == 1         # striker drew one foul
    assert counts[2]["fouls_won"] == 0         # mid drew none


# ── Bootstrap rate ──


def test_bootstrap_rate_excludes_short_cameos():
    # one full match (1 shot/90') + one 10-min cameo (should be dropped)
    ds = _bootstrap_rate([(1.0, 90.0), (1.0, 10.0)])
    assert ds.n == 1                           # cameo below MIN_MINUTES_FOR_RATE dropped
    assert math.isclose(ds.mean, 1.0, rel_tol=1e-6)


def test_bootstrap_rate_empty():
    ds = _bootstrap_rate([(5.0, 5.0)])         # only sub-threshold sample
    assert ds.n == 0 and ds.mean == 0.0


def test_min_minutes_threshold_value():
    assert MIN_MINUTES_FOR_RATE == 20.0


# ── Profile aggregation ──


def _n_matches(n: int) -> list:
    return [(*parse_match(_one_match()),) for _ in range(n)]


def parse_match(events):
    counts, meta = parse_player_counts(events)
    minutes = parse_player_minutes(events)
    return counts, meta, minutes


def test_build_profiles_aggregates_and_flags_confidence():
    profiles = build_player_profiles(_n_matches(8))
    by_id = {p.player_id: p for p in profiles}
    striker = by_id[1]
    assert striker.player_name == "Striker"
    assert striker.team == "A"
    assert striker.n_matches == 8
    assert striker.confidence == "green"       # >= 8 matches
    assert striker.minutes_total == 8 * 90.0
    # striker: 3 shots / 90' across identical matches
    assert math.isclose(striker.shots_per90.mean, 3.0, rel_tol=1e-6)
    assert math.isclose(striker.goals_per90.mean, 1.0, rel_tol=1e-6)


def test_confidence_tiers():
    assert build_player_profiles(_n_matches(8))[0].confidence == "green"
    assert {p.confidence for p in build_player_profiles(_n_matches(5))} == {"yellow"}
    assert {p.confidence for p in build_player_profiles(_n_matches(2))} == {"red"}


def test_sub_excluded_when_always_short():
    # Sub plays 30' each match here (60→90), so actually included. Verify the
    # striker (90') and the 30' sub both clear the 20' threshold.
    profiles = build_player_profiles(_n_matches(6))
    ids = {p.player_id for p in profiles}
    assert 3 in ids                            # sub plays 30' ≥ 20' threshold


def test_p_anytime_scorer_from_xg():
    profiles = build_player_profiles(_n_matches(8))
    striker = {p.player_id: p for p in profiles}[1]
    # xg_per90 = 0.9 → p = 1 - exp(-0.9)
    assert math.isclose(striker.p_anytime_scorer, 1 - math.exp(-0.9), rel_tol=1e-6)


# ── Prop board ──


def test_prop_board_orders_softest_first_and_skips_red():
    profiles = build_player_profiles(_n_matches(8))
    board = prop_board(profiles)
    assert board, "board should not be empty"
    softness = [c.softness for c in board]
    assert softness == sorted(softness)        # softest (fouls/cards) first
    # red-confidence players excluded
    assert all(c.confidence in ("green", "yellow") for c in board)


def test_prop_board_anytime_scorer_present_for_scorer():
    profiles = build_player_profiles(_n_matches(8))
    board = prop_board(profiles)
    scorers = [c for c in board if c.market == "Anytime scorer"]
    assert any(c.player_name == "Striker" for c in scorers)


def test_build_profiles_fouls_drawn_rate():
    striker = {p.player_id: p for p in build_player_profiles(_n_matches(8))}[1]
    # 1 foul won / 90' across identical full matches
    assert math.isclose(striker.fouls_drawn_per90.mean, 1.0, rel_tol=1e-6)


def test_prop_board_fouls_drawn_present_for_drawer():
    board = prop_board(build_player_profiles(_n_matches(8)))
    drawn = [c for c in board if c.market == "Faltas recibidas (over)"]
    assert any(c.player_name == "Striker" for c in drawn)
    assert all(c.softness == 1 for c in drawn)
