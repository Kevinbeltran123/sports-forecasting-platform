"""Tests for per-player advanced metrics (progression/creation/defence)."""
from __future__ import annotations

import math

from bip.evaluation.tournaments.team_style_profiler.player_advanced import (
    build_advanced_profiles,
    extract_match_advanced,
    weak_links,
)


def _pass(pid, team, x0, x1, poss, idx):
    return {"type": {"name": "Pass"}, "player": {"id": pid, "name": f"P{pid}"},
            "team": {"name": team}, "location": [x0, 40], "possession": poss, "index": idx,
            "pass": {"end_location": [x1, 40]}}


def _carry(pid, team, x0, x1, poss, idx):
    return {"type": {"name": "Carry"}, "player": {"id": pid, "name": f"P{pid}"},
            "team": {"name": team}, "location": [x0, 40], "possession": poss, "index": idx,
            "carry": {"end_location": [x1, 40]}}


def _shot(pid, team, poss, idx, outcome="Saved"):
    return {"type": {"name": "Shot"}, "player": {"id": pid, "name": f"P{pid}"},
            "team": {"name": team}, "possession": poss, "index": idx,
            "shot": {"outcome": {"name": outcome}}}


def _duel(pid, team, subtype):
    return {"type": {"name": "Duel"}, "player": {"id": pid, "name": f"P{pid}"},
            "team": {"name": team}, "duel": {"type": {"name": subtype}}}


def _events():
    return [
        _pass(10, "A", 50, 100, poss=1, idx=1),   # progressive (Δ50) — P10
        _carry(11, "A", 60, 90, poss=1, idx=2),   # progressive (Δ30) — P11
        _shot(9, "A", poss=1, idx=3, outcome="Goal"),  # SCA→P11,P10 ; GCA too
        _pass(12, "A", 50, 55, poss=2, idx=1),    # NOT progressive (Δ5)
        _duel(5, "A", "Tackle"),                  # tackle P5
        _duel(6, "A", "Aerial Lost"),             # aerial lost P6
        {"type": {"name": "Clearance"}, "player": {"id": 6, "name": "P6"},
         "team": {"name": "A"}, "clearance": {"aerial_won": True}},  # aerial won P6
        {"type": {"name": "Dribbled Past"}, "player": {"id": 7, "name": "P7"},
         "team": {"name": "A"}},                  # P7 beaten
    ]


def test_progressive_pass_and_carry():
    counts, _ = extract_match_advanced(_events())
    assert counts[10]["prog_pass"] == 1
    assert counts[11]["prog_carry"] == 1
    assert counts[12].get("prog_pass", 0) == 0     # Δ5 < threshold


def test_defensive_counts():
    counts, _ = extract_match_advanced(_events())
    assert counts[5]["tackles"] == 1
    assert counts[6]["aerial_lost"] == 1
    assert counts[6]["aerial_won"] == 1
    assert counts[7]["dribbled_past"] == 1


def test_sca_gca_credited_to_last_two_offensive_actions():
    counts, _ = extract_match_advanced(_events())
    # shot by P9 in possession 1 → credits carry (P11) + pass (P10)
    assert counts[11]["sca"] == 1 and counts[10]["sca"] == 1
    assert counts[9].get("sca", 0) == 0            # shooter not credited
    # it was a Goal → GCA too
    assert counts[11]["gca"] == 1 and counts[10]["gca"] == 1


def test_profile_aerial_win_rate():
    counts, meta = extract_match_advanced(_events())
    minutes = {pid: 90.0 for pid in counts}
    profiles = build_advanced_profiles([(counts, meta, minutes)] * 8)
    p6 = {p.player_id: p for p in profiles}[6]
    assert math.isclose(p6.aerial_win_rate, 0.5, rel_tol=1e-6)   # 1 won / (1 won + 1 lost)


def test_weak_links_flags_beaten_defender():
    counts = {7: {"dribbled_past": 2.0}}
    meta = {7: ("Slow Back", "A", "Right Back")}
    minutes = {7: 90.0}
    profiles = build_advanced_profiles([(counts, meta, minutes)] * 8)
    wl = weak_links(profiles)
    assert any(w["player"] == "Slow Back" and "beaten" in w["reason"] for w in wl)
