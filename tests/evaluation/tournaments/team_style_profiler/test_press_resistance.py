"""Tests for press-resistance (build-up under pressure)."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.press_resistance import (
    FRAGILE_MAX,
    RESISTANT_MIN,
    PressResistance,
    _resistance,
    build_press_resistance,
    extract_match_press,
    press_resistance_note,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat


def _pass(team, under_pressure, completed=True):
    p = {"type": {"name": "Pass"}, "team": {"name": team}, "pass": {}}
    if under_pressure:
        p["under_pressure"] = True
    if not completed:
        p["pass"]["outcome"] = {"name": "Incomplete"}
    return p


def test_extract_counts_passes_and_completions_under_pressure():
    events = [
        _pass("A", under_pressure=True, completed=True),
        _pass("A", under_pressure=True, completed=False),
        _pass("A", under_pressure=False, completed=True),   # not pressed → ignored
        _pass("B", under_pressure=True, completed=True),
    ]
    out = extract_match_press(events)
    assert out["A"] == (2, 1)        # 2 pressed passes, 1 completed
    assert out["B"] == (1, 1)


def test_extract_ignores_non_pass_and_teamless():
    events = [
        {"type": {"name": "Pressure"}, "team": {"name": "A"}, "under_pressure": True},
        {"type": {"name": "Pass"}, "under_pressure": True, "pass": {}},   # no team
    ]
    assert extract_match_press(events) == {}


# ── Resistance classification boundaries ──


@pytest.mark.parametrize("rate,expected", [
    (0.50, "press-fragile"),
    (FRAGILE_MAX, "press-fragile"),          # inclusive
    (FRAGILE_MAX + 0.001, "average"),
    (RESISTANT_MIN - 0.001, "average"),
    (RESISTANT_MIN, "press-resistant"),      # inclusive
    (0.90, "press-resistant"),
])
def test_resistance_boundaries(rate, expected):
    assert _resistance(rate) == expected


def test_build_aggregates_completion_rate_and_confidence():
    # 10 matches, each 10 pressed passes / 8 completed → rate 0.8 (resistant)
    per_match = [{"A": (10, 8)} for _ in range(10)]
    table = build_press_resistance(per_match)
    a = table["A"]
    assert a.n_matches == 10 and a.confidence == "green"   # N_GREEN_MIN = 10
    assert abs(a.completion_under_pressure.mean - 0.8) < 1e-6
    assert a.resistance == "press-resistant"


def test_build_skips_team_with_no_pressed_passes():
    table = build_press_resistance([{"A": (0, 0)}])
    assert "A" not in table          # no usable rate → omitted (not fabricated)


# ── Matchup note ──


def _pr(team, rate, conf="green", res=None):
    ds = DistributionStat(mean=rate, ci_low=rate, ci_high=rate, n=10)
    return PressResistance(
        team=team, n_matches=10, confidence=conf,
        passes_under_pressure_per_match=DistributionStat(mean=70, ci_low=70, ci_high=70, n=10),
        completion_under_pressure=ds, resistance=res or _resistance(rate))


def test_note_flags_fragile_under_high_press():
    note = press_resistance_note("high_press", _pr("Malta", 0.60))
    assert note is not None and "frágil" in note and "BTTS/Over" in note


def test_note_flags_resistant_neutralises_press():
    note = press_resistance_note("high_press", _pr("Spain", 0.80))
    assert note is not None and "resistente" in note and "neutraliza" in note


def test_note_none_when_presser_not_high():
    assert press_resistance_note("low_block", _pr("Malta", 0.60)) is None


def test_note_none_for_low_confidence():
    assert press_resistance_note("high_press", _pr("X", 0.60, conf="red")) is None


def test_note_none_without_profile():
    assert press_resistance_note("high_press", None) is None
