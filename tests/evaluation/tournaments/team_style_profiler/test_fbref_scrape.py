"""Tests for the FBref parser (offline) — comment-strip + data-stat extraction."""
from __future__ import annotations

from bip.evaluation.tournaments.team_style_profiler.fbref_scrape import (
    parse_player_table,
    season_stats_url,
)

# FBref wraps its data tables in HTML comments; the parser must surface them.
_COMMENTED = """
<div id="all_stats_shooting">
<!--
<table id="stats_shooting">
  <thead><tr><th data-stat="player">Player</th></tr></thead>
  <tbody>
    <tr class="thead"><th data-stat="player">Player</th></tr>
    <tr>
      <th data-stat="player">Erling Haaland</th>
      <td data-stat="team">Norway</td>
      <td data-stat="shots">5</td>
      <td data-stat="shots_on_target">3</td>
      <td data-stat="goals">2</td>
    </tr>
    <tr>
      <th data-stat="player">Martin Ødegaard</th>
      <td data-stat="team">Norway</td>
      <td data-stat="shots">2</td>
      <td data-stat="shots_on_target">1</td>
      <td data-stat="goals">0</td>
    </tr>
    <tr><th data-stat="player"></th><td data-stat="team">x</td></tr>
  </tbody>
</table>
-->
</div>
"""

_FIELDS = ["player", "team", "shots", "shots_on_target", "goals", "xg"]


def test_parse_surfaces_commented_table():
    rows = parse_player_table(_COMMENTED, "stats_shooting", _FIELDS)
    assert len(rows) == 2                        # 2 players; thead + empty-player rows skipped
    assert rows[0]["player"] == "Erling Haaland"
    assert rows[0]["shots"] == "5"
    assert rows[1]["player"] == "Martin Ødegaard"


def test_missing_field_is_empty_string():
    rows = parse_player_table(_COMMENTED, "stats_shooting", _FIELDS)
    assert rows[0]["xg"] == ""                   # xG column absent → "" (FBref WC tables lack xG)


def test_unknown_table_returns_empty():
    assert parse_player_table(_COMMENTED, "stats_nonexistent", _FIELDS) == []


def test_season_stats_url():
    assert season_stats_url(1, "2022", "World-Cup", "shooting") == (
        "https://fbref.com/en/comps/1/2022/shooting/2022-World-Cup-Stats"
    )
    assert season_stats_url(9, "2024-2025", "Premier-League", "stats") == (
        "https://fbref.com/en/comps/9/2024-2025/stats/2024-2025-Premier-League-Stats"
    )
