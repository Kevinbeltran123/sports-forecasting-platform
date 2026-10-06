"""Tests for the API-Football player-prop parser (mejora #2)."""
from __future__ import annotations

from bip.evaluation.tournaments.team_style_profiler.player_props_apifootball import parse_fixture

TID = 9        # our team
OPP = 32       # opponent


def _player(pid, name, minutes, *, pos="M", shots=0, on=0, goals=0,
            fouls=None, drawn=0, kp=0, assists=0, pen_scored=0, pen_missed=0):
    return {
        "player": {"id": pid, "name": name},
        "statistics": [{
            "games": {"minutes": minutes, "position": pos},
            "shots": {"total": shots, "on": on},
            "goals": {"total": goals, "assists": assists},
            "fouls": {"committed": fouls, "drawn": drawn},
            "passes": {"key": kp},
            "penalty": {"scored": pen_scored, "missed": pen_missed},
        }],
    }


def _players_resp(team_id, players):
    return [{"team": {"id": team_id, "name": "Us"}, "players": players}]


def _card(pid, team_id, detail, minute):
    return {"type": "Card", "detail": detail, "time": {"elapsed": minute},
            "player": {"id": pid}, "team": {"id": team_id}}


def test_basic_extraction_and_dnp_excluded():
    players = _players_resp(TID, [
        _player(1, "Starter", 90, shots=3, on=2, goals=1, fouls=2, drawn=1, kp=4),
        _player(2, "Benchwarmer", 0, shots=0),  # DNP -> excluded
    ])
    counts, meta, minutes = parse_fixture(players, [], TID)
    assert set(counts) == {1}              # DNP dropped
    assert minutes[1] == 90.0
    assert counts[1]["shots"] == 3.0 and counts[1]["sot"] == 2.0
    assert counts[1]["goals"] == 1.0 and counts[1]["fouls"] == 2.0
    assert counts[1]["fouls_won"] == 1.0 and counts[1]["key_passes"] == 4.0
    assert counts[1]["xg"] == 0.0          # no xG for internationals
    assert meta[1] == ("Starter", "Us", "M")


def test_none_fouls_coerced_to_zero():
    players = _players_resp(TID, [_player(1, "P", 90, fouls=None, drawn=None)])
    counts, _, _ = parse_fixture(players, [], TID)
    assert counts[1]["fouls"] == 0.0
    assert counts[1]["fouls_won"] == 0.0


def test_yellows_counted_from_events_not_players():
    players = _players_resp(TID, [_player(1, "Booked", 90), _player(2, "Clean", 90)])
    events = [
        _card(1, TID, "Yellow Card", 30),
        _card(1, TID, "Yellow Card", 80),   # second yellow -> counts as a shown yellow
        _card(1, TID, "Red Card", 80),       # red ignored for yellows
        _card(99, OPP, "Yellow Card", 40),   # opponent -> ignored
    ]
    counts, _, _ = parse_fixture(players, events, TID)
    assert counts[1]["yellows"] == 2.0
    assert counts[2]["yellows"] == 0.0


def test_penalties_taken_is_scored_plus_missed():
    players = _players_resp(TID, [_player(1, "Taker", 90, pen_scored=1, pen_missed=1)])
    counts, _, _ = parse_fixture(players, [], TID)
    assert counts[1]["pens"] == 2.0


def test_opponent_block_ignored():
    players = [
        {"team": {"id": OPP, "name": "Them"}, "players": [_player(50, "Rival", 90, shots=5)]},
        {"team": {"id": TID, "name": "Us"}, "players": [_player(1, "Ours", 90, shots=1)]},
    ]
    counts, meta, _ = parse_fixture(players, [], TID)
    assert set(counts) == {1}
    assert meta[1][0] == "Ours"


def test_missing_team_block_returns_empty():
    counts, meta, minutes = parse_fixture(_players_resp(OPP, [_player(50, "X", 90)]), [], TID)
    assert counts == {} and meta == {} and minutes == {}
