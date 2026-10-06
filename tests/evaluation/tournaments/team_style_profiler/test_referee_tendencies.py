"""Tests for the referee tendency table + board cross."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.player_props import PropCandidate
from bip.evaluation.tournaments.team_style_profiler.referee_tendencies import (
    CARDS_LENIENT_MAX,
    CARDS_STRICT_MIN,
    FOULS_HIGH_MIN,
    FOULS_LOW_MAX,
    PEN_PRONE_MIN,
    PEN_SHY_MAX,
    MatchDiscipline,
    RefereeTendency,
    _foul_volume,
    _pen_tendency,
    _strictness,
    apply_referee_to_board,
    build_referee_tendencies,
    parse_match_discipline,
    referee_for,
    referee_match_note,
)


def _events():
    return [
        {"type": {"name": "Foul Committed"}, "player": {"id": 1},
         "foul_committed": {"card": {"name": "Yellow Card"}}},
        {"type": {"name": "Foul Committed"}, "player": {"id": 2}, "foul_committed": {}},
        {"type": {"name": "Foul Committed"}, "player": {"id": 3},
         "foul_committed": {"card": {"name": "Second Yellow"}}},   # yellow AND red
        {"type": {"name": "Bad Behaviour"}, "player": {"id": 4},
         "bad_behaviour": {"card": {"name": "Red Card"}}},
        {"type": {"name": "Shot"}, "player": {"id": 5},
         "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}}},
        {"type": {"name": "Shot"}, "player": {"id": 5},
         "shot": {"type": {"name": "Open Play"}, "outcome": {"name": "Goal"}}},
    ]


def test_parse_match_discipline_counts():
    d = parse_match_discipline(_events())
    assert d.fouls == 3
    assert d.yellows == 2          # Yellow Card + Second Yellow
    assert d.reds == 2             # Second Yellow + Red Card
    assert d.penalties == 1        # only the Penalty-type shot


def test_shootout_penalties_excluded():
    # period 5 = shootout: 5 pens that must NOT count toward in-game penalties.
    shootout = [{"type": {"name": "Shot"}, "period": 5,
                 "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}}}
                for _ in range(5)]
    in_game = [{"type": {"name": "Shot"}, "period": 2,
                "shot": {"type": {"name": "Penalty"}, "outcome": {"name": "Goal"}}}]
    d = parse_match_discipline(shootout + in_game)
    assert d.penalties == 1        # only the period-2 in-game penalty


# ── Strictness thresholds (boundary-parametrized) ──


@pytest.mark.parametrize("cards,expected", [
    (2.0, "lenient"),
    (CARDS_LENIENT_MAX, "lenient"),       # 3.2 inclusive
    (CARDS_LENIENT_MAX + 0.01, "average"),
    (CARDS_STRICT_MIN - 0.01, "average"),
    (CARDS_STRICT_MIN, "strict"),         # 3.8 inclusive
    (4.9, "strict"),
])
def test_strictness_boundaries(cards, expected):
    assert _strictness(cards) == expected


def test_build_tendencies_confidence_and_strictness():
    strict_matches = [MatchDiscipline(yellows=4, reds=1, fouls=30, penalties=1)] * 8
    lenient_matches = [MatchDiscipline(yellows=2, reds=0, fouls=22, penalties=0)] * 5
    table = build_referee_tendencies({
        "Strict Ref": ("Spain", strict_matches),
        "Lenient Ref": ("France", lenient_matches),
    })
    s = table["Strict Ref"]
    assert s.confidence == "green"        # 8 matches
    assert s.strictness == "strict"       # 5 cards/match
    assert s.cards_per_match.mean == 5.0
    l = table["Lenient Ref"]
    assert l.confidence == "yellow"       # 5 matches
    assert l.strictness == "lenient"      # 2 cards/match


def test_referee_for_lookup():
    table = build_referee_tendencies({"X": ("Y", [MatchDiscipline(3, 0, 25, 0)] * 4)})
    assert referee_for("X", table) is not None
    assert referee_for("Unknown", table) is None


# ── Board cross ──


def _board():
    return [
        PropCandidate("Faltas cometidas (over)", "Foulguy", "CB", "3.0 faltas/90", 1, "green"),
        PropCandidate("Tarjeta a jugador", "Cardguy", "CM", "0.5 amarillas/90", 1, "green"),
        PropCandidate("Anytime scorer", "Striker", "CF", "xG 0.8", 3, "green"),
    ]


def _ref(cards, name="Ref"):
    return RefereeTendency(
        referee_name=name, country="X", n_matches=8, confidence="green",
        strictness=_strictness(cards),
        cards_per_match=_ds(cards), yellows_per_match=_ds(cards), reds_per_match=_ds(0),
        fouls_per_match=_ds(28), penalties_per_match=_ds(0.5),
    )


def _ds(mean):
    from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat
    return DistributionStat(mean=mean, ci_low=mean, ci_high=mean, n=8)


def test_strict_ref_reinforces_soft_markets_only():
    out = apply_referee_to_board(_board(), _ref(4.9))
    soft = [c for c in out if c.softness == 1]
    hard = [c for c in out if c.softness == 3]
    assert all("estricto" in c.stat and "refuerza" in c.stat for c in soft)
    assert all("estricto" not in c.stat for c in hard)   # scorer untouched


def test_lenient_ref_attenuates_soft_markets():
    out = apply_referee_to_board(_board(), _ref(2.0))
    soft = [c for c in out if c.softness == 1]
    assert all("permisivo" in c.stat and "atenúa" in c.stat for c in soft)


def test_none_ref_leaves_board_unchanged():
    board = _board()
    assert apply_referee_to_board(board, None) == board


# ── Foul-volume + penalty-propensity (new dimensions) ──


@pytest.mark.parametrize("fouls,expected", [
    (20.0, "lets-play"),
    (FOULS_LOW_MAX, "lets-play"),            # inclusive
    (FOULS_LOW_MAX + 0.1, "average"),
    (FOULS_HIGH_MIN - 0.1, "average"),
    (FOULS_HIGH_MIN, "whistle-happy"),       # inclusive
    (35.0, "whistle-happy"),
])
def test_foul_volume_boundaries(fouls, expected):
    assert _foul_volume(fouls) == expected


@pytest.mark.parametrize("pens,expected", [
    (0.0, "pen-shy"),
    (PEN_SHY_MAX, "pen-shy"),                # inclusive
    (PEN_SHY_MAX + 0.01, "average"),
    (PEN_PRONE_MIN - 0.01, "average"),
    (PEN_PRONE_MIN, "pen-prone"),            # inclusive
    (1.0, "pen-prone"),
])
def test_pen_tendency_boundaries(pens, expected):
    assert _pen_tendency(pens) == expected


def test_properties_derive_from_rates():
    r = RefereeTendency(
        referee_name="R", country="X", n_matches=8, confidence="green",
        strictness="strict", cards_per_match=_ds(4.0), yellows_per_match=_ds(4.0),
        reds_per_match=_ds(0.0), fouls_per_match=_ds(30.0), penalties_per_match=_ds(0.6))
    assert r.foul_volume == "whistle-happy"
    assert r.pen_tendency == "pen-prone"


def test_board_foul_market_gets_foul_volume_note():
    out = apply_referee_to_board(_board(), _ref(4.9))   # fouls_per_match=28 → average
    # average foul volume → no foul-volume note (cards tag still applied)
    faltas = [c for c in out if "Faltas" in c.market]
    assert faltas and all(
        "pita muchas faltas" not in c.stat and "deja jugar" not in c.stat for c in faltas)
    # now a whistle-happy ref adds the note on the foul market specifically
    whistle = RefereeTendency(
        referee_name="W", country="X", n_matches=8, confidence="green",
        strictness="strict", cards_per_match=_ds(4.0), yellows_per_match=_ds(4.0),
        reds_per_match=_ds(0.0), fouls_per_match=_ds(32.0), penalties_per_match=_ds(0.3))
    out2 = apply_referee_to_board(_board(), whistle)
    faltas2 = [c for c in out2 if "Faltas" in c.market]
    assert faltas2 and all("pita muchas faltas" in c.stat for c in faltas2)
    # a non-foul soft market (cards) does NOT get the foul-volume note
    cards = [c for c in out2 if c.market == "Tarjeta a jugador"]
    assert cards and all("pita muchas faltas" not in c.stat for c in cards)


def test_referee_match_note_surfaces_all_dims():
    r = RefereeTendency(
        referee_name="Turpin", country="France", n_matches=10, confidence="green",
        strictness="strict", cards_per_match=_ds(4.2), yellows_per_match=_ds(4.0),
        reds_per_match=_ds(0.2), fouls_per_match=_ds(31.0), penalties_per_match=_ds(0.6))
    note = referee_match_note(r)
    assert "Turpin" in note and "penalti" in note and "faltas" in note
    assert "señal débil" in note          # honesty about small-n penalty signal
