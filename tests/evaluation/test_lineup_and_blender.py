"""Phase 2.2 — lineup predictor, recent-form normalization, composition blender."""

from __future__ import annotations

import polars as pl
import pytest

from bip.evaluation.tournaments.data.league_strength import (
    LeagueStrength,
    LeagueStrengthTable,
)
from bip.evaluation.tournaments.models import Player, Position
from bip.evaluation.tournaments.players.lineup_predictor import (
    LineupConfidence,
    LineupPrediction,
    predict_from_confirmed,
    predict_from_qualifier_history,
    predict_from_squad_only,
    starter_count_from_lineups,
)
from bip.evaluation.tournaments.players.recent_form import normalize_player_rates
from bip.evaluation.tournaments.predict.blender import (
    DEFAULT_ALPHA,
    blend_team_rates,
)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


def _build_squad(n: int, *, n_g: int = 3, n_d: int = 8, n_m: int = 8, n_f: int = 3) -> list[Player]:
    """Build a synthetic squad of size n with default position distribution."""
    if n_g + n_d + n_m + n_f != n:
        raise ValueError("position counts must sum to n")
    players: list[Player] = []
    pid = 1
    for _ in range(n_g):
        players.append(Player.from_api_football(pid, f"G{pid}", Position.GOALKEEPER, 6))
        pid += 1
    for _ in range(n_d):
        players.append(Player.from_api_football(pid, f"D{pid}", Position.DEFENDER, 6))
        pid += 1
    for _ in range(n_m):
        players.append(Player.from_api_football(pid, f"M{pid}", Position.MIDFIELDER, 6))
        pid += 1
    for _ in range(n_f):
        players.append(Player.from_api_football(pid, f"F{pid}", Position.FORWARD, 6))
        pid += 1
    return players


# ─────────────────────────────────────────────────────────────────
# lineup_predictor
# ─────────────────────────────────────────────────────────────────


class TestLineupPredictor:
    def test_predict_from_confirmed_round_trip(self):
        squad = _build_squad(22)
        xi = squad[:11]
        pred = predict_from_confirmed(xi)
        assert pred.confidence == LineupConfidence.CONFIRMED
        assert len(pred.players) == 11
        assert pred.players[0].api_football_id == 1

    def test_predict_from_confirmed_rejects_wrong_size(self):
        squad = _build_squad(22)
        with pytest.raises(ValueError, match="11 players"):
            predict_from_confirmed(squad[:10])

    def test_predict_from_qualifier_history_picks_top_starters(self):
        squad = _build_squad(22)
        # All-zero except the first 11 are clear starters.
        starter_counts = {p.api_football_id: 0 for p in squad}
        for p in squad[:11]:
            starter_counts[p.api_football_id] = 8  # high count
        for p in squad[11:]:
            starter_counts[p.api_football_id] = 1
        minutes = {p.api_football_id: 720 for p in squad[:11]}
        minutes.update({p.api_football_id: 90 for p in squad[11:]})

        pred = predict_from_qualifier_history(squad, starter_counts, minutes)
        assert pred.confidence == LineupConfidence.HEURISTIC_QUALIFIER
        chosen_ids = {p.api_football_id for p in pred.players}
        expected_ids = {p.api_football_id for p in squad[:11]}
        assert chosen_ids == expected_ids

    def test_predict_from_qualifier_history_falls_back_when_sparse(self):
        squad = _build_squad(22)
        # Only 5 players have any qualifier history.
        starter_counts = {p.api_football_id: 1 for p in squad[:5]}
        minutes = {p.api_football_id: 90 for p in squad[:5]}
        pred = predict_from_qualifier_history(squad, starter_counts, minutes)
        assert pred.confidence == LineupConfidence.FALLBACK_SQUAD

    def test_predict_from_squad_only_balanced_433(self):
        squad = _build_squad(22)  # 3G/8D/8M/3F
        pred = predict_from_squad_only(squad)
        assert pred.confidence == LineupConfidence.FALLBACK_SQUAD
        positions = [p.position for p in pred.players]
        assert positions.count(Position.GOALKEEPER) == 1
        assert positions.count(Position.DEFENDER) == 4
        assert positions.count(Position.MIDFIELDER) == 3
        assert positions.count(Position.FORWARD) == 3

    def test_squad_only_fails_if_position_short(self):
        # Only 2 forwards available (need 3).
        squad = _build_squad(22, n_g=3, n_d=8, n_m=9, n_f=2)
        with pytest.raises(ValueError, match="lacks 3"):
            predict_from_squad_only(squad)

    def test_lineup_prediction_must_be_eleven(self):
        with pytest.raises(ValueError, match="11 players"):
            LineupPrediction(
                players=tuple(_build_squad(10, n_g=1, n_d=3, n_m=3, n_f=3)),
                confidence=LineupConfidence.CONFIRMED,
            )

    def test_starter_count_from_lineups_aggregates_correctly(self):
        payloads = [
            {
                "response": [
                    {
                        "team": {"id": 6},
                        "startXI": [
                            {"player": {"id": 1}},
                            {"player": {"id": 2}},
                            {"player": {"id": 3}},
                        ],
                    }
                ]
            },
            {
                "response": [
                    {
                        "team": {"id": 6},
                        "startXI": [
                            {"player": {"id": 1}},
                            {"player": {"id": 2}},
                        ],
                    }
                ]
            },
        ]
        starts, minutes = starter_count_from_lineups(payloads, team_id=6)
        assert starts == {1: 2, 2: 2, 3: 1}
        assert minutes == {1: 180, 2: 180, 3: 90}

    def test_starter_count_filters_other_teams(self):
        payloads = [
            {
                "response": [
                    {
                        "team": {"id": 999},  # different team
                        "startXI": [{"player": {"id": 50}}],
                    }
                ]
            },
        ]
        starts, _ = starter_count_from_lineups(payloads, team_id=6)
        assert starts == {}


# ─────────────────────────────────────────────────────────────────
# recent_form normalization
# ─────────────────────────────────────────────────────────────────


def _strength_table() -> LeagueStrengthTable:
    return LeagueStrengthTable(
        [
            LeagueStrength(
                api_football_league_id=39,
                name="EPL",
                multiplier=1.0,
                confidence="high",
            ),
            LeagueStrength(
                api_football_league_id=253,
                name="MLS",
                multiplier=0.5,
                confidence="medium",
            ),
        ]
    )


class TestNormalizePlayerRates:
    def test_epl_player_rate_unchanged(self):
        df = pl.DataFrame(
            {
                "club_league_id": [39],
                "shots_per90": [3.0],
                "sot_per90": [1.5],
                "goals_per90": [0.5],
                "assists_per90": [0.3],
                "key_passes_per90": [2.0],
            }
        )
        out = normalize_player_rates(df, _strength_table())
        assert out["normalized_shots_per90"][0] == pytest.approx(3.0)
        assert out["normalized_goals_per90"][0] == pytest.approx(0.5)

    def test_mls_player_rate_halved(self):
        df = pl.DataFrame(
            {
                "club_league_id": [253],
                "shots_per90": [3.0],
                "sot_per90": [1.5],
                "goals_per90": [0.5],
                "assists_per90": [0.3],
                "key_passes_per90": [2.0],
            }
        )
        out = normalize_player_rates(df, _strength_table())
        assert out["normalized_shots_per90"][0] == pytest.approx(1.5)
        assert out["normalized_goals_per90"][0] == pytest.approx(0.25)

    def test_unknown_league_uses_fallback(self):
        df = pl.DataFrame(
            {
                "club_league_id": [99999],
                "shots_per90": [3.0],
                "goals_per90": [0.5],
            }
        )
        out = normalize_player_rates(df, _strength_table())
        # FALLBACK_MULTIPLIER = 0.70
        assert out["normalized_shots_per90"][0] == pytest.approx(3.0 * 0.70)

    def test_empty_input_returns_empty(self):
        df = pl.DataFrame({"club_league_id": []}, schema={"club_league_id": pl.Int64})
        out = normalize_player_rates(df, _strength_table())
        assert out.is_empty()


# ─────────────────────────────────────────────────────────────────
# blender
# ─────────────────────────────────────────────────────────────────


def _team_baseline(team_id: int = 6) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "team_id": [team_id],
            "adjusted_gf_per90": [1.8],
            "adjusted_shots_for_per90": [13.0],
            "adjusted_sot_for_per90": [4.5],
            "adjusted_fouls_for_per90": [11.0],
            "adjusted_corners_for_per90": [5.5],
        }
    )


def _lineup_form(n: int = 11) -> pl.DataFrame:
    """11 players each contributing ~0.18 normalized goals per 90 -> sum = 2.0."""
    return pl.DataFrame(
        {
            "canonical_id": [f"af-{i}" for i in range(n)],
            "normalized_goals_per90": [2.0 / n] * n,
            "normalized_shots_per90": [12.0 / n] * n,
            "normalized_sot_per90": [4.8 / n] * n,
            "fouls_committed_per90": [10.0 / n] * n,
            "reliable": [True] * n,
        }
    )


class TestBlender:
    def test_default_alpha_blends_correctly(self):
        team = _team_baseline()
        lineup = _lineup_form()
        out = blend_team_rates(team, lineup)
        # alpha=0.35: 0.35*1.8 + 0.65*2.0 = 0.63 + 1.30 = 1.93
        assert out.expected_goals_per90 == pytest.approx(0.35 * 1.8 + 0.65 * 2.0)
        assert out.alpha == DEFAULT_ALPHA
        assert out.team_id == 6
        assert out.team_component_goals == pytest.approx(1.8)
        assert out.lineup_component_goals == pytest.approx(2.0)

    def test_alpha_one_uses_only_team(self):
        out = blend_team_rates(_team_baseline(), _lineup_form(), alpha=1.0)
        assert out.expected_goals_per90 == pytest.approx(1.8)

    def test_alpha_zero_uses_only_lineup(self):
        out = blend_team_rates(_team_baseline(), _lineup_form(), alpha=0.0)
        assert out.expected_goals_per90 == pytest.approx(2.0)

    def test_alpha_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="in.*0.*1"):
            blend_team_rates(_team_baseline(), _lineup_form(), alpha=1.5)

    def test_corners_passed_through_team_only(self):
        out = blend_team_rates(_team_baseline(), _lineup_form())
        # Corners are not blended — team baseline only.
        assert out.expected_corners_for_per90 == pytest.approx(5.5)

    def test_n_starters_with_form_counted(self):
        out = blend_team_rates(_team_baseline(), _lineup_form())
        assert out.n_starters_with_form == 11

    def test_unreliable_players_counted_separately(self):
        lineup = _lineup_form()
        # Mark 4 as unreliable.
        lineup = lineup.with_columns(
            pl.Series("reliable", [True] * 7 + [False] * 4)
        )
        out = blend_team_rates(_team_baseline(), lineup)
        assert out.n_starters_with_form == 7

    def test_falls_back_to_raw_columns_if_adjusted_missing(self):
        """If adjust_team_baselines wasn't called, raw cols are used."""
        team = pl.DataFrame(
            {
                "team_id": [6],
                "gf_per90": [1.5],
                "shots_for_per90": [12.0],
                "sot_for_per90": [4.0],
                "fouls_for_per90": [10.0],
                "corners_for_per90": [5.0],
            }
        )
        out = blend_team_rates(team, _lineup_form(), alpha=1.0)
        assert out.expected_goals_per90 == pytest.approx(1.5)

    def test_baseline_must_be_single_row(self):
        team = pl.concat([_team_baseline(1), _team_baseline(2)])
        with pytest.raises(ValueError, match="exactly 1 row"):
            blend_team_rates(team, _lineup_form())
