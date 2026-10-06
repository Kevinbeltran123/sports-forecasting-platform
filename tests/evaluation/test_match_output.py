"""Phase 3 — match output: player projections + JSON/Markdown emitters + runner."""

from __future__ import annotations

import json
from datetime import datetime

import polars as pl
import pytest

from bip.evaluation.tournaments.models import Player, Position
from bip.evaluation.tournaments.players.lineup_predictor import (
    LineupConfidence,
    LineupPrediction,
)
from bip.evaluation.tournaments.predict.match_runner import (
    MatchPredictionRunner,
    TeamMatchInputs,
    synthesize_lineup_form_for,
)
from bip.evaluation.tournaments.predict.output import (
    JSON_SCHEMA_VERSION,
    MODEL_VERSION,
    to_json_dict,
    to_markdown,
    write_json,
    write_markdown,
)
from bip.evaluation.tournaments.predict.player_projections import (
    project_starters,
)
from bip.evaluation.tournaments.predictors.bivariate_poisson import (
    BivariatePoissonModel,
)
from bip.evaluation.tournaments.predictors.independent_poisson import (
    IndependentPoissonModel,
)


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _build_lineup(team_id: int, start_pid: int = 1) -> LineupPrediction:
    """Standard 4-3-3 lineup of 11 players starting at given pid."""
    players: list[Player] = []
    pid = start_pid
    for _ in range(1):
        players.append(Player.from_api_football(pid, f"GK_{pid}", Position.GOALKEEPER, team_id))
        pid += 1
    for _ in range(4):
        players.append(Player.from_api_football(pid, f"DEF_{pid}", Position.DEFENDER, team_id))
        pid += 1
    for _ in range(3):
        players.append(Player.from_api_football(pid, f"MID_{pid}", Position.MIDFIELDER, team_id))
        pid += 1
    for _ in range(3):
        players.append(Player.from_api_football(pid, f"FWD_{pid}", Position.FORWARD, team_id))
        pid += 1
    return LineupPrediction(players=tuple(players), confidence=LineupConfidence.HEURISTIC_QUALIFIER)


def _lineup_form(canonical_ids: list[str], reliable: bool = True) -> pl.DataFrame:
    n = len(canonical_ids)
    return pl.DataFrame(
        {
            "canonical_id": canonical_ids,
            "position": ["F"] * n,
            "minutes_total": [1500] * n,
            "reliable": [reliable] * n,
            "normalized_shots_per90": [1.5] * n,
            "normalized_sot_per90": [0.6] * n,
            "normalized_goals_per90": [0.18] * n,  # ~ 11 * 0.18 = ~2.0 team goals
            "fouls_committed_per90": [1.0] * n,
            "shots_per90": [1.5] * n,
            "sot_per90": [0.6] * n,
            "goals_per90": [0.18] * n,
        }
    )


def _team_baseline(team_id: int) -> pl.DataFrame:
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


# ─────────────────────────────────────────────────────────────────
# project_starters
# ─────────────────────────────────────────────────────────────────


class TestProjectStarters:
    def test_basic_projection_shape(self):
        ids = [f"af-{i}" for i in range(11)]
        df = _lineup_form(ids)
        out = project_starters(df)
        assert len(out) == 11
        first = out[0]
        assert first.canonical_id == "af-0"
        assert first.lambda_shots == pytest.approx(1.5)
        assert first.lambda_shots_on_target == pytest.approx(0.6)
        assert first.reliable is True

    def test_p_anytime_scorer_formula(self):
        """P(scorer) = 1 - exp(-λ). With λ=0.18 -> ~0.165."""
        df = _lineup_form(["af-1"])
        out = project_starters(df)
        # 1 - exp(-0.18) = 0.16473
        assert out[0].p_anytime_scorer == pytest.approx(0.16473, abs=1e-4)

    def test_zero_goals_zero_scorer_probability(self):
        df = pl.DataFrame(
            {
                "canonical_id": ["af-1"],
                "position": ["F"],
                "minutes_total": [0],
                "reliable": [False],
                "normalized_shots_per90": [0.0],
                "normalized_sot_per90": [0.0],
                "normalized_goals_per90": [0.0],
                "fouls_committed_per90": [0.0],
            }
        )
        out = project_starters(df)
        assert out[0].p_anytime_scorer == 0.0

    def test_name_overrides_applied(self):
        df = _lineup_form(["af-42"])
        out = project_starters(df, name_overrides={"af-42": "Vinicius Jr"})
        assert out[0].name == "Vinicius Jr"

    def test_falls_back_to_unnormalized_columns(self):
        """If normalized_* missing, falls back to raw per-90."""
        df = pl.DataFrame(
            {
                "canonical_id": ["af-1"],
                "position": ["F"],
                "minutes_total": [1000],
                "reliable": [True],
                "shots_per90": [2.5],
                "sot_per90": [1.0],
                "goals_per90": [0.4],
                "fouls_committed_per90": [1.5],
            }
        )
        out = project_starters(df)
        assert out[0].lambda_shots == pytest.approx(2.5)

    def test_empty_input_yields_empty_list(self):
        df = pl.DataFrame(
            {"canonical_id": []},
            schema={"canonical_id": pl.Utf8},
        )
        assert project_starters(df) == []


# ─────────────────────────────────────────────────────────────────
# MatchPredictionRunner
# ─────────────────────────────────────────────────────────────────


def _build_inputs(team_id: int, name: str, fifa: str, start_pid: int) -> TeamMatchInputs:
    lineup = _build_lineup(team_id, start_pid=start_pid)
    canonical_ids = [f"af-{p.api_football_id}" for p in lineup.players]
    return TeamMatchInputs(
        team_id=team_id,
        team_name=name,
        fifa_code=fifa,
        baseline_row=_team_baseline(team_id),
        lineup=lineup,
        lineup_form=_lineup_form(canonical_ids),
    )


class TestMatchPredictionRunner:
    def test_runner_produces_full_prediction(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        home = _build_inputs(6, "Brazil", "BRA", start_pid=1)
        away = _build_inputs(26, "Argentina", "ARG", start_pid=100)

        pred = runner.predict(
            match_id="wc2026_test_001",
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 15, 18, 0, 0),
            home=home,
            away=away,
            predicted_at=datetime(2026, 5, 7, 10, 0, 0),
        )

        assert pred.match_id == "wc2026_test_001"
        assert len(pred.home.starters) == 11
        assert len(pred.away.starters) == 11
        assert pred.home.team_name == "Brazil"
        assert pred.away.team_name == "Argentina"
        # 1X2 must sum to 1.
        total = pred.goals.p_home_win + pred.goals.p_draw + pred.goals.p_away_win
        assert total == pytest.approx(1.0, abs=1e-6)
        # Independent Poisson has no rho.
        assert pred.rho is None

    def test_bivariate_runner_exposes_rho(self):
        runner = MatchPredictionRunner(BivariatePoissonModel(rho=0.04))
        home = _build_inputs(6, "Brazil", "BRA", 1)
        away = _build_inputs(26, "Argentina", "ARG", 100)
        pred = runner.predict(
            match_id="wc2026_test_002",
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 15, 18, 0, 0),
            home=home,
            away=away,
        )
        assert pred.rho == 0.04

    def test_unconfirmed_lineup_warns(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        home = _build_inputs(6, "Brazil", "BRA", 1)  # heuristic
        away = _build_inputs(26, "Argentina", "ARG", 100)  # heuristic
        pred = runner.predict(
            match_id="x",
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 15, 18, 0, 0),
            home=home,
            away=away,
        )
        # Both sides unconfirmed -> 2 warnings.
        assert len([w for w in pred.warnings if "lineup confidence" in w]) == 2

    def test_unreliable_starters_warned(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        home = _build_inputs(6, "Brazil", "BRA", 1)
        # Flip several home starters to unreliable.
        canonical_ids = [f"af-{p.api_football_id}" for p in home.lineup.players]
        home_form_unreliable = home.lineup_form.with_columns(
            pl.Series("reliable", [True] * 8 + [False] * 3)
        )
        home_unreliable = TeamMatchInputs(
            team_id=home.team_id,
            team_name=home.team_name,
            fifa_code=home.fifa_code,
            baseline_row=home.baseline_row,
            lineup=home.lineup,
            lineup_form=home_form_unreliable,
        )
        away = _build_inputs(26, "Argentina", "ARG", 100)
        pred = runner.predict(
            match_id="x",
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 15, 18, 0, 0),
            home=home_unreliable,
            away=away,
        )
        assert any("starters lack reliable" in w for w in pred.warnings)

    def test_player_names_from_lineup_propagate(self):
        runner = MatchPredictionRunner(IndependentPoissonModel())
        home = _build_inputs(6, "Brazil", "BRA", 1)
        away = _build_inputs(26, "Argentina", "ARG", 100)
        pred = runner.predict(
            match_id="x",
            tournament_slug="world_cup_2026",
            kickoff=datetime(2026, 6, 15, 18, 0, 0),
            home=home,
            away=away,
        )
        # First home starter is Player(api_football_id=1, name='GK_1')
        assert pred.home.starters[0].name == "GK_1"


# ─────────────────────────────────────────────────────────────────
# Output emitters
# ─────────────────────────────────────────────────────────────────


def _make_prediction():
    runner = MatchPredictionRunner(BivariatePoissonModel(rho=0.04))
    home = _build_inputs(6, "Brazil", "BRA", 1)
    away = _build_inputs(26, "Argentina", "ARG", 100)
    return runner.predict(
        match_id="wc2026_grpA_001",
        tournament_slug="world_cup_2026",
        kickoff=datetime(2026, 6, 15, 18, 0, 0),
        home=home,
        away=away,
        predicted_at=datetime(2026, 5, 7, 10, 0, 0),
    )


class TestJsonEmitter:
    def test_json_dict_has_expected_top_level(self):
        d = to_json_dict(_make_prediction())
        for key in (
            "schema_version",
            "model_version",
            "match_id",
            "tournament_slug",
            "kickoff",
            "predicted_at",
            "home",
            "away",
            "match_markets",
            "diagnostics",
        ):
            assert key in d

    def test_schema_version_locked(self):
        d = to_json_dict(_make_prediction())
        assert d["schema_version"] == JSON_SCHEMA_VERSION
        assert d["model_version"] == MODEL_VERSION

    def test_starters_serialized_with_projections(self):
        d = to_json_dict(_make_prediction())
        starters = d["home"]["lineup"]["starters"]
        assert len(starters) == 11
        for s in starters:
            for col in (
                "canonical_id",
                "name",
                "position",
                "lambda_shots",
                "p_anytime_scorer",
            ):
                assert col in s

    def test_match_markets_present(self):
        d = to_json_dict(_make_prediction())
        m = d["match_markets"]
        assert m["model"] == "bivariate_poisson"
        # Sum of 1X2 (already renormalized).
        assert m["p_home_win"] + m["p_draw"] + m["p_away_win"] == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= m["p_btts"] <= 1.0

    def test_write_json_produces_valid_file(self, tmp_path):
        path = tmp_path / "pred.json"
        write_json(_make_prediction(), path)
        loaded = json.loads(path.read_text())
        assert loaded["match_id"] == "wc2026_grpA_001"


class TestMarkdownEmitter:
    def test_markdown_contains_team_names(self):
        md = to_markdown(_make_prediction())
        assert "Brazil" in md
        assert "Argentina" in md

    def test_markdown_contains_match_markets_section(self):
        md = to_markdown(_make_prediction())
        assert "## Match markets" in md
        assert "BTTS" in md
        assert "Over 2.5" in md

    def test_markdown_contains_starters_per_side(self):
        md = to_markdown(_make_prediction())
        assert "## Home starters" in md
        assert "## Away starters" in md
        assert "P(scorer)" in md

    def test_markdown_includes_warnings_when_present(self):
        md = to_markdown(_make_prediction())
        # Both lineups are heuristic_qualifier -> 2 warnings expected.
        assert "## Warnings" in md

    def test_write_markdown_produces_file(self, tmp_path):
        path = tmp_path / "pred.md"
        write_markdown(_make_prediction(), path)
        assert path.exists()
        assert "Brazil" in path.read_text()


# ─────────────────────────────────────────────────────────────────
# synthesize_lineup_form_for
# ─────────────────────────────────────────────────────────────────


class TestSynthesizeLineupFormFor:
    def test_filters_to_lineup_members(self):
        lineup = _build_lineup(6, start_pid=1)
        # Form has the 11 starters PLUS some bench players we should drop.
        all_ids = [f"af-{i}" for i in range(1, 25)]
        raw = _lineup_form(all_ids)
        out = synthesize_lineup_form_for(lineup, raw)
        assert out.height == 11
        assert set(out["canonical_id"].to_list()) == {f"af-{i}" for i in range(1, 12)}

    def test_preserves_lineup_order(self):
        lineup = _build_lineup(6, start_pid=1)
        all_ids = [f"af-{i}" for i in range(1, 25)]
        raw = _lineup_form(all_ids)
        out = synthesize_lineup_form_for(lineup, raw)
        # Lineup is GK, DEF×4, MID×3, FWD×3 with pids 1-11.
        assert out["canonical_id"].to_list() == [f"af-{i}" for i in range(1, 12)]

    def test_missing_starter_raises(self):
        lineup = _build_lineup(6, start_pid=1)
        # Form only has 10 of the 11 starters.
        partial_ids = [f"af-{i}" for i in range(1, 11)]
        raw = _lineup_form(partial_ids)
        with pytest.raises(ValueError, match="missing from raw_form"):
            synthesize_lineup_form_for(lineup, raw)

    def test_missing_canonical_id_column_raises(self):
        lineup = _build_lineup(6, start_pid=1)
        bad = pl.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="canonical_id"):
            synthesize_lineup_form_for(lineup, bad)
