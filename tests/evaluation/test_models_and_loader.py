"""Phase 1.2 — Pydantic models + YAML loader for tournament configs."""

from datetime import date

import pytest
from pydantic import ValidationError

from bip.core.errors import ConfigurationError
from bip.evaluation.tournaments.loader import (
    load_qualifying_leagues,
    load_tournament,
)
from bip.evaluation.tournaments.models import (
    Confederation,
    Group,
    Player,
    Position,
    QualifyingLeague,
    Squad,
    Team,
    Tournament,
    TournamentFormat,
)


class TestModels:
    def test_team_requires_three_letter_fifa_code(self):
        with pytest.raises(ValidationError):
            Team(
                api_football_id=6,
                name="Brazil",
                confederation=Confederation.CONMEBOL,
                fifa_code="BR",  # too short
            )

    def test_team_frozen(self):
        t = Team(
            api_football_id=6,
            name="Brazil",
            confederation=Confederation.CONMEBOL,
            fifa_code="BRA",
        )
        with pytest.raises(ValidationError):
            t.name = "Brasil"  # type: ignore[misc]

    def test_player_canonical_id_from_factory(self):
        p = Player.from_api_football(
            af_player_id=42,
            name="Vinicius Jr",
            position=Position.FORWARD,
            national_team_id=6,
        )
        assert p.canonical_id == "af-42"
        assert p.api_football_id == 42

    def test_player_canonical_id_pattern_enforced(self):
        with pytest.raises(ValidationError):
            Player(
                canonical_id="42",  # missing af- prefix
                api_football_id=42,
                name="x",
                position=Position.FORWARD,
                national_team_id=6,
            )

    def test_squad_size_validation(self):
        # Build minimal valid squad (18 players)
        players = [
            Player.from_api_football(i, f"Player {i}", Position.MIDFIELDER, 6)
            for i in range(18)
        ]
        s = Squad(
            team_id=6,
            tournament_slug="world_cup_2026",
            players=players,
            captured_at=date(2026, 5, 1).__str__(),  # type: ignore[arg-type]
        )
        assert len(s.players) == 18

    def test_squad_size_too_small_rejected(self):
        players = [
            Player.from_api_football(i, f"Player {i}", Position.MIDFIELDER, 6)
            for i in range(10)
        ]
        with pytest.raises(ValidationError, match="outside expected range"):
            Squad(
                team_id=6,
                tournament_slug="world_cup_2026",
                players=players,
                captured_at="2026-05-01T00:00:00",
            )

    def test_group_rejects_duplicates(self):
        with pytest.raises(ValidationError, match="duplicate"):
            Group(group_id="A", team_ids=[1, 2, 2, 3])

    def test_group_rejects_wrong_size(self):
        with pytest.raises(ValidationError, match="3 or 4"):
            Group(group_id="A", team_ids=[1, 2])

    def test_tournament_end_after_start(self):
        with pytest.raises(ValidationError, match="before start_date"):
            Tournament(
                slug="bad",
                name="Bad",
                format=TournamentFormat.GROUPS_KNOCKOUT,
                season=2024,
                start_date=date(2024, 6, 20),
                end_date=date(2024, 6, 10),  # before start
                participants_count=16,
            )


class TestLoader:
    def test_load_world_cup_2026(self):
        t = load_tournament("world_cup_2026")
        assert t.slug == "world_cup_2026"
        assert t.participants_count == 48
        assert t.format == TournamentFormat.GROUPS_KNOCKOUT
        assert t.best_third_placed_count == 8

    def test_load_world_cup_2022(self):
        t = load_tournament("world_cup_2022")
        assert t.participants_count == 32
        assert t.best_third_placed_count == 0

    def test_load_euro_2024(self):
        t = load_tournament("euro_2024")
        assert t.participants_count == 24
        assert t.best_third_placed_count == 4

    def test_load_copa_america_2024(self):
        t = load_tournament("copa_america_2024")
        assert t.participants_count == 16

    def test_unknown_tournament_raises(self):
        with pytest.raises(ConfigurationError, match="not found"):
            load_tournament("nonexistent_tournament_xyz")

    def test_load_qualifying_leagues_returns_six_confederations(self):
        leagues = load_qualifying_leagues()
        assert len(leagues) == 6
        confeds = {league.confederation for league in leagues}
        assert confeds == {
            Confederation.UEFA,
            Confederation.CONMEBOL,
            Confederation.AFC,
            Confederation.CAF,
            Confederation.CONCACAF,
            Confederation.OFC,
        }

    def test_qualifying_leagues_are_frozen(self):
        leagues = load_qualifying_leagues()
        with pytest.raises(ValidationError):
            leagues[0].api_football_league_id = 999  # type: ignore[misc]

    def test_qualifying_league_field_types(self):
        leagues = load_qualifying_leagues()
        for league in leagues:
            assert isinstance(league, QualifyingLeague)
            assert isinstance(league.api_football_league_id, int)
            assert 2018 <= league.season <= 2030
