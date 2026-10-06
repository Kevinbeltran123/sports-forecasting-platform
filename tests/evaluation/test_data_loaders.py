"""Phase 1.3/1.4 — qualifying_loader, club_form_loader, identity_matcher, cache.

Loaders are tested with pytest-httpx mocked API-Football responses so no
network or .env is required. Each test verifies aggregation correctness
against payloads that mirror the real API shape (see
https://www.api-football.com/documentation-v3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from bip.evaluation.tournaments.data import cache as cache_module
from bip.evaluation.tournaments.data.club_form_loader import ClubFormLoader
from bip.evaluation.tournaments.data.identity_matcher import (
    from_canonical,
    is_api_football_canonical,
    to_canonical,
)
from bip.evaluation.tournaments.data.qualifying_loader import QualifyingLoader
from bip.sports.football.client import ApiFootballClient


# ─────────────────────────────────────────────────────────────────
# identity_matcher
# ─────────────────────────────────────────────────────────────────


class TestIdentityMatcher:
    def test_to_canonical_simple(self):
        assert to_canonical(42) == "af-42"

    def test_to_canonical_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            to_canonical(0)
        with pytest.raises(ValueError):
            to_canonical(-1)

    def test_from_canonical_round_trip(self):
        for n in (1, 42, 99999):
            assert from_canonical(to_canonical(n)) == n

    def test_from_canonical_rejects_other_prefixes(self):
        with pytest.raises(ValueError, match="not in"):
            from_canonical("fb-12345")

    def test_is_api_football_canonical(self):
        assert is_api_football_canonical("af-42") is True
        assert is_api_football_canonical("fb-42") is False
        assert is_api_football_canonical("42") is False


# ─────────────────────────────────────────────────────────────────
# cache
# ─────────────────────────────────────────────────────────────────


class TestCache:
    def test_round_trip(self, tmp_path: Path):
        import polars as pl

        df = pl.DataFrame({"a": [1, 2, 3]})
        path = cache_module.cache_path(
            "team_qualifier_baselines",
            "world_cup_2026",
            "6",
            root=tmp_path,
        )
        assert cache_module.read(path) is None  # missing -> None
        cache_module.write(df, path)
        out = cache_module.read(path)
        assert out is not None
        assert out.equals(df)

    def test_path_layout(self, tmp_path: Path):
        path = cache_module.cache_path(
            "player_recent_form",
            "world_cup_2026",
            "af-42",
            root=tmp_path,
        )
        assert path == tmp_path / "player_recent_form" / "world_cup_2026" / "af-42.parquet"


# ─────────────────────────────────────────────────────────────────
# qualifying_loader
# ─────────────────────────────────────────────────────────────────


def _team_stats_payload(team_id: int, played: int, gf: int, ga: int) -> dict:
    """Mock payload matching /teams/statistics shape."""
    return {
        "response": {
            "team": {"id": team_id},
            "fixtures": {"played": {"total": played}},
            "goals": {
                "for": {"total": {"total": gf}},
                "against": {"total": {"total": ga}},
            },
        }
    }


def _fixtures_payload(team_id: int, fixture_ids: list[int]) -> dict:
    """Mock /fixtures?league&season payload — all FT fixtures involving team."""
    return {
        "response": [
            {
                "fixture": {"id": fid, "status": {"short": "FT"}},
                "teams": {
                    "home": {"id": team_id},
                    "away": {"id": 999},  # opponent
                },
            }
            for fid in fixture_ids
        ]
    }


def _statistics_payload(team_id: int, corners: int, shots: int, sot: int, fouls: int) -> dict:
    """Mock /fixtures/statistics — one team's stats for one fixture."""
    return {
        "response": [
            {
                "team": {"id": team_id},
                "statistics": [
                    {"type": "Corner Kicks", "value": corners},
                    {"type": "Total Shots", "value": shots},
                    {"type": "Shots on Goal", "value": sot},
                    {"type": "Fouls", "value": fouls},
                    {"type": "Yellow Cards", "value": 1},
                    {"type": "Red Cards", "value": 0},
                ],
            }
        ]
    }


class TestQualifyingLoader:
    async def test_aggregates_per_90_rates_from_team_and_fixture_stats(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        # Redirect cache root to tmp_path so we don't pollute the workspace.
        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)

        team_id = 6
        # Order matters — httpx_mock is a queue:
        # 1) /teams/statistics
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/teams/statistics?league=13&season=2024&team=6",
            json=_team_stats_payload(team_id, played=2, gf=4, ga=2),
        )
        # 2) /fixtures?league&season
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/fixtures?league=13&season=2024",
            json=_fixtures_payload(team_id, fixture_ids=[100, 101]),
        )
        # 3) /fixtures/statistics for fixture 100
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/fixtures/statistics?fixture=100",
            json=_statistics_payload(team_id, corners=6, shots=12, sot=5, fouls=10),
        )
        # 4) /fixtures/statistics for fixture 101
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/fixtures/statistics?fixture=101",
            json=_statistics_payload(team_id, corners=4, shots=8, sot=3, fouls=14),
        )

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = QualifyingLoader(client=client)
            df = await loader.load_team_baseline(
                team_id=team_id, league_id=13, season=2024, tournament_slug="wc2026_test"
            )

        # 2 matches × 90min = 180min. gf=4 -> 4/180*90 = 2.0/90.
        assert df["matches_played"][0] == 2
        assert df["minutes_total"][0] == 180
        assert df["gf_per90"][0] == pytest.approx(2.0)
        assert df["ga_per90"][0] == pytest.approx(1.0)
        # corners = 6+4 = 10 over 180min -> 5.0 per90
        assert df["corners_for_per90"][0] == pytest.approx(5.0)
        # shots 12+8 = 20 -> 10.0 per90
        assert df["shots_for_per90"][0] == pytest.approx(10.0)
        # sot 5+3 = 8 -> 4.0 per90
        assert df["sot_for_per90"][0] == pytest.approx(4.0)
        # fouls 10+14 = 24 -> 12.0 per90
        assert df["fouls_for_per90"][0] == pytest.approx(12.0)

    async def test_cache_hit_skips_api(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        import polars as pl

        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        # Pre-populate cache.
        canned = pl.DataFrame({"team_id": [6], "matches_played": [99]})
        path = cache_module.cache_path(
            "team_qualifier_baselines", "wc2026_test", "6", root=tmp_path
        )
        cache_module.write(canned, path)

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = QualifyingLoader(client=client)
            df = await loader.load_team_baseline(
                team_id=6, league_id=13, season=2024, tournament_slug="wc2026_test"
            )

        assert df["matches_played"][0] == 99
        assert httpx_mock.get_requests() == []  # zero API calls

    async def test_skips_non_ft_fixtures(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        team_id = 6

        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/teams/statistics?league=13&season=2024&team=6",
            json=_team_stats_payload(team_id, played=1, gf=1, ga=1),
        )
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/fixtures?league=13&season=2024",
            json={
                "response": [
                    {
                        "fixture": {"id": 100, "status": {"short": "FT"}},
                        "teams": {"home": {"id": team_id}, "away": {"id": 999}},
                    },
                    {
                        "fixture": {"id": 200, "status": {"short": "NS"}},  # not started
                        "teams": {"home": {"id": team_id}, "away": {"id": 888}},
                    },
                ]
            },
        )
        # Only fixture 100 should be queried for statistics — 200 is skipped.
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/fixtures/statistics?fixture=100",
            json=_statistics_payload(team_id, corners=5, shots=10, sot=4, fouls=8),
        )

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = QualifyingLoader(client=client)
            df = await loader.load_team_baseline(
                team_id=team_id, league_id=13, season=2024, tournament_slug="wc2026_test"
            )

        assert df["matches_played"][0] == 1


# ─────────────────────────────────────────────────────────────────
# club_form_loader
# ─────────────────────────────────────────────────────────────────


def _player_payload(
    player_id: int,
    minutes: int,
    shots: int,
    sot: int,
    goals: int,
    fouls_committed: int,
    fouls_drawn: int,
) -> dict:
    """Mock /players response."""
    return {
        "response": [
            {
                "player": {"id": player_id, "name": "Test Player"},
                "statistics": [
                    {
                        "team": {"id": 50},
                        "league": {"id": 39, "name": "Premier League"},
                        "games": {
                            "appearences": 10,
                            "minutes": minutes,
                            "position": "Attacker",
                        },
                        "shots": {"total": shots, "on": sot},
                        "goals": {"total": goals, "assists": 4},
                        "passes": {"key": 25},
                        "fouls": {
                            "committed": fouls_committed,
                            "drawn": fouls_drawn,
                        },
                        "cards": {"yellow": 2, "red": 0},
                    }
                ],
            }
        ]
    }


class TestClubFormLoader:
    async def test_extracts_per_90_rates(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/players?id=42&season=2025",
            json=_player_payload(
                player_id=42,
                minutes=900,
                shots=30,
                sot=15,
                goals=5,
                fouls_committed=8,
                fouls_drawn=12,
            ),
        )

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = ClubFormLoader(client=client)
            df = await loader.load_player_form(
                player_id=42, season=2025, tournament_slug="wc2026_test"
            )

        assert df["canonical_id"][0] == "af-42"
        assert df["minutes_total"][0] == 900
        assert df["reliable"][0] is True  # 900 >= 600
        # shots 30 / 900min * 90 = 3.0
        assert df["shots_per90"][0] == pytest.approx(3.0)
        # sot 15/900*90 = 1.5
        assert df["sot_per90"][0] == pytest.approx(1.5)
        # goals 5/900*90 = 0.5
        assert df["goals_per90"][0] == pytest.approx(0.5)

    async def test_low_minutes_player_flagged_unreliable(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/players?id=43&season=2025",
            json=_player_payload(
                player_id=43, minutes=400, shots=10, sot=5, goals=1,
                fouls_committed=2, fouls_drawn=3,
            ),
        )

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = ClubFormLoader(client=client)
            df = await loader.load_player_form(
                player_id=43, season=2025, tournament_slug="wc2026_test"
            )

        assert df["reliable"][0] is False  # 400 < 600 threshold
        assert df["minutes_total"][0] == 400

    async def test_empty_response_returns_zero_frame(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        httpx_mock.add_response(
            url="https://v3.football.api-sports.io/players?id=44&season=2025",
            json={"response": []},
        )

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = ClubFormLoader(client=client)
            df = await loader.load_player_form(
                player_id=44, season=2025, tournament_slug="wc2026_test"
            )

        assert df["canonical_id"][0] == "af-44"
        assert df["reliable"][0] is False
        assert df["minutes_total"][0] == 0
        assert df["shots_per90"][0] == 0.0

    async def test_cache_hit_skips_api(
        self, httpx_mock: HTTPXMock, settings, tmp_path, monkeypatch
    ):
        import polars as pl

        monkeypatch.setattr(cache_module, "DEFAULT_CACHE_ROOT", tmp_path)
        canned = pl.DataFrame({"canonical_id": ["af-42"], "minutes_total": [12345]})
        path = cache_module.cache_path(
            "player_recent_form", "wc2026_test", "af-42", root=tmp_path
        )
        cache_module.write(canned, path)

        async with ApiFootballClient(api_key=settings.api_football_key) as client:
            loader = ClubFormLoader(client=client)
            df = await loader.load_player_form(
                player_id=42, season=2025, tournament_slug="wc2026_test"
            )

        assert df["minutes_total"][0] == 12345
        assert httpx_mock.get_requests() == []
