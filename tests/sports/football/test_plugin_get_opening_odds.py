"""FootballPlugin.get_opening_odds() — 1X2 dict for Betano (G-CODE-01)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


def _settings(monkeypatch, model_dir: Path):
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-af-key")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds-key")
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from bip.core.settings import Settings
    return Settings()


def _betano_match_winner_response() -> dict:
    return {
        "response": [
            {
                "bookmakers": [
                    {
                        "name": "Betano",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "1.85"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "4.20"},
                                ],
                            }
                        ],
                    }
                ]
            }
        ]
    }


class _FakeAsyncCM:
    """Async context manager that returns a stub client whose get_odds is mocked."""

    def __init__(self, mock_get_odds: AsyncMock) -> None:
        self._mock_get_odds = mock_get_odds

    async def __aenter__(self):
        class _Client:
            def __init__(self, get_odds):
                self.get_odds = get_odds

        return _Client(self._mock_get_odds)

    async def __aexit__(self, *args):
        return False


def _patch_client(monkeypatch, raw_response: dict) -> AsyncMock:
    """Patch FootballPlugin's ApiFootballClient with one whose get_odds returns raw_response."""
    mock_get_odds = AsyncMock(return_value=raw_response)

    def _factory(*args, **kwargs):  # noqa: ARG001
        return _FakeAsyncCM(mock_get_odds)

    monkeypatch.setattr(
        "bip.sports.football.plugin.ApiFootballClient", _factory
    )
    return mock_get_odds


class TestGetOpeningOdds:
    async def test_get_opening_odds_returns_1x2_dict_for_betano(
        self, tmp_model_dir, monkeypatch
    ):
        from bip.sports.football.plugin import FootballPlugin

        _patch_client(monkeypatch, _betano_match_winner_response())
        settings = _settings(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)

        odds = await plugin.get_opening_odds(999)
        assert odds == {"1": 1.85, "X": 3.40, "2": 4.20}

    async def test_get_opening_odds_returns_empty_dict_when_betano_missing(
        self, tmp_model_dir, monkeypatch
    ):
        from bip.sports.football.plugin import FootballPlugin

        raw = {
            "response": [
                {
                    "bookmakers": [
                        {
                            "name": "Pinnacle",
                            "bets": [
                                {
                                    "name": "Match Winner",
                                    "values": [
                                        {"value": "Home", "odd": "1.85"},
                                        {"value": "Draw", "odd": "3.40"},
                                        {"value": "Away", "odd": "4.20"},
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        _patch_client(monkeypatch, raw)
        settings = _settings(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)

        odds = await plugin.get_opening_odds(999)
        assert odds == {}

    async def test_get_opening_odds_returns_empty_dict_when_no_match_winner_bet(
        self, tmp_model_dir, monkeypatch
    ):
        from bip.sports.football.plugin import FootballPlugin

        raw = {
            "response": [
                {
                    "bookmakers": [
                        {
                            "name": "Betano",
                            "bets": [
                                {
                                    "name": "Both Teams to Score",
                                    "values": [
                                        {"value": "Yes", "odd": "1.75"},
                                        {"value": "No", "odd": "2.05"},
                                    ],
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        _patch_client(monkeypatch, raw)
        settings = _settings(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)

        odds = await plugin.get_opening_odds(999)
        assert odds == {}

    async def test_get_opening_odds_returns_empty_dict_on_empty_response(
        self, tmp_model_dir, monkeypatch
    ):
        from bip.sports.football.plugin import FootballPlugin

        _patch_client(monkeypatch, {"response": []})
        settings = _settings(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)

        odds = await plugin.get_opening_odds(999)
        assert odds == {}

    async def test_get_opening_odds_calls_client_with_fixture_id(
        self, tmp_model_dir, monkeypatch
    ):
        from bip.sports.football.plugin import FootballPlugin

        mock_get_odds = _patch_client(monkeypatch, _betano_match_winner_response())
        settings = _settings(monkeypatch, tmp_model_dir)
        plugin = FootballPlugin(settings=settings)

        await plugin.get_opening_odds(999)
        mock_get_odds.assert_awaited_once_with(fixture_id=999, bookmaker="Betano")
