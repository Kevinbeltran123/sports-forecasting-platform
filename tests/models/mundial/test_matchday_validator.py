"""Tests for bip.models.mundial.matchday_validator.

Sprint 0 Ola C. All tests OFFLINE — ApiFootballClient is mocked via the
ApiFootballClientProtocol. No network calls.

Coverage:
  - Lineup empty → returns valid=True (wait, don't block)
  - Lineup confirmed + no injuries → valid=True
  - Lineup confirmed + critical injury on starter (by id) → valid=False
  - Lineup confirmed + critical injury on starter (by name) → valid=False
  - Lineup confirmed + injury on non-starter → valid=True
  - Lineup fetch error → valid=True (non-blocking)
  - Injuries fetch error → valid=True (non-blocking)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from bip.models.base import PredictionRecord
from bip.models.mundial.matchday_validator import (
    MatchdayValidator,
    _injured_players,
    _starters_from_lineup,
)


def _make_prediction() -> PredictionRecord:
    return PredictionRecord(
        source="mundial",
        fixture_id="wc2026_grp_000",
        competition="WC2026",
        home_team="Mexico",
        away_team="South Africa",
        match_datetime=datetime(2026, 6, 11, 0, 0, 0, tzinfo=UTC),
        market="1x2",
        selection="home",
        p_model=0.42,
        model_version="lock-1589177",
    )


def _lineup_response(starter_ids: list[int]) -> dict:
    return {
        "response": [
            {
                "team": {"id": 1},
                "startXI": [
                    {"player": {"id": pid, "name": f"Player {pid}"}} for pid in starter_ids
                ],
            }
        ]
    }


def _injuries_response(items: list[dict]) -> dict:
    return {"response": [{"player": item} for item in items]}


class TestHelpers:
    def test_starters_from_lineup_extracts_players(self):
        lineup = _lineup_response([10, 11, 12])
        starters = _starters_from_lineup(lineup)
        assert {s["id"] for s in starters} == {10, 11, 12}

    def test_starters_from_lineup_handles_empty(self):
        assert _starters_from_lineup({"response": []}) == []
        assert _starters_from_lineup({}) == []

    def test_injured_players_filters_critical_types(self):
        resp = _injuries_response(
            [
                {"id": 10, "name": "Hurt Starter", "type": "Out"},
                {"id": 20, "name": "Healthy", "type": "Available"},
                {"id": 30, "name": "Doubtful Guy", "type": "Doubtful"},
                {"id": 40, "name": "Missing", "type": "Missing Fixture"},
            ]
        )
        injured = _injured_players(resp)
        assert {p["id"] for p in injured} == {10, 30, 40}


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.get_lineups = AsyncMock()
    client.get_injuries = AsyncMock()
    return client


@pytest.mark.asyncio
class TestValidate:
    async def test_lineup_empty_returns_valid(self, mock_client):
        mock_client.get_lineups.return_value = {"response": []}
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is True
        assert result.reason is None
        assert "lineup_not_confirmed" in result.checks_performed

    async def test_lineup_confirmed_no_injuries_returns_valid(self, mock_client):
        mock_client.get_lineups.return_value = _lineup_response([10, 11, 12])
        mock_client.get_injuries.return_value = {"response": []}
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is True
        assert result.reason is None
        assert "lineup_confirmed" in result.checks_performed
        assert "injuries_fetched" in result.checks_performed

    async def test_critical_injury_on_starter_by_id_blocks(self, mock_client):
        mock_client.get_lineups.return_value = _lineup_response([10, 11, 12])
        mock_client.get_injuries.return_value = _injuries_response(
            [{"id": 10, "name": "Star Striker", "type": "Out"}]
        )
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is False
        assert result.reason is not None
        assert "Star Striker" in result.reason

    async def test_critical_injury_on_starter_by_name_blocks(self, mock_client):
        mock_client.get_lineups.return_value = {
            "response": [
                {"startXI": [{"player": {"name": "Lionel Messi"}}]}
            ]
        }
        mock_client.get_injuries.return_value = _injuries_response(
            [{"name": "Lionel Messi", "type": "Doubtful"}]
        )
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is False
        assert "Lionel Messi" in (result.reason or "")

    async def test_injury_on_non_starter_does_not_block(self, mock_client):
        mock_client.get_lineups.return_value = _lineup_response([10, 11, 12])
        mock_client.get_injuries.return_value = _injuries_response(
            [{"id": 99, "name": "Bench Warmer", "type": "Out"}]
        )
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is True
        assert result.reason is None

    async def test_lineup_fetch_error_is_non_blocking(self, mock_client):
        mock_client.get_lineups.side_effect = RuntimeError("API down")
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is True
        assert result.reason is None

    async def test_injuries_fetch_error_is_non_blocking(self, mock_client):
        mock_client.get_lineups.return_value = _lineup_response([10, 11, 12])
        mock_client.get_injuries.side_effect = RuntimeError("API timeout")
        validator = MatchdayValidator(mock_client)
        result = await validator.validate(_make_prediction(), api_fixture_id=999)
        assert result.valid is True
        assert result.reason is None
