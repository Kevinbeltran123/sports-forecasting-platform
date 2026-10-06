"""FixtureHydrator tests — Sprint 3 Ola B.

Mock ApiFootballClient via AsyncMock. Verifies:
  - All 5 leagues + WC2026 queried in one hydrate call
  - Normalized dicts match the Orchestrator contract
  - API errors per league are non-fatal
  - Malformed fixture records are skipped + counted
  - features=None in Layer-1; FeatureEngineerProtocol hook fills it
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from bip.models.ligas import LEAGUE_ID_MAP
from bip.pipeline.fixture_hydrator import (
    WC2026_COMPETITION_LABEL,
    WC2026_LEAGUE_ID,
    FixtureHydrator,
    HydratorSummary,
)


def _api_fixture(
    *,
    api_id: int = 1001,
    kickoff_iso: str = "2026-06-01T19:00:00+00:00",
    home: str = "Arsenal",
    away: str = "Chelsea",
) -> dict:
    return {
        "fixture": {"id": api_id, "date": kickoff_iso},
        "teams": {
            "home": {"id": 42, "name": home},
            "away": {"id": 47, "name": away},
        },
    }


def _api_response(fixtures: list[dict]) -> dict:
    return {"response": fixtures}


@pytest.mark.asyncio
class TestHydrateForDate:
    async def test_queries_all_5_leagues_plus_wc(self):
        client = AsyncMock()
        client.get_fixtures.return_value = _api_response([])
        h = FixtureHydrator(api_client=client)
        out, summary = await h.hydrate_for_date("2026-06-01")
        # 5 leagues + WC2026 = 6 calls
        assert client.get_fixtures.await_count == 6
        called_ids = sorted(c.args[0] for c in client.get_fixtures.await_args_list)
        expected = sorted(list(LEAGUE_ID_MAP.values()) + [WC2026_LEAGUE_ID])
        assert called_ids == expected
        assert summary.n_leagues_queried == 6
        assert summary.n_fixtures_total == 0
        assert out == []

    async def test_normalized_dict_shape(self):
        client = AsyncMock()
        # Only PL returns one fixture
        async def per_league(league_id, date):
            if league_id == LEAGUE_ID_MAP["PL"]:
                return _api_response([_api_fixture(api_id=42)])
            return _api_response([])
        client.get_fixtures.side_effect = per_league
        h = FixtureHydrator(api_client=client)
        out, summary = await h.hydrate_for_date("2026-06-01")

        assert summary.n_fixtures_total == 1
        fix = out[0]
        assert fix["fixture_id"] == "42"
        assert fix["match_id"] == "42"
        assert fix["competition"] == "PL"
        assert fix["home_team"] == "Arsenal"
        assert fix["away_team"] == "Chelsea"
        assert isinstance(fix["match_datetime"], datetime)
        assert fix["match_datetime"].tzinfo is not None
        assert fix["features"] is None  # Layer-1

    async def test_wc2026_competition_label(self):
        client = AsyncMock()
        async def per_league(league_id, date):
            if league_id == WC2026_LEAGUE_ID:
                return _api_response([_api_fixture(api_id=99)])
            return _api_response([])
        client.get_fixtures.side_effect = per_league
        h = FixtureHydrator(api_client=client)
        out, _ = await h.hydrate_for_date("2026-06-11")
        wc = [f for f in out if f["competition"] == WC2026_COMPETITION_LABEL]
        assert len(wc) == 1

    async def test_disable_wc_only_queries_leagues(self):
        client = AsyncMock()
        client.get_fixtures.return_value = _api_response([])
        h = FixtureHydrator(api_client=client, include_wc2026=False)
        _, summary = await h.hydrate_for_date("2026-06-01")
        assert summary.n_leagues_queried == 5

    async def test_api_error_per_league_is_non_fatal(self):
        client = AsyncMock()
        async def per_league(league_id, date):
            if league_id == LEAGUE_ID_MAP["PL"]:
                raise RuntimeError("API-Football 500")
            return _api_response([_api_fixture(api_id=league_id)])
        client.get_fixtures.side_effect = per_league
        h = FixtureHydrator(api_client=client)
        out, summary = await h.hydrate_for_date("2026-06-01")
        # PL failed, other 4 + WC succeeded
        assert summary.n_fixtures_total == 5
        assert summary.skipped_reasons.get("fetch_error") == 1

    async def test_malformed_fixture_skipped(self):
        client = AsyncMock()
        async def per_league(league_id, date):
            if league_id == LEAGUE_ID_MAP["PL"]:
                return _api_response([
                    _api_fixture(api_id=1),                      # good
                    {"fixture": {}, "teams": {}},                # missing id
                    {"fixture": {"id": 2, "date": "garbage"}},   # bad date
                ])
            return _api_response([])
        client.get_fixtures.side_effect = per_league
        h = FixtureHydrator(api_client=client)
        out, summary = await h.hydrate_for_date("2026-06-01")
        assert summary.n_fixtures_total == 1
        assert summary.n_skipped == 2
        assert summary.skipped_reasons.get("normalize_failed") == 2

    async def test_feature_engineer_hook_invoked(self):
        client = AsyncMock()
        async def per_league(league_id, date):
            if league_id == LEAGUE_ID_MAP["PL"]:
                return _api_response([_api_fixture(api_id=42)])
            return _api_response([])
        client.get_fixtures.side_effect = per_league

        engineer = MagicMock()
        engineer.build_for_fixture.return_value = np.array([1.0, 2.0, 3.0])

        h = FixtureHydrator(api_client=client, feature_engineer=engineer)
        out, _ = await h.hydrate_for_date("2026-06-01")
        assert engineer.build_for_fixture.call_count == 1
        feats = out[0]["features"]
        assert feats is not None
        assert list(feats) == [1.0, 2.0, 3.0]

    async def test_feature_engineer_exception_does_not_drop_fixture(self):
        client = AsyncMock()
        async def per_league(league_id, date):
            if league_id == LEAGUE_ID_MAP["PL"]:
                return _api_response([_api_fixture(api_id=42)])
            return _api_response([])
        client.get_fixtures.side_effect = per_league

        engineer = MagicMock()
        engineer.build_for_fixture.side_effect = RuntimeError("feature crash")

        h = FixtureHydrator(api_client=client, feature_engineer=engineer)
        out, summary = await h.hydrate_for_date("2026-06-01")
        # Fixture still emitted, features stays None
        assert summary.n_fixtures_total == 1
        assert out[0]["features"] is None
        assert summary.skipped_reasons.get("feature_error") == 1


@pytest.mark.asyncio
class TestCustomLeagueMap:
    async def test_custom_leagues_param(self):
        client = AsyncMock()
        client.get_fixtures.return_value = _api_response([])
        h = FixtureHydrator(
            api_client=client,
            leagues={"Eredivisie": 88},
            include_wc2026=False,
        )
        _, summary = await h.hydrate_for_date("2026-06-01")
        assert summary.n_leagues_queried == 1
        assert client.get_fixtures.await_count == 1
        assert client.get_fixtures.await_args.args[0] == 88
