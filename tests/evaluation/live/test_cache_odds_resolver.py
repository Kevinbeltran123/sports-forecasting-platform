"""Tests for SportmonksCache-backed OddsResolver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bip.evaluation.live.cache_odds_resolver import (
    make_cache_odds_resolver,
    resolve_odd_from_fixture,
)
from bip.sports.football.sportmonks.cache import SportmonksCache
from bip.sports.football.sportmonks.schemas import Fixture, Odd
from bip.sports.football.sportmonks.types import MarketID


def _fixture_with_odds(odds_data: list[dict]) -> Fixture:
    """Build a minimal Fixture carrying a list of odds."""
    odds = [Odd(**d) for d in odds_data]
    return Fixture(
        id=12345,
        sport_id=1,
        name="Home FC vs Away FC",
        starting_at=datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc),
        league_id=99,
        season_id=999,
        venue_id=None,
        state_id=22,
        odds=odds,
    )


def _odd(
    *, market_id: int, label: str, value: str,
    bookmaker_id: int = 2, total: str | None = None,
    suspended: bool = False, stopped: bool = False,
) -> dict:
    return {
        "id": hash((market_id, label, bookmaker_id, value)) & 0xFFFFFFFF,
        "fixture_id": 12345,
        "market_id": market_id,
        "bookmaker_id": bookmaker_id,
        "label": label,
        "value": value,
        "total": total,
        "suspended": suspended,
        "stopped": stopped,
    }


# ── pure resolver function ──────────────────────────────────────────────────


class TestResolveOddFromFixture:
    def test_matches_fulltime_home(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.10"),
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Draw", value="3.40"),
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Away", value="3.20"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) == pytest.approx(2.10)

    def test_matches_ou_25_over(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.MATCH_GOALS, label="Over",
                 value="1.85", total="2.5"),
            _odd(market_id=MarketID.MATCH_GOALS, label="Under",
                 value="1.95", total="2.5"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="ou_2_5", selection="over", bookmaker_id=2,
        ) == pytest.approx(1.85)

    def test_returns_none_for_wrong_bookmaker(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="2.10", bookmaker_id=99),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None

    def test_returns_none_for_wrong_selection(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="2.10"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="away",
            bookmaker_id=2,
        ) is None

    def test_skips_suspended(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="2.10", suspended=True),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None

    def test_skips_stopped(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="2.10", stopped=True),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None

    def test_returns_none_for_unknown_market(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="2.10"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="market_we_do_not_support",
            selection="home", bookmaker_id=2,
        ) is None

    def test_returns_none_when_no_odds(self):
        fx = Fixture(
            id=12345, sport_id=1, name="A vs B",
            starting_at=datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc),
            league_id=99, season_id=999, venue_id=None, state_id=22,
            odds=None,
        )
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None

    def test_skips_odd_with_value_below_1(self):
        # Some bookmakers emit weird artifacts; only valid decimal odds count.
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="0.50"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None

    def test_handles_unparseable_value(self):
        fx = _fixture_with_odds([
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home",
                 value="not-a-number"),
        ])
        assert resolve_odd_from_fixture(
            fx, market="fulltime_result", selection="home",
            bookmaker_id=2,
        ) is None


# ── cache-backed resolver ───────────────────────────────────────────────────


class TestMakeCacheOddsResolver:
    @pytest.mark.asyncio
    async def test_resolves_via_latest_snapshot(self, tmp_path: Path):
        cache = SportmonksCache(root=tmp_path)
        # Save a snapshot with a FULLTIME_RESULT/Home odd at 2.15
        payload = {
            "id": 12345,
            "sport_id": 1,
            "name": "A vs B",
            "starting_at": "2026-05-10T20:00:00+00:00",
            "league_id": 99,
            "season_id": 999,
            "venue_id": None,
            "state_id": 22,
            "odds": [
                {
                    "id": 1, "fixture_id": 12345,
                    "market_id": MarketID.FULLTIME_RESULT,
                    "bookmaker_id": 2, "label": "Home", "value": "2.15",
                    "suspended": False, "stopped": False,
                },
            ],
        }
        cache.save_snapshot(12345, {"data": payload})

        resolver = make_cache_odds_resolver(cache)
        odd = await resolver(12345, "fulltime_result", "home", 2)
        assert odd == pytest.approx(2.15)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_snapshot(self, tmp_path: Path):
        cache = SportmonksCache(root=tmp_path)
        resolver = make_cache_odds_resolver(cache)
        odd = await resolver(99999, "fulltime_result", "home", 2)
        assert odd is None

    @pytest.mark.asyncio
    async def test_load_failure_returns_none(self, tmp_path: Path):
        # Write a corrupt snapshot file
        snap_dir = tmp_path / "snapshots" / "12345"
        snap_dir.mkdir(parents=True)
        (snap_dir / "20260510T200000Z.json").write_text("not-valid-json")

        cache = SportmonksCache(root=tmp_path)
        resolver = make_cache_odds_resolver(cache)
        odd = await resolver(12345, "fulltime_result", "home", 2)
        assert odd is None
