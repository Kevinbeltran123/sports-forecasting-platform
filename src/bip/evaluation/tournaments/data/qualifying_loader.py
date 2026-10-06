"""Pull qualifying-stage stats per national team and aggregate to per-90 baselines.

Workflow::

    client = ApiFootballClient(api_key=settings.api_football_key)
    loader = QualifyingLoader(client=client)
    df = await loader.load_team_baseline(
        team_id=6, league_id=13, season=2024, tournament_slug="world_cup_2026"
    )

The loader hits two endpoints per team:
1. /teams/statistics — overall fixtures, goals_for/against, cards, penalties.
2. /fixtures?league&season + /fixtures/statistics per fixture — corners, shots,
   shots_on_target, fouls (NOT in /teams/statistics).

Aggregation is raw per-90 only at this layer. Opponent adjustment lives in
`baselines/team_rates.py` (Phase 2).

Caching: results land in data/cache/evaluation/team_qualifier_baselines/
<tournament_slug>/<team_id>.parquet. The loader is cache-first: if the
parquet exists it is returned directly without re-hitting the API.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl
import structlog

from bip.evaluation.tournaments.data.cache import cache_path, read, write
from bip.sports.football.client import ApiFootballClient

logger = structlog.get_logger(__name__)

CATEGORY = "team_qualifier_baselines"

# API-Football statistics types we care about (as returned by /fixtures/statistics).
# Each fixture's statistics array contains entries with these `type` strings.
STAT_TYPE_CORNERS = "Corner Kicks"
STAT_TYPE_SHOTS_TOTAL = "Total Shots"
STAT_TYPE_SHOTS_ON = "Shots on Goal"
STAT_TYPE_FOULS = "Fouls"
STAT_TYPE_YELLOWS = "Yellow Cards"
STAT_TYPE_REDS = "Red Cards"


class QualifyingLoader:
    """Async pull + aggregate of qualifier stats per team."""

    def __init__(self, client: ApiFootballClient) -> None:
        self._client = client

    async def load_team_baseline(
        self,
        team_id: int,
        league_id: int,
        season: int,
        tournament_slug: str,
        *,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return a single-row DataFrame of per-90 baseline rates.

        Columns: team_id, matches_played, minutes_total, gf_per90, ga_per90,
        corners_for_per90, corners_against_per90, shots_for_per90,
        shots_against_per90, sot_for_per90, sot_against_per90,
        fouls_for_per90, fouls_against_per90, yellows_per90, reds_per90.
        """
        path = cache_path(CATEGORY, tournament_slug, str(team_id))
        if not force_refresh:
            cached = read(path)
            if cached is not None:
                return cached

        team_stats = await self._client.get_team_statistics(
            league_id=league_id, season=season, team_id=team_id
        )
        ts_block = team_stats.get("response", {})

        fixtures_payload = await self._client.get_fixtures_by_season(
            league_id=league_id, season=season
        )
        team_fixtures = _filter_fixtures_for_team(fixtures_payload, team_id)

        # Aggregate per-fixture stats (corners, shots, SoT, fouls) — none of
        # which exist in /teams/statistics.
        per_fixture_stats: list[dict[str, Any]] = []
        for fixture_id in (f["fixture"]["id"] for f in team_fixtures):
            stats_resp = await self._client.get_statistics(fixture_id=fixture_id)
            row = _extract_team_fixture_stats(stats_resp, team_id=team_id)
            if row is not None:
                per_fixture_stats.append(row)

        df = _build_baseline_frame(
            team_id=team_id,
            team_stats_block=ts_block,
            per_fixture=per_fixture_stats,
        )
        write(df, path)
        return df


def _filter_fixtures_for_team(
    payload: dict[str, Any], team_id: int
) -> list[dict[str, Any]]:
    """Return only the fixtures where the given team played, status FT only."""
    out = []
    for fx in payload.get("response", []):
        teams = fx.get("teams", {})
        home_id = teams.get("home", {}).get("id")
        away_id = teams.get("away", {}).get("id")
        status = fx.get("fixture", {}).get("status", {}).get("short")
        if status == "FT" and team_id in (home_id, away_id):
            out.append(fx)
    return out


def _extract_team_fixture_stats(
    statistics_payload: dict[str, Any], team_id: int
) -> dict[str, Any] | None:
    """From /fixtures/statistics?fixture=X, pull the row for `team_id`.

    Returns dict with int values, or None if team not present or incomplete.
    The stat values from API-Football are strings sometimes ("65%") so we
    always coerce; for percentages we ignore them (they aren't counts we want).
    """
    response = statistics_payload.get("response", [])
    for team_stats in response:
        if team_stats.get("team", {}).get("id") != team_id:
            continue

        wanted = {
            STAT_TYPE_CORNERS: "corners_for",
            STAT_TYPE_SHOTS_TOTAL: "shots_for",
            STAT_TYPE_SHOTS_ON: "sot_for",
            STAT_TYPE_FOULS: "fouls_for",
            STAT_TYPE_YELLOWS: "yellows",
            STAT_TYPE_REDS: "reds",
        }

        out: dict[str, Any] = {key: 0 for key in wanted.values()}
        for entry in team_stats.get("statistics", []):
            stat_type = entry.get("type")
            if stat_type in wanted:
                out[wanted[stat_type]] = _coerce_count(entry.get("value"))
        return out
    return None


def _coerce_count(value: Any) -> int:
    """API-Football returns ints as ints, percentages as '65%', None for missing."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.endswith("%"):
            return 0
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _build_baseline_frame(
    team_id: int,
    team_stats_block: dict[str, Any],
    per_fixture: Iterable[dict[str, Any]],
) -> pl.DataFrame:
    """Aggregate per-90 rates from team_statistics + per-fixture rows."""
    fx_played = (
        team_stats_block.get("fixtures", {}).get("played", {}).get("total") or 0
    )
    minutes_total = max(fx_played * 90, 1)  # avoid /0 even on empty teams

    gf = team_stats_block.get("goals", {}).get("for", {}).get("total", {}).get("total", 0) or 0
    ga = team_stats_block.get("goals", {}).get("against", {}).get("total", {}).get("total", 0) or 0

    sum_corners_for = sum(r.get("corners_for", 0) for r in per_fixture)
    sum_shots_for = sum(r.get("shots_for", 0) for r in per_fixture)
    sum_sot_for = sum(r.get("sot_for", 0) for r in per_fixture)
    sum_fouls_for = sum(r.get("fouls_for", 0) for r in per_fixture)
    sum_yellows = sum(r.get("yellows", 0) for r in per_fixture)
    sum_reds = sum(r.get("reds", 0) for r in per_fixture)

    return pl.DataFrame(
        {
            "team_id": [team_id],
            "matches_played": [fx_played],
            "minutes_total": [minutes_total],
            "gf_per90": [gf / minutes_total * 90.0],
            "ga_per90": [ga / minutes_total * 90.0],
            "corners_for_per90": [sum_corners_for / minutes_total * 90.0],
            "shots_for_per90": [sum_shots_for / minutes_total * 90.0],
            "sot_for_per90": [sum_sot_for / minutes_total * 90.0],
            "fouls_for_per90": [sum_fouls_for / minutes_total * 90.0],
            "yellows_per90": [sum_yellows / minutes_total * 90.0],
            "reds_per90": [sum_reds / minutes_total * 90.0],
        }
    )
