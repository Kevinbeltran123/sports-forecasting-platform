"""Pull recent club-football form per player and compute per-90 rolling rates.

Workflow::

    loader = ClubFormLoader(client=client)
    df = await loader.load_player_form(
        player_id=42,
        season=2025,
        tournament_slug="world_cup_2026",
        last_n_matches=10,
    )

The loader hits /players/statistics for the season aggregate and produces
a single-row DataFrame of per-90 rates suitable for blending into a
match-level prediction (see Phase 2 modeling).

R-03 mitigation (SPIKE.md §5): the caller is responsible for filtering
players with <600 minutes — this loader returns zeros for them but tags
`minutes_total` so downstream can shrink to position-average.

Caching: data/cache/evaluation/player_recent_form/<tournament_slug>/
<canonical_id>.parquet. Cache-first.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import structlog

from bip.evaluation.tournaments.data.cache import cache_path, read, write
from bip.evaluation.tournaments.data.identity_matcher import to_canonical
from bip.sports.football.client import ApiFootballClient

logger = structlog.get_logger(__name__)

CATEGORY = "player_recent_form"

MIN_RELIABLE_MINUTES = 600  # R-03 threshold (SPIKE.md §5)


class ClubFormLoader:
    """Async pull + aggregate of player club-form rates."""

    def __init__(self, client: ApiFootballClient) -> None:
        self._client = client

    async def load_player_form(
        self,
        player_id: int,
        season: int,
        tournament_slug: str,
        *,
        team_id: int | None = None,
        force_refresh: bool = False,
    ) -> pl.DataFrame:
        """Return per-90 rates for one player from the season-aggregate endpoint.

        Columns: canonical_id, api_football_id, club_team_id, club_league_id,
        position, matches_played, minutes_total, reliable, shots_per90,
        sot_per90, goals_per90, assists_per90, key_passes_per90,
        fouls_committed_per90, fouls_drawn_per90, yellows_per90, reds_per90.
        """
        canonical = to_canonical(player_id)
        path = cache_path(CATEGORY, tournament_slug, canonical)
        if not force_refresh:
            cached = read(path)
            if cached is not None:
                return cached

        payload = await self._client.get_player_statistics(
            player_id=player_id, season=season, team_id=team_id
        )

        df = _extract_player_rates(payload, canonical=canonical)
        write(df, path)
        return df


def _extract_player_rates(
    payload: dict[str, Any], canonical: str
) -> pl.DataFrame:
    """Parse /players response into a single-row per-90 frame.

    API-Football returns response[0].statistics as a list with one entry per
    league/team the player appeared in this season. We aggregate across
    those entries to handle mid-season transfers.
    """
    response = payload.get("response", [])
    if not response:
        return _empty_player_frame(canonical)

    record = response[0]
    af_player_id = record.get("player", {}).get("id")
    statistics = record.get("statistics", [])

    if not statistics:
        return _empty_player_frame(canonical, af_player_id=af_player_id)

    totals = {
        "matches_played": 0,
        "minutes_total": 0,
        "shots": 0,
        "sot": 0,
        "goals": 0,
        "assists": 0,
        "key_passes": 0,
        "fouls_committed": 0,
        "fouls_drawn": 0,
        "yellows": 0,
        "reds": 0,
    }

    # Take the most recent / largest entry for club + league + position.
    primary = max(
        statistics,
        key=lambda s: (s.get("games", {}) or {}).get("minutes") or 0,
    )
    club_team_id = (primary.get("team", {}) or {}).get("id")
    club_league_id = (primary.get("league", {}) or {}).get("id")
    position = (primary.get("games", {}) or {}).get("position") or "M"

    for entry in statistics:
        games = entry.get("games", {}) or {}
        shots = entry.get("shots", {}) or {}
        goals = entry.get("goals", {}) or {}
        passes = entry.get("passes", {}) or {}
        fouls = entry.get("fouls", {}) or {}
        cards = entry.get("cards", {}) or {}

        totals["matches_played"] += games.get("appearences") or 0
        totals["minutes_total"] += games.get("minutes") or 0
        totals["shots"] += shots.get("total") or 0
        totals["sot"] += shots.get("on") or 0
        totals["goals"] += goals.get("total") or 0
        totals["assists"] += goals.get("assists") or 0
        totals["key_passes"] += passes.get("key") or 0
        totals["fouls_committed"] += fouls.get("committed") or 0
        totals["fouls_drawn"] += fouls.get("drawn") or 0
        totals["yellows"] += cards.get("yellow") or 0
        totals["reds"] += cards.get("red") or 0

    minutes = max(totals["minutes_total"], 1)
    reliable = totals["minutes_total"] >= MIN_RELIABLE_MINUTES

    return pl.DataFrame(
        {
            "canonical_id": [canonical],
            "api_football_id": [af_player_id],
            "club_team_id": [club_team_id],
            "club_league_id": [club_league_id],
            "position": [position],
            "matches_played": [totals["matches_played"]],
            "minutes_total": [totals["minutes_total"]],
            "reliable": [reliable],
            "shots_per90": [totals["shots"] / minutes * 90.0],
            "sot_per90": [totals["sot"] / minutes * 90.0],
            "goals_per90": [totals["goals"] / minutes * 90.0],
            "assists_per90": [totals["assists"] / minutes * 90.0],
            "key_passes_per90": [totals["key_passes"] / minutes * 90.0],
            "fouls_committed_per90": [totals["fouls_committed"] / minutes * 90.0],
            "fouls_drawn_per90": [totals["fouls_drawn"] / minutes * 90.0],
            "yellows_per90": [totals["yellows"] / minutes * 90.0],
            "reds_per90": [totals["reds"] / minutes * 90.0],
        }
    )


def _empty_player_frame(canonical: str, af_player_id: int | None = None) -> pl.DataFrame:
    """Empty / zero frame with the same schema. Tags reliable=False."""
    return pl.DataFrame(
        {
            "canonical_id": [canonical],
            "api_football_id": [af_player_id],
            "club_team_id": [None],
            "club_league_id": [None],
            "position": ["M"],
            "matches_played": [0],
            "minutes_total": [0],
            "reliable": [False],
            "shots_per90": [0.0],
            "sot_per90": [0.0],
            "goals_per90": [0.0],
            "assists_per90": [0.0],
            "key_passes_per90": [0.0],
            "fouls_committed_per90": [0.0],
            "fouls_drawn_per90": [0.0],
            "yellows_per90": [0.0],
            "reds_per90": [0.0],
        },
        schema_overrides={
            "club_team_id": pl.Int64,
            "club_league_id": pl.Int64,
            "api_football_id": pl.Int64,
        },
    )
