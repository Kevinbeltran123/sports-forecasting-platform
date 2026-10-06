"""MatchdayValidator — T-2h pre-kickoff sanity check on a locked Mundial pick.

Sprint 0 Ola C. Implementation of the implementation spec §5.8 with the
realistic API-Football endpoints available today (get_lineups, get_injuries).

Validation policy (intentionally conservative — Mundial picks are
already gated by the lock):

  1. If lineup is NOT confirmed yet, return (True, None) so the caller
     re-runs closer to kickoff. We do NOT invalidate on missing lineup.
  2. If lineup IS confirmed AND a critical injury (status=Doubtful/Out)
     is reported for a starter from either team, return
     (False, "key_player_injury:{name}").
  3. Otherwise return (True, None).

The validator does NOT check odds availability or market closure — that
is the Delivery Worker's concern (Sprint 2). The validator is a SAFETY
gate, not a delivery filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from bip.models.base import PredictionRecord

log = structlog.get_logger(__name__)


class ApiFootballClientProtocol(Protocol):
    """Minimal subset of bip.sports.football.client.ApiFootballClient used here.

    Declared as a Protocol so tests can pass mocks without importing the
    real client (keeps the validator decoupled).
    """

    async def get_lineups(self, fixture_id: int) -> dict: ...
    async def get_injuries(self, league_id: int, season: int) -> dict: ...


@dataclass(frozen=True)
class ValidationResult:
    """Result of a single MatchdayValidator.validate call."""

    valid: bool
    reason: str | None
    checks_performed: list[str]


def _starters_from_lineup(lineup_response: dict) -> list[dict[str, Any]]:
    """Flatten API-Football lineup response into a list of starters.

    API-Football lineup format:
      {"response": [{"team": {...}, "startXI": [{"player": {...}}], ...}, ...]}
    """
    starters: list[dict[str, Any]] = []
    for team_entry in lineup_response.get("response", []):
        for item in team_entry.get("startXI") or []:
            player = item.get("player") or item
            if "name" in player or "id" in player:
                starters.append(player)
    return starters


def _injured_players(injuries_response: dict) -> list[dict[str, Any]]:
    """Filter API-Football injuries response to currently-unavailable players.

    A player is treated as critically injured when player.type is in
    {"Missing Fixture", "Doubtful", "Out"}; the latter two are reported
    directly by the feed for pre-match injuries.
    """
    out: list[dict[str, Any]] = []
    critical_types = {"Missing Fixture", "Doubtful", "Out"}
    for entry in injuries_response.get("response", []):
        player_info = entry.get("player") or {}
        injury_type = player_info.get("type") or entry.get("type")
        if injury_type in critical_types:
            out.append(player_info)
    return out


class MatchdayValidator:
    """T-2h validation gate for Mundial locked picks."""

    def __init__(
        self,
        client: ApiFootballClientProtocol,
        *,
        league_id: int = 1,  # API-Football league id for FIFA World Cup
        season: int = 2026,
    ) -> None:
        self.client = client
        self.league_id = league_id
        self.season = season

    async def validate(
        self,
        prediction: PredictionRecord,
        *,
        api_fixture_id: int,
    ) -> ValidationResult:
        """Run the T-2h checks for one locked prediction.

        Args:
            prediction: the locked Mundial PredictionRecord to validate.
            api_fixture_id: API-Football numeric fixture id (NOT the
                lock's match_id which is the WC slug). The caller maps
                lock match_id → api fixture id via team-name match.

        Returns:
            ValidationResult with valid=False only on a definitive
            blocker (e.g. confirmed key-player injury). Missing or
            partial data returns valid=True so the caller can retry
            closer to kickoff.
        """
        checks: list[str] = []

        # 1. Lineup
        try:
            lineup = await self.client.get_lineups(api_fixture_id)
            checks.append("lineup_fetched")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "matchday_lineup_fetch_error",
                fixture_id=prediction.fixture_id,
                api_fixture_id=api_fixture_id,
                error=str(exc),
            )
            return ValidationResult(True, None, checks)

        starters = _starters_from_lineup(lineup)
        if not starters:
            # Lineup not confirmed yet — non-blocking.
            log.info("matchday_lineup_not_confirmed", fixture_id=prediction.fixture_id)
            return ValidationResult(True, None, checks + ["lineup_not_confirmed"])

        checks.append("lineup_confirmed")

        # 2. Injuries
        try:
            injuries = await self.client.get_injuries(self.league_id, self.season)
            checks.append("injuries_fetched")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "matchday_injuries_fetch_error",
                fixture_id=prediction.fixture_id,
                error=str(exc),
            )
            return ValidationResult(True, None, checks)

        injured = _injured_players(injuries)
        if not injured:
            return ValidationResult(True, None, checks)

        starter_ids = {s.get("id") for s in starters if s.get("id") is not None}
        starter_names = {(s.get("name") or "").strip().lower() for s in starters}
        blocked_player: dict[str, Any] | None = None
        for inj in injured:
            inj_id = inj.get("id")
            inj_name = (inj.get("name") or "").strip().lower()
            if inj_id and inj_id in starter_ids:
                blocked_player = inj
                break
            if inj_name and inj_name in starter_names:
                blocked_player = inj
                break

        if blocked_player is not None:
            display = blocked_player.get("name") or str(blocked_player.get("id"))
            log.info(
                "matchday_key_player_injury_blocked",
                fixture_id=prediction.fixture_id,
                player=display,
            )
            return ValidationResult(False, f"key_player_injury:{display}", checks)

        return ValidationResult(True, None, checks)


__all__ = ["ApiFootballClientProtocol", "MatchdayValidator", "ValidationResult"]
