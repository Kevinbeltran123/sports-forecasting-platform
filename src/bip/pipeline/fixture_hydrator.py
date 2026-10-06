"""FixtureHydrator — API-Football → Orchestrator-shaped fixture dicts.

Sprint 3 Ola B. Bridges ApiFootballClient.get_fixtures(league_id, date)
to the dict shape that LigasModel / MundialModel expect via BaseModel.

Output dict contract (per fixture):
  {
    'fixture_id':     str  (API-Football numeric id, stringified),
    'match_id':       str  (== fixture_id by default; WC2026 may override),
    'competition':    str  (one of SUPPORTED_LEAGUES or 'WC2026'),
    'home_team':      str,
    'away_team':      str,
    'match_datetime': datetime (UTC),
    'features':       np.ndarray | None  (Layer-2 hookup; None in Layer-1),
  }

Layer-1: features=None always. LigasModel returns [] for missing features
(by design, no fallback heuristic). MundialModel ignores the features
field (it reads from the lock directly via match_id).

Layer-2 (xfail in this commit, requires-real-data): inject a
FeatureEngineer that computes ~40-60 features per fixture from the
historical corpus + raw API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import structlog

from bip.models.ligas import LEAGUE_ID_MAP, SUPPORTED_LEAGUES

log = structlog.get_logger(__name__)


# WC2026 league id on API-Football
WC2026_LEAGUE_ID = 1
WC2026_COMPETITION_LABEL = "WC2026"


@runtime_checkable
class ApiFootballClientProtocol(Protocol):
    """Minimal subset of bip.sports.football.client.ApiFootballClient."""

    async def get_fixtures(self, league_id: int, date: str) -> dict: ...


@runtime_checkable
class FeatureEngineerProtocol(Protocol):
    """Layer-2 hook for feature computation.

    Implementations: bip.sports.football.features.FeatureEngineer (legacy,
    559 LOC) or any subset. Returns features as np.ndarray; None signals
    "features unavailable for this fixture".
    """

    def build_for_fixture(self, fixture_raw: dict) -> Any: ...


@dataclass
class HydratorSummary:
    n_leagues_queried: int = 0
    n_fixtures_total: int = 0
    n_skipped: int = 0
    skipped_reasons: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skipped_reasons is None:
            self.skipped_reasons = {}


class FixtureHydrator:
    """Pulls fixtures from API-Football and normalizes them for the v4 stack."""

    def __init__(
        self,
        *,
        api_client: ApiFootballClientProtocol,
        feature_engineer: FeatureEngineerProtocol | None = None,
        leagues: dict[str, int] | None = None,
        include_wc2026: bool = True,
    ) -> None:
        self._client = api_client
        self._feature_engineer = feature_engineer
        self._leagues = dict(leagues) if leagues is not None else dict(LEAGUE_ID_MAP)
        self._include_wc2026 = include_wc2026

    async def hydrate_for_date(
        self, date_iso: str
    ) -> tuple[list[dict[str, Any]], HydratorSummary]:
        """Return (fixture_dicts, summary) for the given YYYY-MM-DD date.

        One API-Football call per league; WC2026 added when include_wc2026.
        """
        summary = HydratorSummary()
        out: list[dict[str, Any]] = []

        queue: list[tuple[str, int]] = list(self._leagues.items())
        if self._include_wc2026:
            queue.append((WC2026_COMPETITION_LABEL, WC2026_LEAGUE_ID))

        for competition, league_id in queue:
            summary.n_leagues_queried += 1
            try:
                raw = await self._client.get_fixtures(league_id, date_iso)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "hydrator_fetch_error",
                    competition=competition,
                    league_id=league_id,
                    error=str(exc),
                )
                summary.skipped_reasons["fetch_error"] = (
                    summary.skipped_reasons.get("fetch_error", 0) + 1
                )
                continue

            for fixture_raw in raw.get("response", []):
                fixture_dict = self._normalize(competition, fixture_raw)
                if fixture_dict is None:
                    summary.n_skipped += 1
                    summary.skipped_reasons["normalize_failed"] = (
                        summary.skipped_reasons.get("normalize_failed", 0) + 1
                    )
                    continue
                # Layer-2: attach features if engineer is wired
                if self._feature_engineer is not None:
                    try:
                        fixture_dict["features"] = self._feature_engineer.build_for_fixture(
                            fixture_raw
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.warning(
                            "hydrator_feature_error",
                            fixture_id=fixture_dict.get("fixture_id"),
                            error=str(exc),
                        )
                        summary.skipped_reasons["feature_error"] = (
                            summary.skipped_reasons.get("feature_error", 0) + 1
                        )
                out.append(fixture_dict)

        summary.n_fixtures_total = len(out)
        log.info(
            "hydrator_run_complete",
            date=date_iso,
            n_leagues=summary.n_leagues_queried,
            n_fixtures=summary.n_fixtures_total,
        )
        return out, summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(competition: str, fixture_raw: dict) -> dict[str, Any] | None:
        """Map API-Football fixture record → orchestrator-shaped dict."""
        fixture = fixture_raw.get("fixture") or {}
        teams = fixture_raw.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}

        api_fixture_id = fixture.get("id")
        if not api_fixture_id:
            return None
        kickoff_iso = fixture.get("date")
        if not kickoff_iso:
            return None
        try:
            kickoff = datetime.fromisoformat(str(kickoff_iso).replace("Z", "+00:00"))
        except ValueError:
            return None

        return {
            "fixture_id": str(api_fixture_id),
            "match_id": str(api_fixture_id),
            "competition": competition,
            "home_team": str(home.get("name", "")),
            "away_team": str(away.get("name", "")),
            "match_datetime": kickoff,
            "features": None,  # Layer-1; FeatureEngineerProtocol hook overrides
        }


__all__ = [
    "ApiFootballClientProtocol",
    "FeatureEngineerProtocol",
    "FixtureHydrator",
    "HydratorSummary",
    "SUPPORTED_LEAGUES",
    "WC2026_COMPETITION_LABEL",
    "WC2026_LEAGUE_ID",
]
