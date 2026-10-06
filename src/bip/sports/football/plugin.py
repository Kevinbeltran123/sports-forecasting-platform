"""Football sport plugin -- implements SportPlugin ABC.

Phase 2: predict() is wired to the ML ensemble loaded from the registry.
Shadow-mode predictions (ML-05) are written alongside production when
the registry has a shadow version configured.

Phase 1 behavior preserved on cold start: when no production model is
registered for a league, predict() returns 1/3-1/3-1/3 with
model_version='stub-v0'.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog

from bip.core.settings import Settings
from bip.core.storage.models import Prediction
from bip.core.storage.parquet_store import ParquetStore
from bip.core.types import MarketKey
from bip.sports import (
    ClaudeContext,
    FeatureMatrix,
    FixtureData,
    ProbabilityMap,
    SportPlugin,
)
from bip.sports.football.client import ApiFootballClient
from bip.sports.football.config.league_registry import LeagueRegistry
from bip.sports.football.config.market_config import load_markets
from bip.sports.football.features import FeatureEngineer
from bip.train.loader import ModelLoader
from bip.train.registry import ModelRegistry

logger = structlog.get_logger(__name__)

_LEAGUES_DIR = Path(__file__).parent / "config" / "leagues"

# CLASSES = [0, 1, 2] from bip.train.stacking -> 1X2 outcome labels
_CLASS_TO_OUTCOME = {0: "1", 1: "X", 2: "2"}

# Map API-Football "Match Winner" value strings -> 1X2 selection keys
_ODDS_VALUE_TO_KEY = {"Home": "1", "Draw": "X", "Away": "2"}


class FootballPlugin(SportPlugin):
    """Football sport plugin -- full SportPlugin implementation."""

    def __init__(
        self,
        settings: Settings,
        prediction_repo: Any | None = None,
    ) -> None:
        """Initialize plugin.

        Args:
            settings: Application settings.
            prediction_repo: Optional PredictionRepository for shadow writes (ML-05).
                             When None, shadow path is silently skipped.
        """
        self._settings = settings
        self._markets = load_markets()
        self._registry = LeagueRegistry(_LEAGUES_DIR)
        self._store = ParquetStore(base_path=Path(settings.parquet_base_path))
        self._engineer = FeatureEngineer()

        # Phase 2: model artifact layer
        self._model_dir = Path(settings.model_dir)
        registry_path = self._model_dir / "football" / "registry.json"
        self._model_registry = ModelRegistry.load(registry_path)
        self._loader = ModelLoader(
            model_dir=self._model_dir, registry=self._model_registry
        )
        self._prediction_repo = prediction_repo

    def get_available_markets(self) -> list[str]:
        """Return market keys from markets.yaml -- CORE-05."""
        return [m["key"] for m in self._markets]

    async def get_fixtures(self, date: datetime) -> list[FixtureData]:
        """Fetch today's football fixtures from API-Football for all 5 leagues."""
        date_str = date.strftime("%Y-%m-%d")
        fixtures: list[FixtureData] = []
        async with ApiFootballClient(api_key=self._settings.api_football_key) as client:
            for league_cfg in self._registry.all_leagues():
                try:
                    raw = await client.get_fixtures(
                        league_id=league_cfg.api_mappings.api_football_league_id,
                        date=date_str,
                    )
                    for item in raw.get("response", []):
                        fixture = self._parse_fixture(item, league_cfg.slug)
                        if fixture:
                            fixtures.append(fixture)
                except Exception as exc:
                    logger.warning(
                        "get_fixtures_failed",
                        league=league_cfg.slug,
                        error=str(exc),
                    )
        logger.info("fixtures_fetched", count=len(fixtures), date=date_str)
        return fixtures

    def _parse_fixture(self, item: dict[str, Any], league_slug: str) -> FixtureData | None:
        try:
            return FixtureData(
                fixture_id=item["fixture"]["id"],
                league=league_slug,
                sport="football",
                home_team=item["teams"]["home"]["name"],
                away_team=item["teams"]["away"]["name"],
                kickoff_utc=datetime.fromisoformat(item["fixture"]["date"]),
            )
        except (KeyError, ValueError) as exc:
            logger.warning(
                "fixture_parse_failed",
                error=str(exc),
                item=str(item)[:200],
            )
            return None

    @staticmethod
    def _parse_matchday(item: dict[str, Any]) -> int:
        """Parse API-Football 'Regular Season - 12' -> 12. Returns 0 on failure."""
        try:
            round_str = item.get("league", {}).get("round") or ""
            tail = round_str.rsplit("-", 1)[-1].strip()
            return int(tail)
        except (ValueError, AttributeError):
            return 0

    async def build_features(self, fixture: FixtureData) -> FeatureMatrix:
        """Build point-in-time feature matrix and write to ParquetStore."""
        computed_at = datetime.now(UTC)
        async with ApiFootballClient(api_key=self._settings.api_football_key) as client:
            league_cfg = next(
                (lc for lc in self._registry.all_leagues() if lc.slug == fixture.league),
                None,
            )
            raw_fixtures: dict[str, Any] = {"response": []}
            if league_cfg is not None:
                date_str = fixture.kickoff_utc.strftime("%Y-%m-%d")
                try:
                    raw_fixtures = await client.get_fixtures(
                        league_id=league_cfg.api_mappings.api_football_league_id,
                        date=date_str,
                    )
                except Exception as exc:
                    logger.warning(
                        "fixtures_refetch_failed",
                        fixture_id=fixture.fixture_id,
                        error=str(exc),
                    )
            raw_stats = await client.get_statistics(fixture_id=fixture.fixture_id)
            raw_lineups = await client.get_lineups(fixture_id=fixture.fixture_id)

        matchday = 0
        for item in raw_fixtures.get("response", []):
            if item.get("fixture", {}).get("id") == fixture.fixture_id:
                matchday = self._parse_matchday(item)
                break
        if matchday == 0:
            matchday = 1

        fm = self._engineer.build_features_for_fixture(
            fixture=fixture,
            raw_stats=raw_stats,
            raw_lineups=raw_lineups,
            computed_at=computed_at,
        )
        season = f"{fixture.kickoff_utc.year}-{fixture.kickoff_utc.year + 1}"
        parquet_row = self._engineer.to_parquet_row(fm, matchday=matchday, season=season)
        self._store.write_features(parquet_row)
        return fm

    # ----------------------------------------------------------
    # Phase 2 -- ML ensemble prediction (ML-01 + ML-05)
    # ----------------------------------------------------------

    async def predict(self, features: FeatureMatrix, market: str | MarketKey) -> ProbabilityMap:
        """Run ML ensemble and return calibrated probabilities.

        ML-01: loads production model from registry, predicts, returns
               ProbabilityMap with model_version sourced from registry.
        ML-05: if registry has a shadow version for this league, also write
               a shadow Prediction row (is_shadow=True). Shadow failures are
               logged and do not raise.
        """
        computed_at = datetime.now(UTC)

        prod_version = self._model_registry.get_production_version(features.league)
        if prod_version is None:
            # Cold start -- preserve Phase 1 stub contract
            logger.info("predict_cold_start", league=features.league)
            return ProbabilityMap(
                fixture_id=features.fixture_id,
                market=market,
                probabilities={"1": 1 / 3, "X": 1 / 3, "2": 1 / 3},
                model_version="stub-v0",
                computed_at=computed_at,
            )

        X = self._features_to_numpy(features)  # noqa: N806 — sklearn convention

        # Production prediction
        prod_probs = self._predict_one(features.league, prod_version, X)
        prod_map = self._probs_to_map(
            fixture_id=features.fixture_id,
            market=market,
            probs=prod_probs,
            model_version=prod_version,
            computed_at=computed_at,
        )

        # Shadow prediction (ML-05) -- best-effort, never raises
        shadow_version = self._model_registry.get_shadow_version(features.league)
        shadow_map: ProbabilityMap | None = None
        if shadow_version is not None:
            try:
                shadow_probs = self._predict_one(features.league, shadow_version, X)
                shadow_map = self._probs_to_map(
                    fixture_id=features.fixture_id,
                    market=market,
                    probs=shadow_probs,
                    model_version=shadow_version,
                    computed_at=computed_at,
                )
            except Exception as exc:
                logger.warning(
                    "shadow_predict_failed",
                    league=features.league,
                    version=shadow_version,
                    error=str(exc),
                )

        # Write to Supabase if repo wired
        if self._prediction_repo is not None:
            self._write_prediction_rows(
                features=features,
                market=market,
                prod_map=prod_map,
                shadow_map=shadow_map,
            )

        return prod_map

    def _predict_one(
        self, league: str, version: str, X: np.ndarray,  # noqa: N803 — sklearn convention
    ) -> np.ndarray:
        """Load a specific model version and produce a single-row (3,) probability vector."""
        ensemble, _meta = self._loader.load(league=league, version=version)
        probs = np.asarray(cast(Any, ensemble).predict_proba(X))
        # Normalize shape: (1, 3) -> (3,)
        if probs.ndim == 2:
            probs = probs[0]
        # Renormalize defensively
        total = float(probs.sum())
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        return (probs / total).astype(float)

    def _features_to_numpy(self, features: FeatureMatrix) -> np.ndarray:
        """Convert FeatureMatrix.features dict to a (1, n) numpy array in sorted-key order."""
        keys = sorted(features.features.keys())
        return np.array([[features.features[k] for k in keys]], dtype=float)

    @staticmethod
    def _probs_to_map(
        fixture_id: int,
        market: str,
        probs: np.ndarray,
        model_version: str,
        computed_at: datetime,
    ) -> ProbabilityMap:
        return ProbabilityMap(
            fixture_id=fixture_id,
            market=market,
            probabilities={
                _CLASS_TO_OUTCOME[0]: float(probs[0]),
                _CLASS_TO_OUTCOME[1]: float(probs[1]),
                _CLASS_TO_OUTCOME[2]: float(probs[2]),
            },
            model_version=model_version,
            computed_at=computed_at,
        )

    async def get_opening_odds(self, fixture_id: int) -> dict[str, float]:
        """Fetch 1X2 opening odds from API-Football for the configured bookmaker.

        G-CODE-01: implements SportPlugin.get_opening_odds. Returns
        ``{"1": <home_odd>, "X": <draw_odd>, "2": <away_odd>}`` for the
        primary "Match Winner" market on Betano. Returns ``{}`` when:
          - API response is empty (``response: []``),
          - Betano is not in the bookmakers list for this fixture,
          - Betano has no "Match Winner" market for this fixture, or
          - the "Match Winner" values are missing/incomplete.

        Network/API failures bubble up (httpx exceptions, status errors)
        because they require retry/back-off at the caller. Coverage gaps
        return empty dict so the orchestrator can route around them.
        """
        bookmaker = "Betano"  # reference bookmaker for opening odds
        async with ApiFootballClient(api_key=self._settings.api_football_key) as client:
            raw = await client.get_odds(fixture_id=fixture_id, bookmaker=bookmaker)
        response = raw.get("response", []) or []
        if not response:
            logger.info("get_opening_odds_no_response", fixture_id=fixture_id)
            return {}
        # Walk: response[].bookmakers[?name==Betano].bets[?name=="Match Winner"].values
        for fixture_block in response:
            for bm in fixture_block.get("bookmakers", []) or []:
                if bm.get("name") != bookmaker:
                    continue
                for bet in bm.get("bets", []) or []:
                    if bet.get("name") != "Match Winner":
                        continue
                    odds: dict[str, float] = {}
                    for v in bet.get("values", []) or []:
                        key = _ODDS_VALUE_TO_KEY.get(v.get("value", ""))
                        if key is None:
                            continue
                        try:
                            odds[key] = float(v.get("odd"))
                        except (TypeError, ValueError):
                            continue
                    if {"1", "X", "2"}.issubset(odds.keys()):
                        return odds
        logger.info(
            "get_opening_odds_no_betano_match_winner", fixture_id=fixture_id
        )
        return {}

    def _write_prediction_rows(
        self,
        features: FeatureMatrix,
        market: str,
        prod_map: ProbabilityMap,
        shadow_map: ProbabilityMap | None,
    ) -> None:
        """Persist production + optional shadow predictions to Supabase (ML-05).

        Failures are logged and swallowed -- prediction writes must never
        break the production path.
        """
        repo = self._prediction_repo
        if repo is None:
            return
        # Production row -- G-MAINT-08/11: real values from FeatureMatrix
        try:
            prod_pred = Prediction(
                fixture_id=features.fixture_id,
                league=features.league,
                sport=features.sport,
                market=market,
                home_team=features.home_team,
                away_team=features.away_team,
                kickoff_utc=features.kickoff_utc,
                probabilities=prod_map.probabilities,
                model_version=prod_map.model_version,
                is_shadow=False,
            )
            repo.insert(prod_pred)
        except Exception as exc:
            logger.warning(
                "production_prediction_write_failed",
                fixture_id=features.fixture_id,
                error=str(exc),
            )

        # Shadow row (when configured) -- G-MAINT-08/11: real values from FeatureMatrix
        if shadow_map is not None:
            try:
                shadow_pred = Prediction(
                    fixture_id=features.fixture_id,
                    league=features.league,
                    sport=features.sport,
                    market=market,
                    home_team=features.home_team,
                    away_team=features.away_team,
                    kickoff_utc=features.kickoff_utc,
                    probabilities=shadow_map.probabilities,
                    model_version=shadow_map.model_version,
                    is_shadow=True,
                )
                repo.insert(shadow_pred)
            except Exception as exc:
                logger.warning(
                    "shadow_prediction_write_failed",
                    fixture_id=features.fixture_id,
                    error=str(exc),
                )

    async def build_claude_context(self, fixture: FixtureData) -> ClaudeContext:
        """Claude context -- stub in Phase 1/2, wired in Phase 3."""
        return ClaudeContext(
            fixture_id=fixture.fixture_id,
            sport=fixture.sport,
            summary="",
        )
