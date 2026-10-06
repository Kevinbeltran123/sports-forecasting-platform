"""One-off historical data seed — D-01.

Pulls fixtures + stats + lineups + H2H across 5 leagues × 3 seasons
(2023-2024, 2024-2025, 2025-2026) via the Phase 1 ApiFootballClient,
writing features to the same Hive-partitioned Parquet store as live
ingestion. Checkpoint-resumable.

NOT invoked by the scheduler. Run manually:
    uv run python scripts/seed_historical.py

Rate-limited to 300 req/min (API-Football Pro plan).

WR-03 — Checkpoint compatibility:
    Any pre-CR-01 seed run wrote an inconsistent checkpoint:
    ``completed_results_fixture_ids`` and ``completed_odds_fixture_ids``
    listed fixtures whose rows were silently wiped from the on-disk
    Parquet store by the next per-fixture write. Resuming such a checkpoint
    after the CR-01 fix would skip those fixtures forever, leaving a
    permanent data hole. The checkpoint now carries a ``schema_version``
    field; loading a file without the post-CR-01 version automatically
    discards the results/odds keys (features key is preserved — that
    write path was never affected by CR-01) and emits a structured-log
    warning. To force a clean re-seed, delete ``data/seed_checkpoint.json``
    before the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

from bip.core.settings import Settings
from bip.core.storage.parquet_store import ParquetStore
from bip.sports import FixtureData
from bip.sports.football.client import ApiFootballClient
from bip.sports.football.config.league_registry import LeagueRegistry
from bip.sports.football.features import FeatureEngineer

logger = structlog.get_logger(__name__)

CHECKPOINT_PATH = Path("data/seed_checkpoint.json")
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
RATE_LIMIT_RPM = 300               # Pro plan
INTER_REQUEST_DELAY_S = 60.0 / RATE_LIMIT_RPM  # ~0.2s → ~300 req/min
_LEAGUES_DIR = (
    Path(__file__).parent.parent
    / "src"
    / "bip"
    / "sports"
    / "football"
    / "config"
    / "leagues"
)


# ----------------------------------------------------------------------
# Checkpoint helpers — atomic write (rename pattern)
# ----------------------------------------------------------------------
#
# Phase 02.1-09: checkpoint expanded from a single set to a 3-key dict so
# results and odds writes can resume independently of feature writes.
#
# Backward-compat: a Phase 2 file containing only ``completed_fixture_ids``
# loads cleanly; the two new keys default to empty sets.

CHECKPOINT_KEYS = (
    "completed_fixture_ids",
    "completed_results_fixture_ids",
    "completed_odds_fixture_ids",
)

# WR-03: bump whenever the on-disk store invariant changes such that a
# pre-existing checkpoint should not be trusted to mirror what is on disk.
# v1 = post-CR-01 fix (per-fixture writes use overwrite=False and actually
# persist). Any checkpoint missing this version (or carrying a lower one)
# has its results/odds tracking sets discarded; the features set is kept
# because the features write path was never affected by CR-01.
CHECKPOINT_SCHEMA_VERSION = 1


def _empty_state() -> dict[str, set[int]]:
    return {key: set() for key in CHECKPOINT_KEYS}


def load_checkpoint() -> dict[str, set[int]]:
    """Return the 3-key checkpoint dict from data/seed_checkpoint.json.

    Keys: ``completed_fixture_ids`` (features), ``completed_results_fixture_ids``,
    ``completed_odds_fixture_ids``. Returns all three as empty sets if the file
    does not exist. If the file exists but only carries the legacy single key
    (Phase 2 shape), the missing keys are populated with empty sets.

    WR-03: a checkpoint written before CR-01 was fixed (i.e. without
    ``schema_version`` >= 1) is treated as untrusted for results/odds
    tracking — those two sets are reset to empty so the next run re-seeds
    every fixture's results+odds row, while the features set is preserved
    (the features write path was never affected by CR-01). Delete
    ``data/seed_checkpoint.json`` to force a full clean reseed.
    """
    if not CHECKPOINT_PATH.exists():
        return _empty_state()
    data = json.loads(CHECKPOINT_PATH.read_text())
    state = {key: set(data.get(key, [])) for key in CHECKPOINT_KEYS}
    on_disk_version = int(data.get("schema_version", 0))
    if on_disk_version < CHECKPOINT_SCHEMA_VERSION:
        logger.warning(
            "seed_checkpoint_pre_cr01_reset",
            on_disk_version=on_disk_version,
            required_version=CHECKPOINT_SCHEMA_VERSION,
            note=(
                "Discarding results/odds tracking sets — pre-CR-01 store "
                "rows were silently overwritten and cannot be trusted."
            ),
        )
        state["completed_results_fixture_ids"] = set()
        state["completed_odds_fixture_ids"] = set()
    return state


def save_checkpoint(state: dict[str, set[int]]) -> None:
    """Atomic write: JSON to .tmp, then Path.replace() onto target.

    Each known set in ``state`` is sorted before serialization for a stable
    on-disk diff. Unknown keys are ignored. WR-03: ``schema_version`` is
    written so a downgrade path can detect a forward-compat file.
    """
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
    payload: dict[str, object] = {
        key: sorted(state.get(key, set())) for key in CHECKPOINT_KEYS
    }
    payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION
    payload["updated_at"] = datetime.now(UTC).isoformat()
    tmp.write_text(json.dumps(payload))
    tmp.replace(CHECKPOINT_PATH)


# ----------------------------------------------------------------------
# Season → date range (API-Football uses YYYY-MM-DD per-day queries)
# ----------------------------------------------------------------------

def season_date_range(season: str) -> tuple[datetime, datetime]:
    """Return (start, end) UTC datetimes for a season string like '2024-2025'.

    Football season window: Aug 1 YYYY → May 31 YYYY+1.
    """
    y1, y2 = season.split("-")
    start = datetime(int(y1), 8, 1, tzinfo=UTC)
    end = datetime(int(y2), 5, 31, 23, 59, 59, tzinfo=UTC)
    return start, end


def _parse_fixture(item: dict, league_slug: str) -> FixtureData | None:
    """Parse API-Football fixture item into FixtureData. Skip malformed."""
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
        logger.warning("fixture_parse_failed", error=str(exc))
        return None


def _parse_matchday(item: dict) -> int | None:
    """Parse 'Regular Season - 12' → 12. Returns None on parse failure.

    WR-04: previously returned ``0`` for any non-numeric round (cup ties,
    play-offs, group stages without a numeric tail). That collapsed every
    unparseable round into a single ``matchday=0`` Hive partition, hiding
    round structure from downstream consumers and amplifying CR-01's
    overwrite pathology. ``None`` now signals "skip this fixture's
    feature row" — the caller filters before writing so cup matches do
    not silently land in matchday=0.
    """
    try:
        round_str = item["league"]["round"]
        # Examples: 'Regular Season - 12', 'Quarter-finals'
        tail = round_str.rsplit("-", 1)[-1].strip()
        return int(tail)
    except (KeyError, ValueError, AttributeError):
        return None


# ----------------------------------------------------------------------
# League allowlist — T-02.1-04 (path traversal prevention).
# Validated before any API call or Parquet write so a malformed slug cannot
# pollute the partitioned store.
# ----------------------------------------------------------------------

ALLOWED_LEAGUE_SLUGS: frozenset[str] = frozenset(
    {
        "premier_league",
        "la_liga",
        "bundesliga",
        "serie_a",
        "ligue_1",
    }
)


# ----------------------------------------------------------------------
# Odds parsing — extract Betano 1X2 from one bulk-/odds entry.
# D-02 (closing proxy = opening in 02.1) + D-03 (skip non-Betano fixtures).
# ----------------------------------------------------------------------

_ODDS_SCHEMA: dict[str, pl.DataType] = {
    "fixture_id": pl.Int64,
    "opening_home": pl.Float64,
    "opening_draw": pl.Float64,
    "opening_away": pl.Float64,
    "closing_home": pl.Float64,
    "closing_draw": pl.Float64,
    "closing_away": pl.Float64,
    "pinnacle_close_home": pl.Float64,
    "pinnacle_close_draw": pl.Float64,
    "pinnacle_close_away": pl.Float64,
}


def _parse_odds_entry(
    entry: dict,
    fixture_id: int,
    league_slug: str,
    season: str,
) -> pl.DataFrame | None:
    """Extract Betano 1X2 opening odds from a single API-Football odds entry.

    Returns ``None`` if Betano is not present in the entry's bookmakers list
    or if the Match Winner market is missing one of the three outcomes.

    D-02: closing proxy = opening for Phase 02.1; ``closing_*`` columns mirror
    ``opening_*`` until the live closing snapshotter lands post-Phase 3.
    D-04: Pinnacle columns are written as null — populated by plan 02.1-07's
    fetch_historical_closing once the Pinnacle CLV path is implemented.
    """
    bookmakers = entry.get("bookmakers", []) or []
    betano = next(
        (b for b in bookmakers if b.get("name", "").lower() == "betano"),
        None,
    )
    if betano is None:
        return None

    bets = {b.get("name"): b.get("values", []) for b in betano.get("bets", []) or []}
    match_winner = bets.get("Match Winner", []) or []
    odds_map: dict[str, float] = {}
    for v in match_winner:
        try:
            odds_map[v["value"]] = float(v["odd"])
        except (KeyError, TypeError, ValueError):
            continue

    home = odds_map.get("Home")
    draw = odds_map.get("Draw")
    away = odds_map.get("Away")
    if home is None or draw is None or away is None:
        return None

    return pl.DataFrame(
        {
            "fixture_id": [fixture_id],
            "sport": ["football"],
            "league": [league_slug],
            "season": [season],
            "bookmaker": ["Betano"],
            "opening_home": [home],
            "opening_draw": [draw],
            "opening_away": [away],
            "closing_home": [home],          # D-02: opening proxies as closing
            "closing_draw": [draw],
            "closing_away": [away],
            "pinnacle_close_home": [None],   # populated by plan 02.1-07
            "pinnacle_close_draw": [None],
            "pinnacle_close_away": [None],
        },
        schema_overrides=_ODDS_SCHEMA,
    )


# ----------------------------------------------------------------------
# Main seed loop
# ----------------------------------------------------------------------

async def seed_one_league(
    client: ApiFootballClient,
    engineer: FeatureEngineer,
    store: ParquetStore,
    league_cfg,
    season: str,
    state: dict[str, set[int]],
) -> int:
    """Fetch fixtures for one (league, season) and write features + results + odds.

    On each fixture this writes three independent stores:

    * **Features** (4-level Hive partitioning) — gated by
      ``state["completed_fixture_ids"]``.
    * **Results** (3-level: sport/league/season) — extracted from the same
      ``/fixtures`` payload (D-06: zero extra API calls), gated by
      ``state["completed_results_fixture_ids"]``.
    * **Odds** (3-level) — populated from a single bulk ``/odds`` call per
      (league, season) (research finding 1: per-fixture mode has 7-day
      lookback), gated by ``state["completed_odds_fixture_ids"]``.

    Returns the count of NEW feature rows written.

    Raises:
        ValueError: if ``league_cfg.slug`` is outside ``ALLOWED_LEAGUE_SLUGS``
            (T-02.1-04 path-traversal guard).
    """
    # T-02.1-04: validate league slug BEFORE any API call or Parquet write.
    if league_cfg.slug not in ALLOWED_LEAGUE_SLUGS:
        raise ValueError(
            f"seed_one_league: unknown league slug '{league_cfg.slug}' — "
            f"must be one of {sorted(ALLOWED_LEAGUE_SLUGS)}"
        )

    completed = state["completed_fixture_ids"]

    # Research finding 1: bulk per-season is the only viable historical mode
    # (per-fixture /odds has a 7-day lookback at API-Football). One call per
    # (league, season) — the response is keyed by fixture_id for O(1) lookup.
    season_year = int(season.split("-")[0])
    try:
        raw_odds_bulk = await client.get_odds(
            league_id=league_cfg.api_mappings.api_football_league_id,
            season=season_year,
            bookmaker="Betano",
        )
        await asyncio.sleep(INTER_REQUEST_DELAY_S)
        odds_by_fixture: dict[int, dict] = {
            entry["fixture"]["id"]: entry
            for entry in raw_odds_bulk.get("response", [])
            if isinstance(entry, dict) and "fixture" in entry
        }
        logger.info(
            "seed_bulk_odds_fetched",
            league=league_cfg.slug,
            season=season,
            n_fixtures_with_odds=len(odds_by_fixture),
        )
    except Exception as exc:
        logger.warning(
            "seed_bulk_odds_failed",
            league=league_cfg.slug,
            season=season,
            error=str(exc),
        )
        odds_by_fixture = {}

    processed = 0
    start, end = season_date_range(season)
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        try:
            raw = await client.get_fixtures(
                league_id=league_cfg.api_mappings.api_football_league_id,
                date=date_str,
            )
        except Exception as exc:
            logger.warning(
                "seed_get_fixtures_failed",
                league=league_cfg.slug,
                date=date_str,
                error=str(exc),
            )
            current += timedelta(days=1)
            await asyncio.sleep(INTER_REQUEST_DELAY_S)
            continue

        await asyncio.sleep(INTER_REQUEST_DELAY_S)

        for item in raw.get("response", []):
            fixture = _parse_fixture(item, league_cfg.slug)
            if fixture is None:
                continue

            # D-06: extract goals + status from the existing /fixtures payload
            # (zero extra API calls). Independent gate so a fixture whose
            # features were written earlier still gets its results row.
            if fixture.fixture_id not in state["completed_results_fixture_ids"]:
                status_short = (
                    item.get("fixture", {}).get("status", {}).get("short", "")
                )
                goals = item.get("goals", {}) or {}
                home_goals = goals.get("home")
                away_goals = goals.get("away")
                result_row = pl.DataFrame(
                    {
                        "fixture_id": [fixture.fixture_id],
                        "sport": ["football"],
                        "league": [league_cfg.slug],
                        "season": [season],
                        "home_goals": [home_goals],
                        "away_goals": [away_goals],
                        "status": [status_short or "unknown"],
                    },
                    schema_overrides={"fixture_id": pl.Int64},
                )
                # CR-01: overwrite=False keeps append semantics. With the
                # default overwrite=True every per-fixture write would
                # shutil.rmtree the (sport, league, season) partition and
                # silently destroy every previously-seeded fixture in the
                # same partition.
                store.write_results(result_row, overwrite=False)
                state["completed_results_fixture_ids"].add(fixture.fixture_id)
                logger.info(
                    "seed_results_written",
                    fixture_id=fixture.fixture_id,
                    status=status_short,
                )

            # D-03: skip silently if Betano not in odds_by_fixture (no coverage)
            # or _parse_odds_entry returns None (Match Winner market missing).
            if fixture.fixture_id not in state["completed_odds_fixture_ids"]:
                odds_entry = odds_by_fixture.get(fixture.fixture_id)
                if odds_entry is not None:
                    odds_row = _parse_odds_entry(
                        odds_entry,
                        fixture.fixture_id,
                        league_cfg.slug,
                        season,
                    )
                    if odds_row is not None:
                        # CR-01: see write_results comment above — append
                        # semantics are mandatory for per-fixture writes.
                        store.write_odds(odds_row, overwrite=False)
                        state["completed_odds_fixture_ids"].add(fixture.fixture_id)
                    else:
                        logger.info(
                            "seed_betano_odds_not_found",
                            fixture_id=fixture.fixture_id,
                        )

            if fixture.fixture_id in completed:
                continue

            matchday = _parse_matchday(item)
            if matchday is None:
                # WR-04: skip cup/play-off rounds whose round string is
                # non-numeric (e.g. "Quarter-finals"). Lumping every such
                # fixture into matchday=0 hides round structure from the
                # Hive partitioner and from any downstream feature that uses
                # matchday as a contextual signal. Logged so a long-tail of
                # skipped fixtures remains visible to the operator.
                logger.info(
                    "seed_skip_unparseable_round",
                    fixture_id=fixture.fixture_id,
                    round=(
                        (item.get("league") or {}).get("round", "<missing>")
                    ),
                )
                continue

            # Pull per-fixture stats + lineups. Errors skip the fixture.
            try:
                raw_stats = await client.get_statistics(fixture_id=fixture.fixture_id)
                await asyncio.sleep(INTER_REQUEST_DELAY_S)
                raw_lineups = await client.get_lineups(fixture_id=fixture.fixture_id)
                await asyncio.sleep(INTER_REQUEST_DELAY_S)
            except Exception as exc:
                logger.warning(
                    "seed_fixture_failed",
                    fixture_id=fixture.fixture_id,
                    error=str(exc),
                )
                continue

            # computed_at is kickoff_utc itself (historical: we treat pre-kickoff
            # as the point-in-time boundary for this fixture).
            fm = engineer.build_features_for_fixture(
                fixture=fixture,
                raw_stats=raw_stats,
                raw_lineups=raw_lineups,
                computed_at=fixture.kickoff_utc,
            )
            parquet_row = engineer.to_parquet_row(fm, matchday=matchday, season=season)
            store.write_features(parquet_row)

            completed.add(fixture.fixture_id)
            processed += 1

            # Persist checkpoint every 25 fixtures
            if processed % 25 == 0:
                save_checkpoint(state)

        current += timedelta(days=1)

    return processed


async def main(seasons: list[str] | None = None) -> None:
    settings = Settings()
    registry = LeagueRegistry(_LEAGUES_DIR)
    engineer = FeatureEngineer()
    store = ParquetStore(base_path=Path(settings.parquet_base_path))
    state = load_checkpoint()
    logger.info(
        "seed_start",
        resumed_features=len(state["completed_fixture_ids"]),
        resumed_results=len(state["completed_results_fixture_ids"]),
        resumed_odds=len(state["completed_odds_fixture_ids"]),
        seasons=seasons or SEASONS,
    )

    total_new = 0
    async with ApiFootballClient(api_key=settings.api_football_key) as client:
        for league_cfg in registry.all_leagues():
            for season in (seasons or SEASONS):
                try:
                    n = await seed_one_league(
                        client=client,
                        engineer=engineer,
                        store=store,
                        league_cfg=league_cfg,
                        season=season,
                        state=state,
                    )
                    total_new += n
                    save_checkpoint(state)
                    logger.info(
                        "seed_league_season_done",
                        league=league_cfg.slug,
                        season=season,
                        new=n,
                        total_features=len(state["completed_fixture_ids"]),
                        total_results=len(state["completed_results_fixture_ids"]),
                        total_odds=len(state["completed_odds_fixture_ids"]),
                    )
                except Exception as exc:
                    logger.warning(
                        "seed_league_failed",
                        league=league_cfg.slug,
                        season=season,
                        error=str(exc),
                    )

    save_checkpoint(state)
    logger.info(
        "seed_done",
        new=total_new,
        total_features=len(state["completed_fixture_ids"]),
        total_results=len(state["completed_results_fixture_ids"]),
        total_odds=len(state["completed_odds_fixture_ids"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical data seed for Phase 2 ML training."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Subset of seasons to seed (default: all 3). Example: --seasons 2024-2025",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(seasons=args.seasons))
