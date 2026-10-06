"""Understat xG client — Phase 0.5 of WC2026 calibration-lock spike.

Wraps `soccerdata.Understat` (sync, file-cached) and exposes Polars DataFrames
matching canonical schemas defined in this module. The wrapper exists so that:

1. Downstream consumers (corners predictor, goal model) depend on a stable
   schema, not on whatever columns soccerdata happens to return today.
2. Tests can substitute a fake reader via the `_reader_factory` injection
   point without monkeypatching the soccerdata module.
3. The cache lives under `data/cache/understat/` (project-local), so backfills
   are reproducible and visible alongside other parquet caches.

Layer 1 (this commit) ships the wrapper + schemas + aggregation helper with
synthetic-data tests. Layer 2 (queued, `# requires-real-data`) backfills
top-5 EU + RFPL 2020/21–2025/26 to parquet and feeds `xg_total_l5` into the
corners and goal models — see SYNTHESIS.md Conclusion 2 (Poisson tail failure)
for why xG matters specifically for those two predictors.

Research grounding:
- internal research notes Part 3 — Understat reclassified GRAY → RECONSIDERED
  because soccerdata wraps it with explicit non-commercial OK from maintainer.
- internal research notes Part 4 — Phase 0.5 formal definition.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# League slugs (soccerdata convention) — top-5 EU + RFPL
# Reference: https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html
# ---------------------------------------------------------------------------
LEAGUE_EPL = "ENG-Premier League"
LEAGUE_LALIGA = "ESP-La Liga"
LEAGUE_SERIE_A = "ITA-Serie A"
LEAGUE_BUNDESLIGA = "GER-Bundesliga"
LEAGUE_LIGUE_1 = "FRA-Ligue 1"
LEAGUE_RFPL = "RUS-Premier League"

TOP_5_EU_LEAGUES: tuple[str, ...] = (
    LEAGUE_EPL,
    LEAGUE_LALIGA,
    LEAGUE_SERIE_A,
    LEAGUE_BUNDESLIGA,
    LEAGUE_LIGUE_1,
)
TOP_5_EU_PLUS_RFPL: tuple[str, ...] = (*TOP_5_EU_LEAGUES, LEAGUE_RFPL)


# ---------------------------------------------------------------------------
# Canonical Polars schemas
# Downstream code MUST consume DataFrames matching these schemas. If
# soccerdata's columns drift in a future release, the normalization happens
# inside this module — consumers stay decoupled.
# ---------------------------------------------------------------------------

# Per-shot event, one row per shot.
# Used by: offline shot-quality model (Phase 7 optional), tail-fitting for
# corners/goals overdispersion checks.
SHOT_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Int64,
    "shot_id": pl.Int64,
    "minute": pl.Int32,
    "season": pl.Utf8,
    "league": pl.Utf8,
    "date": pl.Datetime("ms"),
    "home_team": pl.Utf8,
    "away_team": pl.Utf8,
    "team_side": pl.Utf8,             # 'h' | 'a'
    "player": pl.Utf8,
    "player_id": pl.Int64,
    "x": pl.Float32,                  # normalized field coord 0–1
    "y": pl.Float32,
    "xg": pl.Float32,                 # 0–1 expected-goal probability
    "result": pl.Utf8,                # Goal | MissedShots | SavedShot | BlockedShot | ShotOnPost
    "situation": pl.Utf8,             # OpenPlay | SetPiece | FromCorner | DirectFreekick | Penalty
    "shot_type": pl.Utf8,             # LeftFoot | RightFoot | Head | OtherBodyPart
    "is_penalty": pl.Boolean,         # convenience flag derived from situation
}

# Per-team-per-match aggregate, two rows per match (one home, one away).
# Used by: xg_total_l5 covariate for corners_poisson and goal model.
TEAM_MATCH_STATS_SCHEMA: dict[str, pl.DataType] = {
    "match_id": pl.Int64,
    "season": pl.Utf8,
    "league": pl.Utf8,
    "date": pl.Datetime("ms"),
    "team": pl.Utf8,
    "opponent": pl.Utf8,
    "venue": pl.Utf8,                 # 'home' | 'away'
    "goals": pl.Int32,
    "xg": pl.Float32,
    "npxg": pl.Float32,               # non-penalty xG (preferred for transfer)
    "xg_against": pl.Float32,
    "deep": pl.Int32,                 # passes completed within 20yd of goal
    "ppda": pl.Float32,               # passes per defensive action (pressing)
}


# ---------------------------------------------------------------------------
# Reader protocol — what we need from soccerdata.Understat
# ---------------------------------------------------------------------------

class _UnderstatReader(Protocol):
    """Subset of `soccerdata.Understat` we actually call. Test fakes implement this."""

    def read_shot_events(self) -> Any: ...      # returns a pandas DataFrame
    def read_team_match_stats(self) -> Any: ...  # returns a pandas DataFrame


ReaderFactory = Callable[..., _UnderstatReader]


# ---------------------------------------------------------------------------
# UnderstatClient
# ---------------------------------------------------------------------------

class UnderstatClient:
    """Sync wrapper around `soccerdata.Understat` returning Polars DataFrames.

    Why sync: soccerdata is sync internally and its only side effect is reading
    cached parquet files. Wrapping in asyncio.to_thread adds overhead for
    nothing — Understat is invoked offline (weekly backfill), never on the
    pre-kickoff hot path.

    Why DI on `_reader_factory`: tests can pass a fake reader instead of
    monkeypatching the soccerdata module, which keeps the test surface
    flat and import-safe.

    Cache layout (project-local, reproducible):
        data/cache/understat/
            ENG-Premier League/
                2425/
                    shot_events.parquet
                    team_match_stats.parquet
                ...
    """

    def __init__(
        self,
        leagues: Sequence[str] = TOP_5_EU_LEAGUES,
        seasons: Sequence[str] | str = "2425",
        cache_dir: Path | str = Path("data/cache/understat"),
        *,
        _reader_factory: ReaderFactory | None = None,
    ) -> None:
        self._leagues = tuple(leagues)
        self._seasons = (seasons,) if isinstance(seasons, str) else tuple(seasons)
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._reader_factory = _reader_factory or _default_reader_factory
        self._reader: _UnderstatReader | None = None

    # ----- reader lazy construction --------------------------------------

    def _get_reader(self) -> _UnderstatReader:
        if self._reader is None:
            logger.info(
                "understat_reader_init",
                leagues=self._leagues,
                seasons=self._seasons,
                cache_dir=str(self._cache_dir),
            )
            self._reader = self._reader_factory(
                leagues=list(self._leagues),
                seasons=list(self._seasons),
                data_dir=self._cache_dir,
            )
        return self._reader

    # ----- shot events ---------------------------------------------------

    def read_shot_events(self) -> pl.DataFrame:
        """Return all shot events for configured leagues+seasons as Polars DataFrame.

        The returned DataFrame conforms to `SHOT_EVENT_SCHEMA`. Columns from
        soccerdata that don't map to the canonical schema are dropped;
        missing canonical columns raise `ValueError`.
        """
        raw = self._get_reader().read_shot_events()
        return _normalize_shot_events(_to_polars(raw))

    # ----- team match stats ----------------------------------------------

    def read_team_match_stats(self) -> pl.DataFrame:
        """Return per-team-per-match xG aggregates as Polars DataFrame.

        The returned DataFrame conforms to `TEAM_MATCH_STATS_SCHEMA` with two
        rows per match (one for the home team, one for the away team).
        """
        raw = self._get_reader().read_team_match_stats()
        return _normalize_team_match_stats(_to_polars(raw))


# ---------------------------------------------------------------------------
# Default reader factory — imports soccerdata lazily so tests can run without it
# ---------------------------------------------------------------------------

def _default_reader_factory(
    *,
    leagues: list[str],
    seasons: list[str],
    data_dir: Path,
) -> _UnderstatReader:
    """Construct a real `soccerdata.Understat` instance.

    Imported here (not at module top) so that test environments without the
    soccerdata package installed can still import this module.
    """
    import soccerdata as sd  # type: ignore[import-untyped]

    return sd.Understat(leagues=leagues, seasons=seasons, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Normalization — soccerdata DataFrame → canonical Polars schema
# ---------------------------------------------------------------------------

def _to_polars(df: Any) -> pl.DataFrame:
    """Convert pandas DataFrame to Polars. Pass-through if already Polars."""
    if isinstance(df, pl.DataFrame):
        return df
    return pl.from_pandas(df.reset_index() if hasattr(df, "reset_index") else df)


_SHOT_RENAMES: dict[str, str] = {
    "id": "shot_id",
    "h_a": "team_side",
    "X": "x",
    "Y": "y",
    "xG": "xg",
    "h_team": "home_team",
    "a_team": "away_team",
    "shotType": "shot_type",
}


def _normalize_shot_events(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {src: dst for src, dst in _SHOT_RENAMES.items() if src in df.columns}
    if rename_map:
        df = df.rename(rename_map)

    if "is_penalty" not in df.columns and "situation" in df.columns:
        df = df.with_columns((pl.col("situation") == "Penalty").alias("is_penalty"))

    missing = [col for col in SHOT_EVENT_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(
            f"Understat shot events missing canonical columns: {missing}. "
            f"soccerdata may have changed its schema; update _SHOT_RENAMES "
            f"or SHOT_EVENT_SCHEMA in src/bip/data/understat_client.py."
        )

    return df.select(list(SHOT_EVENT_SCHEMA)).cast(SHOT_EVENT_SCHEMA)  # type: ignore[arg-type]


_TEAM_MATCH_RENAMES: dict[str, str] = {
    "xG": "xg",
    "npxG": "npxg",
    "xGA": "xg_against",
    "PPDA": "ppda",
}


def _normalize_team_match_stats(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {src: dst for src, dst in _TEAM_MATCH_RENAMES.items() if src in df.columns}
    if rename_map:
        df = df.rename(rename_map)

    missing = [col for col in TEAM_MATCH_STATS_SCHEMA if col not in df.columns]
    if missing:
        raise ValueError(
            f"Understat team match stats missing canonical columns: {missing}. "
            f"soccerdata may have changed its schema; update _TEAM_MATCH_RENAMES "
            f"or TEAM_MATCH_STATS_SCHEMA in src/bip/data/understat_client.py."
        )

    return df.select(list(TEAM_MATCH_STATS_SCHEMA)).cast(TEAM_MATCH_STATS_SCHEMA)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Team-form features — multi-dimensional feed for corners_poisson + goal model
# ---------------------------------------------------------------------------
#
# A single scalar `xg_total_l5` collapses information the gradient-boosting
# ensemble needs to see separately. Two teams with identical sums but very
# different shapes (consistent vs spiky, home-loaded vs away-loaded, vs strong
# vs weak defenses) should produce different predictions. We expose the five
# dimensions Glickman & Stern (1998), Held (2005), and Yip (2022) collectively
# argue matter, and let the ensemble learn the weighting.
#
# Dimensions:
#   1. Volume     — totals at l3, l5, l10 windows (npxG and xg_against)
#   2. Recency    — exponential-decay-weighted sum at l5 (β default 0.85;
#                   the *value* lives in this feature so Phase 3's Bayesian
#                   updater stays free to calibrate its own decay separately)
#   3. Venue      — home-only vs away-only per-match averages at l5
#   4. Variance   — stdev of npxG within l5 (form spikiness signal)
#   5. Quality    — xg_diff (own - opponent) at l5 captures opposition strength
#                   indirectly via what they generated against us
#
# Layer 2 will add opponent-strength-normalised xG (z-score vs league baseline)
# once the Phase 4 league_strength lookup is wired in. That feature is queued
# below as `xg_normalized_l5: None` placeholder.

DEFAULT_DECAY = 0.85  # Glickman & Stern range for match-to-match form
DEFAULT_WINDOWS: tuple[int, ...] = (3, 5, 10)


@dataclass(frozen=True)
class TeamFormFeatures:
    """Multi-dimensional team-form feature vector at a point in time.

    All values use npxG (penalties excluded) by convention. Use `xg_total_l5`
    if you need to include penalties — the corners and goal models prefer npxG.
    Returned by `compute_team_form()`.

    Sample-size handling: if `n_matches_used < window`, totals/decay still
    aggregate over what's available; means and stdev return 0.0 rather than
    NaN so downstream models don't have to special-case missing windows.
    Use `n_matches_used` as a confidence weight if needed.
    """

    # Volume
    npxg_total_l3: float
    npxg_total_l5: float
    npxg_total_l10: float
    xg_against_total_l5: float

    # Recency-weighted (β-decay)
    npxg_decay_l5: float
    decay_beta: float

    # Venue split (l5 window, per-match averages)
    npxg_per_match_home_l5: float
    npxg_per_match_away_l5: float

    # Variance
    npxg_stdev_l5: float

    # Quality (relative to opposition)
    xg_diff_l5: float

    # Sample size + raw xG fallback
    n_matches_used: int
    xg_total_l5: float  # includes penalties — kept for callers that don't want npxG

    # Layer 2 placeholder — set when league-strength normalization lands
    xg_normalized_l5: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Flat dict for direct insertion into a feature-row Polars DataFrame."""
        return asdict(self)


def compute_team_form(
    team_match_stats: pl.DataFrame,
    *,
    team: str,
    as_of: Any,
    decay_beta: float = DEFAULT_DECAY,
) -> TeamFormFeatures:
    """Build a `TeamFormFeatures` for `team` from rows strictly before `as_of`.

    The returned features are designed to be the inputs the gradient-boosting
    ensemble needs to learn corners + goals dynamics on its own — the function
    deliberately does NOT pre-collapse multiple windows or pre-weight venue
    contributions, since the model can do that better.

    Args:
        team_match_stats: DataFrame matching TEAM_MATCH_STATS_SCHEMA.
        team: Canonical team name as it appears in the `team` column.
        as_of: Anything comparable with the `date` column — only matches
            strictly before this date are included.
        decay_beta: Recency weight in `npxg_decay_l5` = Σ β^k · npxg_k for
            k=0..n-1 over the last `n` matches (most recent k=0). Default
            0.85 sits in Glickman & Stern's empirical range; tunable per
            league if Phase 5 backtest finds drift.

    Returns:
        `TeamFormFeatures` with all five dimensions populated. Empty windows
        contribute 0.0 — `n_matches_used` carries the confidence signal.
    """
    history = (
        team_match_stats
        .filter(pl.col("team") == team)
        .filter(pl.col("date") < as_of)
        .sort("date", descending=True)
    )
    n_used = min(history.height, 10)

    def head_npxg(n: int) -> list[float]:
        return history.head(n)["npxg"].cast(pl.Float64).to_list()

    npxg_l3 = head_npxg(3)
    npxg_l5 = head_npxg(5)
    npxg_l10 = head_npxg(10)
    xg_l5 = history.head(5)["xg"].cast(pl.Float64).to_list()
    xga_l5 = history.head(5)["xg_against"].cast(pl.Float64).to_list()

    # Decay sum — β^k weighting, k=0 = most recent
    decay_sum = sum(npxg * (decay_beta**k) for k, npxg in enumerate(npxg_l5))

    # Venue split on l5
    l5_window = history.head(5)
    home_npxg = (
        l5_window.filter(pl.col("venue") == "home")["npxg"].cast(pl.Float64).to_list()
    )
    away_npxg = (
        l5_window.filter(pl.col("venue") == "away")["npxg"].cast(pl.Float64).to_list()
    )

    # Variance — population stdev to keep stable with very small n; 0.0 if <2 samples
    npxg_stdev = _stdev_or_zero(npxg_l5)

    return TeamFormFeatures(
        npxg_total_l3=sum(npxg_l3),
        npxg_total_l5=sum(npxg_l5),
        npxg_total_l10=sum(npxg_l10),
        xg_against_total_l5=sum(xga_l5),
        npxg_decay_l5=decay_sum,
        decay_beta=decay_beta,
        npxg_per_match_home_l5=_mean_or_zero(home_npxg),
        npxg_per_match_away_l5=_mean_or_zero(away_npxg),
        npxg_stdev_l5=npxg_stdev,
        xg_diff_l5=sum(npxg_l5) - sum(xga_l5),
        n_matches_used=n_used,
        xg_total_l5=sum(xg_l5),
    )


def _mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev_or_zero(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(var)


# Thin convenience wrapper preserved for callers that only need the scalar.
# The realistic path through the codebase is `compute_team_form(...)`.
def aggregate_xg_last_n(
    team_match_stats: pl.DataFrame,
    *,
    team: str,
    as_of: Any,
    n: int = 5,
    use_npxg: bool = True,
) -> float:
    """Sum a team's (n)pxG over its last `n` matches strictly before `as_of`.

    Kept as a low-cardinality helper for places where only the scalar matters
    (e.g., diagnostic dashboards). For model features, prefer
    `compute_team_form()` which returns the full multi-dimensional vector.
    """
    column = "npxg" if use_npxg else "xg"
    window = (
        team_match_stats
        .filter(pl.col("team") == team)
        .filter(pl.col("date") < as_of)
        .sort("date", descending=True)
        .head(n)
    )
    if window.is_empty():
        return 0.0
    return float(window[column].sum())


# Silence "unused" linter on Literal — exposed for callers that want to opt
# venue into the future API without depending on a Polars literal.
VenueFilter = Literal["home", "away", "all"]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__: Iterable[str] = (
    "UnderstatClient",
    "SHOT_EVENT_SCHEMA",
    "TEAM_MATCH_STATS_SCHEMA",
    "TOP_5_EU_LEAGUES",
    "TOP_5_EU_PLUS_RFPL",
    "LEAGUE_EPL",
    "LEAGUE_LALIGA",
    "LEAGUE_SERIE_A",
    "LEAGUE_BUNDESLIGA",
    "LEAGUE_LIGUE_1",
    "LEAGUE_RFPL",
    "TeamFormFeatures",
    "compute_team_form",
    "aggregate_xg_last_n",
    "DEFAULT_DECAY",
    "DEFAULT_WINDOWS",
    "VenueFilter",
)
