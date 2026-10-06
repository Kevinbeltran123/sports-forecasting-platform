"""MetricsAggregator + drift compute (D-13/D-14/D-15/D-16).

D-13: daily cron computes/upserts metrics for all four periods in one pass.
D-14: weekly drift compares 4-week rolling-mean CLV vs prior 4-week mean.
D-16: aggregator pushes math to Postgres via PerformanceMetricRepository.compute_period;
      drift check uses Polars over rows fetched via PickRepository + ClvRecordRepository.

Sport-agnostic per CORE-02 — every public function takes `sport` as a parameter.

RESEARCH §Drift Statistical Method:
- D-14's stdev-of-per-pick gate is unstable at n=30 (heavy-tailed CLV).
- Mitigation: compute stdev over DAILY-MEAN CLV (n~28 daily means) and floor at
  drift_stdev_floor_pp = 1.0pp (Pitfall 4 prevents zero-stdev tight gate).

WARNING 4 fix (datetime parsing):
- Supabase returns `created_at` as ISO8601 strings with TZ offset (e.g.
  "2026-05-01T12:34:56.789+00:00").
- pl.col("created_at").cast(pl.Datetime) silently produces a NULL column on
  these strings (Polars cast does not parse TZ-bearing ISO strings).
- The fix: pl.col("created_at").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%:z", strict=False).
- strict=False keeps already-parsed datetime values flowing through unchanged
  (in-process tests passing native `datetime` objects).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
import structlog

from bip.core.storage.models import AggregationPeriod, PerformanceMetric
from bip.core.storage.repositories import PerformanceMetricRepository

logger = structlog.get_logger(__name__)


# Supabase ISO8601 format: "2026-05-01T12:34:56.789+00:00"
# %.f handles optional fractional seconds; %:z handles +HH:MM offset.
_SUPABASE_DT_FORMAT = "%Y-%m-%dT%H:%M:%S%.f%:z"


# -----------------------------------------------------------------
# Drift result + computer (D-14, RESEARCH §Drift Statistical Method)
# -----------------------------------------------------------------

@dataclass(frozen=True)
class DriftResult:
    """Output of compute_weekly_drift — enough info for ops Telegram message."""

    sport: str
    league: str
    market: str
    recent_window_start: date
    recent_window_end: date
    recent_mean_pp: float
    recent_n: int
    prior_window_start: date
    prior_window_end: date
    prior_mean_pp: float
    prior_n: int
    delta_pp: float            # recent_mean - prior_mean (negative = drop)
    stdev_pp: float            # stdev of DAILY MEANS in prior window (FLOORED)
    triggered_hard: bool
    triggered_soft: bool

    @property
    def triggered(self) -> bool:
        return self.triggered_hard or self.triggered_soft


def _parse_created_at(df: pl.DataFrame) -> pl.DataFrame:
    """WARNING 4 fix: parse created_at column from Supabase ISO8601 strings to Datetime.

    Handles three input shapes:
    - String column with TZ offset (production Supabase rows): use str.to_datetime
      with format "%Y-%m-%dT%H:%M:%S%.f%:z".
    - Datetime column (in-process tests passing native datetime objects): no-op.
    - Mixed/Object column: coerce via str.to_datetime with strict=False (rows that
      can't parse become null and get filtered out by the window filters below).
    """
    dtype = df.schema.get("created_at")
    if dtype == pl.Datetime or (dtype is not None and dtype.base_type() == pl.Datetime):
        return df
    return df.with_columns(
        pl.col("created_at").cast(pl.Utf8).str.to_datetime(_SUPABASE_DT_FORMAT, strict=False)
    )


def compute_weekly_drift(
    clv_rows: list[dict],
    sport: str,
    league: str,
    market: str,
    today: date,
    min_picks: int = 30,
    abs_threshold_pp: float = 2.0,
    stdev_multiplier: float = 1.5,
    stdev_floor_pp: float = 1.0,
) -> DriftResult | None:
    """D-14: 4-week rolling mean CLV vs prior 4-week mean.

    clv_rows: list of dicts with keys 'sport', 'league', 'market', 'created_at',
    'clv_percentage'. `created_at` may be either an ISO8601 string (Supabase
    response shape) or a native datetime object (in-process tests).
    Caller fetches via repos joining clv_records + picks.

    Returns None if either window has < min_picks settled picks (no false alarm).
    Otherwise returns a DriftResult with both hard/soft trigger flags set.
    """
    if not clv_rows:
        return None

    df = pl.DataFrame(clv_rows).filter(
        (pl.col("sport") == sport)
        & (pl.col("league") == league)
        & (pl.col("market") == market)
    )
    if df.height == 0:
        return None

    recent_start = today - timedelta(days=28)
    prior_start = today - timedelta(days=56)

    # WARNING 4 fix: parse Supabase ISO8601 strings (with +00:00 offset) properly.
    df = _parse_created_at(df)

    recent_start_dt = datetime.combine(recent_start, datetime.min.time(), tzinfo=UTC)
    prior_start_dt = datetime.combine(prior_start, datetime.min.time(), tzinfo=UTC)

    recent = df.filter(pl.col("created_at") >= recent_start_dt)
    prior = df.filter(
        (pl.col("created_at") >= prior_start_dt)
        & (pl.col("created_at") < recent_start_dt)
    )

    if recent.height < min_picks or prior.height < min_picks:
        return None  # insufficient sample — D-14 explicit gate

    recent_mean = float(recent["clv_percentage"].mean() or 0.0)
    prior_mean = float(prior["clv_percentage"].mean() or 0.0)

    # RESEARCH §Drift Statistical Method: daily-mean stdev (NOT per-pick stdev).
    prior_daily = (
        prior.with_columns(pl.col("created_at").dt.date().alias("day"))
        .group_by("day")
        .agg(pl.col("clv_percentage").mean().alias("day_mean"))
    )
    raw_stdev = float(prior_daily["day_mean"].std() or 0.0)
    stdev = max(raw_stdev, stdev_floor_pp)  # FLOOR — Pitfall 4

    delta = recent_mean - prior_mean  # negative = drop
    triggered_hard = delta <= -abs_threshold_pp
    triggered_soft = delta <= -stdev_multiplier * stdev

    return DriftResult(
        sport=sport, league=league, market=market,
        recent_window_start=recent_start, recent_window_end=today,
        recent_mean_pp=recent_mean, recent_n=recent.height,
        prior_window_start=prior_start, prior_window_end=recent_start,
        prior_mean_pp=prior_mean, prior_n=prior.height,
        delta_pp=delta, stdev_pp=stdev,
        triggered_hard=triggered_hard, triggered_soft=triggered_soft,
    )


# -----------------------------------------------------------------
# Daily aggregator (D-13, D-15, D-16)
# -----------------------------------------------------------------

_PERIODS: tuple[tuple[AggregationPeriod, int], ...] = (
    ("daily", 1),
    ("weekly", 7),
    ("monthly", 30),
    ("all_time", 0),  # 0 = special-cased to a fixed start date
)
_ALL_TIME_START: date = date(2020, 1, 1)


def _period_ranges(today: date) -> list[tuple[AggregationPeriod, date, date]]:
    """Compute (period, start, end) tuples for the four daily-cron periods.

    All ranges are HALF-OPEN [start, end) — matches migration 005's >= AND <.
    """
    out: list[tuple[AggregationPeriod, date, date]] = []
    for period, days in _PERIODS:
        if period == "all_time":
            out.append((period, _ALL_TIME_START, today))
        elif period == "daily":
            # Yesterday — D-13.
            out.append((period, today - timedelta(days=1), today))
        else:
            out.append((period, today - timedelta(days=days), today))
    return out


def compute_daily_metrics(
    perf_repo: PerformanceMetricRepository,
    sport: str,
    league: str,
    market: str,
    today: date,
) -> list[PerformanceMetric]:
    """Return PerformanceMetric for daily/weekly/monthly/all_time.

    Caller upserts (this stays pure for testability — D-15).
    """
    out: list[PerformanceMetric] = []
    for period, start, end in _period_ranges(today):
        metric = perf_repo.compute_period(
            sport=sport, league=league, market=market,
            period=period, period_start=start, period_end=end,
        )
        out.append(metric)
    return out


class MetricsAggregator:
    """D-13 cron entrypoint. Iterates (sport,league,market) and upserts metrics.

    Sport-agnostic per CORE-02 — `_discover_tuples` takes no sport branching.
    For Phase 4 v1 (football/1X2 only), the discovery returns the configured
    league x market product. Phase 6 corners adds rows naturally.
    """

    def __init__(
        self,
        perf_repo: PerformanceMetricRepository,
        pick_repo: Any,
    ) -> None:
        self._perf_repo = perf_repo
        self._pick_repo = pick_repo

    async def run(self, today: date | None = None) -> int:
        today = today or datetime.now(UTC).date()
        upserted = 0
        for sport, league, market in self._discover_tuples():
            metrics = compute_daily_metrics(self._perf_repo, sport, league, market, today)
            for m in metrics:
                self._perf_repo.upsert(m)
                upserted += 1
        logger.info("metrics_aggregation_complete", upserted=upserted, date=today.isoformat())
        return upserted

    def _discover_tuples(self) -> list[tuple[str, str, str]]:
        """Discover (sport,league,market) tuples to aggregate.

        v1: hardcoded football x 5 leagues x 1X2. Phase 6 / Phase 7 extend by
        either pulling from `picks.distinct(sport,league,market)` or by config.
        """
        leagues = ("premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1")
        return [("football", league, "onextwo") for league in leagues]
