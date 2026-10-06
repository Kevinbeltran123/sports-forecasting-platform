"""Repository classes for Supabase CRUD operations.

Each repository wraps one table and provides typed methods for
insert, select, update, and upsert operations. Domain logic never
constructs Supabase queries directly — it goes through these repositories.

All operations are wrapped in error handling that converts Supabase
exceptions to StorageError with context (table name, operation).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from bip.core.errors import StorageError
from bip.core.storage.models import (
    ClvRecord,
    OddsSnapshot,
    PerformanceMetric,
    Pick,
    Prediction,
    Result,
)
from bip.core.types import AggregationPeriod
from supabase import Client


@dataclass
class PredictionRepository:
    """Repository for the predictions table."""

    client: Client

    def insert(self, prediction: Prediction) -> dict:
        """Insert a prediction and return the created record."""
        try:
            response = (
                self.client.table("predictions")
                .insert(prediction.to_supabase_dict())
                .execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to insert into predictions: {e}") from e

    def get_by_fixture(
        self, fixture_id: int, market: str | None = None, sport: str | None = None
    ) -> list[dict]:
        """Get predictions for a fixture, optionally filtered by market and sport."""
        try:
            query = (
                self.client.table("predictions").select("*").eq("fixture_id", fixture_id)
            )
            if market is not None:
                query = query.eq("market", market)
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from predictions: {e}") from e

    def get_latest(self, fixture_id: int, market: str) -> dict | None:
        """Get the most recent prediction for a fixture/market combo."""
        try:
            data = (
                self.client.table("predictions")
                .select("*")
                .eq("fixture_id", fixture_id)
                .eq("market", market)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            ).data
            return data[0] if data else None
        except Exception as e:
            raise StorageError(f"Failed to select from predictions: {e}") from e

    def get_production(
        self, fixture_id: int, market: str | None = None
    ) -> list[dict]:
        """Get non-shadow predictions for a fixture (Phase 3 pick engine reads these) — ML-05."""
        try:
            query = (
                self.client.table("predictions")
                .select("*")
                .eq("fixture_id", fixture_id)
                .eq("is_shadow", False)
            )
            if market is not None:
                query = query.eq("market", market)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from predictions: {e}") from e


@dataclass
class PickRepository:
    """Repository for the picks table."""

    client: Client

    def insert(self, pick: Pick) -> dict:
        """Insert a pick and return the created record."""
        try:
            response = (
                self.client.table("picks").insert(pick.to_supabase_dict()).execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to insert into picks: {e}") from e

    def get_by_fixture(self, fixture_id: int) -> list[dict]:
        """Get all picks for a fixture."""
        try:
            return (
                self.client.table("picks")
                .select("*")
                .eq("fixture_id", fixture_id)
                .execute()
            ).data
        except Exception as e:
            raise StorageError(f"Failed to select from picks: {e}") from e

    def get_by_status(self, status: str, sport: str | None = None) -> list[dict]:
        """Get all picks with a given status, optionally filtered by sport."""
        try:
            query = self.client.table("picks").select("*").eq("status", status)
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from picks: {e}") from e

    def update_status(self, pick_id: int, status: str) -> dict:
        """Update the status of a pick."""
        try:
            response = (
                self.client.table("picks")
                .update({"status": status})
                .eq("id", pick_id)
                .execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to update picks: {e}") from e

    def get_window_picks(self, sport: str, hours: int = 168) -> list[dict]:
        """Return SENT picks (not filtered/rejected) for sport in the trailing `hours` window.

        D-09: backs the rolling 168h market-cap query. Uses idx_picks_sport_market_created
        (migration 004) for sub-50ms response. Filters status NOT IN ('filtered', 'rejected')
        — the cap counts what was actually SENT, not attempts.
        """
        from datetime import UTC, datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        try:
            response = (
                self.client.table("picks")
                .select("market, status, created_at")
                .eq("sport", sport)
                .gte("created_at", cutoff)
                .neq("status", "filtered")
                .neq("status", "rejected")
                .execute()
            )
            return response.data or []
        except Exception as e:
            raise StorageError(f"Failed to read picks window for sport={sport}: {e}") from e

    def get_pending_for_fixture(self, fixture_id: int) -> list[dict]:
        """Return all picks with status='pending' for a fixture.

        D-16: result reconciliation reads these and updates won/lost/void/push.
        """
        try:
            response = (
                self.client.table("picks")
                .select("*")
                .eq("fixture_id", fixture_id)
                .eq("status", "pending")
                .execute()
            )
            return response.data or []
        except Exception as e:
            raise StorageError(f"Failed to read pending picks for fixture={fixture_id}: {e}") from e

    def update_status_by_fixture(self, fixture_id: int, new_status: str) -> int:
        """Bulk-update all pending picks for a fixture to `new_status`. Returns row count.

        D-16: used for void path (PST/CANC/ABD).
        """
        try:
            response = (
                self.client.table("picks")
                .update({"status": new_status})
                .eq("fixture_id", fixture_id)
                .eq("status", "pending")
                .execute()
            )
            return len(response.data or [])
        except Exception as e:
            raise StorageError(
                f"Failed to update picks for fixture={fixture_id} → {new_status}: {e}"
            ) from e

    def query_pending_sends(self, sport: str, max_age_minutes: int = 30) -> list[dict]:
        """Return pending picks claude-validated within `max_age_minutes` (Pitfall 6 recovery).

        Pitfall 6: MemoryJobStore loses DateTrigger send jobs on restart. On startup,
        the orchestrator queries this method and re-queues immediate sends for picks
        that were validated but never made it to Telegram before the restart.
        Filter: status='pending' AND claude_validation IS NOT NULL AND created_at > now()-Xmin.
        """
        from datetime import UTC, datetime, timedelta
        cutoff = (datetime.now(UTC) - timedelta(minutes=max_age_minutes)).isoformat()
        try:
            response = (
                self.client.table("picks")
                .select("*")
                .eq("sport", sport)
                .eq("status", "pending")
                .not_.is_("claude_validation", "null")
                .gte("created_at", cutoff)
                .execute()
            )
            return response.data or []
        except Exception as e:
            raise StorageError(f"Failed to query pending sends for sport={sport}: {e}") from e


@dataclass
class OddsSnapshotRepository:
    """Repository for the odds_snapshots table."""

    client: Client

    def insert(self, snapshot: OddsSnapshot) -> dict:
        """Insert an odds snapshot and return the created record."""
        try:
            response = (
                self.client.table("odds_snapshots")
                .insert(snapshot.to_supabase_dict())
                .execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to insert into odds_snapshots: {e}") from e

    def get_by_fixture(
        self, fixture_id: int, market: str | None = None, sport: str | None = None
    ) -> list[dict]:
        """Get odds snapshots for a fixture, optionally filtered by market and sport."""
        try:
            query = (
                self.client.table("odds_snapshots").select("*").eq("fixture_id", fixture_id)
            )
            if market is not None:
                query = query.eq("market", market)
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from odds_snapshots: {e}") from e

    def get_closing(self, fixture_id: int, market: str) -> dict | None:
        """Get the closing odds snapshot for a fixture/market."""
        try:
            data = (
                self.client.table("odds_snapshots")
                .select("*")
                .eq("fixture_id", fixture_id)
                .eq("market", market)
                .eq("is_closing", True)
                .execute()
            ).data
            return data[0] if data else None
        except Exception as e:
            raise StorageError(f"Failed to select from odds_snapshots: {e}") from e


@dataclass
class ResultRepository:
    """Repository for the results table."""

    client: Client

    def insert(self, result: Result) -> dict:
        """Insert a result and return the created record."""
        try:
            response = (
                self.client.table("results").insert(result.to_supabase_dict()).execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to insert into results: {e}") from e

    def get_by_fixture(self, fixture_id: int) -> dict | None:
        """Get the result for a fixture (unique per fixture)."""
        try:
            data = (
                self.client.table("results")
                .select("*")
                .eq("fixture_id", fixture_id)
                .execute()
            ).data
            return data[0] if data else None
        except Exception as e:
            raise StorageError(f"Failed to select from results: {e}") from e

    def get_by_league(
        self, league: str, sport: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Get results for a league with optional sport filter and limit."""
        try:
            query = (
                self.client.table("results").select("*").eq("league", league).limit(limit)
            )
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from results: {e}") from e


@dataclass
class ClvRecordRepository:
    """Repository for the clv_records table."""

    client: Client

    def insert(self, record: ClvRecord) -> dict:
        """Insert a CLV record and return the created record."""
        try:
            response = (
                self.client.table("clv_records")
                .insert(record.to_supabase_dict())
                .execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to insert into clv_records: {e}") from e

    def get_by_pick(self, pick_id: int) -> dict | None:
        """Get the CLV record for a pick (unique per pick)."""
        try:
            data = (
                self.client.table("clv_records")
                .select("*")
                .eq("pick_id", pick_id)
                .execute()
            ).data
            return data[0] if data else None
        except Exception as e:
            raise StorageError(f"Failed to select from clv_records: {e}") from e

    def get_by_market(
        self, market: str, sport: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Get CLV records for a market with optional sport filter and limit."""
        try:
            query = (
                self.client.table("clv_records").select("*").eq("market", market).limit(limit)
            )
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from clv_records: {e}") from e

    def last_n_settled(self, n: int = 50, market: str | None = None) -> list[dict]:
        """Return the last n CLV records joined with picks.status (settled only).

        D-09: feeds compute_rolling_clv_average via ClvTrendChecker. PostgREST
        nested-resource syntax `picks!inner(status)` performs an INNER JOIN to
        picks; combined with .neq filters this excludes non-settled rows
        (filtered/rejected/pending). Per-market filter is OPTIONAL (D-10 — global
        scope when market is None).
        """
        try:
            query = (
                self.client.table("clv_records")
                .select("clv_percentage, market, created_at, picks!inner(status)")
                .neq("picks.status", "filtered")
                .neq("picks.status", "rejected")
                .neq("picks.status", "pending")
                .order("created_at", desc=True)
                .limit(n)
            )
            if market is not None:
                query = query.eq("market", market)
            return query.execute().data or []
        except Exception as e:
            raise StorageError(f"Failed to query last_n_settled clv_records: {e}") from e


@dataclass
class PerformanceMetricRepository:
    """Repository for the performance_metrics table."""

    client: Client

    def upsert(self, metric: PerformanceMetric) -> dict:
        """Upsert a performance metric (insert or update on conflict)."""
        try:
            response = (
                self.client.table("performance_metrics")
                .upsert(
                    metric.to_supabase_dict(),
                    on_conflict="league,market,period,period_start",
                )
                .execute()
            )
            return response.data[0]
        except Exception as e:
            raise StorageError(f"Failed to upsert into performance_metrics: {e}") from e

    def compute_period(
        self,
        sport: str,
        league: str,
        market: str,
        period: AggregationPeriod,
        period_start: date,
        period_end: date,
    ) -> PerformanceMetric:
        """D-16: single Supabase round trip for one (sport,league,market) over a window.

        Calls Postgres function compute_performance_period(p_sport,p_league,p_market,p_start,p_end)
        added by migration 005. LEFT JOIN on clv_records — picks with no CLV row (D-02
        Odds API failure path) are still counted; AVG ignores NULL clv_percentage
        (Pitfall 3).

        T-4-03 mitigation: parameters are bound via the dict argument to .rpc(), NOT
        interpolated. Migration 005's function is LANGUAGE sql STABLE — no EXECUTE.
        """
        try:
            response = self.client.rpc(
                "compute_performance_period",
                {
                    "p_sport":  sport,
                    "p_league": league,
                    "p_market": market,
                    "p_start":  datetime.combine(period_start, time.min, tzinfo=UTC).isoformat(),
                    "p_end":    datetime.combine(period_end, time.min, tzinfo=UTC).isoformat(),
                },
            ).execute()
        except Exception as e:
            raise StorageError(
                f"Failed compute_performance_period(sport={sport},league={league},"
                f"market={market}): {e}"
            ) from e

        row = response.data[0] if response.data else {}
        return PerformanceMetric(
            sport=sport, league=league, market=market,
            period=period, period_start=period_start, period_end=period_end,
            total_picks=int(row.get("total_picks") or 0),
            won=int(row.get("won") or 0),
            lost=int(row.get("lost") or 0),
            void=int(row.get("void") or 0),
            total_staked=float(row.get("total_staked") or 0.0),
            total_pnl=float(row.get("total_pnl") or 0.0),
            roi=float(row["roi"]) if row.get("roi") is not None else None,
            yield_pct=(float(row["roi"]) * 100.0) if row.get("roi") is not None else None,
            avg_clv=float(row["avg_clv"]) if row.get("avg_clv") is not None else None,
            avg_edge=float(row["avg_edge"]) if row.get("avg_edge") is not None else None,
        )

    def get_by_league_market(
        self, league: str, market: str, sport: str | None = None
    ) -> list[dict]:
        """Get performance metrics for a league/market combination."""
        try:
            query = (
                self.client.table("performance_metrics")
                .select("*")
                .eq("league", league)
                .eq("market", market)
            )
            if sport is not None:
                query = query.eq("sport", sport)
            return query.execute().data
        except Exception as e:
            raise StorageError(f"Failed to select from performance_metrics: {e}") from e
