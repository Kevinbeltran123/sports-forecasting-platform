"""ClvWorker — T+2h CLV measurement scaffold.

Sprint 2 Ola C. Reads predictions_raw rows with status='sent' AND
clv IS NULL AND match_datetime < now - 2h, and for each one attempts to:

  1. Fetch the Pinnacle closing odds (via an injected OddsClientProtocol)
  2. Fetch the match result (via an injected ResultClientProtocol)
  3. Compute CLV + P/L and UPDATE the row

Per memory `project_data_blocker`, the real Pinnacle integration is
deferred. This worker ships the LOOP + UPDATE machinery; the actual
fetches are protocol-based so Sprint 2+ can drop in real clients
without changing the worker.

Layer-1 behavior (default OddsClient = None): the worker enumerates
candidate rows and emits structured log events ("clv_would_measure")
without modifying the rows. This is intentionally observable so the
operator can see what WOULD happen when real data lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import structlog

from bip.clv.recorder import calculate_clv_percentage
from bip.pipeline.orchestrator import PREDICTIONS_RAW_TABLE
from bip.pipeline.protocols import SupabaseClientProtocol

log = structlog.get_logger(__name__)


DEFAULT_DELAY_AFTER_KICKOFF = timedelta(hours=2)


@runtime_checkable
class OddsClientProtocol(Protocol):
    """Protocol the worker uses to fetch Pinnacle closing odds.

    Implementations: bip.sports.football.client.ApiFootballClient-style
    wrapper around The Odds API (Sprint 2+ when data unblocked).
    """

    async def fetch_closing_odds(
        self,
        *,
        fixture_id: str,
        market: str,
    ) -> dict[str, float] | None: ...


@runtime_checkable
class ResultClientProtocol(Protocol):
    """Protocol for fetching match outcome."""

    async def fetch_result(self, *, fixture_id: str) -> dict[str, Any] | None: ...


@dataclass
class ClvRunSummary:
    n_candidate_rows: int = 0
    n_measured: int = 0
    n_skipped_no_data: int = 0
    n_errors: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)


def _grade_result(
    *, market: str, selection: str, result: dict[str, Any]
) -> str | None:
    """Coarse grader for the v4 markets currently in scope.

    Returns 'win' / 'loss' / 'void' / None (when grading is unsupported
    for the market). Real implementation lives in CLV recorder for the
    legacy stack; here we expose only what is needed for the goals/btts/
    totals/1x2 picks the v4 system emits.
    """
    home_goals = result.get("home_goals")
    away_goals = result.get("away_goals")
    if home_goals is None or away_goals is None:
        return None
    hg, ag = int(home_goals), int(away_goals)
    total = hg + ag

    if market == "1x2":
        if selection == "home":
            return "win" if hg > ag else "loss"
        if selection == "draw":
            return "win" if hg == ag else "loss"
        if selection == "away":
            return "win" if ag > hg else "loss"
    elif market.startswith("ou_"):
        try:
            line = float(market.split("_", 1)[1])
        except ValueError:
            return None
        if selection == "over":
            if total > line:
                return "win"
            if total < line:
                return "loss"
            return "push"
        if selection == "under":
            if total < line:
                return "win"
            if total > line:
                return "loss"
            return "push"
    elif market == "btts":
        btts = hg > 0 and ag > 0
        if selection == "yes":
            return "win" if btts else "loss"
        if selection == "no":
            return "win" if not btts else "loss"
    return None


def _pl_units(*, graded: str | None, stake_units: float, odds: float) -> float:
    if graded == "win":
        return stake_units * (odds - 1.0)
    if graded == "loss":
        return -stake_units
    return 0.0  # void / push / unknown


class ClvWorker:
    """Polls sent picks past T+2h and (optionally) records CLV + outcome."""

    def __init__(
        self,
        *,
        supabase_client: SupabaseClientProtocol,
        odds_client: OddsClientProtocol | None = None,
        result_client: ResultClientProtocol | None = None,
        kickoff_delay: timedelta = DEFAULT_DELAY_AFTER_KICKOFF,
        table_name: str = PREDICTIONS_RAW_TABLE,
    ) -> None:
        self._client = supabase_client
        self._odds_client = odds_client
        self._result_client = result_client
        self.kickoff_delay = kickoff_delay
        self._table_name = table_name

    async def run_once(self, *, now: datetime | None = None) -> ClvRunSummary:
        now = now or datetime.now(UTC)
        summary = ClvRunSummary()
        rows = self._fetch_candidates(now=now)
        summary.n_candidate_rows = len(rows)

        for row in rows:
            try:
                result = await self._measure(row)
            except Exception as exc:  # noqa: BLE001
                summary.n_errors += 1
                log.warning(
                    "clv_worker_row_error",
                    row_id=row.get("id"),
                    error=str(exc),
                )
                continue
            if result == "measured":
                summary.n_measured += 1
            else:
                summary.n_skipped_no_data += 1
                summary.skip_reasons[result] = (
                    summary.skip_reasons.get(result, 0) + 1
                )

        log.info(
            "clv_worker_run_complete",
            n_candidates=summary.n_candidate_rows,
            n_measured=summary.n_measured,
            n_skipped=summary.n_skipped_no_data,
            n_errors=summary.n_errors,
        )
        return summary

    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------

    def _fetch_candidates(self, *, now: datetime) -> list[dict]:
        cutoff = (now - self.kickoff_delay).isoformat()
        query = (
            self._client.table(self._table_name)
            .select("*")
            .eq("status", "sent")
            .is_("clv", "null")
            .lt("match_datetime", cutoff)
        )
        resp = query.execute()
        return list(getattr(resp, "data", None) or [])

    async def _measure(self, row: dict) -> str:
        """Returns 'measured' on success or a skip-reason string."""
        if self._odds_client is None:
            log.info(
                "clv_would_measure",
                row_id=row.get("id"),
                fixture_id=row.get("fixture_id"),
                market=row.get("market"),
                selection=row.get("selection"),
                reason="no_odds_client_injected_layer1",
            )
            return "no_odds_client_layer1"

        closing = await self._odds_client.fetch_closing_odds(
            fixture_id=str(row.get("fixture_id")),
            market=str(row.get("market")),
        )
        if not closing:
            return "no_closing_odds"

        try:
            clv_pct = calculate_clv_percentage(
                odds_at_pick=float(row.get("odds_at_pick", 0.0) or 0.0),
                closing_odds_dict=closing,
                selection=str(row.get("selection")),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "clv_calc_error",
                row_id=row.get("id"),
                error=str(exc),
            )
            return "clv_calc_error"

        # Result + P/L are optional — if the result client is unwired we
        # still record CLV alone.
        graded: str | None = None
        if self._result_client is not None:
            result = await self._result_client.fetch_result(
                fixture_id=str(row.get("fixture_id"))
            )
            if result:
                graded = _grade_result(
                    market=str(row.get("market")),
                    selection=str(row.get("selection")),
                    result=result,
                )

        stake = float(row.get("stake_units") or 0.0)
        odds = float(row.get("odds_at_pick") or 0.0)
        pl = _pl_units(graded=graded, stake_units=stake, odds=odds)

        payload: dict[str, Any] = {
            "clv": clv_pct,
            "clv_measured_at": datetime.now(UTC).isoformat(),
        }
        if graded is not None:
            payload["result"] = graded
            payload["pl_units"] = pl
        # closing odds (selection-specific): record the canonical key
        closing_key = self._resolve_closing_key(
            selection=str(row.get("selection")), closing=closing
        )
        if closing_key is not None:
            payload["closing_odds"] = float(closing[closing_key])

        try:
            (
                self._client.table(self._table_name)
                .update(payload)
                .eq("id", row.get("id"))
                .execute()
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("clv_update_error", row_id=row.get("id"), error=str(exc))
            return "update_error"
        return "measured"

    @staticmethod
    def _resolve_closing_key(*, selection: str, closing: dict[str, float]) -> str | None:
        sel = selection.strip().lower()
        if sel in closing:
            return sel
        for k in closing:
            if sel in k:
                return k
        return None


__all__ = [
    "ClvRunSummary",
    "ClvWorker",
    "OddsClientProtocol",
    "ResultClientProtocol",
]
