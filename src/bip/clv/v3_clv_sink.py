"""v3 live CLV parquet sink — measures CLV for DELIVERED v3 picks.

CLV is measured only on picks that were SENT (send_result="sent") as recorded
in deliveries.parquet from Wave 1 (DualWriteRuntime._send_alerts_for).

# Supported families
- goals, btts, totals, result_1x2 — Pinnacle closing odds available via
  OddsApiClient (find_event_by_fixture + fetch_pinnacle_closing_odds).

# Unsupported families
- corners, cards, next_goal — OddsApiClient has no corners/Betfair support.
  These families write a row with clv_percentage=None and
  reason="clv_unsupported_family" so downstream analysis sees coverage gaps.

# Partition convention
Rows are persisted to:
    data/cache/v3_shadow/dt=YYYY-MM-DD/clv.parquet
One partition per delivery date (delivery timestamp_utc). Append via the
same diagonal_relaxed strategy as shadow_logger to survive schema evolution.

# Scheduler wiring
The public callable is ``run_clv_sink_for_delivered_picks``. It is NOT wired
to a live scheduler in this commit — the orchestrator/scheduler should invoke
it at kickoff+ (after the closing window closes). Follow-up note: wire this
as a DateTrigger job in watch.py (same flush site as deliveries.parquet).

# Pure-function reuse
calculate_clv_percentage + compute_rolling_clv_average are imported from
bip.clv.recorder without modification. ClvRecorder/Supabase are NOT touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from bip.clv.recorder import calculate_clv_percentage, compute_rolling_clv_average

log = structlog.get_logger(__name__)

# Shadow root where deliveries.parquet is written by the delivery layer
# and where this sink appends clv.parquet siblings. Path is a layout
# convention shared with engine_v3.shadow_logger (which defines an
# identical constant); we redefine here to keep this module
# self-contained — see plan Sprint 0 Ola A (CLV decouple from engine_v3).
DEFAULT_SHADOW_ROOT = Path("data/cache/v3_shadow")

# ──────────────────────────────────────────────────────────────────────
# Supported CLV families — determined by OddsApiClient coverage
# ──────────────────────────────────────────────────────────────────────

_SUPPORTED_FAMILIES = frozenset({"goals", "btts", "totals", "result_1x2"})
_UNSUPPORTED_FAMILIES = frozenset({"corners", "cards", "next_goal"})

# Map internal family → Odds API market key
_FAMILY_TO_ODDS_API_MARKET: dict[str, str] = {
    "goals": "totals",
    "btts": "btts",
    "totals": "totals",
    "result_1x2": "h2h",
}

# The Odds API sport key for European football leagues
_DEFAULT_SPORT_KEY = "soccer_epl"


# ──────────────────────────────────────────────────────────────────────
# CLV record dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class V3ClvRecord:
    """One CLV measurement row for clv.parquet.

    clv_percentage is None when:
    - family is unsupported (corners/cards/next_goal)
    - Pinnacle event not found by find_event_by_fixture
    - Odds API returned no Pinnacle data for the market

    reason is populated only when clv_percentage is None to explain why.
    """

    fixture_id: int
    thesis_id: str
    family: str
    market_id: str
    direction: str
    bookmaker_odd: float | None
    delivery_ts: datetime
    odds_fetched_at: datetime | None
    clv_percentage: float | None
    reason: str | None  # "clv_unsupported_family" | "no_event_match" | "no_pinnacle_data" | None


# ──────────────────────────────────────────────────────────────────────
# Vig-removal / selection-key helpers
# ──────────────────────────────────────────────────────────────────────


def _extract_closing_odds_dict(bookmaker_dict: dict, market_key: str) -> dict[str, float] | None:
    """Extract outcome-keyed decimal odds from a Pinnacle bookmaker dict.

    The Odds API returns outcomes as a list of {name, price} objects.
    We normalise to lowercase name keys.

    Returns None when the market is absent or has fewer than 2 outcomes.
    """
    for market in bookmaker_dict.get("markets") or []:
        if market.get("key") == market_key:
            outcomes = market.get("outcomes") or []
            if len(outcomes) < 2:
                return None
            return {
                str(o["name"]).lower(): float(o["price"])
                for o in outcomes
                if "name" in o and "price" in o
            }
    return None


def _resolve_selection_key(direction: str, closing_dict: dict[str, float]) -> str | None:
    """Map a pick direction to its key in the closing_odds_dict.

    Tries direct match first, then heuristic substring match. Returns
    None if no suitable key is found.
    """
    dir_lc = direction.strip().lower()

    # Direct key match
    if dir_lc in closing_dict:
        return dir_lc

    # Heuristic: over/under in totals markets
    for key in closing_dict:
        if dir_lc in key:
            return key

    # Heuristic: home/away/draw in h2h markets
    direction_map = {"home": "home", "away": "away", "draw": "draw", "yes": "yes", "no": "no"}
    mapped = direction_map.get(dir_lc)
    if mapped:
        for key in closing_dict:
            if mapped in key:
                return key

    return None


# ──────────────────────────────────────────────────────────────────────
# Core CLV computation for one delivery row
# ──────────────────────────────────────────────────────────────────────


async def _compute_clv_for_row(
    row: dict[str, Any],
    *,
    odds_client: Any,
    sport_key: str,
    home_team: str,
    away_team: str,
    kickoff_utc: datetime,
) -> V3ClvRecord:
    """Attempt to compute CLV for one deliveries.parquet row.

    Returns a V3ClvRecord with clv_percentage set or None + reason.
    """
    family = str(row.get("family") or "")
    thesis_id = str(row.get("thesis_id") or "")
    fixture_id = int(row.get("fixture_id") or 0)
    market_id = str(row.get("market_id") or "")
    direction = str(row.get("direction") or "")
    book_odd = row.get("bookmaker_odd")
    if book_odd is not None:
        book_odd = float(book_odd)
    delivery_ts = row.get("timestamp_utc")
    if not isinstance(delivery_ts, datetime):
        delivery_ts = datetime.now(UTC)

    def _record(clv_pct: float | None, reason: str | None, fetched_at: datetime | None = None) -> V3ClvRecord:
        return V3ClvRecord(
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            family=family,
            market_id=market_id,
            direction=direction,
            bookmaker_odd=book_odd,
            delivery_ts=delivery_ts,
            odds_fetched_at=fetched_at,
            clv_percentage=clv_pct,
            reason=reason,
        )

    # Fast path: unsupported family
    if family not in _SUPPORTED_FAMILIES:
        log.info(
            "clv_unsupported_family",
            fixture_id=fixture_id,
            family=family,
            thesis_id=thesis_id,
        )
        return _record(None, "clv_unsupported_family")

    # No bookmaker odd → can't compute CLV
    if book_odd is None or book_odd <= 1.0:
        return _record(None, "no_bookmaker_odd")

    # Resolve The Odds API event id
    try:
        event_id = await odds_client.find_event_by_fixture(
            sport_key=sport_key,
            home_team=home_team,
            away_team=away_team,
            kickoff_utc=kickoff_utc,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "clv_find_event_error",
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            error=str(exc),
        )
        return _record(None, "event_lookup_error")

    if event_id is None:
        log.info(
            "clv_no_event_match",
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            home_team=home_team,
            away_team=away_team,
        )
        return _record(None, "no_event_match")

    # Fetch Pinnacle closing odds
    odds_api_market = _FAMILY_TO_ODDS_API_MARKET.get(family, "h2h")
    fetched_at = datetime.now(UTC)
    try:
        pinnacle_dict = await odds_client.fetch_pinnacle_closing_odds(
            sport_key=sport_key,
            event_id=event_id,
            market_key=odds_api_market,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "clv_fetch_odds_error",
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            event_id=event_id,
            error=str(exc),
        )
        return _record(None, "odds_fetch_error", fetched_at)

    if pinnacle_dict is None:
        return _record(None, "no_pinnacle_data", fetched_at)

    closing_dict = _extract_closing_odds_dict(pinnacle_dict, odds_api_market)
    if closing_dict is None or len(closing_dict) < 2:
        return _record(None, "no_pinnacle_data", fetched_at)

    selection_key = _resolve_selection_key(direction, closing_dict)
    if selection_key is None:
        log.info(
            "clv_no_selection_match",
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            direction=direction,
            closing_keys=sorted(closing_dict.keys()),
        )
        return _record(None, "no_selection_match", fetched_at)

    try:
        clv_pct = calculate_clv_percentage(
            odds_at_pick=book_odd,
            closing_odds_dict=closing_dict,
            selection=selection_key,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "clv_calculation_error",
            fixture_id=fixture_id,
            thesis_id=thesis_id,
            error=str(exc),
        )
        return _record(None, "calculation_error", fetched_at)

    log.info(
        "clv_computed",
        fixture_id=fixture_id,
        thesis_id=thesis_id,
        family=family,
        direction=direction,
        book_odd=book_odd,
        clv_percentage=round(clv_pct, 4),
        selection_key=selection_key,
    )
    return _record(clv_pct, None, fetched_at)


# ──────────────────────────────────────────────────────────────────────
# Parquet persistence
# ──────────────────────────────────────────────────────────────────────


def _date_partition(ts: datetime) -> str:
    return ts.strftime("dt=%Y-%m-%d")


def _to_row(rec: V3ClvRecord) -> dict[str, Any]:
    return {
        "fixture_id": int(rec.fixture_id),
        "thesis_id": rec.thesis_id,
        "family": rec.family,
        "market_id": rec.market_id,
        "direction": rec.direction,
        "bookmaker_odd": (float(rec.bookmaker_odd) if rec.bookmaker_odd is not None else None),
        "delivery_ts": rec.delivery_ts,
        "odds_fetched_at": rec.odds_fetched_at,
        "clv_percentage": (float(rec.clv_percentage) if rec.clv_percentage is not None else None),
        "reason": rec.reason,
    }


def _append_clv_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append CLV rows to a daily partition parquet using diagonal_relaxed."""
    import polars as pl

    new_df = pl.DataFrame(rows)
    if path.exists():
        try:
            existing = pl.read_parquet(path)
            combined = pl.concat([existing, new_df], how="diagonal_relaxed")
        except Exception:  # noqa: BLE001
            combined = new_df
    else:
        combined = new_df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path)


def write_clv_records(
    records: list[V3ClvRecord],
    shadow_root: Path = DEFAULT_SHADOW_ROOT,
) -> dict[str, Path]:
    """Persist V3ClvRecord list to daily clv.parquet partitions.

    Returns dict of partition-key → path written (one per day encountered).
    """
    by_partition: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        key = _date_partition(rec.delivery_ts)
        by_partition.setdefault(key, []).append(_to_row(rec))

    written: dict[str, Path] = {}
    for part, rows in sorted(by_partition.items()):
        path = shadow_root / part / "clv.parquet"
        _append_clv_parquet(path, rows)
        written[part] = path
        log.info(
            "clv_parquet_written",
            partition=part,
            n_rows=len(rows),
            path=str(path),
        )
    return written


# ──────────────────────────────────────────────────────────────────────
# Rolling average helper (reuses compute_rolling_clv_average)
# ──────────────────────────────────────────────────────────────────────


def rolling_clv_from_parquet(shadow_root: Path = DEFAULT_SHADOW_ROOT) -> float | None:
    """Compute rolling 50-pick CLV average from all clv.parquet partitions.

    Returns None when no graded CLV rows exist yet (first run). Returns
    the rolling average (may be negative) when data is present.
    """
    import polars as pl

    root = Path(shadow_root)
    if not root.exists():
        return None

    all_pcts: list[float] = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("dt="):
            continue
        clv_path = part_dir / "clv.parquet"
        if not clv_path.exists():
            continue
        try:
            df = pl.read_parquet(clv_path)
            if "clv_percentage" in df.columns:
                pcts = [
                    float(v)
                    for v in df["clv_percentage"].drop_nulls().to_list()
                ]
                all_pcts.extend(pcts)
        except Exception:  # noqa: BLE001
            continue

    if not all_pcts:
        return None
    return compute_rolling_clv_average(all_pcts)


# ──────────────────────────────────────────────────────────────────────
# Public callable — invoke at kickoff+
# ──────────────────────────────────────────────────────────────────────


async def run_clv_sink_for_delivered_picks(
    *,
    date_iso: str,
    odds_client: Any,
    fixture_metadata: dict[int, dict[str, Any]],
    shadow_root: Path = DEFAULT_SHADOW_ROOT,
    sport_key: str = _DEFAULT_SPORT_KEY,
) -> tuple[list[V3ClvRecord], dict[str, Path]]:
    """Measure CLV for every SENT delivery on date_iso.

    This is the callable the orchestrator/scheduler invokes at kickoff+.
    It is NOT wired to a live scheduler in this commit — wiring is a
    follow-up: add a DateTrigger job at kickoff+5min in watch.py using
    the same flush site as deliveries.parquet.

    Args:
        date_iso: Partition date string e.g. "2026-05-15".
        odds_client: OddsApiClient instance (or mock for tests).
        fixture_metadata: Mapping of fixture_id → dict with keys:
            "home_team" (str), "away_team" (str), "kickoff_utc" (datetime).
            Used to call find_event_by_fixture. Callers build this from
            the Sportmonks fixture objects they already have.
        shadow_root: Root of the v3_shadow partition tree.
        sport_key: The Odds API sport key (default: soccer_epl).

    Returns:
        (records, written_paths) — list of V3ClvRecord computed for every
        SENT delivery row + dict of partition paths written.

    NOTE: Only rows with send_result="sent" are processed. Skipped/failed
    deliveries are excluded — CLV is measured on what was actually bet.
    """
    import polars as pl

    partition_dir = shadow_root / f"dt={date_iso}"
    deliveries_path = partition_dir / "deliveries.parquet"

    if not deliveries_path.exists():
        log.info("clv_sink_no_deliveries", date=date_iso, path=str(deliveries_path))
        return [], {}

    df = pl.read_parquet(deliveries_path)
    # Only process SENT picks — CLV is measured on what was actually delivered
    sent_df = df.filter(pl.col("send_result") == "sent")
    rows = list(sent_df.iter_rows(named=True))

    if not rows:
        log.info("clv_sink_no_sent_rows", date=date_iso)
        return [], {}

    log.info("clv_sink_processing", date=date_iso, n_sent=len(rows))

    records: list[V3ClvRecord] = []
    for row in rows:
        fixture_id = int(row.get("fixture_id") or 0)
        meta = fixture_metadata.get(fixture_id)
        if meta is None:
            log.warning(
                "clv_sink_no_fixture_metadata",
                fixture_id=fixture_id,
                thesis_id=row.get("thesis_id"),
            )
            records.append(
                V3ClvRecord(
                    fixture_id=fixture_id,
                    thesis_id=str(row.get("thesis_id") or ""),
                    family=str(row.get("family") or ""),
                    market_id=str(row.get("market_id") or ""),
                    direction=str(row.get("direction") or ""),
                    bookmaker_odd=(
                        float(row["bookmaker_odd"])
                        if row.get("bookmaker_odd") is not None
                        else None
                    ),
                    delivery_ts=(
                        row["timestamp_utc"]
                        if isinstance(row.get("timestamp_utc"), datetime)
                        else datetime.now(UTC)
                    ),
                    odds_fetched_at=None,
                    clv_percentage=None,
                    reason="no_fixture_metadata",
                )
            )
            continue

        rec = await _compute_clv_for_row(
            row,
            odds_client=odds_client,
            sport_key=sport_key,
            home_team=meta["home_team"],
            away_team=meta["away_team"],
            kickoff_utc=meta["kickoff_utc"],
        )
        records.append(rec)

    written = write_clv_records(records, shadow_root=shadow_root)
    log.info(
        "clv_sink_complete",
        date=date_iso,
        n_records=len(records),
        n_supported=sum(1 for r in records if r.clv_percentage is not None),
        n_unsupported=sum(1 for r in records if r.reason == "clv_unsupported_family"),
    )
    return records, written


__all__ = [
    "V3ClvRecord",
    "rolling_clv_from_parquet",
    "run_clv_sink_for_delivered_picks",
    "write_clv_records",
]
