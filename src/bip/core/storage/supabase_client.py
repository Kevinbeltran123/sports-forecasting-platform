"""Supabase client factory + predictions_raw helpers.

Creates a Supabase Client from application settings.
The caller is responsible for caching the instance if needed.

Sprint 0 Ola B adds thin helpers for the new `predictions_raw` table
(see migration 20260524000000). These wrap supabase-py's chained API
into typed entry points that the v4 plugin-registry producers use.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from supabase import Client, create_client

from bip.core.settings import Settings

if TYPE_CHECKING:
    from bip.models.base import PredictionRecord


PREDICTIONS_RAW_TABLE = "predictions_raw"


def get_supabase_client(settings: Settings) -> Client:
    """Create and return a Supabase client from settings."""
    return create_client(settings.supabase_url, settings.supabase_key)


def insert_prediction_raw(client: Client, record: PredictionRecord) -> dict[str, Any]:
    """Insert a PredictionRecord into `predictions_raw`.

    Returns the inserted row (with server-side `id` and `created_at`).
    """
    payload = record.to_supabase_dict()
    resp = client.table(PREDICTIONS_RAW_TABLE).insert(payload).execute()
    if not resp.data:
        raise RuntimeError("insert_prediction_raw returned no data")
    return resp.data[0]


def select_pending_predictions(
    client: Client,
    *,
    within_hours: int = 4,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Read `predictions_raw` rows with status='pending' kicking off soon.

    Returns rows whose `match_datetime` is within (now, now + within_hours).
    Optionally filter by source ('ligas', 'mundial', ...).
    """
    now = datetime.now(UTC)
    until = now + timedelta(hours=within_hours)

    query = (
        client.table(PREDICTIONS_RAW_TABLE)
        .select("*")
        .eq("status", "pending")
        .gte("match_datetime", now.isoformat())
        .lte("match_datetime", until.isoformat())
    )
    if sources:
        query = query.in_("source", sources)
    resp = query.execute()
    return resp.data or []
