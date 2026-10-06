"""Verify migration 004 (claude_* columns + status CHECK widening + index + unique idempotency constraint) on live Supabase.

Exit criterion for Phase 3 plan 03-01 Task 3 (D-15 mirror). Returns exit code
0 on success, 1 on failure.

Usage:
    SUPABASE_DB_PASSWORD=... uv run python scripts/verify_migration_004.py

Reads SUPABASE_URL from .env via Settings. Reads SUPABASE_DB_PASSWORD directly
from os.environ (NOT in the Settings model — this is the DB password, NOT the
anon/service-role key). Uses psycopg directly (not supabase-py) because
information_schema queries require SQL-level access that the PostgREST layer
does not expose.

Security (T-02.1-02, T-02.1-03):
  - Never log or print the DB password or the full connection string.
  - All SQL queries use parameterized placeholders (%s), never f-strings.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import quote

import psycopg
import structlog

from bip.core.settings import Settings

logger = structlog.get_logger(__name__)


# Copied verbatim from verify_migration_003.py — DO NOT diverge.
def derive_db_url(supabase_url: str, db_password: str) -> str:
    subdomain = supabase_url.removeprefix("https://").removesuffix(".supabase.co")
    return (
        f"postgresql://postgres:{quote(db_password, safe='')}"
        f"@db.{subdomain}.supabase.co:5432/postgres"
    )


def main() -> int:
    settings = Settings()
    db_password = os.environ.get("SUPABASE_DB_PASSWORD")
    if not db_password:
        print("FAIL: SUPABASE_DB_PASSWORD not set in environment")
        return 1

    db_url = derive_db_url(settings.supabase_url, db_password)
    # T-02.1-02 + WR-02: never log the password. rsplit on the LAST '@'
    # so a password containing '@' (now percent-encoded) cannot leak.
    sanitized_host = db_url.rsplit("@", 1)[-1]

    expected_columns = {
        "claude_validation",
        "claude_reasoning",
        "claude_summary",
        "claude_validated_at",
    }

    try:
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            # ---- Query 1: confirm the 4 new columns exist on picks (D-08) ----
            # T-02.1-03: parameterized query — never f-string SQL.
            cur.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name=%s AND column_name = ANY(%s)",
                ("picks", list(expected_columns)),
            )
            col_rows = cur.fetchall()
            found_cols = {r[0] for r in col_rows}
            missing = expected_columns - found_cols
            if missing:
                print(f"FAIL: missing columns on picks: {sorted(missing)}")
                return 1

            # ---- Query 2: confirm picks_status_check accepts 'filtered' and 'rejected' (D-03) ----
            cur.execute(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = %s",
                ("picks_status_check",),
            )
            ck_rows = cur.fetchall()
            if not ck_rows:
                print("FAIL: picks_status_check constraint not found")
                return 1
            ck_def = ck_rows[0][0]
            if "'filtered'" not in ck_def or "'rejected'" not in ck_def:
                print(f"FAIL: picks_status_check does NOT include 'filtered'/'rejected'. Got: {ck_def}")
                return 1

            # ---- Query 3: confirm idx_picks_sport_market_created index exists (D-09) ----
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename=%s AND indexname=%s",
                ("picks", "idx_picks_sport_market_created"),
            )
            idx_rows = cur.fetchall()
            if not idx_rows:
                print("FAIL: idx_picks_sport_market_created index not found")
                return 1

            # ---- Query 4: confirm picks_unique_prediction UNIQUE constraint exists (Specifics §195 / Warning #2) ----
            # DB-layer idempotency on (fixture_id, market, prediction_id) — re-running PickEngine.evaluate
            # for the same prediction must NOT produce a duplicate row.
            cur.execute(
                "SELECT conname FROM pg_constraint WHERE conname = %s",
                ("picks_unique_prediction",),
            )
            uniq_rows = cur.fetchall()
            if not uniq_rows:
                print("FAIL: picks_unique_prediction UNIQUE constraint not found "
                      "(Specifics §195 / Warning #2 — idempotency guard for evaluate())")
                return 1
    except psycopg.OperationalError as exc:
        # Connection failure (wrong host, wrong password, network). Do NOT leak password.
        print(f"FAIL: could not connect to Supabase DB at {sanitized_host}: {exc}")
        return 1

    logger.info(
        "migration_004_verified",
        columns=sorted(found_cols),
        constraint_def=ck_def[:120],
        index=idx_rows[0][0],
        unique_constraint=uniq_rows[0][0],
    )
    print(
        "OK: migration 004 applied — claude_* columns present, "
        "picks_status_check widened to include 'filtered'/'rejected', "
        "idx_picks_sport_market_created index present, "
        "picks_unique_prediction UNIQUE constraint enforces evaluate() idempotency"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
