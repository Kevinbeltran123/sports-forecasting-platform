"""Verify migration 003 (is_shadow column + index) is applied to live Supabase.

Exit criterion for Phase 02.1 (D-15). Returns exit code 0 on success, 1 on failure.

Usage:
    uv run python scripts/verify_migration_003.py

Reads SUPABASE_URL from .env via Settings. Reads SUPABASE_DB_PASSWORD directly
from os.environ (not in the Settings model — this is the DB password, NOT the
anon/service-role key). Uses psycopg directly (not supabase-py) because
information_schema queries require SQL-level access that the PostgREST layer
does not expose (research finding 4).

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


def derive_db_url(supabase_url: str, db_password: str) -> str:
    """Convert SUPABASE_URL (https://<ref>.supabase.co) → direct Postgres URL.

    Format: postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres

    WR-02: Supabase auto-generated passwords commonly contain ``@``, ``#``,
    ``!``, ``?``, ``/``, ``:`` and ``%`` — all of which require percent-
    encoding inside the userinfo component of a URI. ``urllib.parse.quote``
    with ``safe=''`` escapes every reserved character so a password with
    ``@`` does not silently split userinfo prematurely (which would also
    leak part of the password into the sanitized-host log line below).
    """
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
    # T-02.1-02 + WR-02: never log db_url (contains password). Use rsplit on
    # the LAST ``@`` so the host portion is taken from the right of the final
    # delimiter — splitting on the FIRST ``@`` could land inside a password
    # that contains an ``@`` (now percent-encoded by derive_db_url, but the
    # rsplit defends in depth against any future caller passing a raw URL).
    sanitized_host = db_url.rsplit("@", 1)[-1]

    try:
        with psycopg.connect(db_url) as conn, conn.cursor() as cur:
            # T-02.1-03: parameterized queries — no f-strings, no string concat.
            cur.execute(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_name=%s AND column_name=%s",
                ("predictions", "is_shadow"),
            )
            col_rows = cur.fetchall()
            if not col_rows:
                print("FAIL: is_shadow column not found on predictions table")
                return 1

            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename=%s AND indexname=%s",
                ("predictions", "idx_predictions_is_shadow"),
            )
            idx_rows = cur.fetchall()
            if not idx_rows:
                print("FAIL: idx_predictions_is_shadow not found")
                return 1
    except psycopg.OperationalError as exc:
        # Connection failure (wrong host, wrong password, network). Do NOT leak password.
        print(f"FAIL: could not connect to Supabase DB at {sanitized_host}: {exc}")
        return 1

    logger.info(
        "migration_003_verified",
        column=col_rows[0][0],           # "is_shadow"
        data_type=col_rows[0][1],        # "boolean"
        is_nullable=col_rows[0][2],      # "NO"
        column_default=col_rows[0][3],   # "false"
        index=idx_rows[0][0],            # "idx_predictions_is_shadow"
    )
    print("OK: migration 003 applied — is_shadow column + index present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
