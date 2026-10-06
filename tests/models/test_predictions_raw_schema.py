"""Tests that PredictionRecord.to_supabase_dict matches the SQL schema.

Sprint 0 Ola B. Verifies the Python schema and the SQL migration
(supabase/migrations/20260524000000_predictions_raw.sql) stay in sync
by checking column-name parity. This is a static test — it does NOT
connect to a real Supabase instance.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from bip.models.base import PredictionRecord

MIGRATION_PATH = Path(__file__).resolve().parents[2] / (
    "supabase/migrations/20260524000000_predictions_raw.sql"
)


def _extract_column_names_from_sql(sql: str) -> set[str]:
    """Parse CREATE TABLE predictions_raw (...) to extract column names.

    Best-effort regex parser; sufficient for the migration we control.
    """
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS predictions_raw\s*\((.*?)\);",
        sql,
        flags=re.DOTALL,
    )
    assert match, "predictions_raw CREATE TABLE not found"
    body = match.group(1)
    columns: set[str] = set()
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        # Skip CHECK / CONSTRAINT lines
        if stripped.upper().startswith(("CHECK", "CONSTRAINT", "UNIQUE", "PRIMARY", "FOREIGN")):
            continue
        # First token before whitespace is the column name
        token = stripped.split()[0].rstrip(",")
        if token.isidentifier():
            columns.add(token)
    return columns


def test_migration_file_exists():
    assert MIGRATION_PATH.exists(), f"Migration not found at {MIGRATION_PATH}"


def test_to_supabase_dict_keys_match_sql_columns():
    sql_cols = _extract_column_names_from_sql(MIGRATION_PATH.read_text())
    rec = PredictionRecord(
        source="ligas",
        fixture_id="fx-001",
        competition="PL",
        home_team="Arsenal",
        away_team="Chelsea",
        match_datetime=datetime(2026, 6, 1, 19, 0, 0, tzinfo=UTC),
        market="1x2",
        selection="home",
        p_model=0.55,
        ev=0.07,
        odds_at_pick=1.95,
        payload={"k": "v"},
        model_version="ligas-1x2-0.1.0",
    )
    py_keys = set(rec.to_supabase_dict().keys())
    # `id` is server-defaulted (UUID); not emitted by the Python side
    expected_sql = sql_cols - {"id"}
    missing_in_python = expected_sql - py_keys
    extra_in_python = py_keys - sql_cols
    assert not missing_in_python, f"Columns in SQL but not in PredictionRecord: {missing_in_python}"
    assert not extra_in_python, f"Fields in PredictionRecord but not in SQL: {extra_in_python}"


def test_status_default_matches_sql_default():
    # SQL: DEFAULT 'pending'
    rec = PredictionRecord(
        source="ligas",
        fixture_id="fx-001",
        competition="PL",
        home_team="A",
        away_team="B",
        match_datetime=datetime(2026, 6, 1, tzinfo=UTC),
        market="1x2",
        selection="home",
        p_model=0.5,
        model_version="v",
    )
    assert rec.to_supabase_dict()["status"] == "pending"
