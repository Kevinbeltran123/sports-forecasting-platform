-- Migration 004: Add Claude Role C validation columns + widen picks.status CHECK
--                + add picks_unique_prediction UNIQUE for evaluate() idempotency
--
-- Implements D-08 (claude_* columns) and D-03 (PickStatus 'filtered', 'rejected').
-- Adds idx_picks_sport_market_created for D-09 rolling-168h market-cap query.
-- Adds picks_unique_prediction UNIQUE on (fixture_id, market, prediction_id)
-- per RESEARCH Specifics §195 — re-running PickEngine.evaluate for the same
-- prediction (e.g., post-restart T-2h re-execution) must NOT produce a duplicate row.
-- DB-layer constraint is the robust path: scheduler-level replace_existing=True only
-- prevents duplicate sends, not duplicate DB rows. Supabase Python SDK PostgREST does
-- not expose ON CONFLICT cleanly, so a UNIQUE constraint forces rejection regardless
-- of repo path.
--
-- MUST be applied BEFORE any code path writes status='filtered' or status='rejected'
-- to the picks table — Pydantic models would accept those values but the live
-- CHECK constraint (from migration 001) would reject the INSERT at runtime.
--
-- Idempotent: safe to re-run via DROP CONSTRAINT IF EXISTS + ADD COLUMN IF NOT EXISTS.
-- Atomic: wrapped in BEGIN/COMMIT so a failed ADD CONSTRAINT rolls back the prior DROP
-- (Pitfall 5 — without this wrapper, partial failure leaves the table without ANY
-- status CHECK constraint, allowing arbitrary status values until manual repair).

BEGIN;

-- ---------------------------------------------------------------------------
-- D-08: claude_* columns on picks
-- ---------------------------------------------------------------------------
ALTER TABLE picks
    ADD COLUMN IF NOT EXISTS claude_validation VARCHAR(10),
    ADD COLUMN IF NOT EXISTS claude_reasoning TEXT,
    ADD COLUMN IF NOT EXISTS claude_summary VARCHAR(120),
    ADD COLUMN IF NOT EXISTS claude_validated_at TIMESTAMPTZ;

-- ---------------------------------------------------------------------------
-- D-03: widen picks_status_check to accept 'filtered' and 'rejected'
-- (PostgreSQL has no ALTER CONSTRAINT for CHECK — must DROP + ADD.)
-- Constraint name picks_status_check is the PostgreSQL default (table_column_check).
-- If migration 001 used a custom name, this DROP silently no-ops and the OLD
-- constraint remains. Run the verify script after apply to confirm widening took.
-- ---------------------------------------------------------------------------
ALTER TABLE picks
    DROP CONSTRAINT IF EXISTS picks_status_check;
ALTER TABLE picks
    ADD CONSTRAINT picks_status_check
    CHECK (status IN ('pending', 'won', 'lost', 'void', 'push', 'filtered', 'rejected'));

-- D-08 sanity check on claude_validation enum values (NULL allowed for not-yet-validated picks).
ALTER TABLE picks
    DROP CONSTRAINT IF EXISTS picks_claude_validation_check;
ALTER TABLE picks
    ADD CONSTRAINT picks_claude_validation_check
    CHECK (claude_validation IS NULL OR claude_validation IN ('CONFIRM', 'FLAG', 'REJECT', 'SKIPPED'));

-- ---------------------------------------------------------------------------
-- Specifics §195 — Warning #2 fix: DB-layer idempotency on (fixture_id, market, prediction_id).
-- Re-running PickEngine.evaluate for the same prediction (e.g., post-restart re-run
-- of T-2h) must produce the same row, never a duplicate. The scheduler's
-- replace_existing=True prevents duplicate sends but NOT duplicate DB rows; this
-- UNIQUE constraint forces rejection at the DB layer regardless of repo path.
-- DROP CONSTRAINT IF EXISTS first so re-applying the migration is idempotent.
-- ---------------------------------------------------------------------------
ALTER TABLE picks
    DROP CONSTRAINT IF EXISTS picks_unique_prediction;
ALTER TABLE picks
    ADD CONSTRAINT picks_unique_prediction
    UNIQUE (fixture_id, market, prediction_id);

-- ---------------------------------------------------------------------------
-- D-09: index for rolling-168h market-cap query
-- (sport, market, created_at) — supports the WHERE sport=? AND created_at >= ? GROUP BY market query.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_picks_sport_market_created
    ON picks (sport, market, created_at);

COMMIT;
