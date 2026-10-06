-- Migration 007: predictions_raw — CLV / outcome / delivery columns.
--
-- Sprint 2 Ola C of the v4 refactor. See plan:
-- internal design notes (not published)
--
-- Adds the columns the DeliveryWorker (Ola B) writes after a successful
-- send, plus the columns ClvWorker (this migration) will populate at
-- T+2h post-kickoff.
--
-- Coexists with the legacy `clv_records` table from migration 001; the
-- existing engine_v3 → clv_records flow is unchanged. v4 picks live
-- exclusively on predictions_raw.

ALTER TABLE predictions_raw
    ADD COLUMN IF NOT EXISTS kill_reason       TEXT,
    ADD COLUMN IF NOT EXISTS stake_units       DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sent_at           TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS closing_odds      DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS clv               DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS clv_measured_at   TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS result            TEXT
        CHECK (result IS NULL OR result IN ('win', 'loss', 'void', 'push')),
    ADD COLUMN IF NOT EXISTS pl_units          DOUBLE PRECISION;

-- Index for the ClvWorker poll: SELECT WHERE status='sent' AND clv IS NULL
-- AND match_datetime < now() - 2h
CREATE INDEX IF NOT EXISTS idx_predictions_raw_clv_pending
    ON predictions_raw (match_datetime)
    WHERE status = 'sent' AND clv IS NULL;

-- Index for the rolling-CLV KPI query
CREATE INDEX IF NOT EXISTS idx_predictions_raw_clv_measured
    ON predictions_raw (source, clv_measured_at)
    WHERE clv IS NOT NULL;
