-- Migration 002: Add sport column to all 6 tables
-- Default 'football' makes existing Football_analysis rows valid without backfill

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

ALTER TABLE picks
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

ALTER TABLE odds_snapshots
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

ALTER TABLE results
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

ALTER TABLE clv_records
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

-- D-04c: odds_fetched_at for CLV data-freshness measurement
ALTER TABLE clv_records
    ADD COLUMN IF NOT EXISTS odds_fetched_at TIMESTAMPTZ;

ALTER TABLE performance_metrics
    ADD COLUMN IF NOT EXISTS sport VARCHAR(20) NOT NULL DEFAULT 'football';

-- Sport-scoped indexes for query performance
CREATE INDEX IF NOT EXISTS idx_predictions_sport ON predictions (sport);
CREATE INDEX IF NOT EXISTS idx_picks_sport ON picks (sport);
CREATE INDEX IF NOT EXISTS idx_clv_sport ON clv_records (sport);
CREATE INDEX IF NOT EXISTS idx_perf_sport ON performance_metrics (sport);
