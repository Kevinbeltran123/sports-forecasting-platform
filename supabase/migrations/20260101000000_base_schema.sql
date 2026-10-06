-- Migration 001: Base schema for the 6-table betting intelligence platform.
--
-- This file reproduces the pre-Phase-1 schema that originally lived in the
-- upstream Football_analysis project. It is reconstructed here from
-- src/bip/core/storage/models.py (Pydantic models) and internal planning notes
-- for fresh-project bootstrap. Existing Supabase projects that already have
-- these tables should NOT run this migration.
--
-- Migration ordering (timestamps reflect logical history):
--   20260101000000 — base schema           (this file)
--   20260422000000 — add_sport_column       (CORE-03: sport VARCHAR on all 6 tables)
--   20260423000000 — add_is_shadow         (ML-05: is_shadow BOOLEAN on predictions)
--
-- NULL/NOT NULL choices match the Pydantic field types:
--   field: T            -> NOT NULL
--   field: T | None     -> NULL allowed
--   field: T = <default> -> NOT NULL DEFAULT <default>

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid() if needed later

-- ---------------------------------------------------------------------------
-- 1. predictions — model output per fixture/market
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    league VARCHAR(50) NOT NULL,
    market VARCHAR(50) NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    kickoff_utc TIMESTAMPTZ NOT NULL,
    probabilities JSONB NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    is_lineup_adjusted BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_fixture_market
    ON predictions (fixture_id, market);
CREATE INDEX IF NOT EXISTS idx_predictions_kickoff
    ON predictions (kickoff_utc);

-- ---------------------------------------------------------------------------
-- 2. picks — predictions that cleared the EV threshold
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS picks (
    id BIGSERIAL PRIMARY KEY,
    prediction_id BIGINT REFERENCES predictions(id) ON DELETE SET NULL,
    fixture_id BIGINT NOT NULL,
    league VARCHAR(50) NOT NULL,
    market VARCHAR(50) NOT NULL,
    selection VARCHAR(100) NOT NULL,
    model_probability DOUBLE PRECISION NOT NULL,
    implied_probability DOUBLE PRECISION NOT NULL,
    edge DOUBLE PRECISION NOT NULL,
    best_odds DOUBLE PRECISION NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    kelly_fraction DOUBLE PRECISION,
    suggested_stake DOUBLE PRECISION,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'won', 'lost', 'void', 'push')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_picks_fixture ON picks (fixture_id);
CREATE INDEX IF NOT EXISTS idx_picks_status ON picks (status);
CREATE INDEX IF NOT EXISTS idx_picks_created ON picks (created_at);

-- ---------------------------------------------------------------------------
-- 3. odds_snapshots — point-in-time odds captures (open + closing)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL,
    market VARCHAR(50) NOT NULL,
    bookmaker VARCHAR(50) NOT NULL,
    odds JSONB NOT NULL,
    is_closing BOOLEAN NOT NULL DEFAULT false,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_odds_fixture_captured
    ON odds_snapshots (fixture_id, captured_at);
CREATE INDEX IF NOT EXISTS idx_odds_closing
    ON odds_snapshots (fixture_id, market, bookmaker)
    WHERE is_closing = true;

-- ---------------------------------------------------------------------------
-- 4. results — match outcomes for grading picks and feature engineering
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results (
    id BIGSERIAL PRIMARY KEY,
    fixture_id BIGINT NOT NULL UNIQUE,
    league VARCHAR(50) NOT NULL,
    home_team VARCHAR(100) NOT NULL,
    away_team VARCHAR(100) NOT NULL,
    home_goals INT,
    away_goals INT,
    home_corners INT,
    away_corners INT,
    match_stats JSONB,
    kickoff_utc TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_results_kickoff ON results (kickoff_utc);

-- ---------------------------------------------------------------------------
-- 5. clv_records — per-pick CLV (Closing Line Value) measurement
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clv_records (
    id BIGSERIAL PRIMARY KEY,
    pick_id BIGINT NOT NULL REFERENCES picks(id) ON DELETE CASCADE,
    fixture_id BIGINT NOT NULL,
    market VARCHAR(50) NOT NULL,
    odds_at_pick DOUBLE PRECISION NOT NULL,
    pinnacle_closing_odds DOUBLE PRECISION,
    implied_prob_at_pick DOUBLE PRECISION NOT NULL,
    implied_prob_closing DOUBLE PRECISION,
    clv_percentage DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_clv_pick ON clv_records (pick_id);
CREATE INDEX IF NOT EXISTS idx_clv_fixture ON clv_records (fixture_id);

-- ---------------------------------------------------------------------------
-- 6. performance_metrics — aggregated ROI/yield/CLV by league/market/period
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    league VARCHAR(50) NOT NULL,
    market VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL
        CHECK (period IN ('daily', 'weekly', 'monthly', 'season', 'all_time')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_picks INT NOT NULL DEFAULT 0,
    won INT NOT NULL DEFAULT 0,
    lost INT NOT NULL DEFAULT 0,
    void INT NOT NULL DEFAULT 0,
    total_staked DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    roi DOUBLE PRECISION,
    yield_pct DOUBLE PRECISION,
    avg_clv DOUBLE PRECISION,
    avg_edge DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (league, market, period, period_start)
);

CREATE INDEX IF NOT EXISTS idx_perf_period_range
    ON performance_metrics (period, period_start, period_end);
