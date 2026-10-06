-- Migration 006: predictions_raw — v4 prediction-bus table.
--
-- Sprint 0 Ola B of the v4 refactor. See plan:
-- internal design notes (not published)
--
-- This table coexists with the legacy `predictions` (probabilities dict)
-- and `picks` (selection+edge+stake) tables for ~90d during the v3→v4
-- migration. Producers from `src/bip/models/` (LigasModel, MundialModel)
-- INSERT directly into `predictions_raw`. The PickEngine refactor in
-- Sprint 2 will start consuming `pending` rows from here.
--
-- Schema matches bip.models.base.PredictionRecord 1:1.

CREATE TABLE IF NOT EXISTS predictions_raw (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT NOT NULL,         -- 'ligas' | 'mundial'
    fixture_id      TEXT NOT NULL,
    competition     TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    match_datetime  TIMESTAMPTZ NOT NULL,
    market          TEXT NOT NULL,         -- '1x2' | 'ou_2.5' | 'btts' | ...
    selection       TEXT NOT NULL,         -- 'home' | 'over' | ...
    p_model         DOUBLE PRECISION NOT NULL CHECK (p_model >= 0.0 AND p_model <= 1.0),
    ev              DOUBLE PRECISION,      -- NULL when no odds (e.g. Mundial)
    odds_at_pick    DOUBLE PRECISION,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_version   TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'sent', 'killed', 'invalidated', 'shadow')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predictions_raw_status   ON predictions_raw (status);
CREATE INDEX IF NOT EXISTS idx_predictions_raw_match_dt ON predictions_raw (match_datetime);
CREATE INDEX IF NOT EXISTS idx_predictions_raw_source   ON predictions_raw (source);
CREATE INDEX IF NOT EXISTS idx_predictions_raw_fixture  ON predictions_raw (fixture_id);
