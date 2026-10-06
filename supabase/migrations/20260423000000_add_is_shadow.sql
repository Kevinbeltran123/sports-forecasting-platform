-- Migration 003: Add is_shadow for ML-05 shadow-mode prediction logging
-- MUST be applied BEFORE any shadow prediction write (RESEARCH.md anti-pattern warning).

ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS is_shadow BOOLEAN NOT NULL DEFAULT false;

-- Index for production-only reads (Phase 3 pick engine reads is_shadow=false)
CREATE INDEX IF NOT EXISTS idx_predictions_is_shadow ON predictions (is_shadow);
