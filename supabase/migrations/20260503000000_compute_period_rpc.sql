-- Migration 005: compute_performance_period(...) RPC for D-16 single-round-trip aggregation.
--
-- Implements D-16 (Phase 4): one Supabase round trip per (sport, league, market) period
-- replaces N application-level queries. Required because the supabase-py SDK does NOT
-- expose arbitrary SELECT — only .table().select() (PostgREST) or .rpc(name, params).
-- A Postgres function is the supported path for SELECT count(*) FILTER (WHERE ...) with
-- LEFT JOIN on clv_records.
--
-- An inner-join on clv_records would be INCORRECT — picks where Odds API failed
-- (D-02 path) have NO clv_records row; an inner-join silently drops them and ROI
-- under-reports (RESEARCH §Pitfall 3). LEFT JOIN keeps them; AVG(c.clv_percentage)
-- ignores NULLs natively.
--
-- Threat T-4-03 (SQL injection): Function is LANGUAGE sql STABLE — no EXECUTE of dynamic
-- SQL. All parameters are typed (text, timestamptz). Callers MUST use client.rpc(name, dict)
-- (parameterized binding), NEVER string interpolation.
--
-- Idempotent: CREATE OR REPLACE FUNCTION. Atomic: BEGIN/COMMIT.

BEGIN;

CREATE OR REPLACE FUNCTION compute_performance_period(
    p_sport  text,
    p_league text,
    p_market text,
    p_start  timestamptz,
    p_end    timestamptz
)
RETURNS TABLE (
    total_picks  bigint,
    won          bigint,
    lost         bigint,
    void         bigint,
    total_staked double precision,
    total_pnl    double precision,
    roi          double precision,
    avg_clv      double precision,
    avg_edge     double precision
)
LANGUAGE sql STABLE AS $$
    SELECT
        COUNT(*)                                            AS total_picks,
        COUNT(*) FILTER (WHERE p.status = 'won')            AS won,
        COUNT(*) FILTER (WHERE p.status = 'lost')           AS lost,
        COUNT(*) FILTER (WHERE p.status IN ('void','push')) AS void,
        COALESCE(SUM(p.suggested_stake), 0)                 AS total_staked,
        COALESCE(SUM(
            CASE p.status
                WHEN 'won'  THEN p.suggested_stake * (p.best_odds - 1)
                WHEN 'lost' THEN -p.suggested_stake
                ELSE 0
            END
        ), 0)                                               AS total_pnl,
        SUM(
            CASE p.status
                WHEN 'won'  THEN p.suggested_stake * (p.best_odds - 1)
                WHEN 'lost' THEN -p.suggested_stake
                ELSE 0
            END
        ) / NULLIF(SUM(p.suggested_stake), 0)               AS roi,
        AVG(c.clv_percentage)                               AS avg_clv,
        AVG(p.edge)                                         AS avg_edge
    FROM picks p
    LEFT JOIN clv_records c ON c.pick_id = p.id
    WHERE p.sport     = p_sport
      AND p.league    = p_league
      AND p.market    = p_market
      AND p.created_at >= p_start
      AND p.created_at <  p_end
      AND p.status NOT IN ('filtered', 'rejected', 'pending');
$$;

COMMIT;
