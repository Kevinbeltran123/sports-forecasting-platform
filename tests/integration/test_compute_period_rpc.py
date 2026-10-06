"""Integration tests for compute_performance_period RPC (CLV-04 / D-16).

These tests mock the Supabase client to verify the contract between
`PerformanceMetricRepository.compute_period` and the migration-005 function.
The LIVE DB function was smoke-tested manually in 04-00-PLAN Task 4 (apply_migration
via Supabase MCP, then SELECT FROM compute_performance_period(...) returned a row of
zeros for an empty window, confirming the function is deployed).
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from bip.core.storage.repositories import PerformanceMetricRepository


def _mock_rpc_response(client: MagicMock, row: dict | None) -> MagicMock:
    """Wire client.rpc(name, params).execute() → response.data = [row] or []."""
    response = MagicMock()
    response.data = [row] if row is not None else []
    rpc_obj = MagicMock()
    rpc_obj.execute = MagicMock(return_value=response)
    client.rpc = MagicMock(return_value=rpc_obj)
    return rpc_obj


def test_left_join_picks_without_clv_present_in_count():
    """Migration 005 LEFT JOINs clv_records — picks with no CLV row are still counted.

    Simulate: 10 settled picks, only 7 have CLV records (3 had Odds-API failure per D-02).
    The function returns total_picks=10 (counts all settled picks) and avg_clv based on 7.
    """
    client = MagicMock()
    _mock_rpc_response(client, {
        "total_picks": 10, "won": 6, "lost": 3, "void": 1,
        "total_staked": 100.0, "total_pnl": 30.0,
        "roi": 0.30, "avg_clv": 2.5, "avg_edge": 0.07,
    })
    repo = PerformanceMetricRepository(client=client)
    metric = repo.compute_period(
        sport="football", league="premier_league", market="onextwo",
        period="daily", period_start=date(2026, 5, 1), period_end=date(2026, 5, 2),
    )
    # 10 picks counted (LEFT JOIN), avg_clv averaged over the 7 with CLV (NULLs ignored by AVG).
    assert metric.total_picks == 10
    assert metric.avg_clv == pytest.approx(2.5)
    assert metric.roi == pytest.approx(0.30)


def test_status_filter_excludes_filtered_rejected_pending():
    """Migration 005 WHERE clause excludes status NOT IN ('filtered','rejected','pending').

    This test confirms the repo passes through whatever the function returns. The
    function contract (status filter) is enforced by migration 005 itself.
    Verify the call signature uses the correct parameter NAMES.
    """
    client = MagicMock()
    _mock_rpc_response(client, {
        "total_picks": 0, "won": 0, "lost": 0, "void": 0,
        "total_staked": 0.0, "total_pnl": 0.0,
        "roi": None, "avg_clv": None, "avg_edge": None,
    })
    repo = PerformanceMetricRepository(client=client)
    repo.compute_period(
        sport="football", league="premier_league", market="onextwo",
        period="daily", period_start=date(2026, 5, 1), period_end=date(2026, 5, 2),
    )
    # Verify rpc() called with correct function name and parameter dict
    call_args = client.rpc.call_args
    assert call_args.args[0] == "compute_performance_period"
    params = call_args.args[1]
    assert params["p_sport"] == "football"
    assert params["p_league"] == "premier_league"
    assert params["p_market"] == "onextwo"
    assert "p_start" in params and "p_end" in params  # ISO timestamps


def test_won_lost_void_counts_match_seed_data():
    """RPC returns won/lost/void with FILTER (WHERE status='X') counts; repo passes through."""
    client = MagicMock()
    _mock_rpc_response(client, {
        "total_picks": 12, "won": 5, "lost": 4, "void": 3,
        "total_staked": 60.0, "total_pnl": 12.0,
        "roi": 0.20, "avg_clv": 1.8, "avg_edge": 0.06,
    })
    repo = PerformanceMetricRepository(client=client)
    metric = repo.compute_period(
        sport="football", league="premier_league", market="onextwo",
        period="weekly", period_start=date(2026, 4, 26), period_end=date(2026, 5, 3),
    )
    assert metric.won == 5
    assert metric.lost == 4
    assert metric.void == 3  # void status group includes 'void' AND 'push' per migration 005
    assert metric.total_picks == 12
