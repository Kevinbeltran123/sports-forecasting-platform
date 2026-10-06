"""Tests for live.alert_state — rate-limit + re-alert logic."""
from __future__ import annotations

from bip.evaluation.tournaments.team_style_profiler.live.alert_state import (
    EDGE_INCREASE_THRESHOLD_PP,
    AlertHistory,
    should_alert,
)


class TestAlertHistory:
    def test_first_alert_passes(self) -> None:
        h = AlertHistory()
        ok, reason = should_alert(h, 1, "BTTS_yes", 0.10)
        assert ok is True
        assert "first" in reason.lower()

    def test_duplicate_blocked(self) -> None:
        h = AlertHistory()
        h.record(1, "BTTS_yes", edge_pct=0.10, z_score=2.0)
        ok, _ = should_alert(h, 1, "BTTS_yes", 0.12)
        assert ok is False  # only +2pp delta, < 5pp threshold

    def test_re_alert_when_edge_grows(self) -> None:
        h = AlertHistory()
        h.record(1, "BTTS_yes", edge_pct=0.05, z_score=2.0)
        ok, reason = should_alert(h, 1, "BTTS_yes", 0.12)
        # delta = 0.07 >= 0.05 threshold
        assert ok is True
        assert "grew" in reason.lower()

    def test_different_markets_independent(self) -> None:
        h = AlertHistory()
        h.record(1, "BTTS_yes", edge_pct=0.10, z_score=2.0)
        ok, _ = should_alert(h, 1, "O2.5", 0.10)
        # Different market — should pass first-time check
        assert ok is True

    def test_different_fixtures_independent(self) -> None:
        h = AlertHistory()
        h.record(1, "BTTS_yes", edge_pct=0.10, z_score=2.0)
        ok, _ = should_alert(h, 2, "BTTS_yes", 0.10)
        # Different fixture
        assert ok is True

    def test_threshold_value(self) -> None:
        # Locks the 5pp constant per design — protects against drift.
        assert EDGE_INCREASE_THRESHOLD_PP == 0.05

    def test_record_overwrites(self) -> None:
        h = AlertHistory()
        h.record(1, "BTTS_yes", edge_pct=0.05, z_score=2.0)
        h.record(1, "BTTS_yes", edge_pct=0.15, z_score=3.0)
        prev = h.previous(1, "BTTS_yes")
        assert prev is not None
        assert prev.edge_pct_at_emit == 0.15  # latest record
