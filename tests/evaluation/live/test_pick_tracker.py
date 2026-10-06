"""Layer-1 tests for PickTracker SQLite store."""

from __future__ import annotations

from pathlib import Path

import pytest

from bip.evaluation.live.pick_tracker import PickTracker
from bip.evaluation.live.value_detector import LivePick


def _pick(
    *,
    fixture_id: int = 1,
    minute: int = 30,
    market: str = "fulltime_result",
    selection: str = "home",
    edge_pct: float = 5.0,
    bookmaker_odd: float = 2.10,
    our_probability: float = 0.50,
    bookmaker_id: int = 2,
) -> LivePick:
    return LivePick(
        fixture_id=fixture_id,
        minute=minute,
        home_team="Home FC",
        away_team="Away FC",
        market=market,
        selection=selection,
        bookmaker_id=bookmaker_id,
        bookmaker_odd=bookmaker_odd,
        our_probability=our_probability,
        fair_odd=1.0 / our_probability,
        edge_pct=edge_pct,
        kelly_fraction_full=0.05,
        suggested_stake_pct=1.0,
    )


@pytest.fixture
def tracker(tmp_path: Path) -> PickTracker:
    return PickTracker(db_path=tmp_path / "picks.db")


class TestRecordPick:
    def test_first_record_returns_id_and_new_true(self, tracker: PickTracker):
        pid, is_new = tracker.record(_pick())
        assert pid is not None and pid > 0
        assert is_new is True

    def test_duplicate_within_bucket_skipped(self, tracker: PickTracker):
        pid1, new1 = tracker.record(_pick(minute=30))
        # Same fixture/market/selection/bookmaker, minute 32 → same 5-min bucket
        pid2, new2 = tracker.record(_pick(minute=32))
        assert new1 is True
        assert new2 is False
        assert pid2 is None

    def test_different_bucket_inserted(self, tracker: PickTracker):
        # Minute 30 → bucket 6; minute 36 → bucket 7
        _, new1 = tracker.record(_pick(minute=30))
        _, new2 = tracker.record(_pick(minute=36))
        assert new1 is True
        assert new2 is True

    def test_different_selection_inserted(self, tracker: PickTracker):
        _, new1 = tracker.record(_pick(selection="home"))
        _, new2 = tracker.record(_pick(selection="draw"))
        assert new1 is True
        assert new2 is True


class TestQueries:
    def test_list_pending(self, tracker: PickTracker):
        for i in range(3):
            tracker.record(_pick(market=f"market_{i}"))
        pending = tracker.list_pending()
        assert len(pending) == 3
        assert all(p.status == "pending" for p in pending)

    def test_list_for_fixture(self, tracker: PickTracker):
        tracker.record(_pick(fixture_id=100, market="m1"))
        tracker.record(_pick(fixture_id=100, market="m2"))
        tracker.record(_pick(fixture_id=200, market="m1"))
        f100 = tracker.list_for_fixture(100)
        f200 = tracker.list_for_fixture(200)
        assert len(f100) == 2
        assert len(f200) == 1


class TestOutcomes:
    def test_grade_won(self, tracker: PickTracker):
        pid, _ = tracker.record(_pick(bookmaker_odd=2.10))
        tracker.update_outcome(pid, status="won", profit_units=1.10)
        recent = tracker.list_recent()
        assert recent[0].status == "won"
        assert recent[0].profit_units == 1.10

    def test_grade_lost_negative_profit(self, tracker: PickTracker):
        pid, _ = tracker.record(_pick())
        tracker.update_outcome(pid, status="lost", profit_units=-1.0)
        recent = tracker.list_recent()
        assert recent[0].profit_units == -1.0

    def test_invalid_status_rejected(self, tracker: PickTracker):
        pid, _ = tracker.record(_pick())
        with pytest.raises(ValueError):
            tracker.update_outcome(pid, status="banana")


class TestStatsSummary:
    def test_empty_tracker(self, tracker: PickTracker):
        s = tracker.stats_summary()
        assert s["n_total"] == 0
        assert s["win_rate"] is None

    def test_two_wins_one_loss(self, tracker: PickTracker):
        # Two winning picks @ 2.0 odds, one losing @ 2.0 → +1.00 unit total
        ids = []
        for i in range(3):
            pid, _ = tracker.record(_pick(market=f"m_{i}", bookmaker_odd=2.0))
            ids.append(pid)
        tracker.update_outcome(ids[0], status="won", profit_units=1.0)
        tracker.update_outcome(ids[1], status="won", profit_units=1.0)
        tracker.update_outcome(ids[2], status="lost", profit_units=-1.0)
        s = tracker.stats_summary()
        assert s["n_graded"] == 3
        assert s["n_won"] == 2
        assert s["win_rate"] == pytest.approx(2 / 3, abs=1e-6)
        assert s["total_profit_units"] == pytest.approx(1.0, abs=1e-6)
        assert s["roi_pct"] == pytest.approx(33.33, abs=0.1)


class TestPlacedTracking:
    def test_mark_placed(self, tracker: PickTracker):
        pid, _ = tracker.record(_pick())
        tracker.mark_placed(pid, betano_odd=2.05, stake_units=2.5, notes="OK")
        with tracker._connect() as conn:
            row = conn.execute(
                "SELECT placed_at_betano, betano_odd, actual_stake_units, notes "
                "FROM picks WHERE id=?", (pid,),
            ).fetchone()
            assert row["placed_at_betano"] == 1
            assert row["betano_odd"] == 2.05
            assert row["actual_stake_units"] == 2.5
            assert row["notes"] == "OK"
