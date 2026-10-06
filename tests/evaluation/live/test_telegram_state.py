"""Layer-1 tests for TelegramState — persistent state for Bot v2.

Boundary-focus per project rule:
- Duplicate action via same callback_id     (TG redelivery)
- Duplicate action via different callback_id (operator double-tap)
- Mute window before / at / after expiry
- Burst queue dedup
- CLV drift sign for both odds directions
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bip.evaluation.live.pick_tracker import PickTracker
from bip.evaluation.live.telegram_state import (
    Action,
    BurstReason,
    Role,
    TelegramState,
    _utc_iso,
)
from bip.evaluation.live.value_detector import LivePick


# ── Fixtures ────────────────────────────────────────────────────────────────


def _pick(*, fixture_id: int = 1, minute: int = 30,
          market: str = "fulltime_result", selection: str = "home") -> LivePick:
    return LivePick(
        fixture_id=fixture_id,
        minute=minute,
        home_team="Home FC",
        away_team="Away FC",
        market=market,
        selection=selection,
        bookmaker_id=2,
        bookmaker_odd=2.10,
        our_probability=0.50,
        fair_odd=2.00,
        edge_pct=5.0,
        kelly_fraction_full=0.05,
        suggested_stake_pct=1.0,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "picks.db"


@pytest.fixture
def tg(db_path: Path) -> TelegramState:
    # PickTracker creates picks/pick_decisions tables; TelegramState then
    # adds tg_* tables on the same DB. Order matters for picks_awaiting_outcome_reply.
    PickTracker(db_path=db_path)
    return TelegramState(db_path=db_path)


@pytest.fixture
def tracker(db_path: Path) -> PickTracker:
    return PickTracker(db_path=db_path)


# ── Schema ──────────────────────────────────────────────────────────────────


class TestSchema:
    def test_idempotent_init(self, db_path: Path):
        TelegramState(db_path=db_path)
        TelegramState(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"tg_messages", "tg_actions", "tg_state",
                "tg_burst_queue", "tg_clv_track"} <= tables

    def test_does_not_drop_existing_picks_table(
        self, db_path: Path, tracker: PickTracker,
    ):
        tracker.record(_pick())
        TelegramState(db_path=db_path)
        conn = sqlite3.connect(db_path)
        n = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
        assert n == 1


# ── tg_messages ─────────────────────────────────────────────────────────────


class TestMessages:
    def test_record_and_find(self, tg: TelegramState):
        tg.record_message(
            pick_id=42, channel_id="-100123", message_id=7,
            role=Role.PICK, tier=2,
        )
        ref = tg.find_message(pick_id=42, role=Role.PICK)
        assert ref is not None
        assert ref.message_id == 7
        assert ref.tier == 2
        assert ref.channel_id == "-100123"

    def test_upsert_overwrites_message_id(self, tg: TelegramState):
        tg.record_message(pick_id=1, channel_id="-1", message_id=100, role=Role.PICK)
        tg.record_message(pick_id=1, channel_id="-1", message_id=200, role=Role.PICK)
        ref = tg.find_message(pick_id=1, role=Role.PICK)
        assert ref.message_id == 200

    def test_different_role_coexists(self, tg: TelegramState):
        tg.record_message(pick_id=1, channel_id="-1", message_id=10, role=Role.PICK)
        tg.record_message(pick_id=1, channel_id="-1", message_id=11, role=Role.OUTCOME)
        assert tg.find_message(pick_id=1, role=Role.PICK).message_id == 10
        assert tg.find_message(pick_id=1, role=Role.OUTCOME).message_id == 11

    def test_find_returns_none_when_missing(self, tg: TelegramState):
        assert tg.find_message(pick_id=999, role=Role.PICK) is None


class TestAwaitingOutcome:
    def test_returns_settled_picks_without_outcome(
        self, tg: TelegramState, tracker: PickTracker, db_path: Path,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1))
        assert pid is not None
        tg.record_message(pick_id=pid, channel_id="-1",
                          message_id=10, role=Role.PICK)
        # Manually grade
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.5, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        assert tg.picks_awaiting_outcome_reply() == [pid]

    def test_excludes_picks_with_outcome_message(
        self, tg: TelegramState, tracker: PickTracker, db_path: Path,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1))
        tg.record_message(pick_id=pid, channel_id="-1",
                          message_id=10, role=Role.PICK)
        tg.record_message(pick_id=pid, channel_id="-1",
                          message_id=11, role=Role.OUTCOME)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.0, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        assert tg.picks_awaiting_outcome_reply() == []

    def test_excludes_pending(
        self, tg: TelegramState, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1))
        tg.record_message(pick_id=pid, channel_id="-1",
                          message_id=10, role=Role.PICK)
        # status stays 'pending'
        assert tg.picks_awaiting_outcome_reply() == []


# ── tg_actions: idempotency boundaries ──────────────────────────────────────


class TestActionIdempotency:
    def test_first_record_returns_new_true(self, tg: TelegramState):
        was_new, prior = tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-A",
            stake_pct=1.5,
        )
        assert was_new is True
        assert prior is None

    def test_same_callback_id_idempotent(self, tg: TelegramState):
        """TG network retry: identical callback_id arrives twice."""
        tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-A", stake_pct=1.5,
        )
        was_new, prior = tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-A", stake_pct=1.5,
        )
        assert was_new is False
        assert prior is not None
        assert prior.callback_id == "cb-A"
        assert prior.stake_pct == pytest.approx(1.5)

    def test_double_tap_different_callback_blocked(self, tg: TelegramState):
        """Operator clicks PLACE 1.5% then PLACE 0.75% — second rejected."""
        tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-A", stake_pct=1.5,
        )
        was_new, prior = tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-B", stake_pct=0.75,
        )
        assert was_new is False
        assert prior is not None
        assert prior.stake_pct == pytest.approx(1.5), \
            "Original 1.5% action must be preserved; override needs /placed"

    def test_different_action_on_same_pick_ok(self, tg: TelegramState):
        """REMIND-then-PLACE is a legitimate sequence, not a double-tap."""
        tg.record_action(pick_id=1, action=Action.REMINDED,
                         operator_user_id=999, callback_id="cb-r")
        was_new, _ = tg.record_action(
            pick_id=1, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-p", stake_pct=1.5,
        )
        assert was_new is True

    def test_get_action_round_trip(self, tg: TelegramState):
        tg.record_action(pick_id=42, action=Action.SKIPPED,
                         operator_user_id=999, callback_id="cb-s")
        rec = tg.get_action(pick_id=42, action=Action.SKIPPED)
        assert rec is not None
        assert rec.action == Action.SKIPPED
        assert tg.get_action(pick_id=42, action=Action.PLACED) is None


# ── tg_state: scalar bag + mute helpers ─────────────────────────────────────


class TestState:
    @pytest.mark.parametrize("value", [
        "string", 42, 3.14, True, None,
        {"nested": [1, 2, 3]}, ["a", "b"],
    ])
    def test_json_round_trip(self, tg: TelegramState, value):
        tg.set_state("k", value)
        assert tg.get_state("k") == value

    def test_get_state_default(self, tg: TelegramState):
        assert tg.get_state("missing", default=42) == 42

    def test_delete_state(self, tg: TelegramState):
        tg.set_state("k", "v")
        tg.delete_state("k")
        assert tg.get_state("k") is None

    def test_set_overwrites(self, tg: TelegramState):
        tg.set_state("k", "v1")
        tg.set_state("k", "v2")
        assert tg.get_state("k") == "v2"


class TestMarketMute:
    def test_set_and_check(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_market_mute("ou_2_5", future)
        assert tg.is_market_muted("ou_2_5") is True
        assert tg.is_market_muted("btts") is False

    def test_expired_returns_false(self, tg: TelegramState):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        tg.set_market_mute("ou_2_5", past)
        assert tg.is_market_muted("ou_2_5") is False

    def test_clear_market_mute(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_market_mute("ou_2_5", future)
        tg.clear_market_mute("ou_2_5")
        assert tg.is_market_muted("ou_2_5") is False

    def test_muted_markets_prunes_expired(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        tg.set_market_mute("ou_2_5", future)
        tg.set_market_mute("btts", past)
        active = tg.muted_markets()
        assert "ou_2_5" in active
        assert "btts" not in active

    def test_independent_markets(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_market_mute("ou_2_5", future)
        tg.set_market_mute("btts", future)
        tg.clear_market_mute("ou_2_5")
        assert tg.is_market_muted("ou_2_5") is False
        assert tg.is_market_muted("btts") is True


class TestMuteWindowTracking:
    def test_mute_records_started_at(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_mute_until(future)
        window = tg.last_mute_window()
        assert window is not None
        started_at, until = window
        assert started_at  # not empty
        assert until == future.isoformat()

    def test_clear_mute_preserves_started_at(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_mute_until(future)
        tg.clear_mute()
        # mute_until cleared, but started_at retained for /missed lookback
        assert tg.is_muted() is False
        assert tg.get_state("mute_started_at") is not None

    def test_last_mute_window_none_when_never_set(self, tg: TelegramState):
        assert tg.last_mute_window() is None


class TestMuteWindow:
    def test_not_muted_when_unset(self, tg: TelegramState):
        assert tg.is_muted() is False

    def test_muted_just_inside_window(self, tg: TelegramState):
        future = datetime.now(timezone.utc) + timedelta(minutes=30)
        tg.set_mute_until(future)
        assert tg.is_muted() is True

    def test_not_muted_just_after_window(self, tg: TelegramState):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        tg.set_mute_until(past)
        assert tg.is_muted() is False

    def test_not_muted_at_exact_boundary(self, tg: TelegramState):
        """is_muted = now < until. At the boundary, mute has expired."""
        now = datetime.now(timezone.utc)
        tg.set_mute_until(now)
        assert tg.is_muted(now=now) is False

    def test_clear_mute(self, tg: TelegramState):
        tg.set_mute_until(datetime.now(timezone.utc) + timedelta(minutes=30))
        tg.clear_mute()
        assert tg.is_muted() is False


# ── tg_burst_queue ──────────────────────────────────────────────────────────


class TestBurstQueue:
    def test_enqueue_returns_true_on_first(self, tg: TelegramState):
        assert tg.enqueue_burst(1, BurstReason.BURST) is True

    def test_enqueue_duplicate_returns_false(self, tg: TelegramState):
        tg.enqueue_burst(1, BurstReason.BURST)
        assert tg.enqueue_burst(1, BurstReason.MUTED) is False
        # Original reason preserved, count unchanged
        assert tg.burst_queue_size() == 1

    def test_drain_returns_oldest_first(self, tg: TelegramState):
        # Use record_action's acted_at parameter style — pass explicit
        # timestamps so ordering is deterministic on fast machines.
        import time
        tg.enqueue_burst(1, BurstReason.BURST)
        time.sleep(0.01)
        tg.enqueue_burst(2, BurstReason.BURST)
        time.sleep(0.01)
        tg.enqueue_burst(3, BurstReason.BURST)
        drained = tg.drain_burst_queue()
        assert drained == [1, 2, 3]
        assert tg.burst_queue_size() == 0

    def test_drain_with_limit(self, tg: TelegramState):
        for i in range(5):
            tg.enqueue_burst(i, BurstReason.BURST)
        drained = tg.drain_burst_queue(limit=2)
        assert len(drained) == 2
        assert tg.burst_queue_size() == 3


# ── tg_clv_track ────────────────────────────────────────────────────────────


class TestClvTracking:
    def test_drift_positive_when_odd_rises(self, tg: TelegramState):
        drift = tg.record_clv_sample(
            pick_id=1, bookmaker_odd=2.20, emit_odd=2.00,
        )
        assert drift == pytest.approx(10.0)

    def test_drift_negative_when_odd_falls(self, tg: TelegramState):
        drift = tg.record_clv_sample(
            pick_id=1, bookmaker_odd=1.90, emit_odd=2.00,
        )
        assert drift == pytest.approx(-5.0)

    def test_drift_zero_when_unchanged(self, tg: TelegramState):
        drift = tg.record_clv_sample(
            pick_id=1, bookmaker_odd=2.00, emit_odd=2.00,
        )
        assert drift == pytest.approx(0.0)

    def test_zero_emit_odd_rejected(self, tg: TelegramState):
        with pytest.raises(ValueError):
            tg.record_clv_sample(pick_id=1, bookmaker_odd=2.00, emit_odd=0.0)

    def test_latest_returns_most_recent(self, tg: TelegramState):
        tg.record_clv_sample(
            pick_id=1, bookmaker_odd=2.10, emit_odd=2.00,
            sampled_at="2026-05-10T10:00:00+00:00",
        )
        tg.record_clv_sample(
            pick_id=1, bookmaker_odd=1.90, emit_odd=2.00,
            sampled_at="2026-05-10T11:00:00+00:00",
        )
        assert tg.latest_clv_drift(1) == pytest.approx(-5.0)

    def test_latest_none_when_no_samples(self, tg: TelegramState):
        assert tg.latest_clv_drift(999) is None
