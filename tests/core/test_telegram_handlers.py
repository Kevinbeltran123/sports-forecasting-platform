"""Layer-1 tests for Telegram Bot v2 handlers.

Focused on the deterministic surface area:
- callback-data codec round-trip + boundary (64-byte budget)
- on_callback_query idempotency (network retry, double-tap)
- /mute, /resume state effects
- /status, /top, /bankroll SQL shape vs a populated picks.db
- /placed reconciliation into picks + tg_actions

PTB Update/Context machinery mocked via AsyncMock — we never hit the
network. Operator user-id filter is bypassed by calling handlers
directly (PTB does the filter dispatch, we test the handler body).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bip.core.telegram.handlers import (
    cmd_bankroll,
    cmd_missed,
    cmd_mute,
    cmd_mute_market,
    cmd_odd,
    cmd_placed,
    cmd_resume,
    cmd_resume_market,
    cmd_status,
    cmd_top,
    cmd_undo_place,
    decode_callback,
    encode_callback,
    on_callback_query,
)
from bip.evaluation.live.pick_tracker import PickTracker
from bip.evaluation.live.telegram_state import (
    Action,
    Role,
    TelegramState,
    _utc_iso,
)
from bip.evaluation.live.value_detector import LivePick


# ── Fixtures ────────────────────────────────────────────────────────────────


def _pick(
    *, fixture_id: int = 1, minute: int = 30, market: str = "ou_2_5",
    selection: str = "over", edge_pct: float = 5.0,
    bookmaker_odd: float = 2.10, our_probability: float = 0.50,
) -> LivePick:
    return LivePick(
        fixture_id=fixture_id, minute=minute,
        home_team="Home FC", away_team="Away FC",
        market=market, selection=selection,
        bookmaker_id=2, bookmaker_odd=bookmaker_odd,
        our_probability=our_probability,
        fair_odd=1.0 / our_probability,
        edge_pct=edge_pct,
        kelly_fraction_full=0.05,
        suggested_stake_pct=1.0,
        logical_score=0.85,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "picks.db"


@pytest.fixture
def tracker(db_path: Path) -> PickTracker:
    return PickTracker(db_path=db_path)


@pytest.fixture
def state(db_path: Path, tracker: PickTracker) -> TelegramState:
    return TelegramState(db_path=db_path)


@pytest.fixture
def ctx(state: TelegramState, db_path: Path) -> MagicMock:
    """Build a minimal mock ContextTypes.DEFAULT_TYPE.

    The handlers only touch ``ctx.application.bot_data``, ``ctx.args``,
    and ``ctx.bot.edit_message_text``. Everything else is stubbed.
    """
    c = MagicMock()
    c.application.bot_data = {
        "state": state,
        "db_path": db_path,
        "operator_user_id": 12345,
    }
    c.args = []
    c.bot.edit_message_text = AsyncMock()
    return c


def _make_update_message() -> MagicMock:
    """Build a minimal mock Update with a .message that has reply_html."""
    u = MagicMock()
    u.message.reply_html = AsyncMock()
    u.effective_user.id = 12345
    return u


def _make_update_callback(callback_data: str, callback_id: str = "cb-1") -> MagicMock:
    """Build a minimal mock Update carrying a callback_query."""
    u = MagicMock()
    u.callback_query.data = callback_data
    u.callback_query.id = callback_id
    u.callback_query.from_user.id = 12345
    u.callback_query.answer = AsyncMock()
    u.callback_query.message.chat_id = 67890
    u.callback_query.message.message_id = 111
    u.callback_query.message.text_html = "<b>PICK</b> · Some content"
    u.callback_query.message.text = "PICK · Some content"
    return u


# ── Callback codec ──────────────────────────────────────────────────────────


class TestCallbackCodec:
    def test_round_trip_no_arg(self):
        data = encode_callback("skip", 42)
        verb, pid, arg = decode_callback(data)
        assert verb == "skip"
        assert pid == 42
        assert arg is None

    def test_round_trip_with_arg(self):
        data = encode_callback("place", 99, "1.5")
        verb, pid, arg = decode_callback(data)
        assert verb == "place"
        assert pid == 99
        assert arg == "1.5"

    def test_version_prefix(self):
        data = encode_callback("skip", 1)
        assert data.startswith("v1:")

    def test_decode_rejects_unknown_version(self):
        with pytest.raises(ValueError):
            decode_callback("v9:skip:1")

    def test_decode_rejects_malformed(self):
        with pytest.raises(ValueError):
            decode_callback("garbage")

    @pytest.mark.parametrize("pick_id,arg", [
        (1, None),
        (999999999, "1.5"),
        (2**31 - 1, "99.99"),
    ])
    def test_within_64_byte_budget(self, pick_id, arg):
        data = encode_callback("place", pick_id, arg)
        assert len(data.encode("utf-8")) <= 64

    def test_rejects_over_budget(self):
        # A 60-char arg pushes over the limit
        with pytest.raises(ValueError):
            encode_callback("place", 1, "x" * 80)


# ── on_callback_query: idempotency & state ──────────────────────────────────


class TestCallbackQueryPlace:
    @pytest.mark.asyncio
    async def test_first_place_records_action(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_callback(
            encode_callback("place", pid, "1.5"), callback_id="cb-A",
        )
        await on_callback_query(update, ctx)
        rec = state.get_action(pick_id=pid, action=Action.PLACED)
        assert rec is not None
        assert rec.stake_pct == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_first_place_marks_picks_db(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
        db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_callback(
            encode_callback("place", pid, "1.5"), callback_id="cb-A",
        )
        await on_callback_query(update, ctx)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT placed_at_betano, actual_stake_units "
                "FROM picks WHERE id=?", (pid,),
            ).fetchone()
        assert row["placed_at_betano"] == 1
        assert row["actual_stake_units"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_double_tap_ignored(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
    ):
        """Network retry: same callback_id arrives twice."""
        pid, _ = tracker.record(_pick())
        data = encode_callback("place", pid, "1.5")
        u1 = _make_update_callback(data, callback_id="cb-A")
        u2 = _make_update_callback(data, callback_id="cb-A")
        await on_callback_query(u1, ctx)
        await on_callback_query(u2, ctx)
        # Both invocations completed; state still shows the original stake.
        rec = state.get_action(pick_id=pid, action=Action.PLACED)
        assert rec.stake_pct == pytest.approx(1.5)
        # The second call answered with "Already placed" (string match best-effort)
        u2.callback_query.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_different_stake_blocked(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
    ):
        """Operator double-tap: PLACE 1.5% then PLACE 0.75% — second rejected,
        state preserves 1.5%.
        """
        pid, _ = tracker.record(_pick())
        u1 = _make_update_callback(
            encode_callback("place", pid, "1.5"), callback_id="cb-A",
        )
        u2 = _make_update_callback(
            encode_callback("place", pid, "0.75"), callback_id="cb-B",
        )
        await on_callback_query(u1, ctx)
        await on_callback_query(u2, ctx)
        rec = state.get_action(pick_id=pid, action=Action.PLACED)
        assert rec.stake_pct == pytest.approx(1.5), \
            "First placement wins; second PLACE click rejected"


class TestCallbackQuerySkip:
    @pytest.mark.asyncio
    async def test_skip_records_action(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_callback(
            encode_callback("skip", pid), callback_id="cb-s",
        )
        await on_callback_query(update, ctx)
        rec = state.get_action(pick_id=pid, action=Action.SKIPPED)
        assert rec is not None


class TestCallbackQueryRemind:
    @pytest.mark.asyncio
    async def test_remind_persists_in_state(
        self, state: TelegramState, ctx: MagicMock, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_callback(
            encode_callback("remind", pid, "5"), callback_id="cb-r",
        )
        await on_callback_query(update, ctx)
        reminders = state.get_state("reminders", default=[])
        assert len(reminders) == 1
        assert reminders[0]["pick_id"] == pid


class TestCallbackQueryMalformed:
    @pytest.mark.asyncio
    async def test_unknown_version_answered_gracefully(
        self, ctx: MagicMock,
    ):
        update = _make_update_callback("v9:place:1:1.5", callback_id="cb-x")
        await on_callback_query(update, ctx)
        update.callback_query.answer.assert_awaited()


# ── /mute and /resume ───────────────────────────────────────────────────────


class TestMuteResume:
    @pytest.mark.asyncio
    async def test_mute_default_30m(
        self, state: TelegramState, ctx: MagicMock,
    ):
        update = _make_update_message()
        ctx.args = []
        await cmd_mute(update, ctx)
        assert state.is_muted() is True

    @pytest.mark.asyncio
    async def test_mute_custom_minutes(
        self, state: TelegramState, ctx: MagicMock,
    ):
        update = _make_update_message()
        ctx.args = ["60"]
        await cmd_mute(update, ctx)
        assert state.is_muted() is True

    @pytest.mark.asyncio
    async def test_mute_zero_clears(
        self, state: TelegramState, ctx: MagicMock,
    ):
        from datetime import timedelta
        state.set_mute_until(
            datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        update = _make_update_message()
        ctx.args = ["0"]
        await cmd_mute(update, ctx)
        assert state.is_muted() is False

    @pytest.mark.asyncio
    async def test_mute_rejects_garbage(
        self, ctx: MagicMock,
    ):
        update = _make_update_message()
        ctx.args = ["abc"]
        await cmd_mute(update, ctx)
        update.message.reply_html.assert_awaited_once()
        called = update.message.reply_html.await_args.args[0]
        assert "Usage" in called

    @pytest.mark.asyncio
    async def test_resume_clears_mute(
        self, state: TelegramState, ctx: MagicMock,
    ):
        from datetime import timedelta
        state.set_mute_until(
            datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        update = _make_update_message()
        await cmd_resume(update, ctx)
        assert state.is_muted() is False


# ── /status, /top, /bankroll, /placed ───────────────────────────────────────


class TestStatus:
    @pytest.mark.asyncio
    async def test_empty_db_reports_zeros(self, ctx: MagicMock):
        update = _make_update_message()
        await cmd_status(update, ctx)
        update.message.reply_html.assert_awaited_once()
        msg = update.message.reply_html.await_args.args[0]
        assert "STATUS" in msg
        assert "<b>0</b>" in msg

    @pytest.mark.asyncio
    async def test_settled_picks_counted(
        self, ctx: MagicMock, tracker: PickTracker, db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.5, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        update = _make_update_message()
        await cmd_status(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Won <b>1</b>" in msg
        assert "+1.50u" in msg


class TestTop:
    @pytest.mark.asyncio
    async def test_returns_message_when_empty(self, ctx: MagicMock):
        update = _make_update_message()
        await cmd_top(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "No pending picks" in msg

    @pytest.mark.asyncio
    async def test_orders_by_edge_desc(
        self, ctx: MagicMock, tracker: PickTracker,
    ):
        tracker.record(_pick(fixture_id=1, edge_pct=5.0))
        tracker.record(_pick(fixture_id=2, edge_pct=12.0, market="m_high"))
        tracker.record(_pick(fixture_id=3, edge_pct=8.0, market="m_mid"))
        update = _make_update_message()
        await cmd_top(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        # Top edge first
        idx_high = msg.find("m_high")
        idx_mid = msg.find("m_mid")
        assert 0 < idx_high < idx_mid


class TestBankroll:
    @pytest.mark.asyncio
    async def test_empty_returns_zero(self, ctx: MagicMock):
        update = _make_update_message()
        await cmd_bankroll(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "BANKROLL" in msg
        assert "+0.00u" in msg

    @pytest.mark.asyncio
    async def test_includes_baseline_when_set(
        self, ctx: MagicMock, state: TelegramState,
    ):
        state.set_state("bankroll_baseline", 100.0)
        update = _make_update_message()
        await cmd_bankroll(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Baseline" in msg
        assert "100.0u" in msg


class TestPlacedCommand:
    @pytest.mark.asyncio
    async def test_records_placement(
        self, ctx: MagicMock, tracker: PickTracker,
        state: TelegramState, db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_message()
        ctx.args = [str(pid), "1.5", "2.05"]
        await cmd_placed(update, ctx)
        rec = state.get_action(pick_id=pid, action=Action.PLACED)
        assert rec is not None
        assert rec.stake_pct == pytest.approx(1.5)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT placed_at_betano, betano_odd, actual_stake_units "
                "FROM picks WHERE id=?", (pid,),
            ).fetchone()
        assert row["placed_at_betano"] == 1
        assert row["betano_odd"] == pytest.approx(2.05)
        assert row["actual_stake_units"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_rejects_missing_args(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = ["42"]  # missing stake
        await cmd_placed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Usage" in msg

    @pytest.mark.asyncio
    async def test_rejects_garbage_args(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = ["abc", "xyz"]
        await cmd_placed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Bad arguments" in msg

    @pytest.mark.asyncio
    async def test_double_placed_returns_already(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED,
            operator_user_id=12345, callback_id="prior",
            stake_pct=1.0,
        )
        update = _make_update_message()
        ctx.args = [str(pid), "1.5"]
        await cmd_placed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "already placed" in msg


# ── Tier C: /odd ────────────────────────────────────────────────────────────


class TestOddCommand:
    @pytest.mark.asyncio
    async def test_records_fill_price(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
        db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED, operator_user_id=12345,
            callback_id="cb-1", stake_pct=1.5,
        )
        update = _make_update_message()
        ctx.args = [str(pid), "2.05"]
        await cmd_odd(update, ctx)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT betano_odd FROM picks WHERE id=?", (pid,),
            ).fetchone()
        assert row["betano_odd"] == pytest.approx(2.05)
        msg = update.message.reply_html.await_args.args[0]
        assert "Fill price recorded" in msg

    @pytest.mark.asyncio
    async def test_rejects_unplaced_pick(
        self, ctx: MagicMock, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_message()
        ctx.args = [str(pid), "2.05"]
        await cmd_odd(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "not placed" in msg

    @pytest.mark.asyncio
    async def test_rejects_missing_args(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = ["42"]
        await cmd_odd(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Usage" in msg

    @pytest.mark.asyncio
    async def test_rejects_odd_le_one(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED, operator_user_id=12345,
            callback_id="cb-1", stake_pct=1.5,
        )
        update = _make_update_message()
        ctx.args = [str(pid), "0.95"]
        await cmd_odd(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "greater than 1.0" in msg

    @pytest.mark.asyncio
    async def test_rejects_garbage_args(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = ["abc", "xyz"]
        await cmd_odd(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Bad arguments" in msg


# ── Tier C: /undo_place ─────────────────────────────────────────────────────


class TestUndoPlaceCommand:
    @pytest.mark.asyncio
    async def test_removes_action_and_resets_db(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
        db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED, operator_user_id=12345,
            callback_id="cb-1", stake_pct=1.5,
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET placed_at_betano=1, "
                "actual_stake_units=1.5, betano_odd=2.05 WHERE id=?", (pid,),
            )
        update = _make_update_message()
        ctx.args = [str(pid)]
        await cmd_undo_place(update, ctx)
        # tg_actions row gone
        assert state.get_action(pick_id=pid, action=Action.PLACED) is None
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT placed_at_betano, actual_stake_units, betano_odd "
                "FROM picks WHERE id=?", (pid,),
            ).fetchone()
        assert row["placed_at_betano"] == 0
        assert row["actual_stake_units"] is None
        assert row["betano_odd"] is None

    @pytest.mark.asyncio
    async def test_nothing_to_undo(
        self, ctx: MagicMock, tracker: PickTracker,
    ):
        pid, _ = tracker.record(_pick())
        update = _make_update_message()
        ctx.args = [str(pid)]
        await cmd_undo_place(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "nothing to undo" in msg

    @pytest.mark.asyncio
    async def test_after_undo_can_replace(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED, operator_user_id=12345,
            callback_id="cb-1", stake_pct=1.5,
        )
        ctx.args = [str(pid)]
        await cmd_undo_place(_make_update_message(), ctx)
        # Now /placed should succeed
        ctx.args = [str(pid), "0.75"]
        await cmd_placed(_make_update_message(), ctx)
        rec = state.get_action(pick_id=pid, action=Action.PLACED)
        assert rec is not None
        assert rec.stake_pct == pytest.approx(0.75)


# ── Tier C: /missed ─────────────────────────────────────────────────────────


class TestMissedCommand:
    @pytest.mark.asyncio
    async def test_empty_when_nothing_missed(self, ctx: MagicMock):
        update = _make_update_message()
        await cmd_missed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Caught everything" in msg

    @pytest.mark.asyncio
    async def test_lists_won_picks_with_no_action(
        self, ctx: MagicMock, tracker: PickTracker, db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        # Settle as won, no action recorded
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.5, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        update = _make_update_message()
        await cmd_missed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "MISSED" in msg
        assert "1 unplaced" in msg
        assert "+1.50u" in msg

    @pytest.mark.asyncio
    async def test_excludes_won_picks_that_were_placed(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
        db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.PLACED, operator_user_id=12345,
            callback_id="cb-1", stake_pct=1.5,
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.5, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        update = _make_update_message()
        await cmd_missed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Caught everything" in msg

    @pytest.mark.asyncio
    async def test_excludes_won_picks_that_were_skipped(
        self, ctx: MagicMock, tracker: PickTracker, state: TelegramState,
        db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_action(
            pick_id=pid, action=Action.SKIPPED, operator_user_id=12345,
            callback_id="cb-1",
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.5, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        update = _make_update_message()
        await cmd_missed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Caught everything" in msg

    @pytest.mark.asyncio
    async def test_excludes_lost_picks(
        self, ctx: MagicMock, tracker: PickTracker, db_path: Path,
    ):
        pid, _ = tracker.record(_pick())
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='lost', profit_units=-1.0, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        update = _make_update_message()
        await cmd_missed(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        # Lost picks shouldn't appear in the regret list
        assert "Caught everything" in msg


# ── Tier C: /mute_market ────────────────────────────────────────────────────


class TestMuteMarketCommand:
    @pytest.mark.asyncio
    async def test_default_60m_when_no_minutes(
        self, ctx: MagicMock, state: TelegramState,
    ):
        update = _make_update_message()
        ctx.args = ["ou_2_5"]
        await cmd_mute_market(update, ctx)
        assert state.is_market_muted("ou_2_5") is True
        msg = update.message.reply_html.await_args.args[0]
        assert "60m" in msg

    @pytest.mark.asyncio
    async def test_custom_minutes(
        self, ctx: MagicMock, state: TelegramState,
    ):
        update = _make_update_message()
        ctx.args = ["btts", "30"]
        await cmd_mute_market(update, ctx)
        assert state.is_market_muted("btts") is True

    @pytest.mark.asyncio
    async def test_zero_minutes_clears(
        self, ctx: MagicMock, state: TelegramState,
    ):
        from datetime import datetime, timedelta, timezone
        state.set_market_mute(
            "ou_2_5", datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        update = _make_update_message()
        ctx.args = ["ou_2_5", "0"]
        await cmd_mute_market(update, ctx)
        assert state.is_market_muted("ou_2_5") is False

    @pytest.mark.asyncio
    async def test_no_args_lists_muted_markets(
        self, ctx: MagicMock, state: TelegramState,
    ):
        from datetime import datetime, timedelta, timezone
        state.set_market_mute(
            "ou_2_5", datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        update = _make_update_message()
        ctx.args = []
        await cmd_mute_market(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "MUTED MARKETS" in msg
        assert "ou_2_5" in msg

    @pytest.mark.asyncio
    async def test_no_args_no_mutes(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = []
        await cmd_mute_market(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "No markets currently muted" in msg

    @pytest.mark.asyncio
    async def test_rejects_garbage_minutes(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = ["ou_2_5", "abc"]
        await cmd_mute_market(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "non-negative integer" in msg


class TestResumeMarketCommand:
    @pytest.mark.asyncio
    async def test_clears_mute(
        self, ctx: MagicMock, state: TelegramState,
    ):
        from datetime import datetime, timedelta, timezone
        state.set_market_mute(
            "ou_2_5", datetime.now(timezone.utc) + timedelta(minutes=30),
        )
        update = _make_update_message()
        ctx.args = ["ou_2_5"]
        await cmd_resume_market(update, ctx)
        assert state.is_market_muted("ou_2_5") is False

    @pytest.mark.asyncio
    async def test_no_args_rejects(self, ctx: MagicMock):
        update = _make_update_message()
        ctx.args = []
        await cmd_resume_market(update, ctx)
        msg = update.message.reply_html.await_args.args[0]
        assert "Usage" in msg
