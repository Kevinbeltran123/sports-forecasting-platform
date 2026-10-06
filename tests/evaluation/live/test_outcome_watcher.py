"""Tests for OutcomeWatcher periodic task.

Focus areas:
- One reply per settled pick (idempotent across ticks)
- Outcome reply uses reply_to_message_id from tg_messages
- Scoreboard edits when state has scoreboard_msg_id; reposts on edit fail
- P/L computation handles won / lost / void
- Failure-safe: bot errors don't propagate
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from bip.evaluation.live.outcome_watcher import (
    OutcomeWatcher,
    _actual_profit_units,
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


def _pick(*, fixture_id: int = 1, minute: int = 30,
          market: str = "ou_2_5", selection: str = "over",
          edge_pct: float = 8.0) -> LivePick:
    return LivePick(
        fixture_id=fixture_id, minute=minute,
        home_team="Home", away_team="Away",
        market=market, selection=selection,
        bookmaker_id=2, bookmaker_odd=2.00,
        our_probability=0.55, fair_odd=1.82,
        edge_pct=edge_pct,
        kelly_fraction_full=0.05, suggested_stake_pct=1.0,
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
def bot() -> AsyncMock:
    b = AsyncMock()
    b.send_html = AsyncMock(return_value=99000)  # fresh message_id per send
    b.edit_html = AsyncMock()
    return b


@pytest.fixture
def watcher(bot, state, db_path) -> OutcomeWatcher:
    return OutcomeWatcher(
        bot=bot, state=state, db_path=db_path,
        primary_channel_id="-100PRIMARY",
    )


def _settle(db_path: Path, pick_id: int, *, status: str,
            profit_units: float) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE picks SET status=?, profit_units=?, settled_at=? "
            "WHERE id=?",
            (status, profit_units, _utc_iso(), pick_id),
        )


# ── _actual_profit_units helper ─────────────────────────────────────────────


class TestActualProfitUnits:
    @pytest.mark.parametrize("stake,odd,expected", [
        (1.5, 2.00, 1.5),    # won: stake * (odd-1)
        (1.0, 2.10, 1.1),
        (1.5, 1.50, 0.75),
    ])
    def test_won(self, stake, odd, expected):
        assert _actual_profit_units(
            status="won", stake_pct=stake, odd=odd,
        ) == pytest.approx(expected)

    def test_lost_returns_negative_stake(self):
        assert _actual_profit_units(
            status="lost", stake_pct=1.5, odd=2.00,
        ) == pytest.approx(-1.5)

    def test_void_returns_zero(self):
        assert _actual_profit_units(
            status="void", stake_pct=1.5, odd=2.00,
        ) == pytest.approx(0.0)


# ── outcome replies ─────────────────────────────────────────────────────────


class TestOutcomeReplies:
    @pytest.mark.asyncio
    async def test_no_settled_picks_no_replies(
        self, watcher, bot, tracker, state,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        # status stays 'pending' → no reply, but scoreboard still updates
        n = await watcher.tick()
        assert n == 0
        # Verify no outcome-reply send specifically (only scoreboard).
        reply_calls = [
            c for c in bot.send_html.await_args_list
            if c.kwargs.get("reply_to_message_id") is not None
        ]
        assert reply_calls == []

    @pytest.mark.asyncio
    async def test_won_pick_replies(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="won", profit_units=1.00)

        n = await watcher.tick()
        assert n == 1
        bot.send_html.assert_awaited()
        call = bot.send_html.await_args_list[0]
        text = call.args[0] if call.args else call.kwargs.get("text", "")
        assert "WON" in text
        assert call.kwargs.get("reply_to_message_id") == 500
        assert call.kwargs.get("chat_id") == "-100PRIMARY"

    @pytest.mark.asyncio
    async def test_outcome_persisted_no_double_send(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="won", profit_units=1.00)

        n1 = await watcher.tick()
        n2 = await watcher.tick()
        assert n1 == 1
        assert n2 == 0, "Second tick must not re-fire the reply"

    @pytest.mark.asyncio
    async def test_outcome_message_id_recorded(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="lost", profit_units=-1.00)
        bot.send_html.return_value = 99123  # fake outcome reply message_id

        await watcher.tick()
        ref = state.find_message(pick_id=pid, role=Role.OUTCOME)
        assert ref is not None
        assert ref.message_id == 99123

    @pytest.mark.asyncio
    async def test_placement_info_included_when_present(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        # Record placement
        state.record_action(
            pick_id=pid, action=Action.PLACED,
            operator_user_id=12345, callback_id="cb-1",
            stake_pct=1.5,
        )
        # Set the betano_odd in picks.db
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET placed_at_betano=1, "
                "actual_stake_units=1.5, betano_odd=2.05 WHERE id=?", (pid,),
            )
        _settle(db_path, pid, status="won", profit_units=1.00)

        await watcher.tick()
        # Outcome reply is the call with reply_to_message_id; scoreboard
        # is a separate call.
        reply_call = next(
            c for c in bot.send_html.await_args_list
            if c.kwargs.get("reply_to_message_id") is not None
        )
        text = reply_call.args[0]
        assert "Placed" in text
        assert "1.50%" in text
        assert "2.05" in text

    @pytest.mark.asyncio
    async def test_skips_picks_without_pick_message(
        self, watcher, bot, tracker, state, db_path,
    ):
        # Settled pick but no tg_messages entry → cannot reply, must skip.
        pid, _ = tracker.record(_pick())
        _settle(db_path, pid, status="won", profit_units=1.0)
        n = await watcher.tick()
        assert n == 0
        # No outcome reply (scoreboard still fires)
        reply_calls = [
            c for c in bot.send_html.await_args_list
            if c.kwargs.get("reply_to_message_id") is not None
        ]
        assert reply_calls == []

    @pytest.mark.asyncio
    async def test_failure_logged_not_raised(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="won", profit_units=1.0)
        bot.send_html.side_effect = RuntimeError("net down")
        # Must not raise
        await watcher.tick()
        assert watcher.n_failures >= 1


# ── scoreboard ──────────────────────────────────────────────────────────────


class TestScoreboard:
    @pytest.mark.asyncio
    async def test_first_render_creates_pin(
        self, watcher, bot, tracker, state,
    ):
        tracker.record(_pick(fixture_id=1))
        bot.send_html.return_value = 77777

        await watcher.update_scoreboard()
        assert state.get_state("scoreboard_msg_id") == 77777
        bot.send_html.assert_awaited()

    @pytest.mark.asyncio
    async def test_subsequent_render_edits_pin(
        self, watcher, bot, tracker, state,
    ):
        tracker.record(_pick(fixture_id=1))
        state.set_state("scoreboard_msg_id", 77777)

        await watcher.update_scoreboard()
        bot.edit_html.assert_awaited_once()
        bot.send_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_failure_reposts(
        self, watcher, bot, tracker, state,
    ):
        tracker.record(_pick(fixture_id=1))
        state.set_state("scoreboard_msg_id", 77777)
        bot.edit_html.side_effect = RuntimeError("message too old")
        bot.send_html.return_value = 88888

        await watcher.update_scoreboard()
        assert state.get_state("scoreboard_msg_id") == 88888

    @pytest.mark.asyncio
    async def test_snapshot_counts_today_only(
        self, watcher, bot, tracker, state, db_path,
    ):
        # Two picks today; backdate one outside today
        pid1, _ = tracker.record(_pick(fixture_id=1))
        pid2, _ = tracker.record(_pick(fixture_id=2))
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET emitted_at='2020-01-01T00:00:00+00:00' "
                "WHERE id=?", (pid2,),
            )
        snap = watcher._compute_scoreboard_snapshot()
        assert snap["n_emit"] == 1

    @pytest.mark.asyncio
    async def test_day_rollover_reposts_fresh_pin(
        self, watcher, bot, tracker, state,
    ):
        """When scoreboard_date is older than today, abandon and repost."""
        tracker.record(_pick(fixture_id=1))
        state.set_state("scoreboard_msg_id", 55555)
        state.set_state("scoreboard_date", "2020-01-01")
        bot.send_html.return_value = 99999

        await watcher.update_scoreboard()
        # Should NOT edit (yesterday's pin abandoned)
        bot.edit_html.assert_not_awaited()
        # Should send fresh
        bot.send_html.assert_awaited_once()
        # State updated to today's date + new msg id
        assert state.get_state("scoreboard_msg_id") == 99999
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        assert state.get_state("scoreboard_date") == today

    @pytest.mark.asyncio
    async def test_day_rollover_resets_daily_peak(
        self, watcher, bot, tracker, state,
    ):
        """today_peak_pl is per-day; rollover clears it so drawdown
        recomputes against today's path, not yesterday's high-water.
        """
        tracker.record(_pick(fixture_id=1))
        state.set_state("scoreboard_msg_id", 55555)
        state.set_state("scoreboard_date", "2020-01-01")
        state.set_state("today_peak_pl", 12.5)
        state.set_state("drawdown_alerted_date", "2020-01-01")

        await watcher.update_scoreboard()
        assert state.get_state("today_peak_pl") in (None, 0.0)
        assert state.get_state("drawdown_alerted_date") is None

    @pytest.mark.asyncio
    async def test_same_day_does_not_rollover(
        self, watcher, bot, tracker, state,
    ):
        tracker.record(_pick(fixture_id=1))
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        state.set_state("scoreboard_msg_id", 55555)
        state.set_state("scoreboard_date", today)

        await watcher.update_scoreboard()
        bot.edit_html.assert_awaited_once()
        bot.send_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pl_placed_won(
        self, watcher, bot, tracker, db_path,
    ):
        pid, _ = tracker.record(_pick())
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.0, "
                "placed_at_betano=1, actual_stake_units=1.5, "
                "betano_odd=2.00, settled_at=? WHERE id=?",
                (_utc_iso(), pid),
            )
        snap = watcher._compute_scoreboard_snapshot()
        # actual P/L = 1.5 * (2.0-1) = 1.5
        assert snap["pl_placed"] == pytest.approx(1.5)
        assert snap["pl_emit"] == pytest.approx(1.0)


# ── tick() composes both ────────────────────────────────────────────────────


# ── V2.1: streak alerts (B4) ────────────────────────────────────────────────


class TestStreakAlerts:
    @pytest.mark.asyncio
    async def test_win_streak_fires_at_exactly_3(
        self, watcher, bot, tracker, state, db_path,
    ):
        # Pre-record three wins, each with a pick message
        pids = []
        for i in range(3):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="won", profit_units=1.0)
            pids.append(pid)
        await watcher.tick()
        # One streak alert + 3 outcome replies + 1 scoreboard repost
        streak_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "HOT STREAK" in c.args[0]
        ]
        assert len(streak_sends) == 1
        # 3 in a row
        assert "3" in streak_sends[0].args[0]

    @pytest.mark.asyncio
    async def test_loss_streak_fires_at_exactly_3(
        self, watcher, bot, tracker, state, db_path,
    ):
        for i in range(3):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        streak_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "COLD STREAK" in c.args[0]
        ]
        assert len(streak_sends) == 1

    @pytest.mark.asyncio
    async def test_streak_alert_debounced_at_4_and_5(
        self, watcher, bot, tracker, state, db_path,
    ):
        # Cross threshold, then settle 2 more wins — must NOT re-alert.
        for i in range(5):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="won", profit_units=1.0)
        await watcher.tick()
        streak_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "HOT STREAK" in c.args[0]
        ]
        assert len(streak_sends) == 1, \
            "Only one alert per streak, regardless of how long it gets"

    @pytest.mark.asyncio
    async def test_streak_resets_on_opposite_outcome(
        self, watcher, bot, tracker, state, db_path,
    ):
        # 3 wins → fire HOT, then a loss → streak resets to loss/1.
        for i, status in enumerate(["won", "won", "won", "lost"]):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status=status,
                    profit_units=1.0 if status == "won" else -1.0)
        await watcher.tick()
        streak = state.get_state("current_streak")
        assert streak["kind"] == "loss"
        assert streak["count"] == 1
        # No COLD alert yet (count<3)
        cold_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "COLD STREAK" in c.args[0]
        ]
        assert cold_sends == []

    @pytest.mark.asyncio
    async def test_void_does_not_break_streak(
        self, watcher, bot, tracker, state, db_path,
    ):
        # 2 wins, then a void, then 1 more win → should fire HOT at win 3.
        for i, status in enumerate(["won", "won", "void", "won"]):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            pl = (1.0 if status == "won"
                  else 0.0 if status == "void"
                  else -1.0)
            _settle(db_path, pid, status=status, profit_units=pl)
        await watcher.tick()
        hot_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "HOT STREAK" in c.args[0]
        ]
        assert len(hot_sends) == 1
        assert state.get_state("current_streak")["kind"] == "win"
        assert state.get_state("current_streak")["count"] == 3

    @pytest.mark.asyncio
    async def test_streak_routes_to_diag_channel(
        self, bot, state, db_path, tracker,
    ):
        # Construct watcher with explicit diag channel
        watcher = OutcomeWatcher(
            bot=bot, state=state, db_path=db_path,
            primary_channel_id="-100PRIMARY",
            diag_channel_id="-100DIAG",
        )
        for i in range(3):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="won", profit_units=1.0)
        await watcher.tick()
        streak_call = next(
            c for c in bot.send_html.await_args_list
            if c.args and "HOT STREAK" in c.args[0]
        )
        assert streak_call.kwargs.get("chat_id") == "-100DIAG"


# ── V2.1: drawdown alerts (B4) ──────────────────────────────────────────────


class TestDrawdownAlerts:
    @pytest.mark.asyncio
    async def test_fires_below_units_fallback(
        self, watcher, bot, tracker, state, db_path,
    ):
        # No baseline → -3.0u absolute fallback
        for i in range(4):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        dd_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "DRAWDOWN ALERT" in c.args[0]
        ]
        assert len(dd_sends) == 1

    @pytest.mark.asyncio
    async def test_fires_below_pct_threshold_when_baseline_set(
        self, watcher, bot, tracker, state, db_path,
    ):
        state.set_state("bankroll_baseline", 100.0)
        # P/L = -4u of 100u = -4% → past -3% threshold
        for i in range(4):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        dd_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "DRAWDOWN ALERT" in c.args[0]
        ]
        assert len(dd_sends) == 1
        # P/L line should include bankroll percentage
        assert "bankroll" in dd_sends[0].args[0]

    @pytest.mark.asyncio
    async def test_no_fire_when_above_threshold(
        self, watcher, bot, tracker, state, db_path,
    ):
        state.set_state("bankroll_baseline", 100.0)
        # P/L = -1u of 100u = -1% → above -3% threshold
        pid, _ = tracker.record(_pick(fixture_id=1))
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY",
            message_id=500, role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        dd_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "DRAWDOWN ALERT" in c.args[0]
        ]
        assert dd_sends == []

    @pytest.mark.asyncio
    async def test_debounced_once_per_day(
        self, watcher, bot, tracker, state, db_path,
    ):
        # Two ticks, both with crossing P/L — second tick must NOT re-fire.
        for i in range(4):
            pid, _ = tracker.record(_pick(fixture_id=i + 1))
            state.record_message(
                pick_id=pid, channel_id="-100PRIMARY",
                message_id=500 + i, role=Role.PICK, tier=2,
            )
            _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        bot.send_html.reset_mock()

        # Settle one more loss; tick again.
        pid, _ = tracker.record(_pick(fixture_id=99))
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY",
            message_id=999, role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="lost", profit_units=-1.0)
        await watcher.tick()
        dd_sends = [
            c for c in bot.send_html.await_args_list
            if c.args and "DRAWDOWN ALERT" in c.args[0]
        ]
        assert dd_sends == [], \
            "Drawdown debounced after first crossing per day"


class TestTickComposition:
    @pytest.mark.asyncio
    async def test_tick_runs_outcomes_then_scoreboard(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="won", profit_units=1.0)

        await watcher.tick()
        # send_html called twice: 1) outcome reply, 2) scoreboard pin (first
        # time, no edit yet)
        assert bot.send_html.await_count == 2

    @pytest.mark.asyncio
    async def test_tick_resilient_to_scoreboard_failure(
        self, watcher, bot, tracker, state, db_path,
    ):
        pid, _ = tracker.record(_pick())
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        _settle(db_path, pid, status="won", profit_units=1.0)
        # First send_html (outcome reply) succeeds, second (scoreboard) fails
        bot.send_html.side_effect = [99999, RuntimeError("pin failed")]

        # Must not raise
        n = await watcher.tick()
        assert n == 1
        assert watcher.n_failures >= 1
