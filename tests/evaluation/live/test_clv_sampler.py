"""Layer-1 tests for CLVSampler (§G #1).

Focus:
- record_clv_sample called once per open pick
- Footer edit fires only when |drift| >= threshold
- Footer edit skipped when operator already acted (placed/skipped)
- Footer edit skipped when no PICK message recorded
- Resolver errors logged, not raised
- Lookback window excludes stale picks
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bip.evaluation.live.clv_sampler import CLVSampler
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
    *, fixture_id: int = 1, minute: int = 30,
    market: str = "ou_2_5", selection: str = "over",
    bookmaker_odd: float = 2.00,
) -> LivePick:
    return LivePick(
        fixture_id=fixture_id, minute=minute,
        home_team="Home", away_team="Away",
        market=market, selection=selection,
        bookmaker_id=2, bookmaker_odd=bookmaker_odd,
        our_probability=0.55, fair_odd=1.82,
        edge_pct=10.0,
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
    b.send_html = AsyncMock(return_value=99000)
    b.edit_html = AsyncMock()
    return b


def _make_sampler(bot, state, db_path, *,
                  odds_resolver=None, threshold_pct=5.0) -> CLVSampler:
    return CLVSampler(
        bot=bot, state=state, db_path=db_path,
        odds_resolver=odds_resolver or AsyncMock(return_value=None),
        threshold_pct=threshold_pct,
    )


# ── basic sampling ──────────────────────────────────────────────────────────


class TestSampling:
    @pytest.mark.asyncio
    async def test_no_open_picks_no_calls(self, bot, state, db_path):
        resolver = AsyncMock(return_value=2.10)
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 0
        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolver_called_per_open_pick(
        self, bot, state, db_path, tracker,
    ):
        tracker.record(_pick(fixture_id=1))
        tracker.record(_pick(fixture_id=2))
        resolver = AsyncMock(return_value=2.10)
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 2
        assert resolver.await_count == 2

    @pytest.mark.asyncio
    async def test_resolver_none_skipped(
        self, bot, state, db_path, tracker,
    ):
        tracker.record(_pick(fixture_id=1))
        resolver = AsyncMock(return_value=None)  # no fresh odd
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 0
        # No CLV row written
        assert state.latest_clv_drift(1) is None

    @pytest.mark.asyncio
    async def test_resolver_zero_or_negative_skipped(
        self, bot, state, db_path, tracker,
    ):
        tracker.record(_pick(fixture_id=1))
        resolver = AsyncMock(side_effect=[0.0, -1.0])
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 0

    @pytest.mark.asyncio
    async def test_resolver_error_logged_not_raised(
        self, bot, state, db_path, tracker,
    ):
        tracker.record(_pick(fixture_id=1))
        resolver = AsyncMock(side_effect=RuntimeError("api down"))
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        await sampler.tick()  # must not raise
        assert sampler.n_resolver_errors == 1

    @pytest.mark.asyncio
    async def test_stale_picks_excluded_by_lookback(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1))
        # Backdate the pick well outside the 3h lookback
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET emitted_at='2020-01-01T00:00:00+00:00' "
                "WHERE id=?", (pid,),
            )
        resolver = AsyncMock(return_value=2.10)
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 0
        resolver.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_settled_picks_excluded(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1))
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE picks SET status='won', profit_units=1.0, "
                "settled_at=? WHERE id=?", (_utc_iso(), pid),
            )
        resolver = AsyncMock(return_value=2.10)
        sampler = _make_sampler(bot, state, db_path, odds_resolver=resolver)
        n = await sampler.tick()
        assert n == 0


# ── footer edit logic ──────────────────────────────────────────────────────


class TestFooterEdit:
    @pytest.mark.asyncio
    async def test_no_edit_when_drift_below_threshold(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        # Drift 1% → below 5% threshold
        resolver = AsyncMock(return_value=2.02)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        bot.edit_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_when_drift_above_threshold_positive(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100PRIMARY", message_id=500,
            role=Role.PICK, tier=2,
        )
        # Drift +10% (line moved up)
        resolver = AsyncMock(return_value=2.20)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        bot.edit_html.assert_awaited_once()
        call = bot.edit_html.await_args
        assert call.kwargs.get("chat_id") == "-100PRIMARY"
        assert call.kwargs.get("message_id") == 500
        text = call.kwargs.get("text")
        assert "CLV" in text
        assert "+10.0%" in text or "10.0%" in text
        assert "2.00" in text and "2.20" in text

    @pytest.mark.asyncio
    async def test_edit_when_drift_above_threshold_negative(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        # Drift -10%
        resolver = AsyncMock(return_value=1.80)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        bot.edit_html.assert_awaited_once()
        text = bot.edit_html.await_args.kwargs.get("text")
        assert "-10.0%" in text

    @pytest.mark.asyncio
    async def test_no_edit_after_operator_placed(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        state.record_action(
            pick_id=pid, action=Action.PLACED,
            operator_user_id=999, callback_id="cb-1", stake_pct=1.5,
        )
        resolver = AsyncMock(return_value=2.20)  # big drift
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        # Drift recorded, but no edit (operator already acted)
        assert state.latest_clv_drift(pid) == pytest.approx(10.0)
        bot.edit_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_edit_after_operator_skipped(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        state.record_action(
            pick_id=pid, action=Action.SKIPPED,
            operator_user_id=999, callback_id="cb-2",
        )
        resolver = AsyncMock(return_value=2.20)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        bot.edit_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_edit_when_no_pick_message(
        self, bot, state, db_path, tracker,
    ):
        # CLV row written but no edit attempted (nothing to edit)
        tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        resolver = AsyncMock(return_value=2.20)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        bot.edit_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_edit_error_logged_not_raised(
        self, bot, state, db_path, tracker,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        bot.edit_html.side_effect = RuntimeError("message too old")
        resolver = AsyncMock(return_value=2.20)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        assert sampler.n_edit_errors == 1


class TestThresholdBoundary:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("current_odd,should_edit", [
        (2.099, False),   # +4.95% — just below 5%
        (2.10, True),     # +5.00% — exact boundary, included
        (1.901, False),   # -4.95%
        (1.90, True),     # -5.00%
    ])
    async def test_threshold_boundary(
        self, bot, state, db_path, tracker, current_odd, should_edit,
    ):
        pid, _ = tracker.record(_pick(fixture_id=1, bookmaker_odd=2.00))
        state.record_message(
            pick_id=pid, channel_id="-100", message_id=500,
            role=Role.PICK, tier=2,
        )
        resolver = AsyncMock(return_value=current_odd)
        sampler = _make_sampler(
            bot, state, db_path, odds_resolver=resolver, threshold_pct=5.0,
        )
        await sampler.tick()
        if should_edit:
            bot.edit_html.assert_awaited_once()
        else:
            bot.edit_html.assert_not_awaited()
