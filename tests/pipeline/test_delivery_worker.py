"""DeliveryWorker tests — Sprint 2 Ola B.

Tests:
  - Shadow mode marks rows as 'shadow', sender never called
  - Live mode sends Telegram + marks 'sent'
  - Claude REJECT kills all rows of a fixture
  - Missing EV/odds → killed
  - EV below threshold (after confidence modifier) → killed
  - Daily exposure cap fires after enough picks consumed
  - Validator None or exception is non-fatal
  - One validator call per fixture (not per pick)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from bip.pipeline.delivery_worker import DeliveryRunSummary, DeliveryWorker
from bip.pipeline.orchestrator import PREDICTIONS_RAW_TABLE
from tests.pipeline.conftest import FakeQuery, FakeSupabaseClient


def _pending_row(
    *,
    row_id: str = "row-1",
    fixture_id: str = "fx-1",
    market: str = "ou_2.5",
    selection: str = "over",
    p_model: float = 0.6,
    ev: float | None = 0.08,
    odds: float | None = 1.95,
    match_dt: datetime | None = None,
    source: str = "ligas",
    competition: str = "PL",
) -> dict:
    if match_dt is None:
        match_dt = datetime.now(UTC) + timedelta(hours=2)
    return {
        "id": row_id,
        "source": source,
        "fixture_id": fixture_id,
        "competition": competition,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "match_datetime": match_dt.isoformat(),
        "market": market,
        "selection": selection,
        "p_model": p_model,
        "ev": ev,
        "odds_at_pick": odds,
        "payload": {"stub": True},
        "model_version": "test-0.0",
        "status": "pending",
    }


def _seed_pending(client: FakeSupabaseClient, rows: list[dict]) -> None:
    client.table(PREDICTIONS_RAW_TABLE).rows = list(rows)


def _make_verdict(verdict: str, confidence_modifier: float = 0.0):
    class V:
        pass
    v = V()
    v.verdict = verdict
    v.confidence_modifier = confidence_modifier
    return v


@pytest.fixture
def now_utc() -> datetime:
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.asyncio
class TestShadowMode:
    async def test_shadow_marks_status_shadow(self, fake_client, now_utc):
        _seed_pending(
            fake_client,
            [_pending_row(match_dt=now_utc + timedelta(hours=2))],
        )
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        sender = AsyncMock()

        worker = DeliveryWorker(
            supabase_client=fake_client,
            validator=validator,
            sender=sender,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)

        assert summary.n_shadow == 1
        assert summary.n_sent == 0
        sender.send_pick.assert_not_called()
        # Update was an UPDATE with status='shadow'
        updates = fake_client.tables[PREDICTIONS_RAW_TABLE].updates
        assert any(payload.get("status") == "shadow" for _filt, payload in updates)


@pytest.mark.asyncio
class TestLiveMode:
    async def test_live_sends_telegram_and_marks_sent(self, fake_client, now_utc):
        _seed_pending(fake_client, [_pending_row(match_dt=now_utc + timedelta(hours=2))])
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        sender = AsyncMock()

        worker = DeliveryWorker(
            supabase_client=fake_client,
            validator=validator,
            sender=sender,
            shadow_mode=False,
        )
        summary = await worker.run_once(now=now_utc)

        assert summary.n_sent == 1
        assert summary.n_shadow == 0
        sender.send_pick.assert_awaited_once()
        # Telegram body includes pick info
        args = sender.send_pick.await_args
        text = args.kwargs.get("text") if args.kwargs else (args.args[0] if args.args else "")
        assert "Arsenal" in text and "Chelsea" in text
        # No emojis allowed (project convention)
        for forbidden in ("⚽", "🎯", "🏆", "🚀"):
            assert forbidden not in text

    async def test_live_telegram_failure_marks_killed(self, fake_client, now_utc):
        _seed_pending(fake_client, [_pending_row(match_dt=now_utc + timedelta(hours=2))])
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        sender = AsyncMock()
        sender.send_pick.side_effect = RuntimeError("Telegram timeout")

        worker = DeliveryWorker(
            supabase_client=fake_client,
            validator=validator,
            sender=sender,
            shadow_mode=False,
        )
        summary = await worker.run_once(now=now_utc)

        assert summary.n_sent == 0
        assert summary.n_killed == 1
        assert summary.kill_reasons.get("telegram_failure") == 1


@pytest.mark.asyncio
class TestClaudeGates:
    async def test_claude_reject_kills_all_fixture_picks(self, fake_client, now_utc):
        rows = [
            _pending_row(row_id="r1", fixture_id="fx-A", market="ou_2.5", selection="over",
                         match_dt=now_utc + timedelta(hours=2)),
            _pending_row(row_id="r2", fixture_id="fx-A", market="1x2", selection="home",
                         match_dt=now_utc + timedelta(hours=2)),
        ]
        _seed_pending(fake_client, rows)
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("REJECT")

        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)
        assert summary.n_killed == 2
        assert summary.kill_reasons.get("claude_reject") == 2
        assert summary.n_shadow == 0
        # Only one validator call (per fixture, not per pick)
        validator.validate.assert_awaited_once()

    async def test_one_validator_call_per_fixture_even_with_many_picks(
        self, fake_client, now_utc
    ):
        rows = [
            _pending_row(row_id=f"r{i}", fixture_id="fx-Z", market="1x2",
                         selection=("home", "draw", "away")[i % 3],
                         match_dt=now_utc + timedelta(hours=2))
            for i in range(5)
        ]
        _seed_pending(fake_client, rows)
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")

        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)
        validator.validate.assert_awaited_once()
        assert summary.n_validator_calls == 1
        assert summary.n_pending_rows == 5

    async def test_confidence_modifier_can_promote_ev(self, fake_client, now_utc):
        # EV starts below threshold (0.02 < 0.03); +0.05 modifier raises it
        _seed_pending(
            fake_client,
            [_pending_row(ev=0.02, odds=2.0, match_dt=now_utc + timedelta(hours=2))],
        )
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM", confidence_modifier=0.05)

        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True, ev_threshold=0.03,
        )
        summary = await worker.run_once(now=now_utc)
        assert summary.n_shadow == 1
        assert summary.n_killed == 0

    async def test_validator_exception_is_non_fatal(self, fake_client, now_utc):
        _seed_pending(fake_client, [_pending_row(match_dt=now_utc + timedelta(hours=2))])
        validator = AsyncMock()
        validator.validate.side_effect = RuntimeError("Anthropic down")

        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)
        # Validator error counted; SKIPPED verdict treated as no-op modifier
        assert summary.n_errors == 1
        # The pick still goes through (no modifier, EV=0.08 > 0.03)
        assert summary.n_shadow == 1


@pytest.mark.asyncio
class TestGatesAndSafety:
    async def test_missing_ev_or_odds_is_killed(self, fake_client, now_utc):
        _seed_pending(
            fake_client,
            [_pending_row(ev=None, odds=None, match_dt=now_utc + timedelta(hours=2))],
        )
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)
        assert summary.n_killed == 1
        assert summary.kill_reasons.get("missing_ev_or_odds") == 1

    async def test_ev_below_threshold_killed(self, fake_client, now_utc):
        _seed_pending(
            fake_client,
            [_pending_row(ev=0.01, odds=2.0, match_dt=now_utc + timedelta(hours=2))],
        )
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True, ev_threshold=0.05,
        )
        summary = await worker.run_once(now=now_utc)
        assert summary.n_killed == 1
        assert summary.kill_reasons.get("ev_below_threshold") == 1

    async def test_daily_exposure_cap_fires(self, fake_client, now_utc):
        # 20 picks at 5% each would exceed 15% cap quickly. With per-pick cap
        # set high we deliberately fire the daily cap on later rows.
        rows = [
            _pending_row(
                row_id=f"r{i}", fixture_id=f"fx-{i}", ev=0.15, odds=2.0,
                match_dt=now_utc + timedelta(hours=2),
            )
            for i in range(20)
        ]
        _seed_pending(fake_client, rows)
        validator = AsyncMock()
        validator.validate.return_value = _make_verdict("CONFIRM")
        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
            ev_threshold=0.03,
            daily_exposure_cap=0.10,
            max_stake_per_pick=0.05,
        )
        summary = await worker.run_once(now=now_utc)
        # Some early picks pass, later ones killed by exposure cap
        assert summary.n_killed >= 1
        assert summary.kill_reasons.get("daily_exposure_cap", 0) >= 1
        # The first few picks went through shadow
        assert summary.n_shadow > 0


@pytest.mark.asyncio
class TestEmptyPending:
    async def test_empty_pending_returns_zero_summary(self, fake_client, now_utc):
        validator = AsyncMock()
        worker = DeliveryWorker(
            supabase_client=fake_client, validator=validator, sender=None,
            shadow_mode=True,
        )
        summary = await worker.run_once(now=now_utc)
        assert summary == DeliveryRunSummary()  # all zeros
        validator.validate.assert_not_called()
