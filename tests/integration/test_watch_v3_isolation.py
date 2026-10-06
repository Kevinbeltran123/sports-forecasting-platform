"""End-to-end isolation test: v3 dual-write CANNOT affect v2's pick path.

The dual-write unit tests in tests/evaluation/live/engine_v3/test_dual_write.py
prove the WRAPPER catches exceptions. This test proves the END-TO-END
ordering invariant of scan_round in an offline research script (not published):
v2 generates the pick AND sends it to Telegram BEFORE v3 dual-write runs
on the same frame. So even if v3 raises catastrophically, v2's effects
have already happened — Telegram has received the alert, the pick tracker
has recorded the pick.

Why this matters operationally: a Telegram-driven betting workflow has a
single end-user contract — when a pick fires, it shows up in chat. v3 is
a shadow experiment. The contract MUST NOT break because the experiment
broke. This test asserts that contract directly with adversarial inputs.

Scenario simulated:
- Mocked LiveAlertSender records every send_pick_safe(pick).
- DualWriteRuntime is built with an _ExplodingPipeline that raises on
  every run.
- We invoke the relevant slice of scan_round per-fixture: send the v2
  pick to Telegram, then call await v3_runtime.run_shadow(...).
- Assertions:
  - Telegram sender received the pick exactly once.
  - v3 runtime error_count incremented (proving v3 actually tried).
  - No exception escaped the slice.
  - v3 success_count is 0.
- Additional sanity: even with a SLOW pipeline that hits the wall-clock
  timeout, the same invariant holds.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bip.evaluation.live.engine_v3 import (
    ShadowLogger,
    V3Pipeline,
)
from bip.evaluation.live.engine_v3.runtime.dual_write import DualWriteRuntime
from tests.evaluation.live.engine_v3.conftest import make_state


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ──────────────────────────────────────────────────────────────────────
# Recording stubs
# ──────────────────────────────────────────────────────────────────────


class _RecordingTelegramSender:
    """Stand-in for LiveAlertSender. Records every send_pick_safe call."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def send_pick_safe(
        self,
        pick: Any,
        *,
        pick_id: int | None = None,
        home_score: int | None = None,
        away_score: int | None = None,
    ) -> bool:
        self.received.append(
            {
                "pick": pick,
                "pick_id": pick_id,
                "home_score": home_score,
                "away_score": away_score,
            }
        )
        return True


class _ExplodingPipeline:
    """V3 pipeline that always raises. Worst-case adversary."""

    def run(self, *args: Any, **kwargs: Any):
        raise RuntimeError("synthetic v3 crash — must not reach v2")


class _SlowPipeline:
    """V3 pipeline that exceeds the wall-clock timeout."""

    def run(self, *args: Any, **kwargs: Any):
        time.sleep(0.3)
        return None


def _build_runtime(
    tmp_path: Path,
    *,
    pipeline_cls: type,
    timeout_sec: float = 0.8,
) -> DualWriteRuntime:
    runtime = DualWriteRuntime(
        pipeline=V3Pipeline(),
        logger=ShadowLogger(output_root=tmp_path / "shadow"),
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_ISO_TEST",
        timeout_sec=timeout_sec,
    )
    runtime.pipeline = pipeline_cls()  # type: ignore[assignment]
    return runtime


def _odd(market_id: str = "ou_2_5_over", value: str = "2.10") -> Any:
    """Minimal Sportmonks-shaped odd row."""
    return SimpleNamespace(
        market_id=market_id,
        value=value,
        market_description=market_id,
        bookmaker_id=2,
    )


# ──────────────────────────────────────────────────────────────────────
# The order-of-operations invariant
# ──────────────────────────────────────────────────────────────────────


async def _scan_round_slice(
    *,
    telegram_sender: _RecordingTelegramSender,
    v2_pick: Any,
    v3_runtime: DualWriteRuntime,
    state: Any,
    fixture: Any,
    odds: list,
) -> None:
    """Mirror of the per-fixture slice from an offline research script (not published).

    The order matches the real watch loop:
      1. v2 generates pick (already done by the caller — passed in).
      2. Send to Telegram.
      3. Run v3 dual-write.

    No try/except wraps step 2 OR step 3 here on purpose: we want to
    prove that the runtime CONTAINS its own failures so the caller
    doesn't need to defend.
    """
    await telegram_sender.send_pick_safe(
        v2_pick,
        pick_id=42,
        home_score=state.home_goals,
        away_score=state.away_goals,
    )
    await v3_runtime.run_shadow(
        state=state, fixture=fixture, odds=odds, now_utc=datetime.now(UTC)
    )


@pytest.mark.anyio("asyncio")
async def test_v3_explosion_does_not_block_v2_telegram_send(tmp_path, monkeypatch):
    """The headline invariant: even if v3 raises, v2's pick already reached Telegram."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "true")
    sender = _RecordingTelegramSender()
    runtime = _build_runtime(tmp_path, pipeline_cls=_ExplodingPipeline)

    state = make_state(home_goals=1, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=99, predictions=None)
    odds = [_odd()]
    pick = SimpleNamespace(name="v2-pick-1", edge_pct=5.5)

    await _scan_round_slice(
        telegram_sender=sender,
        v2_pick=pick,
        v3_runtime=runtime,
        state=state,
        fixture=fixture,
        odds=odds,
    )

    # v2 path completed: Telegram received the pick.
    assert len(sender.received) == 1
    assert sender.received[0]["pick"].name == "v2-pick-1"
    assert sender.received[0]["pick_id"] == 42

    # v3 path tried AND failed cleanly: error counter incremented,
    # no success, no exception bubbled up.
    assert runtime.error_count == 1
    assert runtime.success_count == 0


@pytest.mark.anyio("asyncio")
async def test_v3_timeout_does_not_block_v2_telegram_send(tmp_path, monkeypatch):
    """Same invariant under timeout (slow v3 pipeline)."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "true")
    sender = _RecordingTelegramSender()
    runtime = _build_runtime(
        tmp_path, pipeline_cls=_SlowPipeline, timeout_sec=0.05
    )

    state = make_state(home_goals=0, away_goals=0, minute=15)
    fixture = SimpleNamespace(id=100, predictions=None)
    odds = [_odd()]
    pick = SimpleNamespace(name="v2-pick-2", edge_pct=4.0)

    await _scan_round_slice(
        telegram_sender=sender,
        v2_pick=pick,
        v3_runtime=runtime,
        state=state,
        fixture=fixture,
        odds=odds,
    )

    assert len(sender.received) == 1
    assert sender.received[0]["pick"].name == "v2-pick-2"
    assert runtime.error_count == 1


@pytest.mark.anyio("asyncio")
async def test_kill_switch_engaged_keeps_v2_running(tmp_path, monkeypatch):
    """If the kill switch is engaged, v3 doesn't even try — v2 still works."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "true")
    sender = _RecordingTelegramSender()
    runtime = _build_runtime(tmp_path, pipeline_cls=_ExplodingPipeline)
    (tmp_path / "kill.flag").touch()  # engage kill switch

    state = make_state(home_goals=0, away_goals=0, minute=20)
    fixture = SimpleNamespace(id=101, predictions=None)
    pick = SimpleNamespace(name="v2-pick-3", edge_pct=3.5)

    await _scan_round_slice(
        telegram_sender=sender,
        v2_pick=pick,
        v3_runtime=runtime,
        state=state,
        fixture=fixture,
        odds=[_odd()],
    )

    # v2 still received its pick.
    assert len(sender.received) == 1
    # v3 was suspended by kill switch, so error_count stayed 0,
    # skip_count incremented.
    assert runtime.error_count == 0
    assert runtime.skip_count == 1
    assert runtime.success_count == 0


@pytest.mark.anyio("asyncio")
async def test_env_var_disabled_keeps_v2_running(tmp_path, monkeypatch):
    """If V3_SHADOW_ENABLED is false, v3 doesn't run — v2 still works."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "false")
    sender = _RecordingTelegramSender()
    runtime = _build_runtime(tmp_path, pipeline_cls=_ExplodingPipeline)

    state = make_state(home_goals=0, away_goals=0, minute=10)
    fixture = SimpleNamespace(id=102, predictions=None)
    pick = SimpleNamespace(name="v2-pick-4", edge_pct=6.0)

    await _scan_round_slice(
        telegram_sender=sender,
        v2_pick=pick,
        v3_runtime=runtime,
        state=state,
        fixture=fixture,
        odds=[_odd()],
    )

    assert len(sender.received) == 1
    assert runtime.error_count == 0
    assert runtime.skip_count == 1


@pytest.mark.anyio("asyncio")
async def test_multiple_fixtures_sequential_invariant_holds(tmp_path, monkeypatch):
    """Loop over 5 fixtures with ALWAYS-broken v3 → v2 sends 5 picks, all succeed."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "true")
    sender = _RecordingTelegramSender()
    runtime = _build_runtime(tmp_path, pipeline_cls=_ExplodingPipeline)

    for i in range(5):
        state = make_state(home_goals=i % 2, away_goals=0, minute=10 + i)
        fixture = SimpleNamespace(id=200 + i, predictions=None)
        pick = SimpleNamespace(name=f"v2-pick-loop-{i}", edge_pct=4.0 + i)
        await _scan_round_slice(
            telegram_sender=sender,
            v2_pick=pick,
            v3_runtime=runtime,
            state=state,
            fixture=fixture,
            odds=[_odd()],
        )

    # All 5 v2 picks reached Telegram in order.
    assert len(sender.received) == 5
    assert [r["pick"].name for r in sender.received] == [
        f"v2-pick-loop-{i}" for i in range(5)
    ]
    # All 5 v3 attempts failed cleanly.
    assert runtime.error_count == 5
    assert runtime.success_count == 0


@pytest.mark.anyio("asyncio")
async def test_v3_success_does_not_disturb_v2_send(tmp_path, monkeypatch):
    """Sanity: when v3 SUCCEEDS, v2 still wins. Both paths complete."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_ISO_TEST", "true")
    sender = _RecordingTelegramSender()
    # Use the real V3Pipeline which on empty/no-thesis input completes
    # without exception (success path with allowed_picks=0).
    runtime = DualWriteRuntime(
        pipeline=V3Pipeline(),
        logger=ShadowLogger(output_root=tmp_path / "shadow"),
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_ISO_TEST",
    )

    state = make_state(home_goals=0, away_goals=0, minute=25)
    fixture = SimpleNamespace(id=300, predictions=None)
    pick = SimpleNamespace(name="v2-success-coexist", edge_pct=5.0)

    await _scan_round_slice(
        telegram_sender=sender,
        v2_pick=pick,
        v3_runtime=runtime,
        state=state,
        fixture=fixture,
        odds=[_odd()],
    )

    assert len(sender.received) == 1
    assert sender.received[0]["pick"].name == "v2-success-coexist"
    # v3 may report success OR no-op depending on pipeline outcome;
    # what matters is that no error escaped.
    assert runtime.error_count == 0
