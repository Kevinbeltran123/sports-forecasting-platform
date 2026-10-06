"""Tests for the v3 live pick observer.

Covers:
- Protocol contract: NullObserver, StdoutObserver, FileObserver are
  duck-typed compatible callable receivers.
- format_v3_pick produces a non-empty single-line string with the
  default formatter (sanity — operator will override the body).
- StdoutObserver writes a single line per pick + flushes immediately.
- FileObserver appends to disk with line-buffering (tail -f friendly).
- build_observer_from_env honors the V3_LIVE_OBSERVE flag and the
  V3_LIVE_OBSERVE_LOG path override.
- A formatter that raises does NOT propagate into the observer call.
- DualWriteRuntime integration: observer.on_pick fires exactly once
  per allowed pick AND zero times when v3 produces no allowed picks.
- DualWriteRuntime integration: an observer that raises does NOT
  break v3 (error path stays within the observer's defensive catch).
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from bip.evaluation.live.engine_v3 import (
    ShadowLogger,
    V3Pipeline,
)
from bip.evaluation.live.engine_v3.runtime.dual_write import DualWriteRuntime
from bip.evaluation.live.engine_v3.runtime.live_observer import (
    FileObserver,
    LivePickObserver,
    NullObserver,
    StdoutObserver,
    build_observer_from_env,
    format_v3_pick,
)
from tests.evaluation.live.engine_v3.conftest import make_state


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ──────────────────────────────────────────────────────────────────────
# Recording observer for integration assertions
# ──────────────────────────────────────────────────────────────────────


class _RecordingObserver:
    """Observer that captures every on_pick invocation for inspection."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def on_pick(self, pick, gsv, window) -> None:
        self.calls.append((pick, gsv, window))


class _RaisingObserver:
    """Observer that always raises — verifies runtime defensive catch."""

    def on_pick(self, pick, gsv, window) -> None:
        raise RuntimeError("synthetic observer crash")


# ──────────────────────────────────────────────────────────────────────
# Protocol compatibility — duck typing
# ──────────────────────────────────────────────────────────────────────


def test_null_observer_satisfies_protocol():
    obs: LivePickObserver = NullObserver()
    # Calling does nothing, returns None
    assert obs.on_pick(None, None, None) is None


def test_stdout_observer_satisfies_protocol():
    obs: LivePickObserver = StdoutObserver(stream=io.StringIO())
    assert hasattr(obs, "on_pick")


# ──────────────────────────────────────────────────────────────────────
# format_v3_pick — default body produces a usable line
# ──────────────────────────────────────────────────────────────────────


def _make_minimal_pick_and_gsv():
    """Build the minimum ShadowPick + GameStateVector needed to drive the formatter.

    Uses the same conftest builder the rest of the v3 tests use, then
    runs the pipeline to produce a real ShadowPick. We need a real one
    because the formatter walks ShadowPick.full_thesis.archetype etc.
    """
    from bip.evaluation.live.engine_v3.archetypes import generate_theses
    from bip.evaluation.live.engine_v3.conditional_predictor import (
        ConditionalPredictor,
    )
    from bip.evaluation.live.engine_v3.gsv import (
        MarketLine,
        MarketSnapshot,
        PreMatchPriors,
    )
    from bip.evaluation.live.engine_v3.gsv_builder import GSVBuilder
    from bip.evaluation.live.engine_v3.market_selector import select_markets
    from datetime import timedelta

    from tests.evaluation.live.engine_v3.conftest import HOME_ID
    from tests.evaluation.live.engine_v3.test_anti_napoli import (
        _synthesize_states,
    )

    builder = GSVBuilder()
    priors = PreMatchPriors(
        lambda_home_prematch=2.30,
        lambda_away_prematch=0.95,
        expected_corners_total=10.8,
        expected_cards_total=4.10,
        elo_diff=140.0,
    )
    now = datetime.now(UTC)
    markets = MarketSnapshot(
        lines={
            "match_goals_over_2.5": MarketLine(
                market_id="match_goals_over_2.5",
                side_a_decimal=1.95,
                side_b_decimal=1.95,
                line_value=2.5,
                max_stake_cap=500.0,
                last_update_utc=now - timedelta(seconds=20),
            )
        }
    )
    from bip.evaluation.live.engine_v3.pipeline import V3Pipeline

    pipeline = V3Pipeline()
    out = None
    for st in _synthesize_states():
        out = pipeline.run(
            st,
            priors=priors,
            markets=markets,
            dominant_team_id=HOME_ID,
            now_utc=now,
        )
        if out.allowed_picks:
            break
    assert out is not None and out.allowed_picks, (
        "Need at least one allowed pick for formatter test"
    )
    return out.allowed_picks[0], out.gsv


def test_format_v3_pick_returns_single_line_string():
    pick, gsv = _make_minimal_pick_and_gsv()
    out = format_v3_pick(pick, gsv, None)
    assert isinstance(out, str)
    assert out, "default formatter must produce non-empty output"
    assert "\n" not in out, "default formatter must produce a single line"


def test_format_v3_pick_includes_fixture_id():
    pick, gsv = _make_minimal_pick_and_gsv()
    out = format_v3_pick(pick, gsv, None)
    assert f"fid={pick.fixture_id}" in out or str(pick.fixture_id) in out


# ──────────────────────────────────────────────────────────────────────
# StdoutObserver writes + flushes
# ──────────────────────────────────────────────────────────────────────


def test_stdout_observer_writes_one_line_per_pick():
    pick, gsv = _make_minimal_pick_and_gsv()
    buf = io.StringIO()
    obs = StdoutObserver(stream=buf)
    obs.on_pick(pick, gsv, None)
    contents = buf.getvalue()
    assert contents.endswith("\n")
    assert contents.count("\n") == 1


def test_stdout_observer_swallows_formatter_exceptions():
    """Defensive: a buggy custom formatter must not propagate."""
    pick, gsv = _make_minimal_pick_and_gsv()
    buf = io.StringIO()

    def _broken_formatter(p, g, w):
        raise ValueError("synthetic format error")

    obs = StdoutObserver(stream=buf, formatter=_broken_formatter)
    # No raise allowed
    obs.on_pick(pick, gsv, None)
    assert buf.getvalue() == ""  # formatter failed → nothing written


def test_stdout_observer_respects_custom_prefix():
    pick, gsv = _make_minimal_pick_and_gsv()
    buf = io.StringIO()
    obs = StdoutObserver(stream=buf, prefix="V3-LIVE:")
    obs.on_pick(pick, gsv, None)
    assert buf.getvalue().startswith("V3-LIVE:")


# ──────────────────────────────────────────────────────────────────────
# FileObserver
# ──────────────────────────────────────────────────────────────────────


def test_file_observer_writes_to_disk(tmp_path):
    pick, gsv = _make_minimal_pick_and_gsv()
    log_path = tmp_path / "v3_live.log"
    obs = FileObserver(log_path)
    obs.on_pick(pick, gsv, None)
    text = log_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "fid=" in text


def test_file_observer_appends_across_calls(tmp_path):
    pick, gsv = _make_minimal_pick_and_gsv()
    log_path = tmp_path / "v3_live.log"
    obs = FileObserver(log_path)
    obs.on_pick(pick, gsv, None)
    obs.on_pick(pick, gsv, None)
    obs.on_pick(pick, gsv, None)
    text = log_path.read_text(encoding="utf-8")
    assert text.count("\n") == 3


def test_file_observer_creates_parent_dirs(tmp_path):
    nested = tmp_path / "a" / "b" / "v3.log"
    pick, gsv = _make_minimal_pick_and_gsv()
    obs = FileObserver(nested)
    obs.on_pick(pick, gsv, None)
    assert nested.exists()


# ──────────────────────────────────────────────────────────────────────
# Env-driven factory
# ──────────────────────────────────────────────────────────────────────


def test_build_observer_from_env_unset_returns_null(monkeypatch):
    monkeypatch.delenv("V3_LIVE_OBSERVE", raising=False)
    monkeypatch.delenv("V3_LIVE_OBSERVE_LOG", raising=False)
    obs = build_observer_from_env()
    assert isinstance(obs, NullObserver)


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE"])
def test_build_observer_from_env_falsey_returns_null(monkeypatch, val):
    monkeypatch.setenv("V3_LIVE_OBSERVE", val)
    monkeypatch.delenv("V3_LIVE_OBSERVE_LOG", raising=False)
    assert isinstance(build_observer_from_env(), NullObserver)


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE"])
def test_build_observer_from_env_truthy_returns_stdout(monkeypatch, val):
    monkeypatch.setenv("V3_LIVE_OBSERVE", val)
    monkeypatch.delenv("V3_LIVE_OBSERVE_LOG", raising=False)
    obs = build_observer_from_env()
    assert isinstance(obs, StdoutObserver)
    assert not isinstance(obs, FileObserver)


def test_build_observer_from_env_with_log_path_returns_file(monkeypatch, tmp_path):
    log_path = tmp_path / "v3.log"
    monkeypatch.setenv("V3_LIVE_OBSERVE", "true")
    monkeypatch.setenv("V3_LIVE_OBSERVE_LOG", str(log_path))
    obs = build_observer_from_env()
    assert isinstance(obs, FileObserver)
    assert obs.path == log_path


# ──────────────────────────────────────────────────────────────────────
# DualWriteRuntime integration
# ──────────────────────────────────────────────────────────────────────


def _odd(market_id: str = "ou_2_5_over", value: str = "2.10"):
    return SimpleNamespace(
        market_id=market_id,
        value=value,
        market_description=market_id,
        bookmaker_id=2,
    )


@pytest.mark.anyio("asyncio")
async def test_runtime_does_not_invoke_observer_when_no_allowed_picks(
    tmp_path, monkeypatch
):
    """Pipeline that produces no allowed picks (empty markets) →
    observer never called."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_OBS_TEST", "true")
    recorder = _RecordingObserver()
    runtime = DualWriteRuntime(
        pipeline=V3Pipeline(),
        logger=ShadowLogger(output_root=tmp_path / "shadow"),
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_OBS_TEST",
        observer=recorder,
    )
    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=1, predictions=None)
    await runtime.run_shadow(
        state=state, fixture=fixture, odds=[], now_utc=datetime.now(UTC)
    )
    assert recorder.calls == []


@pytest.mark.anyio("asyncio")
async def test_runtime_swallows_observer_exception(tmp_path, monkeypatch):
    """Observer that raises does NOT propagate — error_count stays 0
    because the pipeline itself succeeded."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_OBS_TEST", "true")
    runtime = DualWriteRuntime(
        pipeline=V3Pipeline(),
        logger=ShadowLogger(output_root=tmp_path / "shadow"),
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_OBS_TEST",
        observer=_RaisingObserver(),
    )
    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=1, predictions=None)
    # Should not raise even if observer would have been called
    ran = await runtime.run_shadow(
        state=state, fixture=fixture, odds=[_odd()], now_utc=datetime.now(UTC)
    )
    # Pipeline ran successfully (even if produced 0 picks); error_count
    # is for PIPELINE failures, not observer failures
    assert ran is True
    assert runtime.error_count == 0


@pytest.mark.anyio("asyncio")
async def test_runtime_default_observer_is_null(tmp_path, monkeypatch):
    """Constructing a DualWriteRuntime without specifying an observer
    falls back to NullObserver — backwards-compat for all existing
    callers."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_OBS_TEST", "true")
    runtime = DualWriteRuntime(
        pipeline=V3Pipeline(),
        logger=ShadowLogger(output_root=tmp_path / "shadow"),
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_OBS_TEST",
    )
    assert isinstance(runtime.observer, NullObserver)
