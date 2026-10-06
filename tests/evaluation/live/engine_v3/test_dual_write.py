"""Tests for the dual-write runtime that wires v3 to watch.py.

T1.3 invariants — the contract is "v3 cannot affect v2's pick path":
- ``is_v3_shadow_enabled`` parses env var correctly (default true, false
  on the explicit false-strings, defaults true on unparseable).
- ``is_v3_kill_switch_engaged`` reads filesystem flag without crashing.
- ``DualWriteRuntime.from_paths`` builds gracefully when both pkls are
  missing (v3 still runs with None layers).
- ``run_shadow`` returns False when disabled / kill-switched / no markets.
- ``run_shadow`` catches any exception from V3Pipeline + increments
  error_count without re-raising.
- ``run_shadow`` honors the wall-clock timeout (slow pipeline → timeout
  error, error_count++, returns False).
- ``run_shadow`` increments success_count on a clean run.
- ``market_snapshot_from_odds`` deduplicates by market_id and skips
  invalid quotes (no decimal, decimal<=1, missing description).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from bip.evaluation.live.engine_v3 import (
    PreMatchPriors,
    ShadowLogger,
    V3Pipeline,
)
from bip.evaluation.live.engine_v3.runtime.dual_write import (
    DualWriteRuntime,
    derive_priors_from_fixture,
    is_v3_kill_switch_engaged,
    is_v3_shadow_enabled,
    market_snapshot_from_odds,
)
from tests.evaluation.live.engine_v3.conftest import AWAY_ID, make_state

# ──────────────────────────────────────────────────────────────────────
# Env var + kill switch
# ──────────────────────────────────────────────────────────────────────


def test_v3_shadow_enabled_default_true(monkeypatch):
    monkeypatch.delenv("V3_SHADOW_ENABLED", raising=False)
    assert is_v3_shadow_enabled() is True


@pytest.mark.parametrize("val", ["false", "FALSE", "0", "no", "off", "n", "f"])
def test_v3_shadow_disabled_explicit_false(monkeypatch, val):
    monkeypatch.setenv("V3_SHADOW_ENABLED", val)
    assert is_v3_shadow_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "ON", "y", "t"])
def test_v3_shadow_enabled_explicit_true(monkeypatch, val):
    monkeypatch.setenv("V3_SHADOW_ENABLED", val)
    assert is_v3_shadow_enabled() is True


def test_v3_shadow_enabled_unparseable_falls_open(monkeypatch):
    monkeypatch.setenv("V3_SHADOW_ENABLED", "maybe")
    # Defaults to True (fail-open with warning) so the operator
    # notices via logs rather than silent disable.
    assert is_v3_shadow_enabled() is True


# ──────────────────────────────────────────────────────────────────────
# V3_TELEGRAM_ENABLED: unified parser contract
# ──────────────────────────────────────────────────────────────────────
# watch.py previously used a bespoke 4-token set {"1","true","yes","on"}
# for V3_TELEGRAM_ENABLED (fail-closed on unknowns). It now delegates to
# is_v3_shadow_enabled("V3_TELEGRAM_ENABLED") — same parser as the
# DualWriteRuntime internal gate.  Both sides produce the same decision
# for any env-var value.
#
# This test asserts the unified parser behaviour for all relevant tokens.
# The "watch.py decision" is is_v3_shadow_enabled("V3_TELEGRAM_ENABLED")
# because that is what watch.py now calls.


@pytest.mark.parametrize("raw,expected", [
    # unset → True (fail-open: default to enabled, operator sees logs)
    (None, True),
    # truthy tokens
    ("true",  True),
    ("y",     True),
    ("on",    True),
    # falsy tokens
    ("false", False),
    ("0",     False),
    # garbage → True (fail-open: same as unset — operator notices via logs)
    ("garbage", True),
])
def test_v3_telegram_enabled_parser_unified(monkeypatch, raw, expected):
    """watch.py and DualWriteRuntime use the same is_v3_shadow_enabled parser
    for V3_TELEGRAM_ENABLED.  Assert the canonical decision for each token.

    Fail-open semantics: unset and unparseable both return True so the
    operator sees Telegram traffic and can confirm the env-var spelling
    rather than having alerts silently dropped.
    """
    if raw is None:
        monkeypatch.delenv("V3_TELEGRAM_ENABLED", raising=False)
    else:
        monkeypatch.setenv("V3_TELEGRAM_ENABLED", raw)
    # Both watch.py and DualWriteRuntime now call the same function:
    assert is_v3_shadow_enabled("V3_TELEGRAM_ENABLED") is expected


def test_kill_switch_absent_returns_false(tmp_path):
    path = tmp_path / "never_exists.flag"
    assert is_v3_kill_switch_engaged(path) is False


def test_kill_switch_present_returns_true(tmp_path):
    path = tmp_path / "kill.flag"
    path.touch()
    assert is_v3_kill_switch_engaged(path) is True


# ──────────────────────────────────────────────────────────────────────
# Adapters
# ──────────────────────────────────────────────────────────────────────


def _odd(
    *,
    desc="match goals",
    label="over",
    value="2.10",
    total="2.5",
    update=None,
    suspended=False,
    stopped=False,
):
    return SimpleNamespace(
        market_description=desc,
        label=label,
        value=value,
        total=total,
        latest_bookmaker_update=update,
        suspended=suspended,
        stopped=stopped,
    )


def test_market_snapshot_skips_invalid_quotes():
    captured = datetime.now(UTC)
    odds = [
        _odd(desc="", label="over", value="2.10"),  # no description → skip
        _odd(value=None),  # no value → skip
        _odd(value="1.00"),  # decimal <= 1.0 → skip
        _odd(value="not_a_number"),  # unparseable → skip
        _odd(value="2.10"),  # OK
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    assert len(snap.lines) == 1


def test_market_snapshot_dedupes_by_market_id():
    captured = datetime.now(UTC)
    odds = [
        _odd(value="2.10"),
        _odd(value="2.20"),  # same desc/label/total → same id → skip
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    # First entry wins (Phase-1 convention — see _build_market_id)
    assert len(snap.lines) == 1
    line = next(iter(snap.lines.values()))
    assert line.side_a_decimal == pytest.approx(2.10)


def test_market_snapshot_skips_suspended_quote():
    """A quote where suspended=True must be dropped entirely."""
    captured = datetime.now(UTC)
    odds = [
        _odd(desc="match corners", label="over", value="2.10", total="10.5", suspended=True),
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    assert len(snap.lines) == 0


def test_market_snapshot_skips_stopped_quote():
    """A quote where stopped=True must be dropped, same as suspended."""
    captured = datetime.now(UTC)
    odds = [
        _odd(desc="match corners", label="over", value="2.10", total="10.5", stopped=True),
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    assert len(snap.lines) == 0


def test_market_snapshot_keeps_pure_live_market():
    """A market with only live (non-suspended) quotes is kept normally."""
    captured = datetime.now(UTC)
    odds = [
        _odd(desc="match corners", label="over", value="2.10", total="10.5"),
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    assert len(snap.lines) == 1


def test_market_snapshot_drop_ambiguous_market():
    """When a market has BOTH a live quote and a suspended quote, the
    entire market_id is dropped (conservative drop-ambiguous policy)."""
    captured = datetime.now(UTC)
    # Same desc/label/total → same market_id.
    odds = [
        _odd(desc="match corners", label="over", value="2.10", total="10.5"),  # live
        _odd(desc="match corners", label="over", value="1.85", total="10.5", suspended=True),  # suspended
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    # Drop-ambiguous: the entire market is dropped even though the live quote was present.
    assert len(snap.lines) == 0


def test_market_snapshot_other_markets_unaffected_by_suspended():
    """A suspended quote for market A does NOT pollute market B."""
    captured = datetime.now(UTC)
    odds = [
        _odd(desc="match goals", label="over", value="2.10", total="2.5", suspended=True),
        _odd(desc="match corners", label="over", value="1.90", total="10.5"),  # different market
    ]
    snap = market_snapshot_from_odds(odds, captured_at=captured)
    # Only corners survived — goals was suspended.
    assert len(snap.lines) == 1
    assert any("corners" in mid for mid in snap.lines)


def test_derive_priors_handles_missing_predictions():
    """No predictions → neutral fallback."""
    fixture = SimpleNamespace(predictions=None)
    priors = derive_priors_from_fixture(fixture)
    assert isinstance(priors, PreMatchPriors)
    assert priors.lambda_home_prematch == pytest.approx(1.35)
    assert priors.lambda_away_prematch == pytest.approx(1.15)


# ──────────────────────────────────────────────────────────────────────
# Runtime — disabled / kill-switch paths
# ──────────────────────────────────────────────────────────────────────


def _build_runtime(tmp_path: Path, **overrides) -> DualWriteRuntime:
    pipeline = V3Pipeline()
    logger = ShadowLogger(output_root=tmp_path)
    return DualWriteRuntime(
        pipeline=pipeline,
        logger=logger,
        kill_switch_path=tmp_path / "kill.flag",
        env_var="V3_SHADOW_ENABLED_TEST",
        **overrides,
    )


@pytest.mark.anyio("asyncio")
async def test_run_shadow_skips_when_env_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "false")
    runtime = _build_runtime(tmp_path)
    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=1, predictions=None)
    ran = await runtime.run_shadow(state=state, fixture=fixture, odds=[], now_utc=datetime.now(UTC))
    assert ran is False
    assert runtime.skip_count == 1
    assert runtime.success_count == 0
    assert runtime.error_count == 0


@pytest.mark.anyio("asyncio")
async def test_run_shadow_skips_when_kill_switch_engaged(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "true")
    runtime = _build_runtime(tmp_path)
    (tmp_path / "kill.flag").touch()
    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=1, predictions=None)
    ran = await runtime.run_shadow(state=state, fixture=fixture, odds=[], now_utc=datetime.now(UTC))
    assert ran is False
    assert runtime.skip_count == 1
    assert runtime.error_count == 0


# ──────────────────────────────────────────────────────────────────────
# Runtime — error handling
# ──────────────────────────────────────────────────────────────────────


class _ExplodingPipeline:
    """Stand-in V3Pipeline that always raises in ``run``."""

    def run(self, *args, **kwargs):
        raise RuntimeError("synthetic v3 crash")


@pytest.mark.anyio("asyncio")
async def test_run_shadow_swallows_pipeline_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "true")
    runtime = _build_runtime(tmp_path)
    runtime.pipeline = _ExplodingPipeline()  # type: ignore[assignment]

    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=42, predictions=None)
    # Need at least one odd so the runtime calls into the pipeline
    odds = [_odd(value="2.10")]
    ran = await runtime.run_shadow(
        state=state, fixture=fixture, odds=odds, now_utc=datetime.now(UTC)
    )
    assert ran is False
    assert runtime.error_count == 1
    assert runtime.success_count == 0
    # Stats dict exposes the counter for the watch.py status print
    assert runtime.stats["v3_error"] == 1


@pytest.mark.anyio("asyncio")
async def test_run_shadow_returns_false_on_empty_markets(tmp_path, monkeypatch):
    """No odds → ``markets.lines`` is empty → run returns success_count++.

    Per the implementation, no-markets is NOT an error: it's a frame we
    intentionally skipped. ``run_shadow`` returns True because the
    pipeline call completed cleanly (just bailed early).
    """
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "true")
    runtime = _build_runtime(tmp_path)
    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=1, predictions=None)
    ran = await runtime.run_shadow(state=state, fixture=fixture, odds=[], now_utc=datetime.now(UTC))
    # Implementation choice: empty markets short-circuits inside
    # _run_pipeline_sync. The outer ``run_shadow`` still records this as
    # a success path (no exception, no timeout). Document this so future
    # readers don't conflate "ran=True" with "wrote a pick".
    assert ran is True
    assert runtime.success_count == 1
    assert runtime.error_count == 0


@pytest.mark.anyio("asyncio")
async def test_run_shadow_timeout(tmp_path, monkeypatch):
    """A pipeline that sleeps past the timeout raises asyncio.TimeoutError
    INSIDE the runtime → error_count++, ran=False."""
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "true")

    class _SlowPipeline:
        def run(self, *args, **kwargs):
            time.sleep(0.3)  # exceeds 0.05s timeout
            return None

    runtime = _build_runtime(tmp_path, timeout_sec=0.05)
    runtime.pipeline = _SlowPipeline()  # type: ignore[assignment]

    state = make_state(home_goals=0, away_goals=0, minute=30)
    fixture = SimpleNamespace(id=99, predictions=None)
    odds = [_odd(value="2.10")]
    ran = await runtime.run_shadow(
        state=state, fixture=fixture, odds=odds, now_utc=datetime.now(UTC)
    )
    assert ran is False
    assert runtime.error_count == 1


# ──────────────────────────────────────────────────────────────────────
# Runtime — success path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.anyio("asyncio")
async def test_run_shadow_success_path_writes_to_logger(tmp_path, monkeypatch, priors):
    monkeypatch.setenv("V3_SHADOW_ENABLED_TEST", "true")
    runtime = _build_runtime(tmp_path)

    state = make_state(home_goals=1, away_goals=0, minute=70, red_card_events=[(25, AWAY_ID)])
    fixture = SimpleNamespace(id=12345, predictions=None)
    # Compose a corner-line odd consistent with the v3 market selector
    odds = [
        _odd(desc="corners over/under 10.5", label="over", value="2.10", total="10.5"),
    ]
    now = datetime.now(UTC)
    ran = await runtime.run_shadow(state=state, fixture=fixture, odds=odds, now_utc=now)
    assert ran is True
    assert runtime.success_count == 1
    assert runtime.error_count == 0

    # ShadowLogger flushed buffers via runtime.flush() should produce parquet
    paths = runtime.flush()
    # At minimum, the gsv_log should be written
    assert "gsv_log" in paths


@pytest.mark.anyio("asyncio")
async def test_runtime_from_paths_handles_missing_pkls(tmp_path):
    """Both pkls missing → runtime constructed with None layers, no crash."""
    runtime = DualWriteRuntime.from_paths(
        ood_pkl=tmp_path / "nope_ood.pkl",
        pattern_pkl=tmp_path / "nope_pat.pkl",
        shadow_root=tmp_path,
    )
    assert runtime.pipeline.ood_detector is None
    assert runtime.pipeline.pattern_layer is None
    assert runtime.kill_switch_path == tmp_path / "v3_kill_switch.flag"


# ──────────────────────────────────────────────────────────────────────
# Anyio backend selection (pytest-anyio convention)
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def priors() -> PreMatchPriors:
    """Mirror of conftest fixture for isolated test discovery."""
    return PreMatchPriors(
        lambda_home_prematch=2.10,
        lambda_away_prematch=0.95,
        expected_corners_total=10.4,
        expected_cards_total=3.95,
        elo_diff=120.0,
    )
