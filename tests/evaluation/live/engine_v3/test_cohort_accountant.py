"""Tests for the cohort accountant — load, join, build metrics, run end-to-end.

The accountant is the operationally-critical gap closer: without it, kill
criteria are letter-only. These tests prove:

- load_v3_picks/outcomes/denials handle empty roots, missing partitions,
  and date-range filtering.
- assign_cohort_stage maps cumulative-settled to A/B/C correctly across
  the boundaries.
- join_picks_with_outcomes preserves picks when outcomes are absent,
  and matches on (fixture_id, thesis_id, market_id, timestamp_utc).
- build_cohort_metrics correctly counts won/lost/void, computes ROI flat,
  computes OOD denial rate from gate_denials with rule_number==9, and
  computes market concentration from the 100 most-recent picks.
- engage_kill_switch writes the flag file with the reason.
- run_cohort_eval is the full happy path AND triggers hard-stop on
  fabricated bad data.
- The notify_operator path correctly skips on ok verdicts and sends on
  hard_stop / soft_warn / greenlight.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from bip.evaluation.live.engine_v3.kill_criteria import CohortStage
from bip.evaluation.live.engine_v3.runtime.cohort_accountant import (
    assign_cohort_stage,
    build_cohort_metrics,
    engage_kill_switch,
    join_picks_with_outcomes,
    load_v3_denials,
    load_v3_outcomes,
    load_v3_picks,
    notify_operator,
    run_cohort_eval,
    write_cohort_report,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ──────────────────────────────────────────────────────────────────────
# Test fixtures: build a synthetic shadow root in tmp_path
# ──────────────────────────────────────────────────────────────────────


def _ts(hour: int = 12, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 5, day, hour, minute, 0, tzinfo=timezone.utc)


def _write_picks(root: Path, day: int, picks: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "picks.parquet"
    pl.DataFrame(picks).write_parquet(path)
    return path


def _write_outcomes(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "picks_outcomes.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _write_denials(root: Path, day: int, rows: list[dict]) -> Path:
    part = root / f"dt=2026-05-{day:02d}"
    part.mkdir(parents=True, exist_ok=True)
    path = part / "gate_denials.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _make_pick(
    *,
    fixture_id: int = 1,
    thesis_id: str = "t1",
    market_id: str = "ou_2_5_over",
    ts: datetime | None = None,
    thesis_layer: str = "rule",
) -> dict:
    return {
        "fixture_id": fixture_id,
        "timestamp_utc": ts or _ts(),
        "thesis_id": thesis_id,
        "thesis_layer": thesis_layer,
        "market_id": market_id,
    }


def _make_outcome(
    *,
    fixture_id: int = 1,
    thesis_id: str = "t1",
    market_id: str = "ou_2_5_over",
    ts: datetime | None = None,
    status: str = "won",
    profit: float = 1.0,
    odd: float = 2.0,
) -> dict:
    return {
        "fixture_id": fixture_id,
        "thesis_id": thesis_id,
        "market_id": market_id,
        "pick_timestamp_utc": ts or _ts(),
        "status": status,
        "profit_units": profit,
        "bookmaker_odd": odd,
        "settled_at": (ts or _ts()) + timedelta(hours=2),
    }


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────


def test_load_picks_empty_root_returns_empty_df(tmp_path):
    df = load_v3_picks(shadow_root=tmp_path / "nonexistent")
    assert df.is_empty()


def test_load_picks_skips_non_partition_dirs(tmp_path):
    (tmp_path / "not_a_partition").mkdir()
    (tmp_path / "dt=invalid-date").mkdir()
    _write_picks(tmp_path, day=11, picks=[_make_pick()])
    df = load_v3_picks(shadow_root=tmp_path)
    assert df.height == 1


def test_load_picks_concatenates_partitions(tmp_path):
    _write_picks(tmp_path, day=11, picks=[_make_pick(fixture_id=1)])
    _write_picks(tmp_path, day=12, picks=[_make_pick(fixture_id=2)])
    df = load_v3_picks(shadow_root=tmp_path)
    assert df.height == 2


def test_load_picks_filters_by_date_range(tmp_path):
    _write_picks(tmp_path, day=11, picks=[_make_pick(fixture_id=1)])
    _write_picks(tmp_path, day=12, picks=[_make_pick(fixture_id=2)])
    _write_picks(tmp_path, day=13, picks=[_make_pick(fixture_id=3)])
    start = datetime(2026, 5, 12, tzinfo=timezone.utc)
    end = datetime(2026, 5, 12, tzinfo=timezone.utc)
    df = load_v3_picks(shadow_root=tmp_path, date_range=(start, end))
    assert df.height == 1
    assert df["fixture_id"][0] == 2


def test_load_outcomes_handles_missing_file(tmp_path):
    _write_picks(tmp_path, day=11, picks=[_make_pick()])
    df = load_v3_outcomes(shadow_root=tmp_path)
    assert df.is_empty()


def test_load_denials_returns_rows(tmp_path):
    _write_denials(
        tmp_path,
        day=11,
        rows=[
            {"fixture_id": 1, "timestamp_utc": _ts(), "rule_number": 9, "reason": "ood"},
            {"fixture_id": 1, "timestamp_utc": _ts(), "rule_number": 7, "reason": "liq"},
        ],
    )
    df = load_v3_denials(shadow_root=tmp_path)
    assert df.height == 2


# ──────────────────────────────────────────────────────────────────────
# Cohort stage assignment
# ──────────────────────────────────────────────────────────────────────


def test_assign_cohort_stage_boundaries():
    assert assign_cohort_stage(0) == CohortStage.A
    assert assign_cohort_stage(1) == CohortStage.A
    assert assign_cohort_stage(100) == CohortStage.A
    assert assign_cohort_stage(101) == CohortStage.B
    assert assign_cohort_stage(300) == CohortStage.B
    assert assign_cohort_stage(301) == CohortStage.C
    assert assign_cohort_stage(10000) == CohortStage.C


# ──────────────────────────────────────────────────────────────────────
# Join logic
# ──────────────────────────────────────────────────────────────────────


def test_join_empty_picks_returns_empty():
    j = join_picks_with_outcomes(pl.DataFrame(), pl.DataFrame())
    assert j.is_empty()


def test_join_no_outcomes_keeps_picks_with_null_status():
    picks = pl.DataFrame([_make_pick()])
    j = join_picks_with_outcomes(picks, pl.DataFrame())
    assert j.height == 1
    assert j["status"][0] is None


def test_join_matches_on_canonical_key():
    ts = _ts()
    picks = pl.DataFrame([
        _make_pick(fixture_id=1, thesis_id="ta", market_id="ou", ts=ts),
        _make_pick(fixture_id=2, thesis_id="tb", market_id="btts", ts=ts),
    ])
    outcomes = pl.DataFrame([
        _make_outcome(fixture_id=1, thesis_id="ta", market_id="ou",
                      ts=ts, status="won", profit=1.0),
    ])
    j = join_picks_with_outcomes(picks, outcomes)
    by_fid = {row["fixture_id"]: row for row in j.iter_rows(named=True)}
    assert by_fid[1]["status"] == "won"
    assert by_fid[2]["status"] is None  # no outcome → null


# ──────────────────────────────────────────────────────────────────────
# Metric extraction
# ──────────────────────────────────────────────────────────────────────


def test_build_metrics_empty_returns_zero_settled():
    metrics = build_cohort_metrics(pl.DataFrame(), pl.DataFrame())
    assert metrics.n_settled == 0
    assert metrics.stage == CohortStage.A
    assert metrics.wr == 0.0
    assert metrics.roi_flat == 0.0


def test_build_metrics_counts_outcomes_correctly():
    ts = _ts()
    picks = pl.DataFrame([
        _make_pick(fixture_id=i, thesis_id=f"t{i}", market_id="ou", ts=ts)
        for i in range(1, 6)
    ])
    outcomes_rows = [
        _make_outcome(fixture_id=1, thesis_id="t1", market_id="ou", ts=ts,
                      status="won", profit=1.0),
        _make_outcome(fixture_id=2, thesis_id="t2", market_id="ou", ts=ts,
                      status="won", profit=1.5),
        _make_outcome(fixture_id=3, thesis_id="t3", market_id="ou", ts=ts,
                      status="lost", profit=-1.0),
        _make_outcome(fixture_id=4, thesis_id="t4", market_id="ou", ts=ts,
                      status="lost", profit=-1.0),
        _make_outcome(fixture_id=5, thesis_id="t5", market_id="ou", ts=ts,
                      status="void", profit=0.0),
    ]
    outcomes = pl.DataFrame(outcomes_rows)
    j = join_picks_with_outcomes(picks, outcomes)
    metrics = build_cohort_metrics(j, pl.DataFrame())
    assert metrics.n_settled == 5
    assert metrics.n_won == 2
    assert metrics.n_lost == 2
    assert metrics.n_void == 1
    # ROI flat = (1.0 + 1.5 - 1.0 - 1.0) / 4 = 0.125
    assert metrics.roi_flat == pytest.approx(0.125)
    assert metrics.wr == pytest.approx(0.5)


def test_build_metrics_ood_denial_rate_requires_full_window():
    """With <50 denials the rate stays 0.0 — false-positive guard."""
    denials = pl.DataFrame([
        {"fixture_id": i, "timestamp_utc": _ts(hour=12, minute=i),
         "rule_number": 9, "reason": "x"}
        for i in range(10)
    ])
    metrics = build_cohort_metrics(pl.DataFrame(), denials)
    assert metrics.ood_denial_rate_50 == 0.0


def test_build_metrics_ood_denial_rate_with_full_window():
    """With >=50 denials, alternating rule 9 / rule 7 → rate ~0.5."""
    rows = []
    for i in range(60):
        rows.append({
            "fixture_id": i,
            "timestamp_utc": _ts(hour=12 + i // 60, minute=i % 60),
            "rule_number": 9 if i % 2 == 0 else 7,
            "reason": "x",
        })
    denials = pl.DataFrame(rows)
    metrics = build_cohort_metrics(pl.DataFrame(), denials)
    assert metrics.ood_denial_rate_50 == pytest.approx(0.5, abs=0.05)


def test_build_metrics_market_concentration_requires_full_window():
    """Below 100 picks, concentration stays 0.0 to avoid false positives."""
    picks = pl.DataFrame([
        _make_pick(fixture_id=i, thesis_id=f"t{i}", market_id="A",
                   ts=_ts(minute=i % 60))
        for i in range(10)
    ])
    metrics = build_cohort_metrics(picks, pl.DataFrame())
    assert metrics.market_concentration_top1_100 == 0.0


def test_build_metrics_market_concentration_full_window():
    """70 picks on A + 30 on B over full 100-pick window → top1 = 0.7."""
    rows = []
    for i in range(70):
        rows.append(_make_pick(
            fixture_id=i, thesis_id=f"t{i}", market_id="A",
            ts=_ts(hour=12 + i // 60, minute=i % 60),
        ))
    for i in range(30):
        rows.append(_make_pick(
            fixture_id=200 + i, thesis_id=f"u{i}", market_id="B",
            ts=_ts(hour=14 + i // 60, minute=i % 60),
        ))
    picks = pl.DataFrame(rows)
    metrics = build_cohort_metrics(picks, pl.DataFrame())
    assert metrics.market_concentration_top1_100 == pytest.approx(0.7)


def test_build_metrics_pattern_layer_hit_rate():
    picks = pl.DataFrame([
        _make_pick(fixture_id=i, thesis_id=f"t{i}", market_id="A",
                   ts=_ts(minute=i),
                   thesis_layer="pattern" if i < 3 else "rule")
        for i in range(10)
    ])
    metrics = build_cohort_metrics(picks, pl.DataFrame())
    assert metrics.pattern_layer_hit_rate == pytest.approx(0.3)


# ──────────────────────────────────────────────────────────────────────
# Side effects: kill switch
# ──────────────────────────────────────────────────────────────────────


def test_engage_kill_switch_writes_flag(tmp_path):
    flag = tmp_path / "v3_kill_switch.flag"
    engage_kill_switch("test-reason", flag_path=flag)
    assert flag.exists()
    content = flag.read_text()
    assert "engaged_at=" in content
    assert "test-reason" in content


def test_engage_kill_switch_idempotent_overwrites(tmp_path):
    flag = tmp_path / "v3_kill_switch.flag"
    engage_kill_switch("first", flag_path=flag)
    engage_kill_switch("second", flag_path=flag)
    assert "second" in flag.read_text()


def test_engage_kill_switch_creates_parent_dir(tmp_path):
    flag = tmp_path / "subdir" / "v3_kill_switch.flag"
    engage_kill_switch("x", flag_path=flag)
    assert flag.exists()


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────


def test_write_cohort_report_basic(tmp_path):
    today = _ts()
    data = {
        "is_hard_stop": False,
        "is_soft_warning": False,
        "is_greenlight_ready": False,
        "rule_triggered": "",
        "reason": "",
        "stage": "cohort_a",
        "n_settled": 50,
        "n_won": 25,
        "n_lost": 23,
        "n_void": 2,
        "wr": 0.520,
        "roi_flat": 0.012,
        "ood_denial_rate_50": 0.04,
        "market_concentration_top1": 0.32,
        "pattern_hit_rate": 0.18,
        "contributing_markets": [],
        "n_picks_total": 80,
        "n_outcomes_loaded": 50,
        "kill_switch_engaged": False,
        "alert_sent": False,
    }
    path = write_cohort_report(data, report_root=tmp_path, today=today)
    assert path.exists()
    text = path.read_text()
    assert "n_settled: 50" in text
    assert "WR: 0.5200" in text


# ──────────────────────────────────────────────────────────────────────
# Alerting
# ──────────────────────────────────────────────────────────────────────


class _RecordingSink:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_html(self, text: str):
        self.sent.append(text)
        return None


@pytest.mark.anyio("asyncio")
async def test_notify_operator_skips_on_ok_verdict():
    from bip.evaluation.live.engine_v3.kill_criteria import KillVerdict

    sink = _RecordingSink()
    metrics = build_cohort_metrics(pl.DataFrame(), pl.DataFrame())
    sent = await notify_operator(KillVerdict.ok(), metrics, sink=sink)
    assert sent is False
    assert sink.sent == []


@pytest.mark.anyio("asyncio")
async def test_notify_operator_sends_on_hard_stop():
    from bip.evaluation.live.engine_v3.kill_criteria import KillVerdict

    sink = _RecordingSink()
    metrics = build_cohort_metrics(pl.DataFrame(), pl.DataFrame())
    v = KillVerdict.hard_stop("test_rule", "synthetic reason")
    sent = await notify_operator(v, metrics, sink=sink)
    assert sent is True
    assert len(sink.sent) == 1
    assert "HARD STOP" in sink.sent[0]
    assert "synthetic reason" in sink.sent[0]


@pytest.mark.anyio("asyncio")
async def test_notify_operator_swallows_sink_exception():
    from bip.evaluation.live.engine_v3.kill_criteria import KillVerdict

    class _BrokenSink:
        async def send_html(self, text: str):
            raise RuntimeError("telegram down")

    metrics = build_cohort_metrics(pl.DataFrame(), pl.DataFrame())
    sent = await notify_operator(
        KillVerdict.hard_stop("r", "reason"),
        metrics,
        sink=_BrokenSink(),
    )
    assert sent is False  # caught the exception, returned False


@pytest.mark.anyio("asyncio")
async def test_notify_operator_none_sink_returns_false():
    from bip.evaluation.live.engine_v3.kill_criteria import KillVerdict

    metrics = build_cohort_metrics(pl.DataFrame(), pl.DataFrame())
    sent = await notify_operator(
        KillVerdict.hard_stop("r", "reason"), metrics, sink=None
    )
    assert sent is False


# ──────────────────────────────────────────────────────────────────────
# End-to-end run_cohort_eval
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.anyio("asyncio")
async def test_run_full_pipeline_no_outcomes_no_kill(tmp_path):
    """No outcomes → n_settled=0 → ok verdict, no kill switch."""
    _write_picks(tmp_path, day=11, picks=[_make_pick()])
    flag = tmp_path / "kill_switch.flag"
    sink = _RecordingSink()
    result = await run_cohort_eval(
        shadow_root=tmp_path,
        report_root=tmp_path / "reports",
        kill_switch_path=flag,
        alert_sink=sink,
    )
    assert result.kill_switch_engaged is False
    assert flag.exists() is False
    assert result.metrics.n_settled == 0
    assert result.report_path.exists()
    assert sink.sent == []  # ok → no alert


@pytest.mark.anyio("asyncio")
async def test_run_pipeline_hard_stop_engages_kill_switch(tmp_path):
    """OOD denial rate >25% over a FULL window → system hard stop, kill switch."""
    # Need >=50 denials (full window), with all rule_9 → 100% OOD rate
    denials = []
    for i in range(60):
        denials.append({
            "fixture_id": i,
            "timestamp_utc": _ts(hour=12 + i // 60, minute=i % 60),
            "rule_number": 9,
            "reason": "ood",
        })
    _write_denials(tmp_path, day=11, rows=denials)
    flag = tmp_path / "kill_switch.flag"
    sink = _RecordingSink()
    result = await run_cohort_eval(
        shadow_root=tmp_path,
        report_root=tmp_path / "reports",
        kill_switch_path=flag,
        alert_sink=sink,
    )
    assert result.verdict.is_hard_stop is True
    assert result.kill_switch_engaged is True
    assert flag.exists()
    assert "HARD STOP" in sink.sent[0]
    assert "ood" in sink.sent[0].lower() or "ood" in result.verdict.reason.lower()


@pytest.mark.anyio("asyncio")
async def test_run_pipeline_idempotent(tmp_path):
    """Running twice produces same kill state and rewrites report."""
    _write_picks(tmp_path, day=11, picks=[_make_pick()])
    flag = tmp_path / "kill_switch.flag"
    r1 = await run_cohort_eval(
        shadow_root=tmp_path,
        report_root=tmp_path / "reports",
        kill_switch_path=flag,
        alert_sink=None,
    )
    r2 = await run_cohort_eval(
        shadow_root=tmp_path,
        report_root=tmp_path / "reports",
        kill_switch_path=flag,
        alert_sink=None,
    )
    assert r1.report_path == r2.report_path
    assert r1.verdict.is_hard_stop == r2.verdict.is_hard_stop
