"""Cohort accounting + auto-suspend for the Phase-4 shadow run.

Closes the most operationally-critical gap left after the gaps-closed
ship: the kill criteria in ``kill_criteria.py`` are documented and
tested, but nothing **invokes** them automatically. Without an
accountant, hard stops are letter-only — the operator has to remember
to grade picks, bucket them by cohort, and run the cohort scorer each
night. That is exactly the kind of "human checklist" that fails
predictably during a 4-week run.

What this module does:

1. Loads v3 shadow picks (from the daily Parquet partitions written
   by ``ShadowLogger``).
2. Joins them with operator-provided outcomes (a sibling parquet,
   ``picks_outcomes.parquet``, with the same partitioning).
3. Buckets settled picks by cohort sequence (first 100 -> A, next 200
   -> B, rest -> C) using **cumulative settled count**, not date
   windows. A picks-light week doesn't shrink the cohort; a picks-heavy
   week doesn't inflate it.
4. Builds the ``CohortMetrics`` data class consumed by the cohort scorer.
5. Runs the verdict and:
   - On hard stop -> engages the filesystem kill switch (``v3_kill_switch.flag``)
     so the dual-write loop suspends on the next iteration, AND emits
     a Telegram alert if a bot is configured.
   - On soft warn -> emits a Telegram alert only.
   - On greenlight -> emits an info Telegram message.
6. Writes a daily report to ``reports/v3/cohort_evals/YYYY-MM-DD.md``
   regardless of verdict, so the operator has an audit trail.

What this module deliberately does NOT do:

- It does not GRADE picks. The shadow logger writes raw picks; some
  upstream process (operator-run ``analyze_jornada`` for v3 in a
  follow-on, or manual settlement) populates ``picks_outcomes.parquet``.
  When that parquet is absent, the accountant reports "0 settled"
  and exits without firing any verdict — graceful degradation.
- It does not compute drift. The KS drift detector lives in
  ``phase3/drift_detector.py``; persisting its rolling state is a
  separate piece of plumbing. We default ``drift_active_50=False``
  for now and document this in the system-metrics caveats section.
- It does not collect runtime error counts. ``DualWriteRuntime``
  tracks ``error_count`` in memory; persisting that to disk is a
  one-line follow-on. Default ``pipeline_error_count_100=0``.

Everything that the accountant CAN compute deterministically from
parquet contents — performance metrics, OOD denial rate (via
``gate_denials.parquet`` with ``rule_number==9``), market
concentration, pattern-layer hit rate — IS computed.

# CLI

The CLI in an offline research script (not published) is the operator's
entry point. APScheduler integration lives in ``cohort_scheduler.py``
and registers a nightly cron firing.

# Why filesystem kill switch instead of an in-process flag

The dual-write runtime polls ``v3_kill_switch.flag`` on every
iteration (already implemented in ``dual_write.py``). The accountant
runs in a separate process (the cron job). Filesystem is the only
shared channel that doesn't require IPC — and the ship note already
documents the flag as the official suspend mechanism.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from bip.evaluation.live.engine_v3.kill_criteria import (
    COHORT_A_MIN_SETTLED,
    COHORT_B_MIN_SETTLED,
    CohortMetrics,
    CohortStage,
    KillVerdict,
    evaluate_cohort,
)
from bip.evaluation.live.engine_v3.runtime.dual_write import (
    DEFAULT_KILL_SWITCH_PATH,
)
from bip.evaluation.live.engine_v3.shadow_logger import DEFAULT_SHADOW_ROOT

log = logging.getLogger("v3.cohort_accountant")


DEFAULT_REPORT_ROOT = Path("reports/v3/cohort_evals")


# ──────────────────────────────────────────────────────────────────────
# Telegram alert protocol — duck-typed so tests can pass a minimal stub
# ──────────────────────────────────────────────────────────────────────


class AlertSink(Protocol):
    """Minimal contract for an alert backend.

    The real ``TelegramBot.send_html`` method matches this signature.
    Tests inject a recording stub.
    """

    async def send_html(self, text: str) -> Any:  # noqa: D401
        ...


# ──────────────────────────────────────────────────────────────────────
# Result records
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CohortEvalResult:
    """Output of one accountant invocation."""

    verdict: KillVerdict
    metrics: CohortMetrics
    n_picks_total: int  # total picks in the parquet (settled + pending)
    n_outcomes_loaded: int
    kill_switch_engaged: bool
    report_path: Path | None
    alert_sent: bool


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────


def load_v3_picks(
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    date_range: tuple[datetime, datetime] | None = None,
) -> Any:
    """Load all picks from the daily partitions under ``shadow_root``.

    ``date_range`` filters by partition name (``dt=YYYY-MM-DD``).
    Returns an empty Polars DataFrame if no partitions exist — caller
    is responsible for ``.is_empty()`` checks.
    """
    return _load_partitioned(
        Path(shadow_root), filename="picks.parquet", date_range=date_range
    )


def load_v3_outcomes(
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    date_range: tuple[datetime, datetime] | None = None,
) -> Any:
    """Load all outcomes from ``picks_outcomes.parquet`` partitions.

    Schema contract (the operator/grader is responsible for writing
    rows that conform):

    - ``fixture_id`` (i64)
    - ``thesis_id`` (str)
    - ``market_id`` (str)
    - ``pick_timestamp_utc`` (datetime[us, UTC]) — must match
      ``timestamp_utc`` in the corresponding picks row
    - ``status`` (str): one of ``"won"``, ``"lost"``, ``"void"``, ``"pending"``
    - ``profit_units`` (f64): signed P/L in stake-units
    - ``bookmaker_odd`` (f64): the decimal odd at pick time (for ROI calc)
    - ``settled_at`` (datetime[us, UTC])

    Returns empty DataFrame if no outcomes parquets exist anywhere.
    """
    return _load_partitioned(
        Path(shadow_root),
        filename="picks_outcomes.parquet",
        date_range=date_range,
    )


def load_v3_denials(
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    date_range: tuple[datetime, datetime] | None = None,
) -> Any:
    """Load gate denials. Used for OOD denial rate computation."""
    return _load_partitioned(
        Path(shadow_root),
        filename="gate_denials.parquet",
        date_range=date_range,
    )


def _load_partitioned(
    root: Path,
    *,
    filename: str,
    date_range: tuple[datetime, datetime] | None,
) -> Any:
    import polars as pl

    if not root.exists():
        return pl.DataFrame()

    frames = []
    for part_dir in sorted(root.iterdir()):
        if not part_dir.is_dir() or not part_dir.name.startswith("dt="):
            continue
        try:
            part_date = datetime.strptime(part_dir.name[3:], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if date_range is not None:
            if part_date.date() < date_range[0].date():
                continue
            if part_date.date() > date_range[1].date():
                continue
        path = part_dir / filename
        if not path.exists():
            continue
        frames.append(pl.read_parquet(path))

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


# ──────────────────────────────────────────────────────────────────────
# Cohort bucketing
# ──────────────────────────────────────────────────────────────────────


def assign_cohort_stage(cumulative_settled: int) -> CohortStage:
    """Map the cumulative settled count to the active cohort stage.

    A pick is in cohort A if it is the 1st through 100th settled pick
    of the shadow run. B if 101..300. C if 301..

    The cohort STAGE for evaluation is determined by where the **most
    recent** settled pick falls.
    """
    if cumulative_settled <= COHORT_A_MIN_SETTLED:
        return CohortStage.A
    if cumulative_settled <= COHORT_B_MIN_SETTLED:
        return CohortStage.B
    return CohortStage.C


# ──────────────────────────────────────────────────────────────────────
# Metric extraction from joined dataframe
# ──────────────────────────────────────────────────────────────────────


_JOIN_KEYS = ("fixture_id", "thesis_id", "market_id", "timestamp_utc")


def join_picks_with_outcomes(picks: Any, outcomes: Any) -> Any:
    """Inner-join picks with outcomes on the canonical key.

    Outcomes use ``pick_timestamp_utc`` to match ``timestamp_utc`` in
    picks. Returns an empty DataFrame if either side is empty (caller
    handles).

    The join is left so that picks without outcomes show up with a
    null status — useful for "n_pending" reports.
    """
    import polars as pl

    if picks.is_empty():
        return pl.DataFrame()
    if outcomes.is_empty():
        return picks.with_columns(
            pl.lit(None).cast(pl.Utf8).alias("status"),
            pl.lit(None).cast(pl.Float64).alias("profit_units"),
            pl.lit(None).cast(pl.Float64).alias("bookmaker_odd"),
        )

    outcomes_renamed = outcomes.rename(
        {"pick_timestamp_utc": "timestamp_utc"}
    )
    return picks.join(
        outcomes_renamed,
        on=list(_JOIN_KEYS),
        how="left",
    )


def build_cohort_metrics(
    joined: Any,
    denials: Any,
    *,
    runtime_pipeline_errors_100: int = 0,
    runtime_log_failure_rate_100: float = 0.0,
    drift_active_50: bool = False,
) -> CohortMetrics:
    """Compute ``CohortMetrics`` from a joined picks-outcomes table.

    Pipeline-error count and shadow-log failure rate are NOT derivable
    from parquet alone — they live in the dual-write runtime. The
    accountant accepts them as parameters; defaults are conservative
    (0) so the system hard-stops on these dimensions can only fire when
    the operator wires runtime-stats persistence.

    drift_active_50 likewise comes from the drift detector's persisted
    state (when wired). Default False.
    """
    import polars as pl

    if joined.is_empty() or "status" not in joined.columns:
        return CohortMetrics(
            stage=CohortStage.A,
            n_settled=0,
            n_won=0,
            n_lost=0,
            n_void=0,
            roi_flat=0.0,
            roi_kelly=0.0,
            ood_denial_rate_50=_compute_ood_denial_rate(denials),
            drift_active_50=drift_active_50,
            pipeline_error_count_100=runtime_pipeline_errors_100,
            shadow_log_failure_rate_100=runtime_log_failure_rate_100,
            market_concentration_top1_100=_compute_market_concentration(joined),
            avg_clv=0.0,
            contributing_markets=[],
            drift_quiet_picks=0,
            pattern_layer_hit_rate=_compute_pattern_hit_rate(joined),
        )

    settled = joined.filter(pl.col("status").is_in(["won", "lost", "void"]))
    n_settled = settled.height
    stage = assign_cohort_stage(n_settled)

    if n_settled == 0:
        return CohortMetrics(
            stage=stage,
            n_settled=0,
            n_won=0,
            n_lost=0,
            n_void=0,
            roi_flat=0.0,
            roi_kelly=0.0,
            ood_denial_rate_50=_compute_ood_denial_rate(denials),
            drift_active_50=drift_active_50,
            pipeline_error_count_100=runtime_pipeline_errors_100,
            shadow_log_failure_rate_100=runtime_log_failure_rate_100,
            market_concentration_top1_100=_compute_market_concentration(joined),
            avg_clv=0.0,
            contributing_markets=[],
            drift_quiet_picks=0,
            pattern_layer_hit_rate=_compute_pattern_hit_rate(joined),
        )

    n_won = settled.filter(pl.col("status") == "won").height
    n_lost = settled.filter(pl.col("status") == "lost").height
    n_void = settled.filter(pl.col("status") == "void").height

    profit_sum = float(settled["profit_units"].sum() or 0.0)
    n_non_void = n_won + n_lost
    roi_flat = profit_sum / n_non_void if n_non_void > 0 else 0.0

    # Kelly ROI is not derivable here without per-pick kelly fractions;
    # we use roi_flat as a placeholder until the dual-write logs kelly.
    roi_kelly = roi_flat

    contributing_markets = _compute_contributing_markets(settled)

    return CohortMetrics(
        stage=stage,
        n_settled=n_settled,
        n_won=n_won,
        n_lost=n_lost,
        n_void=n_void,
        roi_flat=roi_flat,
        roi_kelly=roi_kelly,
        ood_denial_rate_50=_compute_ood_denial_rate(denials),
        drift_active_50=drift_active_50,
        pipeline_error_count_100=runtime_pipeline_errors_100,
        shadow_log_failure_rate_100=runtime_log_failure_rate_100,
        market_concentration_top1_100=_compute_market_concentration(joined),
        avg_clv=0.0,  # CLV requires Pinnacle closing line join — Phase 5.
        contributing_markets=contributing_markets,
        drift_quiet_picks=0,
        pattern_layer_hit_rate=_compute_pattern_hit_rate(joined),
    )


def _compute_ood_denial_rate(denials: Any) -> float:
    """Last-50 denials rolling OOD denial rate.

    Returns 0.0 unless the window is FULL (>=50 denials total). With
    fewer samples the rate is not statistically meaningful and we'd
    produce false-positive hard stops on the first few denials of a
    fresh shadow run. The kill criteria spec
    (SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW=50) explicitly requires a
    sustained window before firing.
    """
    import polars as pl

    from bip.evaluation.live.engine_v3.kill_criteria import (
        SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW,
    )

    if denials.is_empty() or "rule_number" not in denials.columns:
        return 0.0
    if denials.height < SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW:
        return 0.0
    sorted_d = denials.sort("timestamp_utc", descending=True).head(
        SYSTEM_OOD_DENIAL_SUSTAINED_WINDOW
    )
    n_ood = sorted_d.filter(pl.col("rule_number") == 9).height
    return n_ood / sorted_d.height


def _compute_market_concentration(joined: Any) -> float:
    """Top-1 market share over the most-recent 100 picks.

    Returns 0.0 unless we have a FULL 100-pick window. A single pick
    with no diversity would otherwise produce 100% concentration and
    trip a false positive on day 1 of shadow. The kill criteria says
    "over last 100 picks" — we honor that as a minimum sample, not a
    max.
    """
    if joined.is_empty() or "market_id" not in joined.columns:
        return 0.0
    if joined.height < 100:
        return 0.0
    sorted_p = joined.sort("timestamp_utc", descending=True).head(100)
    counts = Counter(sorted_p["market_id"].to_list())
    if not counts:
        return 0.0
    return max(counts.values()) / sorted_p.height


def _compute_contributing_markets(settled: Any) -> list[str]:
    """Markets with >=30 settled picks AND non-negative ROI (cohort C input)."""
    import polars as pl

    from bip.evaluation.live.engine_v3.kill_criteria import (
        GREENLIGHT_MIN_PICKS_PER_CONTRIBUTING_MARKET,
    )

    if settled.is_empty() or "market_id" not in settled.columns:
        return []

    contributing: list[str] = []
    for market_id in settled["market_id"].unique().to_list():
        sub = settled.filter(pl.col("market_id") == market_id)
        if sub.height < GREENLIGHT_MIN_PICKS_PER_CONTRIBUTING_MARKET:
            continue
        non_void = sub.filter(pl.col("status").is_in(["won", "lost"]))
        if non_void.height == 0:
            continue
        roi = float(sub["profit_units"].sum() or 0.0) / non_void.height
        if roi >= 0.0:
            contributing.append(str(market_id))
    return contributing


def _compute_pattern_hit_rate(joined: Any) -> float:
    """Fraction of picks whose thesis came from the pattern layer."""
    import polars as pl

    if joined.is_empty() or "thesis_layer" not in joined.columns:
        return 0.0
    n_pattern = joined.filter(pl.col("thesis_layer") == "pattern").height
    return n_pattern / joined.height if joined.height > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
# Side effects: kill switch + alerting + report
# ──────────────────────────────────────────────────────────────────────


def engage_kill_switch(
    reason: str,
    *,
    flag_path: Path = DEFAULT_KILL_SWITCH_PATH,
) -> None:
    """Write the kill-switch flag file. Idempotent — re-engaging just
    rewrites the reason with a fresh timestamp.
    """
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text(
        f"engaged_at={datetime.now(timezone.utc).isoformat()}\nreason={reason}\n",
        encoding="utf-8",
    )


def write_cohort_report(
    result_data: dict[str, Any],
    *,
    report_root: Path = DEFAULT_REPORT_ROOT,
    today: datetime | None = None,
) -> Path:
    """Write a one-page markdown report for the day's eval."""
    today = today or datetime.now(timezone.utc)
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / f"{today.strftime('%Y-%m-%d')}.md"

    lines = [
        f"# v3 cohort eval — {today.strftime('%Y-%m-%d')} UTC",
        "",
        "## Verdict",
        f"- hard_stop: **{result_data['is_hard_stop']}**",
        f"- soft_warn: **{result_data['is_soft_warning']}**",
        f"- greenlight: **{result_data['is_greenlight_ready']}**",
        f"- rule: `{result_data['rule_triggered']}`",
        f"- reason: {result_data['reason']}",
        "",
        "## Cohort metrics",
        f"- stage: `{result_data['stage']}`",
        f"- n_settled: {result_data['n_settled']}",
        f"- n_won / n_lost / n_void: {result_data['n_won']} / "
        f"{result_data['n_lost']} / {result_data['n_void']}",
        f"- WR: {result_data['wr']:.4f}",
        f"- ROI flat: {result_data['roi_flat']:.4f}",
        f"- ood_denial_rate_50: {result_data['ood_denial_rate_50']:.4f}",
        f"- market_concentration_top1: {result_data['market_concentration_top1']:.4f}",
        f"- pattern_layer_hit_rate: {result_data['pattern_hit_rate']:.4f}",
        f"- contributing_markets: {result_data['contributing_markets']}",
        "",
        "## Loading status",
        f"- picks total in parquet: {result_data['n_picks_total']}",
        f"- outcomes loaded: {result_data['n_outcomes_loaded']}",
        f"- kill switch engaged this run: {result_data['kill_switch_engaged']}",
        f"- alert dispatched: {result_data['alert_sent']}",
        "",
        "## Caveats (system metrics not yet auto-derived)",
        "- pipeline_error_count_100 defaulted to 0 — wire dual-write counter persistence",
        "- shadow_log_failure_rate_100 defaulted to 0 — same caveat",
        "- drift_active_50 defaulted to False — wire drift detector state persistence",
        "- avg_clv defaulted to 0 — wire Pinnacle closing line join (Phase 5)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _format_alert(verdict: KillVerdict, metrics: CohortMetrics) -> str:
    if verdict.is_hard_stop:
        header = "v3 HARD STOP — kill switch engaged"
    elif verdict.is_soft_warning:
        header = "v3 soft warning"
    elif verdict.is_greenlight_ready:
        header = "v3 GREENLIGHT — Tier-D ready"
    else:
        header = "v3 cohort eval"

    return (
        f"<b>{header}</b>\n"
        f"stage={metrics.stage.value} n_settled={metrics.n_settled}\n"
        f"WR={metrics.wr:.3f} ROI={metrics.roi_flat:.3f}\n"
        f"rule=<code>{verdict.rule_triggered}</code>\n"
        f"reason={verdict.reason}"
    )


async def notify_operator(
    verdict: KillVerdict,
    metrics: CohortMetrics,
    *,
    sink: AlertSink | None,
) -> bool:
    """Dispatch alert via the provided sink. Returns True on send."""
    if sink is None:
        return False
    if not (verdict.is_hard_stop or verdict.is_soft_warning or verdict.is_greenlight_ready):
        return False
    try:
        await sink.send_html(_format_alert(verdict, metrics))
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("cohort_alert_send_failed err=%s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Top-level entry point
# ──────────────────────────────────────────────────────────────────────


async def run_cohort_eval(
    *,
    shadow_root: Path | str = DEFAULT_SHADOW_ROOT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    kill_switch_path: Path = DEFAULT_KILL_SWITCH_PATH,
    alert_sink: AlertSink | None = None,
    runtime_pipeline_errors_100: int = 0,
    runtime_log_failure_rate_100: float = 0.0,
    drift_active_50: bool = False,
    today: datetime | None = None,
) -> CohortEvalResult:
    """One full cohort evaluation pass.

    Loads picks + outcomes + denials, builds metrics, scores the cohort,
    engages kill switch on hard stop, alerts on any non-ok verdict, writes
    a daily report. Idempotent — running twice in the same day overwrites
    the report and (if needed) refreshes the kill switch flag.
    """
    picks = load_v3_picks(shadow_root=shadow_root)
    outcomes = load_v3_outcomes(shadow_root=shadow_root)
    denials = load_v3_denials(shadow_root=shadow_root)

    n_picks_total = picks.height if not picks.is_empty() else 0
    n_outcomes_loaded = outcomes.height if not outcomes.is_empty() else 0

    joined = join_picks_with_outcomes(picks, outcomes)
    metrics = build_cohort_metrics(
        joined,
        denials,
        runtime_pipeline_errors_100=runtime_pipeline_errors_100,
        runtime_log_failure_rate_100=runtime_log_failure_rate_100,
        drift_active_50=drift_active_50,
    )
    verdict = evaluate_cohort(metrics)

    kill_switch_engaged = False
    if verdict.is_hard_stop:
        engage_kill_switch(verdict.reason, flag_path=kill_switch_path)
        kill_switch_engaged = True
        log.error(
            "v3_cohort_hard_stop rule=%s reason=%s",
            verdict.rule_triggered,
            verdict.reason,
        )

    alert_sent = await notify_operator(verdict, metrics, sink=alert_sink)

    report_path = write_cohort_report(
        {
            "is_hard_stop": verdict.is_hard_stop,
            "is_soft_warning": verdict.is_soft_warning,
            "is_greenlight_ready": verdict.is_greenlight_ready,
            "rule_triggered": verdict.rule_triggered,
            "reason": verdict.reason,
            "stage": metrics.stage.value,
            "n_settled": metrics.n_settled,
            "n_won": metrics.n_won,
            "n_lost": metrics.n_lost,
            "n_void": metrics.n_void,
            "wr": metrics.wr,
            "roi_flat": metrics.roi_flat,
            "ood_denial_rate_50": metrics.ood_denial_rate_50,
            "market_concentration_top1": metrics.market_concentration_top1_100,
            "pattern_hit_rate": metrics.pattern_layer_hit_rate,
            "contributing_markets": metrics.contributing_markets,
            "n_picks_total": n_picks_total,
            "n_outcomes_loaded": n_outcomes_loaded,
            "kill_switch_engaged": kill_switch_engaged,
            "alert_sent": alert_sent,
        },
        report_root=report_root,
        today=today,
    )

    return CohortEvalResult(
        verdict=verdict,
        metrics=metrics,
        n_picks_total=n_picks_total,
        n_outcomes_loaded=n_outcomes_loaded,
        kill_switch_engaged=kill_switch_engaged,
        report_path=report_path,
        alert_sent=alert_sent,
    )


__all__ = [
    "AlertSink",
    "CohortEvalResult",
    "DEFAULT_REPORT_ROOT",
    "assign_cohort_stage",
    "build_cohort_metrics",
    "engage_kill_switch",
    "join_picks_with_outcomes",
    "load_v3_denials",
    "load_v3_outcomes",
    "load_v3_picks",
    "notify_operator",
    "run_cohort_eval",
    "write_cohort_report",
]
