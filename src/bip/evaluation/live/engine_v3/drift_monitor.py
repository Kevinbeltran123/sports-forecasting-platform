"""Calibration drift monitor — sec 5.6 of the v3 engine design notes.

Watches each ``(market_family, minute_bucket)`` cell with a rolling
window of recent settled picks. On every new observation:

1. Compute the empirical win rate over the last ``window_size``
   observations in the cell.
2. Compute the expected win rate (mean of predicted_p over the same
   window).
3. Compute the reliability gap: |expected_wr - empirical_wr|. When the
   gap exceeds ``reliability_gap_threshold`` the cell is flagged as
   calibration-drifted — the model is systematically over- or
   under-confident for this cell.
4. Compute the rolling P&L sign: if the sum of ``profit_units`` over the
   window is below ``pnl_floor`` the cell is flagged as P&L-drifted.

Either condition → ``is_drifted=True``.

Why reliability gap, not KS:
- KS is a distributional divergence test; it is sensitive to the shape
  of the predicted distribution vs the binary outcomes, but it measures
  more than we care about. The reliability gap directly measures
  "is the model's calibration broken for this cell?" without importing
  scipy stats machinery into the gate hot-path.
- The per-cell rolling average reduces noise from individual outliers.

Why P&L in addition to calibration:
- A long-shot archetype (e.g., DOMINANT_LOSING_NAPOLI at predicted 0.20)
  may have a realized win rate of 0.20 (well-calibrated) but still lose
  money if the bookmaker odds are below fair value. The P&L floor catches
  that case without requiring a separate signal.
- Conversely, a cell that is slightly miscalibrated (gap 0.10) but
  strongly profitable should not be blocked.

Why this design removes the need for a napoli exemption:
- The old rule_11 compared WR against a single prior (0.67). That prior
  is correct for "average" archetypes but wrong for long-shot archetypes
  like DOMINANT_LOSING_NAPOLI (natural WR ~0.2). The reliability gap
  comparison is symmetric: predicted 0.2, realized 0.2 → gap = 0, no
  drift triggered, regardless of the absolute WR.
- Positive P&L is the second safety valve: if napoli is genuinely +EV
  at the bookmaker odds, the P&L sum stays above the floor and the gate
  stays open. REVOKE THIS EXEMPTION REMOVAL if P&L turns negative —
  monitored weekly via an offline research script (not published).
  (Historical comment preserved: the WR-based napoli exemption was added
  2026-05-xx, citing +96u/14 Day-3 P/L. It is SUPERSEDED by the
  principled calibration+P&L gate below.)

Per-cell rolling window:
- ``observe(family, minute, predicted_p, outcome, profit_units)``
  appends to the cell's window, oldest dropped past ``window_size``.
- ``status(family, minute)`` returns a ``DriftStatus`` snapshot.
- ``drifted_cells()`` returns all (family, bucket) pairs currently
  failing the test.

Thread-safety: a single ``threading.Lock`` guards mutation. Live
ingestion is the only writer; readers call ``status`` / ``drifted_cells``
freely.

Settlement feed:
- ``feed_graded_picks_to_monitor(monitor, graded_picks, family_fn)``
  is the live feed: given GradedPick records from the canonical
  v3_grader, it calls ``monitor.observe(...)`` for each settled pick.
  This is the connection between grading and drift detection that was
  missing before Wave-3.

Phase-3 will:
- Persist observation history to parquet (currently in-memory only).
- Auto-suspend the cell in the no-bet gate when ``is_drifted`` flips.
- Recalibrate the IsotonicCalibrator cell from the new observations.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

from bip.evaluation.live.engine_v3.calibrator import minute_bucket
from bip.evaluation.live.engine_v3.thesis import MarketFamily


# ──────────────────────────────────────────────────────────────────────
# DriftStatus — per-cell snapshot
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DriftStatus:
    """Per-cell calibration health snapshot."""

    family: str
    minute_bucket: str
    n: int
    empirical_win_rate: float
    expected_win_rate: float
    reliability_gap: float       # |expected_wr - empirical_wr|
    rolling_pnl: float           # sum of profit_units in window
    is_drifted: bool
    is_warm: bool  # True once ``n >= min_observations``
    drift_reason: str = ""       # "calibration" | "pnl" | "calibration+pnl" | ""


# ──────────────────────────────────────────────────────────────────────
# Per-cell rolling window
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _CellWindow:
    """Rolling buffer of (predicted_p, outcome, profit_units) triples."""

    predicted_ps: Deque[float] = field(default_factory=deque)
    outcomes: Deque[int] = field(default_factory=deque)
    profit_units_deque: Deque[float] = field(default_factory=deque)

    def append(
        self, p: float, outcome: int, profit: float, max_size: int
    ) -> None:
        self.predicted_ps.append(p)
        self.outcomes.append(outcome)
        self.profit_units_deque.append(profit)
        while len(self.predicted_ps) > max_size:
            self.predicted_ps.popleft()
            self.outcomes.popleft()
            self.profit_units_deque.popleft()

    def __len__(self) -> int:
        return len(self.predicted_ps)


# ──────────────────────────────────────────────────────────────────────
# Monitor
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_WINDOW = 200
_DEFAULT_MIN_OBS = 30
# Reliability gap threshold: if |predicted_avg - empirical_wr| > this,
# the cell's calibration is considered unreliable.
_DEFAULT_RELIABILITY_GAP = 0.15
# P&L floor: if rolling P&L (sum of profit_units) drops below this per
# unit of sample size, the cell is considered P&L-drifted.
# Floor is expressed as P/L per observation to be window-size-agnostic.
# -0.10 per obs = losing 0.1 units per pick on average.
_DEFAULT_PNL_FLOOR_PER_OBS = -0.10


@dataclass
class CalibrationDriftMonitor:
    """Per-cell rolling drift monitor.

    Use one monitor per process. Persistence is the caller's job
    (Phase 3); for Phase 2, the monitor is rebuildable by replaying
    settled picks.
    """

    window_size: int = _DEFAULT_WINDOW
    min_observations: int = _DEFAULT_MIN_OBS
    reliability_gap_threshold: float = _DEFAULT_RELIABILITY_GAP
    pnl_floor_per_obs: float = _DEFAULT_PNL_FLOOR_PER_OBS
    _cells: dict[tuple[str, str], _CellWindow] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ── ingestion ──────────────────────────────────────────────────────

    def observe(
        self,
        family: MarketFamily,
        minute: int,
        predicted_p: float,
        outcome: int,
        profit_units: float = 0.0,
    ) -> None:
        """Record one settled-pick observation in the appropriate cell.

        ``outcome`` is 1 for win, 0 for loss. Push / void picks should
        not be recorded — they're not informative for calibration.
        ``profit_units`` is the realized P&L for this pick (e.g., odd-1
        for a win, -1 for a loss). Default 0.0 is backward-compatible
        with callers that don't track P&L.
        """
        if outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1, got {outcome!r}")
        bucket = minute_bucket(minute)
        key = (family.value, bucket)
        with self._lock:
            cell = self._cells.setdefault(key, _CellWindow())
            cell.append(float(predicted_p), int(outcome), float(profit_units), self.window_size)

    # ── inspection ────────────────────────────────────────────────────

    def status(self, family: MarketFamily, minute: int) -> DriftStatus:
        bucket = minute_bucket(minute)
        key = (family.value, bucket)
        with self._lock:
            cell = self._cells.get(key)
            if cell is None or len(cell) == 0:
                return DriftStatus(
                    family=family.value,
                    minute_bucket=bucket,
                    n=0,
                    empirical_win_rate=0.0,
                    expected_win_rate=0.0,
                    reliability_gap=0.0,
                    rolling_pnl=0.0,
                    is_drifted=False,
                    is_warm=False,
                )
            xs = list(cell.predicted_ps)
            ys = list(cell.outcomes)
            ps = list(cell.profit_units_deque)
        n = len(xs)
        is_warm = n >= self.min_observations
        empirical = float(sum(ys) / n) if n else 0.0
        expected = float(sum(xs) / n) if n else 0.0
        pnl_sum = float(sum(ps))
        gap = abs(expected - empirical)

        if is_warm:
            cal_drifted = gap > self.reliability_gap_threshold
            pnl_per_obs = pnl_sum / n
            pnl_drifted = pnl_per_obs < self.pnl_floor_per_obs
        else:
            cal_drifted = False
            pnl_drifted = False

        is_drifted = cal_drifted or pnl_drifted
        if cal_drifted and pnl_drifted:
            drift_reason = "calibration+pnl"
        elif cal_drifted:
            drift_reason = "calibration"
        elif pnl_drifted:
            drift_reason = "pnl"
        else:
            drift_reason = ""

        return DriftStatus(
            family=family.value,
            minute_bucket=bucket,
            n=n,
            empirical_win_rate=empirical,
            expected_win_rate=expected,
            reliability_gap=gap,
            rolling_pnl=pnl_sum,
            is_drifted=is_drifted,
            is_warm=is_warm,
            drift_reason=drift_reason,
        )

    def drifted_cells(self) -> list[DriftStatus]:
        """All cells currently flagged drifted. Operator-facing summary."""
        out: list[DriftStatus] = []
        with self._lock:
            keys = list(self._cells.keys())
        for fam_str, bucket in keys:
            mid = _bucket_midpoint(bucket)
            try:
                fam = MarketFamily(fam_str)
            except ValueError:
                continue
            s = self.status(fam, mid)
            if s.is_drifted:
                out.append(s)
        return out

    def coverage(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        with self._lock:
            for (fam, bucket), cell in self._cells.items():
                out.setdefault(fam, {})[bucket] = len(cell)
        return out

    # ── warm-up from historical settled picks ──────────────────────────

    def warm_up_from_v2_history(
        self, picks_parquet_path: "str | Path",
    ) -> int:
        """Seed the monitor with v2 ``picks_graded.parquet`` observations.

        Reads every settled pick (status ∈ {won, lost}), maps the v2
        market string to a v3 MarketFamily via ``map_v2_market_to_family``,
        and inserts (predicted_p, outcome, profit_units) into the
        appropriate cell.

        Returns the number of observations ingested. Picks with unknown
        family or missing fields are skipped.
        """
        from pathlib import Path

        import polars as pl

        from bip.evaluation.live.engine_v3.calibrator import (
            map_v2_market_to_family,
        )

        p = Path(picks_parquet_path)
        if not p.exists():
            return 0
        df = pl.read_parquet(p).filter(
            pl.col("status").is_in(["won", "lost"])
            & pl.col("our_probability").is_not_null()
            & pl.col("minute").is_not_null()
        )
        n = 0
        for r in df.iter_rows(named=True):
            fam = map_v2_market_to_family(r["market"])
            if fam is None:
                continue
            status = r["status"]
            won = status == "won"
            outcome = 1 if won else 0
            odd = r.get("odd") or r.get("bookmaker_odd")
            if won and odd is not None:
                profit = float(odd) - 1.0
            elif not won:
                profit = -1.0
            else:
                profit = 0.0
            self.observe(
                family=fam,
                minute=int(r["minute"] or 0),
                predicted_p=float(r["our_probability"]),
                outcome=outcome,
                profit_units=profit,
            )
            n += 1
        return n


def _bucket_midpoint(bucket: str) -> int:
    """Approximate minute representative for a bucket label."""
    table = {
        "0-15": 7,
        "15-30": 22,
        "30-45": 37,
        "45-60": 52,
        "60-75": 67,
        "75-90": 82,
        "90+": 95,
    }
    return table.get(bucket, 45)


# ──────────────────────────────────────────────────────────────────────
# Settlement feed — live connection between grader and drift monitor
# ──────────────────────────────────────────────────────────────────────


def feed_graded_picks_to_monitor(
    monitor: CalibrationDriftMonitor,
    graded_picks: "list",
    *,
    family_fn: "callable[[str], MarketFamily | None] | None" = None,
    predicted_p_by_thesis_id: "dict[str, float] | None" = None,
    default_predicted_p: float = 0.5,
    default_minute: int = 45,
) -> int:
    """Feed a list of GradedPick records (from v3_grader) into the monitor.

    This is the missing live settlement feed: the canonical grader
    produces GradedPick records; this function routes them into the
    CalibrationDriftMonitor so rule_11 gates on fresh data.

    Arguments:
        monitor: the live CalibrationDriftMonitor to update.
        graded_picks: list of GradedPick (or dict-like) records. Each
            must have: ``market_id``, ``status`` (won/lost/void/pending),
            ``profit_units``, and optionally ``predicted_p`` and
            ``minute``.
        family_fn: callable that maps a market_id string to a
            MarketFamily (or None to skip). Defaults to the calibrator's
            ``map_v2_market_to_family``.
        predicted_p_by_thesis_id: optional dict mapping thesis_id →
            predicted_p from the pick's original fair_prob. When present,
            overrides the pick's own ``predicted_p`` field.
        default_predicted_p: fallback predicted_p when none is available.
        default_minute: fallback minute when none is available.

    Returns the number of observations ingested.
    """
    from bip.evaluation.live.engine_v3.calibrator import (
        map_v2_market_to_family,
    )

    if family_fn is None:
        family_fn = map_v2_market_to_family

    n = 0
    for pick in graded_picks:
        # Support both dataclass and dict access
        if hasattr(pick, "__dict__") or hasattr(pick, "status"):
            status = getattr(pick, "status", None) or ""
            market_id = getattr(pick, "market_id", None) or ""
            profit = float(getattr(pick, "profit_units", 0.0) or 0.0)
            thesis_id = getattr(pick, "thesis_id", None) or ""
            p_raw = getattr(pick, "predicted_p", None)
            minute_raw = getattr(pick, "minute", None)
        else:
            status = pick.get("status") or ""
            market_id = pick.get("market_id") or ""
            profit = float(pick.get("profit_units", 0.0) or 0.0)
            thesis_id = pick.get("thesis_id") or ""
            p_raw = pick.get("predicted_p")
            minute_raw = pick.get("minute")

        # Only settled picks are informative for calibration
        if status not in ("won", "lost"):
            continue

        fam = family_fn(market_id)
        if fam is None:
            continue

        outcome = 1 if status == "won" else 0
        # Predicted probability: from lookup first, then pick field, then default
        if predicted_p_by_thesis_id and thesis_id in predicted_p_by_thesis_id:
            predicted_p = float(predicted_p_by_thesis_id[thesis_id])
        elif p_raw is not None:
            predicted_p = float(p_raw)
        else:
            predicted_p = default_predicted_p

        minute = int(minute_raw) if minute_raw is not None else default_minute

        monitor.observe(
            family=fam,
            minute=minute,
            predicted_p=predicted_p,
            outcome=outcome,
            profit_units=profit,
        )
        n += 1
    return n


__all__ = [
    "CalibrationDriftMonitor",
    "DriftStatus",
    "feed_graded_picks_to_monitor",
]
