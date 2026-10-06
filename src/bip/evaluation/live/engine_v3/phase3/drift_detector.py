"""Calibration drift detector — section 5.6 of the v3 engine design notes.

> Por cada (régimen, market), ventana rodante de 200 picks. KS-test
> entre distribución predicha y empírica. Si rechaza al 1%, suspende
> esa celda hasta recalibración semanal.

The implementation is straightforward:

- Rolling deque per ``(regime, market_family)`` cell, max length
  ``window_size`` (default 200).
- Each entry is ``(predicted_prob, observed_outcome)`` (outcome ∈ {0,1}).
- ``check_drift`` runs a two-sample Kolmogorov-Smirnov test between
  the empirical hit-rate distribution and the predicted distribution.
- If p_value < ``alpha`` (default 0.01), the cell is **suspended**.

The suspension is sticky: cells remain suspended until ``resume(cell)``
is explicitly called — meant to happen during weekly recalibration.
A separate ``snapshot_state`` API exposes the suspension list for
Telegram telemetry / dashboards.

What this is **not**: a tool for re-fitting calibrators. That's the
job of ``calibration.py``. This is purely diagnostic — it tells the
operator which cells need attention.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from scipy import stats

from bip.evaluation.live.engine_v3.thesis import MarketFamily

DriftCell = tuple[str, str]  # (regime, family_value)


@dataclass
class DriftReport:
    cell: DriftCell
    n: int
    ks_statistic: float
    p_value: float
    rejected: bool


@dataclass
class CalibrationDriftDetector:
    """KS-based drift detector with rolling window per cell.

    The detector is stateful — it accumulates predicted/outcome pairs
    via ``record`` and exposes ``check_drift`` to evaluate one or all
    cells at any point.

    ``alpha`` is the rejection threshold for the KS test (default 0.01).
    Lower alpha = more lenient (harder to reject the null of "predictor
    is well-calibrated").
    """

    window_size: int = 200
    alpha: float = 0.01
    min_samples: int = 50  # don't fire below this even if KS rejects

    _data: dict[DriftCell, deque[tuple[float, float]]] = field(default_factory=dict)
    _suspended: dict[DriftCell, datetime] = field(default_factory=dict)
    _last_report: dict[DriftCell, DriftReport] = field(default_factory=dict)

    # ── recording ───────────────────────────────────────────────────────

    def record(
        self,
        regime: str,
        family: MarketFamily,
        predicted_p: float,
        outcome: float,
    ) -> None:
        key = (regime, family.value)
        buf = self._data.get(key)
        if buf is None:
            buf = deque(maxlen=self.window_size)
            self._data[key] = buf
        buf.append((float(predicted_p), float(outcome)))

    # ── KS evaluation ───────────────────────────────────────────────────

    def check_drift(self, regime: str, family: MarketFamily) -> DriftReport | None:
        """Return a report or None when the cell hasn't accumulated
        ``min_samples`` observations.

        The KS test is two-sample: predicted CDF vs empirical CDF.
        We use predicted probabilities as one sample and outcomes as
        the other — a coarse but standard calibration check. For
        binary outcomes, the alternative is the Hosmer-Lemeshow test;
        we prefer KS for monotonicity-only diagnostic.
        """
        key = (regime, family.value)
        buf = self._data.get(key)
        if not buf or len(buf) < self.min_samples:
            return None
        predicted = [p for p, _ in buf]
        outcomes = [o for _, o in buf]
        ks, p_value = stats.ks_2samp(predicted, outcomes)
        rejected = bool(p_value < self.alpha)
        report = DriftReport(
            cell=key, n=len(buf),
            ks_statistic=float(ks), p_value=float(p_value),
            rejected=rejected,
        )
        self._last_report[key] = report
        if rejected:
            # Suspend on rejection (sticky).
            self._suspended[key] = datetime.now(timezone.utc)
        return report

    def check_all(self) -> list[DriftReport]:
        out: list[DriftReport] = []
        for (regime, fam_val) in list(self._data.keys()):
            family = MarketFamily(fam_val)
            r = self.check_drift(regime, family)
            if r is not None:
                out.append(r)
        return out

    # ── suspension state ────────────────────────────────────────────────

    def is_suspended(self, regime: str, family: MarketFamily) -> bool:
        return (regime, family.value) in self._suspended

    def resume(self, regime: str, family: MarketFamily) -> bool:
        """Lift the suspension on this cell. Returns True when something
        was actually un-suspended."""
        return self._suspended.pop((regime, family.value), None) is not None

    def snapshot_state(self) -> dict:
        """Expose state for dashboards / Telegram telemetry."""
        return {
            "n_cells_tracked": len(self._data),
            "n_cells_suspended": len(self._suspended),
            "suspended_cells": sorted(self._suspended.keys()),
            "last_reports": {
                f"{r}:{f}": {
                    "n": rep.n,
                    "ks": rep.ks_statistic,
                    "p_value": rep.p_value,
                    "rejected": rep.rejected,
                }
                for (r, f), rep in self._last_report.items()
            },
        }


__all__ = ["CalibrationDriftDetector", "DriftReport"]
