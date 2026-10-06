"""Phase-2 calibration layer — sec 5.4 of the v3 engine design notes.

Time-bucketed isotonic calibration per market family. Maps raw predictor
output ``p_raw`` to a calibrated ``p_cal`` whose value is monotone-aligned
with the empirical win rate of past picks in the same (family,
minute_bucket) cell.

Why isotonic, not Platt:
- Isotonic makes no parametric assumption (sigmoid). With small samples
  per (family, bucket) cell, sigmoid over-smooths.
- Monotone increasing matches the contract a fair-prob predictor must
  satisfy: higher raw probability ⇒ higher actual probability.

Why per (family, minute_bucket) rather than global:
- Sec 5.4: "calibra isotónicamente por bucket de minuto (0-15, 15-30,
  30-45, 45-60, 60-75, 75-90+). Captura monotonías temporales que la
  calibración global aplasta — exactamente el sesgo Under/No."

Fallback policy:
- Cell with <30 settled observations → fall back to family-global isotonic
  fit (no minute bucketing).
- Family with <30 observations → identity transform (no calibration),
  logged so the operator knows the family is uncalibrated.

The fit is offline (one shot from settled picks); the live path is a
``transform(p_raw, family, minute) → p_cal`` lookup that takes ~µs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import numpy as np
import structlog
from sklearn.isotonic import IsotonicRegression

from bip.evaluation.live.engine_v3.thesis import MarketFamily

_log = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Minute bucketing
# ──────────────────────────────────────────────────────────────────────


_MINUTE_BUCKET_EDGES: tuple[int, ...] = (0, 15, 30, 45, 60, 75, 90, 200)
_MINUTE_BUCKET_LABELS: tuple[str, ...] = (
    "0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+",
)


def minute_bucket(minute: int) -> str:
    """Map a match minute to one of the 7 buckets defined in sec 5.4."""
    if minute < 0:
        return "0-15"
    for i in range(len(_MINUTE_BUCKET_EDGES) - 1):
        lo = _MINUTE_BUCKET_EDGES[i]
        hi = _MINUTE_BUCKET_EDGES[i + 1]
        if lo <= minute < hi:
            return _MINUTE_BUCKET_LABELS[i]
    return _MINUTE_BUCKET_LABELS[-1]


# ──────────────────────────────────────────────────────────────────────
# Per-cell observation record
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationSample:
    """One settled pick observation used for fitting."""

    family: MarketFamily
    minute: int
    predicted_p: float
    outcome: int  # 1 (won) / 0 (lost). Push / void picks should be filtered out.


# ──────────────────────────────────────────────────────────────────────
# Calibrator
# ──────────────────────────────────────────────────────────────────────


_MIN_CELL_SAMPLES = 30
_MIN_FAMILY_SAMPLES = 30


@dataclass
class IsotonicCalibrator:
    """Time-bucketed isotonic calibration per market family.

    Internal state: a dict ``(family, bucket) → IsotonicRegression`` for
    cells with enough samples, plus ``family → IsotonicRegression`` global
    fallback. ``transform`` looks up the cell first, falls back to family,
    then to identity.

    Fitting requires the global-family fallback to be the union of all
    minute buckets, so an unknown bucket still gets the best-available
    estimator.
    """

    per_cell: dict[tuple[str, str], IsotonicRegression] = field(default_factory=dict)
    per_family: dict[str, IsotonicRegression] = field(default_factory=dict)
    n_per_cell: dict[tuple[str, str], int] = field(default_factory=dict)
    n_per_family: dict[str, int] = field(default_factory=dict)
    is_fitted: bool = False

    # ── fitting ────────────────────────────────────────────────────────

    def fit(self, samples: Iterable[CalibrationSample]) -> "IsotonicCalibrator":
        by_family: dict[str, list[CalibrationSample]] = {}
        by_cell: dict[tuple[str, str], list[CalibrationSample]] = {}
        for s in samples:
            fam = s.family.value
            bucket = minute_bucket(s.minute)
            by_family.setdefault(fam, []).append(s)
            by_cell.setdefault((fam, bucket), []).append(s)

        for fam, items in by_family.items():
            self.n_per_family[fam] = len(items)
            if len(items) < _MIN_FAMILY_SAMPLES:
                _log.info("calibrator_family_skipped_low_n", family=fam, n=len(items))
                continue
            xs = np.asarray([i.predicted_p for i in items], dtype=np.float64)
            ys = np.asarray([i.outcome for i in items], dtype=np.float64)
            reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            reg.fit(xs, ys)
            self.per_family[fam] = reg

        for (fam, bucket), items in by_cell.items():
            self.n_per_cell[(fam, bucket)] = len(items)
            if len(items) < _MIN_CELL_SAMPLES:
                continue
            xs = np.asarray([i.predicted_p for i in items], dtype=np.float64)
            ys = np.asarray([i.outcome for i in items], dtype=np.float64)
            reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            reg.fit(xs, ys)
            self.per_cell[(fam, bucket)] = reg

        self.is_fitted = bool(self.per_family) or bool(self.per_cell)
        return self

    # ── inference ──────────────────────────────────────────────────────

    def transform(
        self,
        p_raw: float,
        family: MarketFamily,
        minute: int,
    ) -> float:
        """Return calibrated probability for ``p_raw`` in cell
        ``(family, minute_bucket(minute))``. Falls back to family-global
        then to identity. Always returns a value in [0, 1]."""
        if not self.is_fitted:
            return p_raw
        bucket = minute_bucket(minute)
        fam = family.value
        cell_key = (fam, bucket)
        if cell_key in self.per_cell:
            return float(self.per_cell[cell_key].transform([p_raw])[0])
        if fam in self.per_family:
            return float(self.per_family[fam].transform([p_raw])[0])
        return p_raw

    # ── reporting ─────────────────────────────────────────────────────

    def coverage(self) -> dict[str, dict[str, int]]:
        """Returns ``{family: {bucket: n, '_total': n_family}}`` so the
        operator can audit which cells are actively calibrated."""
        out: dict[str, dict[str, int]] = {}
        for (fam, bucket), n in self.n_per_cell.items():
            out.setdefault(fam, {})[bucket] = n
        for fam, n in self.n_per_family.items():
            out.setdefault(fam, {})["_total"] = n
        return out

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"loaded object is {type(obj)!r}, not IsotonicCalibrator")
        return obj


def map_v2_market_to_family(market: str) -> MarketFamily | None:
    """Best-effort mapping from v2 ``picks_graded.market`` strings to the
    v3 MarketFamily enum. Returns None for v2-only markets that don't
    have a v3 counterpart (e.g. draw_no_bet, fulltime_result variants
    fall into RESULT_1X2 in v3 but no archetype predicts there yet)."""
    m = market.lower()
    if "corner" in m:
        return MarketFamily.CORNERS
    if "btts" in m or "both_teams" in m:
        return MarketFamily.BTTS
    if "card" in m:
        return MarketFamily.CARDS
    if "team_to_score_first" in m or "next_goal" in m or "first_goal" in m:
        return MarketFamily.NEXT_GOAL
    if "ou_" in m or "goal" in m or "over" in m or "under" in m:
        return MarketFamily.GOALS
    if "handicap" in m:
        return MarketFamily.ASIAN_HANDICAP
    if "fulltime_result" in m or "draw_no_bet" in m or "result" in m:
        return MarketFamily.RESULT_1X2
    return None


__all__ = [
    "CalibrationSample",
    "IsotonicCalibrator",
    "map_v2_market_to_family",
    "minute_bucket",
]
