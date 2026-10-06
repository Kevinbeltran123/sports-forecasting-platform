"""Time-bucketed isotonic calibration + hierarchical fallback.

Sections 5.1–5.4 of the v3 engine design notes.

Hierarchy of lookups when calibrating a predictor's raw probability:

1. ``(regime, market_family, minute_bucket)`` — most specific.
2. ``(market_family, minute_bucket)`` — family-level (regime-agnostic).
3. Identity (no calibration, return raw) — when no calibrator fitted.

When a target cell has n < ``min_samples`` (default 30), the fitter
delegates to the next-coarser cell. The end of the chain is the
identity calibrator — we never *invent* a calibration where data is
absent.

Persistence: a fitted ``TimeBucketedIsotonic`` is JSON-serialisable
(via the underlying ``CalibrationTable``). Load semantics are
backwards-compatible: an older calibrator file with fewer cells still
parses; missing cells fall back through the chain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.phase3.regimes import bucket_gsv
from bip.evaluation.live.engine_v3.thesis import MarketFamily


# ──────────────────────────────────────────────────────────────────────
# Single-cell calibrator
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IsotonicCell:
    """One isotonic regression fit on (predicted_p, outcome) pairs."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    n: int

    def apply(self, p: float) -> float:
        if self.n < 2 or not self.x:
            return float(p)
        return float(np.interp(p, self.x, self.y))

    @classmethod
    def fit(cls, predicted: Iterable[float], outcome: Iterable[float]) -> IsotonicCell:
        """Fit a standard isotonic regression on the data.

        ``outcome`` is 0/1 for win/loss; the fit produces a monotone
        increasing function from predicted to empirical hit rate."""
        preds = np.asarray(list(predicted), dtype=float)
        outs = np.asarray(list(outcome), dtype=float)
        n = int(preds.size)
        if n < 2:
            return cls(x=(), y=(), n=n)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(preds, outs)
        # Sample the function on a dense grid for serialisation
        grid = np.linspace(0.0, 1.0, 101)
        mapped = iso.transform(grid)
        return cls(x=tuple(grid.tolist()), y=tuple(mapped.tolist()), n=n)


# ──────────────────────────────────────────────────────────────────────
# Top-level calibrator
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TimeBucketedIsotonic:
    """Hierarchical isotonic calibrator.

    Cells are keyed by ``(regime, family, minute_bucket)``. Fallback
    cells are keyed by ``(family, minute_bucket)`` and store
    family-wide calibration when the regime-specific cell is sparse.

    The ``min_samples`` threshold gates promotion: a cell with fewer
    observations doesn't get its own calibrator — the next coarser cell
    serves it instead.
    """

    cells: dict[tuple[str, str, str], IsotonicCell] = field(default_factory=dict)
    family_cells: dict[tuple[str, str], IsotonicCell] = field(default_factory=dict)
    min_samples: int = 30

    # ── lookup ──────────────────────────────────────────────────────────

    def calibrate(
        self,
        raw_prob: float,
        *,
        regime: str,
        family: MarketFamily,
        minute_bucket: str,
    ) -> float:
        """Apply the most specific calibrator available."""
        key = (regime, family.value, minute_bucket)
        cell = self.cells.get(key)
        if cell is None or cell.n < self.min_samples:
            cell = self.family_cells.get((family.value, minute_bucket))
        if cell is None or cell.n < self.min_samples:
            return float(raw_prob)
        return cell.apply(raw_prob)

    def calibrate_for_gsv(
        self,
        raw_prob: float,
        gsv: GameStateVector,
        family: MarketFamily,
    ) -> float:
        """Convenience wrapper — derive regime & minute bucket from the GSV."""
        regime = bucket_gsv(gsv).as_str()
        # Re-derive the minute bucket from the same source the regime used
        # (kept simple — bucket_gsv already encodes it).
        mb = regime.split(":min-")[1].split(":")[0]
        return self.calibrate(raw_prob, regime=regime, family=family, minute_bucket=mb)

    # ── fitting ─────────────────────────────────────────────────────────

    def fit_cell(
        self,
        regime: str,
        family: MarketFamily,
        minute_bucket: str,
        predicted: Iterable[float],
        outcome: Iterable[float],
    ) -> IsotonicCell:
        cell = IsotonicCell.fit(predicted, outcome)
        if cell.n >= 2:
            self.cells[(regime, family.value, minute_bucket)] = cell
        return cell

    def fit_family_cell(
        self,
        family: MarketFamily,
        minute_bucket: str,
        predicted: Iterable[float],
        outcome: Iterable[float],
    ) -> IsotonicCell:
        cell = IsotonicCell.fit(predicted, outcome)
        if cell.n >= 2:
            self.family_cells[(family.value, minute_bucket)] = cell
        return cell

    # ── persistence ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "min_samples": self.min_samples,
            "cells": [
                {
                    "regime": r, "family": f, "minute_bucket": mb,
                    "x": list(cell.x), "y": list(cell.y), "n": cell.n,
                }
                for (r, f, mb), cell in self.cells.items()
            ],
            "family_cells": [
                {
                    "family": f, "minute_bucket": mb,
                    "x": list(cell.x), "y": list(cell.y), "n": cell.n,
                }
                for (f, mb), cell in self.family_cells.items()
            ],
        }

    def save(self, path: Path | str) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))
        return p

    @classmethod
    def from_dict(cls, payload: dict) -> TimeBucketedIsotonic:
        cells: dict[tuple[str, str, str], IsotonicCell] = {}
        family_cells: dict[tuple[str, str], IsotonicCell] = {}
        for c in payload.get("cells", []):
            cells[(c["regime"], c["family"], c["minute_bucket"])] = IsotonicCell(
                x=tuple(c.get("x", [])), y=tuple(c.get("y", [])), n=int(c["n"]),
            )
        for c in payload.get("family_cells", []):
            family_cells[(c["family"], c["minute_bucket"])] = IsotonicCell(
                x=tuple(c.get("x", [])), y=tuple(c.get("y", [])), n=int(c["n"]),
            )
        return cls(
            cells=cells, family_cells=family_cells,
            min_samples=int(payload.get("min_samples", 30)),
        )

    @classmethod
    def load(cls, path: Path | str) -> TimeBucketedIsotonic:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ──────────────────────────────────────────────────────────────────────
# Market-transfer matrix — sec 5.3
# ──────────────────────────────────────────────────────────────────────


# Empirically-anchored cross-market mappings used to bootstrap calibration
# for cold-start markets. Key format: ``(src, dst) → (a, b)`` where the
# transform is ``p_dst ≈ a * p_src + b`` (linear).
# Transfer is asymmetric in general — we encode each direction explicitly
# rather than auto-inverting. Replace with learned transforms after the
# Phase 3 sec 5.3 backfill produces enough cross-market resolved data.
_TRANSFER_OFFSETS: dict[
    tuple[MarketFamily, MarketFamily], tuple[float, float]
] = {
    # 2H corners share ≈ 55% of total corners. Mapping total→2H shifts down ~5pp.
    (MarketFamily.CORNERS, MarketFamily.NEXT_CORNER): (0.85, -0.05),
    # Reverse direction: 2H over → total over implies more corners overall.
    (MarketFamily.NEXT_CORNER, MarketFamily.CORNERS): (1.10, 0.05),
    # BTTS yes lifts P(over 2.5) by ~0.07 on average (BTTS requires both
    # sides to score, total goals over 2.5 is a weaker condition).
    (MarketFamily.BTTS, MarketFamily.GOALS): (1.0, 0.07),
    # The reverse — P(BTTS yes) is a tighter event than P(over 2.5), drop ~5pp.
    (MarketFamily.GOALS, MarketFamily.BTTS): (1.0, -0.05),
    # Next-goal P(any team scores in window) ↔ goals-over correlation.
    (MarketFamily.GOALS, MarketFamily.NEXT_GOAL): (1.0, 0.0),
    (MarketFamily.NEXT_GOAL, MarketFamily.GOALS): (1.0, 0.0),
}


def transfer_calibration(
    src: MarketFamily, dst: MarketFamily, src_p: float,
) -> float | None:
    """Linear-shift transfer between adjacent market families.

    Returns ``None`` when the pair has no defined transfer (the caller
    should fall back to identity or shadow-mode in that case).
    """
    coef = _TRANSFER_OFFSETS.get((src, dst))
    if coef is None:
        return None
    a, b = coef
    return max(0.0, min(1.0, a * src_p + b))


__all__ = [
    "IsotonicCell",
    "TimeBucketedIsotonic",
    "transfer_calibration",
]
