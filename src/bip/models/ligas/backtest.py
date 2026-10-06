"""Walk-forward backtest harness for LigasModel — Sprint 1 Ola C.

Wraps the existing WalkForwardSplitter (which already enforces the
temporal leakage assertion) and runs the StackedEnsemble.fit_fold inside
each fold to produce OOF test-window probabilities. The harness then:

  1. Computes the Brier-score CI on the concatenated OOF predictions
  2. Computes the ROI CI on the picks that pass the EV threshold
  3. Calls evaluate_gate() to produce a GateDecision
  4. Returns (ModelMetrics, GateDecision) for the caller

Layer-1 (this commit): synthetic-data harness — verifies the wiring,
no-leakage invariants, and bootstrap-CI gate end-to-end. Layer-2 real
historical-corpus runs land in Sprint 1+ once data ingest is wired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from bip.models.base import ModelMetrics
from bip.models.ligas.gate import (
    GateDecision,
    bootstrap_brier_ci,
    bootstrap_roi_ci,
    evaluate_gate,
)
from bip.train.backtest import EDGE_THRESHOLD_PCT, simulate_pick
from bip.train.stacking import StackedEnsemble
from bip.train.walkforward import WalkForwardSplitter

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FoldResult:
    fold_idx: int
    n_train: int
    n_test: int
    test_indices: np.ndarray
    test_probs: np.ndarray  # (n_test, 3) for 1x2 in this harness
    test_y: np.ndarray


def _validate_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"X", "y", "dates"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Missing payload keys: {sorted(missing)}")
    X = np.asarray(payload["X"], dtype=float)
    y = np.asarray(payload["y"], dtype=int)
    dates = np.asarray(payload["dates"])
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError(
            f"Shape mismatch: X={X.shape}, y={y.shape}, dates={dates.shape}"
        )
    # Optional opening_odds: shape (n, 3) for 1x2; default to no odds (NaNs)
    odds = payload.get("opening_odds")
    if odds is None:
        opening_odds = np.full((len(X), 3), np.nan, dtype=float)
    else:
        opening_odds = np.asarray(odds, dtype=float)
        if opening_odds.shape != (len(X), 3):
            raise ValueError(
                f"opening_odds must be shape ({len(X)}, 3); got {opening_odds.shape}"
            )
    return X, y, dates, opening_odds


def _run_folds(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    n_splits: int,
) -> list[FoldResult]:
    """Run StackedEnsemble.fit_fold across walk-forward folds.

    Asserts dates_tr.max() < dates_te.min() via WalkForwardSplitter; the
    inner OOF in fit_fold also asserts. Together this is double-leakage
    defense.
    """
    splitter = WalkForwardSplitter(n_splits=n_splits)
    out: list[FoldResult] = []
    for fold_idx, train_idx, test_idx in splitter.split(X, dates):
        ens = StackedEnsemble()
        test_probs = ens.fit_fold(
            X_tr=X[train_idx],
            y_tr=y[train_idx],
            X_te=X[test_idx],
            dates_tr=dates[train_idx],
            n_inner=min(3, max(2, n_splits - 2)),  # keep inner folds bounded
        )
        out.append(
            FoldResult(
                fold_idx=fold_idx,
                n_train=len(train_idx),
                n_test=len(test_idx),
                test_indices=test_idx,
                test_probs=test_probs,
                test_y=y[test_idx],
            )
        )
    return out


def _select_picks_for_fold(
    fold: FoldResult,
    opening_odds: np.ndarray,
    *,
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Return list of (global_row_idx, selection_idx, edge) per fold pick.

    Uses simulate_pick (single source of truth from train/backtest.py)
    so backtest CLV math and live picks share the same logic.
    """
    picks: list[tuple[int, int, float]] = []
    for local_i in range(fold.n_test):
        gi = int(fold.test_indices[local_i])
        odds_row = opening_odds[gi]
        idx = simulate_pick(fold.test_probs[local_i], odds_row, threshold=threshold)
        if idx is None:
            continue
        edge = float(fold.test_probs[local_i, idx] * odds_row[idx] - 1.0)
        picks.append((gi, int(idx), edge))
    return picks


def _pl_units_for_pick(
    pick_idx_in_y: int,
    selection_idx: int,
    y: np.ndarray,
    opening_odds: np.ndarray,
) -> float:
    """Compute realized P/L in units for one pick (1u stake).

    Returns +odds-1 on win, -1.0 on loss. The opening odds row is the
    decimal price paid at stake.
    """
    odds = float(opening_odds[pick_idx_in_y, selection_idx])
    if int(y[pick_idx_in_y]) == selection_idx:
        return odds - 1.0
    return -1.0


def run_walkforward(
    payload: dict[str, Any],
    *,
    n_splits: int = 5,
    ev_threshold: float = EDGE_THRESHOLD_PCT,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> tuple[ModelMetrics, GateDecision]:
    """Run walk-forward CV + bootstrap-CI gate on a payload.

    Payload keys:
      - X (n, d), y (n,), dates (n,) — REQUIRED
      - opening_odds (n, 3) — OPTIONAL; without it, no picks are emitted
        and the gate runs on Brier alone (SHADOW iff calibration OK).

    Returns (ModelMetrics, GateDecision).
    """
    X, y, dates, opening_odds = _validate_payload(payload)

    folds = _run_folds(X, y, dates, n_splits=n_splits)

    # Concat OOF predictions for Brier CI
    all_probs = np.concatenate([f.test_probs for f in folds], axis=0)
    all_y = np.concatenate([f.test_y for f in folds], axis=0)
    brier_ci = bootstrap_brier_ci(
        all_y, all_probs, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
    )

    # Walk picks + realized P/L
    picks_pl: list[float] = []
    n_hits = 0
    for f in folds:
        for gi, sel, _edge in _select_picks_for_fold(
            f, opening_odds, threshold=ev_threshold
        ):
            pl = _pl_units_for_pick(gi, sel, y, opening_odds)
            picks_pl.append(pl)
            if pl > 0:
                n_hits += 1

    n_picks = len(picks_pl)
    roi_ci = (
        bootstrap_roi_ci(picks_pl, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
        if n_picks > 0
        else None
    )
    hit_rate = (n_hits / n_picks) if n_picks > 0 else 0.0
    roi_point = float(np.mean(picks_pl)) if n_picks > 0 else 0.0

    decision = evaluate_gate(brier_ci=brier_ci, roi_ci=roi_ci, n_picks=n_picks)

    metrics = ModelMetrics(
        n_picks=n_picks,
        hit_rate=hit_rate,
        roi=roi_point,
        brier_score=brier_ci[1],
        clv_mean=None,  # Sprint 1 doesn't have CLV — needs closing odds (Sprint 2+)
        notes=(
            f"verdict={decision.verdict.value}; "
            f"brier_ci=({brier_ci[0]:.4f}, {brier_ci[1]:.4f}, {brier_ci[2]:.4f}); "
            f"roi_ci={roi_ci if roi_ci else 'n/a'}"
        ),
    )
    log.info(
        "ligas_walkforward_done",
        n_folds=len(folds),
        n_picks=n_picks,
        verdict=decision.verdict.value,
    )
    return metrics, decision


__all__ = ["FoldResult", "run_walkforward"]
