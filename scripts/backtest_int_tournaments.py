"""International tournament backtest — Phase 5 pre-lock gate evaluation.

Drives the full pre-WC2026-lock backtest: replays historical international
tournaments through every predictor, accumulates per-(predictor × market)
``CalibrationReport``s via the Phase 1 audit harness, then aggregates them
into a single ``LockDecision`` artifact via Phase 5's ``evaluate_lock``.

Layer-1 ship (this commit) — exposes pure functions consumable by tests:

- ``run_backtest_layer1(snapshots, predictors)`` replays a list of
  ``BacktestSnapshot`` objects through the given predictors and returns
  the list of ``CalibrationReport``s.
- ``aggregate_to_lock_decision(reports, ...)`` is a thin wrapper around
  ``evaluate_lock`` that adds default held-out tournaments and writes the
  artifact to disk.
- The CLI ``main()`` is gated ``# requires-real-data`` and raises
  ``NotImplementedError`` until the operator approves the API-Football
  pull of WC 2018, Euro 2024, Copa 2024 fixtures.

Layer-2 (queued, deadline 2026-06-05) — wires the above to:

- ``data/cache/international_history.parquet`` (populated by
  ``scripts/seed_international_history.py``)
- The 4 production predictors (BivariatePoisson, IndependentPoisson,
  EloLogistic, corners_poisson)
- A real walk-forward replay where each tournament's matches are scored
  using priors trained ONLY on fixtures preceding the tournament

Held-out set (operator-approved 2026-05-08): Copa 2024 + Euro 2024 — most
recent and closest in style to WC2026.

Usage when Layer-2 lands:

    uv run python scripts/backtest_int_tournaments.py \\
        --history-parquet data/cache/international_history.parquet \\
        --held-out copa_2024 euro_2024 \\
        --output data/cache/lock_decision.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bip.evaluation.tournaments.backtest.calibration_audit import CalibrationAudit
from bip.evaluation.tournaments.backtest.calibration_report import (
    MARKET_BTTS,
    MARKET_OU_2_5,
    CalibrationReport,
)
from bip.evaluation.tournaments.backtest.lock_gate import (
    DEFAULT_COVERAGE_THRESHOLD,
    LockDecision,
    evaluate_lock,
)

# Operator-approved held-out tournaments (spike doc Phase 5 open Q3, 2026-05-08).
# These two are the most recent + closest in style to WC2026.
DEFAULT_HELD_OUT_TOURNAMENTS = ("copa_2024", "euro_2024")


# ── Snapshot + prediction protocol (testable seam) ───────────────────────────


@dataclass(frozen=True)
class BacktestSnapshot:
    """Minimal payload the harness needs to score one historical fixture.

    Layer-1 keeps the snapshot abstract — the predictor protocol takes it
    and returns a ``MatchPrediction``-like object exposing per-market
    probabilities. Layer-2 instantiates concrete snapshots from the
    StatsBomb Parquet store (see ``scripts/seed_statsbomb_tournaments.py``).

    Phase 5 Layer-2 added the optional ``observed_home_goals`` and
    ``observed_away_goals`` so stateful predictors (e.g.,
    ``BayesianPoissonPredictor`` in ``scripts/run_phase5_backtest.py``)
    can update their internal state after scoring. Defaults are None so
    Layer-1 stub-predictor tests stay backward-compatible.
    """

    match_id: str
    tournament: str  # slug used to filter held-out vs train
    home_team_id: int
    away_team_id: int
    # Observed outcomes (post-match, used for calibration scoring).
    observed_1x2: int  # 0=home win, 1=draw, 2=away win
    observed_total_goals: int
    observed_btts: int  # 0=no, 1=yes
    # Optional split-goal observation (needed by stateful predictors)
    observed_home_goals: int | None = None
    observed_away_goals: int | None = None
    # Optional per-team corner counts (for corners O/U markets)
    observed_home_corners: int | None = None
    observed_away_corners: int | None = None
    # Optional per-team xG totals (forward-information signal — see
    # `aggregate_xg_per_team` in seed_statsbomb_tournaments.py). Predictors
    # using xG-blended rates consume these; predictors that don't, ignore.
    observed_home_xg: float | None = None
    observed_away_xg: float | None = None
    # Optional ISO date string for between-window decay computation
    match_date: str | None = None


@dataclass(frozen=True)
class FixturePrediction:
    """Probabilities a predictor emits for one fixture across markets.

    Markets the predictor doesn't support are set to None; the harness
    skips those when recording.
    """

    p_home_win: float
    p_draw: float
    p_away_win: float
    p_btts: float | None = None
    p_over_2_5: float | None = None


class PredictorProtocol(Protocol):
    """Layer-1 narrow interface — Layer-2 wraps the production predictors."""

    name: str

    def predict_fixture(self, snapshot: BacktestSnapshot) -> FixturePrediction:
        ...


# ── Pure replay (Layer-1 testable) ───────────────────────────────────────────


def run_backtest_layer1(
    snapshots: Iterable[BacktestSnapshot],
    predictors: Iterable[PredictorProtocol],
    *,
    git_sha: str | None = None,
    progress_cb: Callable[[str, int], None] | None = None,
    n_bins: int | None = None,
    min_bin_fill: float | None = None,
) -> list[CalibrationReport]:
    """Replay snapshots through each predictor; return one report per (predictor, market).

    The harness is intentionally pure: it does not load data, hit APIs, or
    write to disk. Tests construct synthetic snapshots and stub predictors
    to verify the recording / aggregation pipeline. Layer-2 will provide
    real snapshots from the international-history Parquet.

    ``progress_cb(predictor_name, n_processed)`` is invoked once per snapshot
    if provided — for CLI progress bars when Layer-2 is wired.

    ``n_bins`` and ``min_bin_fill`` override the audit's defaults — useful
    when calibration on football data needs fewer bins or a lower bin-fill
    threshold (probabilities concentrate in 0.40-0.60, making the default
    20-bin / 80%-fill scheme empirically unreachable). Defaults preserve
    Layer-1 test compatibility when not specified.
    """
    snapshots_list = list(snapshots)
    reports: list[CalibrationReport] = []

    audit_kwargs: dict[str, object] = {}
    if n_bins is not None:
        audit_kwargs["n_bins"] = n_bins
    if min_bin_fill is not None:
        audit_kwargs["min_bin_fill"] = min_bin_fill

    for predictor in predictors:
        audit = CalibrationAudit(
            predictor_name=predictor.name, git_sha=git_sha, **audit_kwargs
        )
        for i, snap in enumerate(snapshots_list, start=1):
            pred = predictor.predict_fixture(snap)
            audit.record_1x2(
                p_home=pred.p_home_win,
                p_draw=pred.p_draw,
                p_away=pred.p_away_win,
                outcome=snap.observed_1x2,
            )
            if pred.p_btts is not None:
                audit.record_binary(
                    MARKET_BTTS,
                    p_yes=pred.p_btts,
                    outcome=snap.observed_btts,
                )
            if pred.p_over_2_5 is not None:
                audit.record_binary(
                    MARKET_OU_2_5,
                    p_yes=pred.p_over_2_5,
                    outcome=int(snap.observed_total_goals > 2),
                )
            if progress_cb is not None:
                progress_cb(predictor.name, i)
        reports.extend(audit.compute_reports())

    return reports


def aggregate_to_lock_decision(
    reports: Iterable[CalibrationReport],
    *,
    n_fixtures_total: int,
    n_fixtures_with_predictions: int,
    held_out_tournaments: tuple[str, ...] = DEFAULT_HELD_OUT_TOURNAMENTS,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    structural_only: bool = False,
    allow_below_gate: bool = False,
    git_sha: str | None = None,
    output_path: Path | None = None,
) -> LockDecision:
    """Wrap evaluate_lock + optional disk persistence."""
    decision = evaluate_lock(
        reports,
        held_out_tournaments=held_out_tournaments,
        n_fixtures_total=n_fixtures_total,
        n_fixtures_with_predictions=n_fixtures_with_predictions,
        coverage_threshold=coverage_threshold,
        structural_only=structural_only,
        allow_below_gate=allow_below_gate,
        git_sha=git_sha,
    )
    if output_path is not None:
        decision.to_json(output_path)
    return decision


# ── CLI (Layer-2 gated) ──────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--history-parquet",
        type=Path,
        default=Path("data/cache/international_history.parquet"),
        help="Path to the seeded international-history Parquet store",
    )
    p.add_argument(
        "--held-out",
        nargs="+",
        default=list(DEFAULT_HELD_OUT_TOURNAMENTS),
        help="Tournament slugs withheld for the final gate test set",
    )
    p.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
    )
    p.add_argument(
        "--allow-below-gate",
        action="store_true",
        help="Operator override: ship the lock even if a predictor is below gate",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/lock_decision.json"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "scripts/backtest_int_tournaments.py is Layer-2 (requires-real-data). "
        "Wiring depends on (a) scripts/seed_international_history.py output "
        "Parquet, and (b) production predictor adapters implementing "
        "PredictorProtocol over real BlendedRates / lineup data. "
        f"Args parsed OK: {vars(args)}"
    )


if __name__ == "__main__":
    main()
