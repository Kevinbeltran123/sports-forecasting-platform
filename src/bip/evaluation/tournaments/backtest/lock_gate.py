"""Lock-gate aggregator: roll up per-predictor-per-market reports into one verdict.

Phase 5 (2026-05-15) deliverable for the WC2026 calibration-lock spike. After
the historical-tournament backtest emits N ``CalibrationReport``s — one per
(predictor × market) combination — this module aggregates them into a
``LockDecision`` artifact consumable by Phase 6's lock JSON emitter.

Aggregation philosophy (SYNTHESIS Conclusion 1, Walsh & Joshi 2024):
- Per-predictor verdicts (not a single global pass/fail) so the operator
  can keep one predictor's well-calibrated markets even when another
  predictor's market fails.
- Per-market breakdowns surfaced in the JSON so post-tournament forensics
  can identify exactly which (predictor, market) drove a CLV miss.
- Three escape hatches for honest reporting when data is missing or partial:
  ``structural-only``, ``coverage-partial``, ``no-reports``.

Status propagation rules (worst-case across markets within a predictor):

    below-gate > unreliable-bins > marginal > pass

The aggregate ``calibration_status`` then takes the worst across predictors,
gated by coverage. Marginal results don't block the lock — Phase 6's emitter
ships them with the verdict surfaced. Below-gate results DO block by
default, but the operator can override via ``allow_below_gate=True`` (the
override is recorded in the LockDecision's ``operator_overrides`` field).

References:
- internal research notes Conclusion 1 (three-step CLV chain)
- internal research notes Walsh & Joshi
- internal planning notes §4 Phase 5, §7 success criteria
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.tournaments.backtest.calibration_report import CalibrationReport

# Default coverage threshold for the lock gate (≥ 90% of WC2026 group-stage
# matches must have predictions per spike §7). Override via parameter when
# stricter / looser policies apply (e.g., for a friendly-only audit).
DEFAULT_COVERAGE_THRESHOLD = 0.90


# ── Status precedence ────────────────────────────────────────────────────────

# Lower number = better. Used to pick the worst status across markets/predictors.
_STATUS_PRECEDENCE: dict[str, int] = {
    "pass": 0,
    "marginal": 1,
    "unreliable-bins": 2,
    "below-gate": 3,
}

LOCK_STATUSES = ("pass", "marginal", "unreliable-bins", "below-gate")
META_STATUSES = ("structural-only", "coverage-partial", "no-reports")
ALL_STATUSES = LOCK_STATUSES + META_STATUSES


def _worst(statuses: Iterable[str]) -> str:
    """Return the worst status by precedence; defaults to 'pass' if empty."""
    seen = list(statuses)
    if not seen:
        return "pass"
    unknown = [s for s in seen if s not in _STATUS_PRECEDENCE]
    if unknown:
        raise ValueError(f"Unknown status(es): {unknown}")
    return max(seen, key=lambda s: _STATUS_PRECEDENCE[s])


# ── Per-predictor verdict ────────────────────────────────────────────────────


class PredictorGateVerdict(BaseModel):
    """Verdict for one predictor across all its markets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    predictor_name: str
    n_markets: int = Field(ge=0)
    market_statuses: dict[str, str]  # market → status from CalibrationReport.gate_status()
    market_brier_for_gate: dict[str, float]
    market_classwise_ece: dict[str, float]
    overall_status: str  # worst across market_statuses

    def passes_lock(self) -> bool:
        """True iff every market is 'pass' or 'marginal'."""
        return self.overall_status in ("pass", "marginal")


# ── Lock decision artifact ───────────────────────────────────────────────────


class LockDecision(BaseModel):
    """Aggregate verdict over all (predictor × market) reports for a backtest run.

    Persisted as JSON alongside ``locked_predictions/`` so post-tournament
    scoring can reconstruct exactly which gates passed and which were
    overridden.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str | None = None

    # Run scope
    held_out_tournaments: tuple[str, ...]  # e.g., ('copa_2024', 'euro_2024')
    n_fixtures_total: int = Field(ge=0)
    n_fixtures_with_predictions: int = Field(ge=0)
    coverage_threshold: float = Field(default=DEFAULT_COVERAGE_THRESHOLD, ge=0.0, le=1.0)

    # Per-predictor breakdown (sorted by predictor_name for deterministic JSON)
    predictor_verdicts: tuple[PredictorGateVerdict, ...]

    # Aggregate status — drives the lock JSON's calibration_status field
    calibration_status: str = Field(
        description=(
            "One of: pass | marginal | below-gate | unreliable-bins | "
            "coverage-partial | structural-only | no-reports"
        ),
    )

    # Provenance — operator override surface
    operator_overrides: tuple[str, ...] = Field(
        default=(),
        description=(
            "Free-form notes documenting any operator overrides "
            "(e.g., 'allow_below_gate=True for corners_poisson based on R-06')."
        ),
    )

    # ── derived ───────────────────────────────────────────────────────────

    @property
    def coverage_pct(self) -> float:
        if self.n_fixtures_total == 0:
            return 0.0
        return self.n_fixtures_with_predictions / self.n_fixtures_total

    @property
    def passes_lock(self) -> bool:
        """True iff calibration_status is 'pass' or 'marginal' (lock proceeds).

        ``below-gate``, ``unreliable-bins``, ``coverage-partial``, and
        ``no-reports`` block the lock by default; ``structural-only`` is a
        narrow case that proceeds with the predictions un-validated (Layer-1
        ship while Layer-2 data is queued).
        """
        return self.calibration_status in ("pass", "marginal", "structural-only")

    # ── persistence ───────────────────────────────────────────────────────

    def to_json(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        return cls.model_validate_json(Path(path).read_text())


# ── Aggregator ───────────────────────────────────────────────────────────────


def _verdict_from_reports(
    predictor_name: str,
    reports: list[CalibrationReport],
) -> PredictorGateVerdict:
    """Build a PredictorGateVerdict from this predictor's CalibrationReports."""
    market_statuses: dict[str, str] = {}
    market_brier: dict[str, float] = {}
    market_ece: dict[str, float] = {}
    for r in reports:
        market_statuses[r.market] = r.gate_status()
        market_brier[r.market] = r.brier_for_gate
        market_ece[r.market] = r.classwise_ece

    overall = _worst(market_statuses.values()) if market_statuses else "pass"

    return PredictorGateVerdict(
        predictor_name=predictor_name,
        n_markets=len(market_statuses),
        market_statuses=market_statuses,
        market_brier_for_gate=market_brier,
        market_classwise_ece=market_ece,
        overall_status=overall,
    )


def evaluate_lock(
    reports: Iterable[CalibrationReport],
    *,
    held_out_tournaments: tuple[str, ...],
    n_fixtures_total: int,
    n_fixtures_with_predictions: int,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    structural_only: bool = False,
    allow_below_gate: bool = False,
    git_sha: str | None = None,
) -> LockDecision:
    """Aggregate per-(predictor × market) reports into a lock decision.

    Args:
        reports: One CalibrationReport per (predictor, market). Multiple
            reports for the same predictor are grouped automatically.
        held_out_tournaments: Tournament slugs withheld from training and
            used as the final test set (operator default: Copa 2024 + Euro 2024).
        n_fixtures_total: Total historical fixtures available in the backtest
            window (denominator for coverage).
        n_fixtures_with_predictions: Fixtures the predictors successfully
            scored (numerator for coverage). Lower than total when player
            availability data is missing for some matches.
        coverage_threshold: Minimum coverage fraction. Default 0.90 per
            spike §7.
        structural_only: Set True when shipping the spike before Layer-2
            data is approved. Returns calibration_status='structural-only'
            regardless of report contents — caller is asserting that the
            structural code is complete and the validation step is queued.
        allow_below_gate: When True, a below-gate predictor does NOT block
            the lock (the verdict surfaces it but ``passes_lock`` returns
            True if ``marginal`` would otherwise hold). Operator override
            recorded in ``operator_overrides``.
        git_sha: Optional commit hash for reproducibility.

    Returns:
        LockDecision with per-predictor verdicts and aggregate
        calibration_status.

    Aggregate status ordering (most to least restrictive):

        no-reports → calibration_status='no-reports' (structural failure)
        structural_only=True → 'structural-only' (intentional Layer-1 ship)
        coverage < threshold → 'coverage-partial'
        any below-gate (no override) → 'below-gate'
        any unreliable-bins → 'unreliable-bins'
        any marginal → 'marginal'
        else → 'pass'
    """
    if structural_only:
        return LockDecision(
            git_sha=git_sha,
            held_out_tournaments=held_out_tournaments,
            n_fixtures_total=n_fixtures_total,
            n_fixtures_with_predictions=n_fixtures_with_predictions,
            coverage_threshold=coverage_threshold,
            predictor_verdicts=(),
            calibration_status="structural-only",
            operator_overrides=(
                "Layer-1 ship: structural code complete, "
                "Layer-2 backtest data not yet loaded.",
            ),
        )

    grouped: dict[str, list[CalibrationReport]] = defaultdict(list)
    for r in reports:
        grouped[r.predictor_name].append(r)

    if not grouped:
        return LockDecision(
            git_sha=git_sha,
            held_out_tournaments=held_out_tournaments,
            n_fixtures_total=n_fixtures_total,
            n_fixtures_with_predictions=n_fixtures_with_predictions,
            coverage_threshold=coverage_threshold,
            predictor_verdicts=(),
            calibration_status="no-reports",
        )

    verdicts = tuple(
        sorted(
            (_verdict_from_reports(name, recs) for name, recs in grouped.items()),
            key=lambda v: v.predictor_name,
        )
    )

    overrides: list[str] = []

    # Coverage gate runs FIRST — if data is sparse, the calibration claim
    # is uninterpretable regardless of report contents.
    coverage = (
        n_fixtures_with_predictions / n_fixtures_total
        if n_fixtures_total > 0
        else 0.0
    )
    if coverage < coverage_threshold:
        return LockDecision(
            git_sha=git_sha,
            held_out_tournaments=held_out_tournaments,
            n_fixtures_total=n_fixtures_total,
            n_fixtures_with_predictions=n_fixtures_with_predictions,
            coverage_threshold=coverage_threshold,
            predictor_verdicts=verdicts,
            calibration_status="coverage-partial",
            operator_overrides=tuple(overrides),
        )

    # Aggregate: worst status across predictors.
    aggregate = _worst(v.overall_status for v in verdicts)

    if aggregate == "below-gate" and allow_below_gate:
        # Operator chose to ship anyway — record the override and demote
        # to 'marginal' so passes_lock returns True.
        overrides.append(
            "allow_below_gate=True: at least one predictor below gate; "
            "operator accepts the risk and surfaces the breakdown in the "
            "lock JSON's per-predictor verdicts."
        )
        aggregate = "marginal"

    return LockDecision(
        git_sha=git_sha,
        held_out_tournaments=held_out_tournaments,
        n_fixtures_total=n_fixtures_total,
        n_fixtures_with_predictions=n_fixtures_with_predictions,
        coverage_threshold=coverage_threshold,
        predictor_verdicts=verdicts,
        calibration_status=aggregate,
        operator_overrides=tuple(overrides),
    )
