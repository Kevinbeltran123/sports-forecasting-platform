"""CalibrationReport — the canonical artifact of Phase 1's audit.

One report per (predictor, market) combination. Holds every metric the lock
gate cares about plus the bin-fill diagnostic so a downstream consumer can
distinguish "ECE looks fine because the model is calibrated" from "ECE looks
fine because half the bins are empty".

Persisted to JSON under `data/cache/evaluation/calibration/<predictor>/<market>.json`
so successive Phase 2/3/4 calibration changes can be A/B'd against the
baseline captured here.

Hard gates (LOCK criteria from SPIKE-wc2026-calibration-lock.md §7):

    1X2:           classwise-ECE ≤ 5%   AND   multiclass-Brier ≤ 0.21
    Corners O/U:   classwise-ECE ≤ 5%   AND   binary-Brier     ≤ 0.20

A report with `bin_fill_passes=False` does NOT count toward the gate even
if its ECE is below threshold — Walsh & Joshi 2024 explicitly warn against
that pattern.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bip.evaluation.tournaments.backtest.calibration_metrics import (
    DEFAULT_MIN_BIN_FILL,
    DEFAULT_N_BINS,
    bin_fill_check,
    classwise_ece,
)
from bip.evaluation.tournaments.backtest.metrics import (
    brier_score,
    log_loss,
    multiclass_brier_score,
    multiclass_log_loss,
)

# ---------------------------------------------------------------------------
# Lock gate thresholds — single source of truth for all phases
# ---------------------------------------------------------------------------

LOCK_GATE_ECE_MAX = 0.05
LOCK_GATE_BRIER_MAX_1X2 = 0.21
LOCK_GATE_BRIER_MAX_CORNERS = 0.20

# Markets we evaluate. Must match GoalsDistribution / CornersDistribution
# fields so the audit harness can pull probabilities by name.
MARKET_1X2 = "1X2"
MARKET_OU_2_5 = "goals_total_o_u_2_5"
MARKET_BTTS = "btts"
MARKET_CORNERS_OU_9_5 = "corners_total_o_u_9_5"
MARKET_CORNERS_OU_10_5 = "corners_total_o_u_10_5"
MARKET_CORNERS_OU_11_5 = "corners_total_o_u_11_5"

CORNER_MARKETS: frozenset[str] = frozenset(
    {MARKET_CORNERS_OU_9_5, MARKET_CORNERS_OU_10_5, MARKET_CORNERS_OU_11_5}
)
MULTICLASS_MARKETS: frozenset[str] = frozenset({MARKET_1X2})


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------


class CalibrationReport(BaseModel):
    """One predictor × one market — every metric needed by the lock gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Identifiers
    predictor_name: str
    market: str
    class_names: tuple[str, ...] = Field(
        description=(
            "Ordered class labels matching the probability columns. For 1X2: "
            "('home','draw','away'); for binary markets: ('no','yes') or "
            "('under','over')."
        )
    )
    n_samples: int = Field(ge=0)
    n_bins: int = Field(default=DEFAULT_N_BINS, ge=2)

    # Calibration — classwise (per Walsh & Joshi)
    classwise_ece: float = Field(ge=0.0, le=1.0)
    per_class_ece: tuple[float, ...]

    # Bin-fill diagnostic — gate-eligible only when this passes
    bin_fill_pct: float = Field(ge=0.0, le=1.0)
    bin_fill_passes: bool

    # Proper scores
    brier: float = Field(ge=0.0)          # binary or multiclass per market
    log_loss_value: float = Field(ge=0.0)

    # Reproducibility
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    git_sha: str | None = None
    notes: str | None = None

    # ---- factories ------------------------------------------------------

    @classmethod
    def from_predictions(
        cls,
        *,
        predictor_name: str,
        market: str,
        probs: Sequence[Sequence[float]] | Sequence[float],
        outcomes: Sequence[int],
        class_names: Sequence[str],
        n_bins: int = DEFAULT_N_BINS,
        min_bin_fill: float = DEFAULT_MIN_BIN_FILL,
        git_sha: str | None = None,
        notes: str | None = None,
    ) -> CalibrationReport:
        """Compute every metric from raw probabilities + outcomes.

        Binary markets (Over/Under, BTTS, Corners O/U): pass `probs` as a
        sequence of floats representing P(class=1), `outcomes` in {0,1},
        and `class_names` of length 2 (the "yes" / "over" class is index 1).

        Multiclass markets (1X2): pass `probs` as a sequence of (p0,p1,p2)
        tuples summing to ~1, `outcomes` in {0,1,2}, and `class_names` of
        length 3.
        """
        if len(probs) != len(outcomes):
            raise ValueError(
                f"probs/outcomes length mismatch: {len(probs)} vs {len(outcomes)}"
            )

        is_multiclass = bool(probs) and isinstance(probs[0], (list, tuple))

        if is_multiclass:
            probs_matrix = [list(row) for row in probs]  # type: ignore[arg-type]
            if not all(len(row) == len(class_names) for row in probs_matrix):
                raise ValueError(
                    f"each probability row must have {len(class_names)} entries"
                )
            cw = classwise_ece(probs_matrix, outcomes, n_bins=n_bins)
            brier_val = multiclass_brier_score(probs_matrix, outcomes)
            ll = multiclass_log_loss(probs_matrix, outcomes)
            # Bin-fill on the most-loaded class — the strictest signal for 1X2.
            class_loadings = [
                sum(row[k] for row in probs_matrix) for k in range(len(class_names))
            ]
            heaviest = int(max(range(len(class_names)), key=lambda k: class_loadings[k]))
            bf = bin_fill_check(
                [row[heaviest] for row in probs_matrix],
                n_bins=n_bins,
                min_fill=min_bin_fill,
            )
        else:
            if len(class_names) != 2:
                raise ValueError(
                    f"binary market requires 2 class_names, got {len(class_names)}"
                )
            probs_binary = [float(p) for p in probs]  # type: ignore[arg-type]
            # Build virtual 2-column matrix to reuse classwise_ece.
            probs_matrix = [[1.0 - p, p] for p in probs_binary]
            cw = classwise_ece(probs_matrix, outcomes, n_bins=n_bins)
            brier_val = brier_score(probs_binary, outcomes)
            ll = log_loss(probs_binary, outcomes)
            bf = bin_fill_check(probs_binary, n_bins=n_bins, min_fill=min_bin_fill)

        return cls(
            predictor_name=predictor_name,
            market=market,
            class_names=tuple(class_names),
            n_samples=len(outcomes),
            n_bins=n_bins,
            classwise_ece=cw.classwise_ece,
            per_class_ece=cw.per_class_ece,
            bin_fill_pct=bf.fill_pct,
            bin_fill_passes=bf.passes,
            brier=brier_val,
            log_loss_value=ll,
            git_sha=git_sha,
            notes=notes,
        )

    # ---- gate evaluation ------------------------------------------------

    @property
    def is_multiclass(self) -> bool:
        return self.market in MULTICLASS_MARKETS

    @property
    def gate_brier_max(self) -> float:
        """Per-market Brier ceiling for the lock gate (per-class-averaged scale)."""
        if self.market in CORNER_MARKETS:
            return LOCK_GATE_BRIER_MAX_CORNERS
        return LOCK_GATE_BRIER_MAX_1X2

    @property
    def brier_for_gate(self) -> float:
        """Per-class-averaged Brier comparable to gate ceiling (range [0, 1]).

        `self.brier` from `multiclass_brier_score` is the sum over classes
        per match (range [0, K]). The lock-gate ceilings (0.21 / 0.20) are
        in per-class-averaged scale to match Walsh & Joshi 2024 and the
        Macrì-Demartino reference values (per-class-averaged Brier of
        0.499 La Liga corresponds to sum-form ~1.50). Binary Brier needs
        no normalization.
        """
        if self.is_multiclass and self.class_names:
            return self.brier / len(self.class_names)
        return self.brier

    def gate_status(self) -> str:
        """Return one of: 'pass', 'marginal', 'below-gate', 'unreliable-bins'.

        - 'pass'             — ECE ≤ 5% AND Brier ≤ market-ceiling AND bins reliable
        - 'marginal'         — within 0.5pp / 0.005 of either gate (operator decides)
        - 'below-gate'       — clearly fails one or both metric gates and bins are
                               reliable enough to trust the failure
        - 'unreliable-bins'  — bin-fill < 80% so ECE/Brier cannot be trusted EXCEPT
                               in the catastrophic regime (Brier ≥ 2× gate, where
                               the failure is so large that bin choice cannot
                               rescue it); in that case we still emit 'below-gate'
        """
        ece_ok = self.classwise_ece <= LOCK_GATE_ECE_MAX
        brier_ok = self.brier_for_gate <= self.gate_brier_max
        catastrophic = self.brier_for_gate >= 2.0 * self.gate_brier_max

        if not self.bin_fill_passes and not catastrophic:
            return "unreliable-bins"

        if ece_ok and brier_ok:
            return "pass"

        ece_marginal = self.classwise_ece <= LOCK_GATE_ECE_MAX + 0.005
        brier_marginal = self.brier_for_gate <= self.gate_brier_max + 0.005
        if ece_marginal and brier_marginal:
            return "marginal"

        return "below-gate"

    # ---- serialization --------------------------------------------------

    def to_json(self, path: Path | str) -> None:
        """Write report to JSON. Parent directory is created if missing."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path | str) -> Self:
        return cls.model_validate_json(Path(path).read_text())

    @model_validator(mode="after")
    def _validate_per_class_consistent(self) -> CalibrationReport:
        if len(self.per_class_ece) and len(self.per_class_ece) != len(self.class_names):
            raise ValueError(
                f"per_class_ece length ({len(self.per_class_ece)}) does not "
                f"match class_names length ({len(self.class_names)})"
            )
        return self
