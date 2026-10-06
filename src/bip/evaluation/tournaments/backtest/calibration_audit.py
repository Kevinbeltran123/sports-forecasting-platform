"""Per-class evaluation harness — accumulates predictions, emits reports.

The harness is the loop that turns a stream of (market, probabilities,
outcome) records into one `CalibrationReport` per market. Decoupled from
the tournament-evaluator's `MatchPrediction` dataclass so it can be reused
by:
  - synthetic-logits property tests (Layer 1)
  - the real WC2026 audit (Layer 2 — runs once historical fixtures load)
  - Phase 5 backtest (consumes the same harness, different inputs)
  - production CLV pipeline (post-WC graduation, if `LogisticLogitCalibrator`
    measurably improves baseline)

Usage::

    audit = CalibrationAudit(predictor_name="bivariate_poisson")
    for fixture in walk_forward_results:
        audit.record_1x2(
            p_home=fixture.dist.p_home_win,
            p_draw=fixture.dist.p_draw,
            p_away=fixture.dist.p_away_win,
            outcome=fixture.observed_1x2,
        )
        audit.record_binary(
            market=MARKET_OU_2_5,
            p_yes=fixture.dist.p_over_2_5,
            outcome=int(fixture.observed_total_goals > 2.5),
        )
    reports = audit.compute_reports()
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from bip.evaluation.tournaments.backtest.calibration_metrics import (
    DEFAULT_MIN_BIN_FILL,
    DEFAULT_N_BINS,
)
from bip.evaluation.tournaments.backtest.calibration_report import (
    MARKET_1X2,
    CalibrationReport,
)


@dataclass
class _BinaryRecord:
    """Mutable accumulator for one binary market."""

    class_names: tuple[str, str]
    probs_yes: list[float] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)


@dataclass
class _MulticlassRecord:
    """Mutable accumulator for one multiclass market."""

    class_names: tuple[str, ...]
    probs_matrix: list[list[float]] = field(default_factory=list)
    outcomes: list[int] = field(default_factory=list)


class CalibrationAudit:
    """Accumulator that builds `CalibrationReport`s per market for one predictor.

    Markets are identified by the canonical strings exported from
    `calibration_report` (MARKET_1X2, MARKET_OU_2_5, MARKET_BTTS,
    MARKET_CORNERS_OU_*). Add custom markets by passing the canonical name —
    `compute_reports` returns one report per market that received any data.

    Thread-safety: not safe for concurrent writes. Run audits sequentially
    or instantiate one audit per worker.
    """

    def __init__(
        self,
        predictor_name: str,
        *,
        n_bins: int = DEFAULT_N_BINS,
        min_bin_fill: float = DEFAULT_MIN_BIN_FILL,
        git_sha: str | None = None,
    ) -> None:
        self.predictor_name = predictor_name
        self.n_bins = n_bins
        self.min_bin_fill = min_bin_fill
        self.git_sha = git_sha
        self._binary: dict[str, _BinaryRecord] = {}
        self._multiclass: dict[str, _MulticlassRecord] = {}

    # ----- recording -----------------------------------------------------

    def record_1x2(
        self,
        *,
        p_home: float,
        p_draw: float,
        p_away: float,
        outcome: int,
        class_names: tuple[str, str, str] = ("home", "draw", "away"),
    ) -> None:
        """Record one 1X2 prediction.

        outcome must be 0 (home win), 1 (draw), or 2 (away win).
        """
        if outcome not in (0, 1, 2):
            raise ValueError(f"1X2 outcome must be 0/1/2, got {outcome}")
        rec = self._multiclass.setdefault(
            MARKET_1X2, _MulticlassRecord(class_names=class_names)
        )
        rec.probs_matrix.append([p_home, p_draw, p_away])
        rec.outcomes.append(outcome)

    def record_binary(
        self,
        market: str,
        *,
        p_yes: float,
        outcome: int,
        class_names: tuple[str, str] = ("no", "yes"),
    ) -> None:
        """Record one binary prediction (Over/Under, BTTS, Corners O/U, etc.).

        outcome must be 0 (no/under) or 1 (yes/over).
        """
        if outcome not in (0, 1):
            raise ValueError(f"binary outcome must be 0/1, got {outcome}")
        rec = self._binary.setdefault(market, _BinaryRecord(class_names=class_names))
        rec.probs_yes.append(p_yes)
        rec.outcomes.append(outcome)

    def record_multiclass(
        self,
        market: str,
        *,
        probs: Sequence[float],
        outcome: int,
        class_names: tuple[str, ...],
    ) -> None:
        """Record one multiclass prediction for an arbitrary market.

        Useful for double-chance, half-time/full-time, or any other K-class
        market that isn't 1X2. `class_names` defines K and probability order.
        """
        if outcome < 0 or outcome >= len(class_names):
            raise ValueError(
                f"outcome {outcome} out of range for {len(class_names)} classes"
            )
        if len(probs) != len(class_names):
            raise ValueError(
                f"probs length {len(probs)} does not match "
                f"class_names length {len(class_names)}"
            )
        rec = self._multiclass.setdefault(
            market, _MulticlassRecord(class_names=class_names)
        )
        rec.probs_matrix.append(list(probs))
        rec.outcomes.append(outcome)

    # ----- queries -------------------------------------------------------

    @property
    def markets(self) -> tuple[str, ...]:
        """All markets that have received at least one record."""
        return tuple(sorted(set(self._binary) | set(self._multiclass)))

    def n_recorded(self, market: str) -> int:
        if market in self._multiclass:
            return len(self._multiclass[market].outcomes)
        if market in self._binary:
            return len(self._binary[market].outcomes)
        return 0

    # ----- emit ----------------------------------------------------------

    def compute_reports(self) -> list[CalibrationReport]:
        """Build one CalibrationReport per market with accumulated records.

        Empty markets are skipped (returning a report with n_samples=0 would
        produce uninformative gate verdicts).
        """
        reports: list[CalibrationReport] = []

        for market, mc in self._multiclass.items():
            if not mc.outcomes:
                continue
            reports.append(
                CalibrationReport.from_predictions(
                    predictor_name=self.predictor_name,
                    market=market,
                    probs=mc.probs_matrix,
                    outcomes=mc.outcomes,
                    class_names=mc.class_names,
                    n_bins=self.n_bins,
                    min_bin_fill=self.min_bin_fill,
                    git_sha=self.git_sha,
                )
            )

        for market, br in self._binary.items():
            if not br.outcomes:
                continue
            reports.append(
                CalibrationReport.from_predictions(
                    predictor_name=self.predictor_name,
                    market=market,
                    probs=br.probs_yes,
                    outcomes=br.outcomes,
                    class_names=br.class_names,
                    n_bins=self.n_bins,
                    min_bin_fill=self.min_bin_fill,
                    git_sha=self.git_sha,
                )
            )

        return sorted(reports, key=lambda r: r.market)
