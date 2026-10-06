"""Phase 1 Layer 1 — calibration baseline audit (structural).

Tests the Walsh & Joshi 2024 classwise calibration machinery against
synthetic logits where the correct answer is known by construction:

  - perfectly calibrated probabilities → ECE ≈ 0
  - max miscalibration → ECE ≈ 1
  - sparse-bin scenarios → bin_fill_passes=False
  - the marginal-vs-classwise distinction (the whole point of using classwise)
  - lock-gate verdicts ('pass' / 'marginal' / 'below-gate' / 'unreliable-bins')

Layer 2 tests at the bottom (xfail with `# requires-real-data`) cover the
audit run on real BivariatePoisson / CornersPoisson outputs once the
historical fixture set is loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bip.evaluation.tournaments.backtest.calibration_audit import CalibrationAudit
from bip.evaluation.tournaments.backtest.calibration_metrics import (
    bin_fill_check,
    classwise_ece,
    per_class_reliability,
)
from bip.evaluation.tournaments.backtest.calibration_report import (
    LOCK_GATE_BRIER_MAX_1X2,
    LOCK_GATE_BRIER_MAX_CORNERS,
    LOCK_GATE_ECE_MAX,
    MARKET_1X2,
    MARKET_BTTS,
    MARKET_CORNERS_OU_11_5,
    MARKET_OU_2_5,
    CalibrationReport,
)

# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------

def _perfectly_calibrated_1x2(
    n: int = 1000, *, seed: int = 42
) -> tuple[list[list[float]], list[int]]:
    """Generate realistic football-shaped 1X2 probs (one favorite per match)
    with outcomes sampled from those probs.

    Concentrating around (0.55-0.70, 0.18-0.25, 0.12-0.20) reflects how real
    match-level predictors look — without that concentration, expected Brier
    sits well above the lock gate even for perfect calibration. The whole
    point of the lock gate is that informed predictors (which produce
    concentrated probs) can clear it; uniform-Dirichlet noise cannot.
    """
    rng = np.random.default_rng(seed)
    probs: list[list[float]] = []
    outcomes: list[int] = []
    for _ in range(n):
        p_fav = rng.uniform(0.50, 0.70)
        p_draw = rng.uniform(0.18, 0.28)
        p_other = 1.0 - p_fav - p_draw
        # Random which class is favorite; class 1 (draw) is never the favorite
        # in this synthetic setup, matching the empirical fact that draws are
        # rarely the modal outcome.
        fav_class = int(rng.choice([0, 2]))
        row = [0.0, p_draw, 0.0]
        row[fav_class] = p_fav
        row[2 - fav_class] = p_other
        probs.append(row)
        outcomes.append(int(rng.choice(3, p=row)))
    return probs, outcomes


def _perfectly_calibrated_binary(
    n: int = 1000, *, seed: int = 7
) -> tuple[list[float], list[int]]:
    """P(y=1 | p) = p by construction."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.05, 0.95, size=n)
    outcomes = [int(rng.uniform() < p) for p in probs]
    return probs.tolist(), outcomes


def _max_miscalibrated_binary(n: int = 200) -> tuple[list[float], list[int]]:
    """Always says p=0.99 but the outcome is always 0."""
    return [0.99] * n, [0] * n


# ---------------------------------------------------------------------------
# bin_fill_check
# ---------------------------------------------------------------------------

class TestBinFillCheck:
    def test_uniform_full_fill(self) -> None:
        """20 evenly distributed probs → 100% fill, passes."""
        probs = list(np.linspace(0.0, 1.0, 200))
        result = bin_fill_check(probs, n_bins=20, min_fill=0.8)
        assert result.fill_pct == 1.0
        assert result.passes is True

    def test_concentrated_probs_fail(self) -> None:
        """All probs in [0.40, 0.45] → only 1 bin filled, fails."""
        probs = [0.42] * 100
        result = bin_fill_check(probs, n_bins=20, min_fill=0.8)
        assert result.n_filled == 1
        assert result.fill_pct == 0.05
        assert result.passes is False

    def test_empty_probs_fail(self) -> None:
        result = bin_fill_check([], n_bins=20)
        assert result.fill_pct == 0.0
        assert result.passes is False

    def test_min_fill_customizable(self) -> None:
        """Halfway-filled passes a relaxed 0.4 threshold but fails 0.8."""
        probs = list(np.linspace(0.0, 0.5, 100))
        relaxed = bin_fill_check(probs, n_bins=20, min_fill=0.4)
        strict = bin_fill_check(probs, n_bins=20, min_fill=0.8)
        assert relaxed.passes is True
        assert strict.passes is False

    def test_zero_n_bins_raises(self) -> None:
        with pytest.raises(ValueError):
            bin_fill_check([0.5], n_bins=0)


# ---------------------------------------------------------------------------
# classwise_ece
# ---------------------------------------------------------------------------

class TestClasswiseEce:
    def test_perfect_calibration_low_ece(self) -> None:
        probs, outs = _perfectly_calibrated_1x2(n=2000)
        result = classwise_ece(probs, outs, n_bins=10)
        # With n=2000 and proper calibration, ECE should sit well under 5%.
        assert result.classwise_ece < 0.05
        assert result.n_classes == 3
        assert len(result.per_class_ece) == 3

    def test_fully_overconfident_high_ece(self) -> None:
        """Always P(home)=0.99 but actual is always away wins → ECE ~ 0.99 for class 0."""
        n = 100
        probs = [[0.99, 0.005, 0.005]] * n
        outs = [2] * n  # always away wins
        result = classwise_ece(probs, outs, n_bins=10)
        # Class 0 (home): predicted 0.99, actual rate 0 → ECE for class 0 ≈ 0.99
        assert result.per_class_ece[0] > 0.95
        # Class 2 (away): predicted 0.005, actual 1.0 → ECE for class 2 ≈ 0.995
        assert result.per_class_ece[2] > 0.95
        assert result.classwise_ece > 0.6

    def test_marginal_vs_classwise_distinction(self) -> None:
        """Construct a case where marginal-ECE looks fine but classwise reveals miscalibration.

        Predict (home=0.34, draw=0.33, away=0.33) for every match. Real
        outcome is *always* draw. Marginal ECE on the heaviest class (home)
        is small because home is rarely the modal pick. Classwise ECE on
        the draw class is much larger because the model said 0.33 when the
        true rate is 1.0.
        """
        n = 100
        probs = [[0.34, 0.33, 0.33]] * n
        outs = [1] * n  # always draw
        result = classwise_ece(probs, outs, n_bins=10)
        # Draw class miscalibration is the headline.
        assert result.per_class_ece[1] > 0.5
        # Average is dragged down by classes 0/2 but still substantial.
        assert result.classwise_ece > 0.3

    def test_outcome_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            classwise_ece([[0.5, 0.5]], [3], n_bins=10)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            classwise_ece([[0.5, 0.5]] * 3, [0, 1], n_bins=10)

    def test_empty_input(self) -> None:
        result = classwise_ece([], [], n_bins=10)
        assert result.classwise_ece == 0.0
        assert result.per_class_ece == ()


# ---------------------------------------------------------------------------
# per_class_reliability
# ---------------------------------------------------------------------------

class TestPerClassReliability:
    def test_returns_one_curve_per_class(self) -> None:
        probs, outs = _perfectly_calibrated_1x2(n=300)
        curves = per_class_reliability(probs, outs, n_bins=10)
        assert len(curves) == 3

    def test_curves_have_matching_arrays(self) -> None:
        probs, outs = _perfectly_calibrated_1x2(n=300)
        curves = per_class_reliability(probs, outs, n_bins=10)
        for c in curves:
            assert len(c.bucket_mean_pred) == len(c.bucket_freq_actual)
            assert len(c.bucket_mean_pred) == len(c.bucket_count)

    def test_empty_input(self) -> None:
        curves = per_class_reliability([], [], n_bins=10)
        assert curves == ()


# ---------------------------------------------------------------------------
# CalibrationReport — factory + gate verdicts
# ---------------------------------------------------------------------------

class TestCalibrationReport1X2:
    def test_from_predictions_perfect_passes_gate(self) -> None:
        """Realistic football-shaped probs + perfect calibration → gate pass.

        Note on min_bin_fill: real football probs concentrate around the
        favorite line (~0.50–0.70), so 20-bin uniform coverage cannot reach
        Walsh & Joshi's 80% default. We relax to 50% here — that's the open
        question §9 #3 in SPIKE-wc2026-calibration-lock.md flags for the
        Phase 5 backtest run. The framework still surfaces the bin coverage
        in `bin_fill_pct` so the operator can audit it independently.
        """
        probs, outs = _perfectly_calibrated_1x2(n=2000)
        report = CalibrationReport.from_predictions(
            predictor_name="synthetic",
            market=MARKET_1X2,
            probs=probs,
            outcomes=outs,
            class_names=("home", "draw", "away"),
            n_bins=10,
            min_bin_fill=0.5,
        )
        assert report.is_multiclass is True
        assert report.n_samples == 2000
        assert len(report.per_class_ece) == 3
        assert report.classwise_ece < LOCK_GATE_ECE_MAX
        # brier (sum form) divided by K classes should clear the per-class gate.
        assert report.brier_for_gate < LOCK_GATE_BRIER_MAX_1X2
        assert report.bin_fill_passes is True
        assert report.gate_status() == "pass"

    def test_from_predictions_overconfident_fails_gate(self) -> None:
        """Catastrophic miscalibration → 'below-gate' even with sparse bins.

        gate_status guards against bin-fill False-Negatives in the catastrophic
        regime (Brier ≥ 2× ceiling) — a model that's THIS wrong fails the gate
        regardless of bin coverage, because no realistic bin layout can rescue
        a sum-Brier ≈ 1.98 (per-class avg ≈ 0.66) from the 0.21 ceiling.
        """
        n = 200
        probs = [[0.99, 0.005, 0.005]] * n
        outs = [2] * n
        report = CalibrationReport.from_predictions(
            predictor_name="overconfident",
            market=MARKET_1X2,
            probs=probs,
            outcomes=outs,
            class_names=("home", "draw", "away"),
            n_bins=10,
        )
        assert report.gate_status() == "below-gate"

    def test_sparse_bins_yield_unreliable_bins_status(self) -> None:
        """All probs identical → only 1 bin filled even on the heaviest class."""
        n = 100
        probs = [[0.50, 0.25, 0.25]] * n
        # Outcomes match probs roughly so ECE is small in absolute terms.
        outs = [0] * 50 + [1] * 25 + [2] * 25
        report = CalibrationReport.from_predictions(
            predictor_name="degenerate",
            market=MARKET_1X2,
            probs=probs,
            outcomes=outs,
            class_names=("home", "draw", "away"),
            n_bins=20,
        )
        assert report.bin_fill_passes is False
        assert report.gate_status() == "unreliable-bins"


class TestCalibrationReportBinary:
    def test_binary_market_perfect_calibration(self) -> None:
        probs, outs = _perfectly_calibrated_binary(n=1000)
        report = CalibrationReport.from_predictions(
            predictor_name="synthetic",
            market=MARKET_OU_2_5,
            probs=probs,
            outcomes=outs,
            class_names=("under", "over"),
            n_bins=10,
        )
        assert report.is_multiclass is False
        assert report.classwise_ece < 0.05
        assert report.gate_status() == "pass"

    def test_binary_market_max_miscalibration(self) -> None:
        """Always p=0.99, always y=0 → Brier ≈ 0.98, far past the 0.21 gate.

        The catastrophic-regime escape hatch in gate_status() means a model
        this wrong is flagged as 'below-gate' even though all probs land in a
        single bin and bin_fill_passes=False — because no bin layout can
        produce a Brier ≤ 0.21 from this data.
        """
        probs, outs = _max_miscalibrated_binary(n=200)
        report = CalibrationReport.from_predictions(
            predictor_name="overconfident",
            market=MARKET_BTTS,
            probs=probs,
            outcomes=outs,
            class_names=("no", "yes"),
            n_bins=10,
        )
        assert report.gate_status() == "below-gate"
        # Brier ≈ (0.99 - 0)^2 ≈ 0.98 — well above the 0.21 ceiling.
        assert report.brier > LOCK_GATE_BRIER_MAX_1X2

    def test_corners_market_uses_corner_brier_ceiling(self) -> None:
        """Corners O/U should use the 0.20 ceiling, not the 0.21 1X2 ceiling."""
        # Construct a Brier in (0.20, 0.21] — passes 1X2 gate but fails corners.
        probs = [0.6] * 100
        # Y=0 50% of the time → Brier ≈ 0.5*0.36 + 0.5*0.16 = 0.26 (well past)
        # Easier: target Brier ≈ 0.205 via mixed outcomes.
        # Let's just check the property: a corners report's gate_brier_max == 0.20.
        report = CalibrationReport.from_predictions(
            predictor_name="x",
            market=MARKET_CORNERS_OU_11_5,
            probs=probs,
            outcomes=[1] * 60 + [0] * 40,
            class_names=("under", "over"),
            n_bins=10,
        )
        assert report.gate_brier_max == LOCK_GATE_BRIER_MAX_CORNERS


class TestCalibrationReportSerialization:
    def test_json_round_trip(self, tmp_path: Path) -> None:
        probs, outs = _perfectly_calibrated_1x2(n=500)
        original = CalibrationReport.from_predictions(
            predictor_name="round_trip",
            market=MARKET_1X2,
            probs=probs,
            outcomes=outs,
            class_names=("home", "draw", "away"),
            n_bins=10,
            git_sha="abc123",
            notes="round-trip test",
        )
        path = tmp_path / "report.json"
        original.to_json(path)
        loaded = CalibrationReport.from_json(path)
        assert loaded.predictor_name == "round_trip"
        assert loaded.market == MARKET_1X2
        assert loaded.classwise_ece == pytest.approx(original.classwise_ece, abs=1e-6)
        assert loaded.brier == pytest.approx(original.brier, abs=1e-6)
        assert loaded.git_sha == "abc123"
        assert loaded.notes == "round-trip test"


class TestCalibrationReportValidation:
    def test_per_class_length_must_match_class_names(self) -> None:
        with pytest.raises(ValueError, match="per_class_ece length"):
            CalibrationReport(
                predictor_name="x",
                market=MARKET_1X2,
                class_names=("home", "draw", "away"),
                n_samples=10,
                n_bins=20,
                classwise_ece=0.01,
                per_class_ece=(0.01, 0.01),  # only 2 entries for 3 classes
                bin_fill_pct=0.9,
                bin_fill_passes=True,
                brier=0.1,
                log_loss_value=1.0,
            )


# ---------------------------------------------------------------------------
# CalibrationAudit harness
# ---------------------------------------------------------------------------

class TestCalibrationAudit:
    def test_records_1x2_and_binary_in_one_audit(self) -> None:
        audit = CalibrationAudit(predictor_name="bivariate_poisson", n_bins=10)
        for _ in range(50):
            audit.record_1x2(p_home=0.45, p_draw=0.27, p_away=0.28, outcome=0)
            audit.record_binary(MARKET_OU_2_5, p_yes=0.55, outcome=1)
            audit.record_binary(MARKET_BTTS, p_yes=0.50, outcome=0)

        assert audit.n_recorded(MARKET_1X2) == 50
        assert audit.n_recorded(MARKET_OU_2_5) == 50
        assert audit.n_recorded(MARKET_BTTS) == 50

        reports = audit.compute_reports()
        markets = {r.market for r in reports}
        assert markets == {MARKET_1X2, MARKET_OU_2_5, MARKET_BTTS}
        for r in reports:
            assert r.predictor_name == "bivariate_poisson"
            assert r.n_samples == 50

    def test_invalid_outcomes_raise(self) -> None:
        audit = CalibrationAudit("p")
        with pytest.raises(ValueError):
            audit.record_1x2(p_home=0.5, p_draw=0.3, p_away=0.2, outcome=3)
        with pytest.raises(ValueError):
            audit.record_binary("market", p_yes=0.5, outcome=2)

    def test_record_multiclass_arbitrary_k(self) -> None:
        audit = CalibrationAudit("multi")
        # Double-chance: 3 classes (1X, 12, X2)
        for _ in range(30):
            audit.record_multiclass(
                "double_chance",
                probs=(0.45, 0.30, 0.25),
                outcome=0,
                class_names=("1X", "12", "X2"),
            )
        reports = audit.compute_reports()
        assert len(reports) == 1
        assert reports[0].class_names == ("1X", "12", "X2")
        assert reports[0].is_multiclass is False  # not in MULTICLASS_MARKETS set

    def test_empty_audit_returns_empty_reports(self) -> None:
        audit = CalibrationAudit("empty")
        assert audit.compute_reports() == []

    def test_markets_property_sorted_and_unique(self) -> None:
        audit = CalibrationAudit("p")
        audit.record_binary(MARKET_OU_2_5, p_yes=0.5, outcome=1)
        audit.record_1x2(p_home=0.4, p_draw=0.3, p_away=0.3, outcome=0)
        audit.record_binary(MARKET_BTTS, p_yes=0.5, outcome=1)
        markets = audit.markets
        assert markets == tuple(sorted([MARKET_1X2, MARKET_OU_2_5, MARKET_BTTS]))


# ---------------------------------------------------------------------------
# Layer 2 — real-predictor audit (queued: requires-real-data)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="requires-real-data: walk-forward backtest output not yet generated"
)
def test_layer2_bivariate_poisson_passes_lock_gate() -> None:
    """Run BivariatePoisson on historical international tournaments and assert
    the 1X2 report passes the lock gate (classwise-ECE ≤ 5%, Brier ≤ 0.21).

    Unblocks when:
      - WC 2018 / Euro 2024 / Copa 2024 fixture+result data is loaded
      - `scripts/backtest_int_tournaments.py` exists (Phase 5)
    """
    raise NotImplementedError("queued — see SPIKE Phase 5")


@pytest.mark.xfail(
    reason="requires-real-data: corners walk-forward backtest output not yet generated"
)
def test_layer2_corners_poisson_passes_lock_gate() -> None:
    """Run CornersPoissonModel on historical corner data; assert corners O/U
    classwise-ECE ≤ 5% AND Brier ≤ 0.20.

    Unblocks when:
      - API-Football corner history is loaded for top-5 EU last 5 seasons
      - The corners-specific walk-forward harness is wired up
    """
    raise NotImplementedError("queued — see SYNTHESIS.md Conclusion 2")
