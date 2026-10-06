"""Tests for lock_gate.py — Phase 5 aggregator over CalibrationReports.

Layer-1: synthetic CalibrationReports verify the aggregation rules
(per-predictor verdicts, worst-case status propagation, coverage gate,
escape hatches: structural-only / coverage-partial / no-reports).
"""

from __future__ import annotations

import pytest

from bip.evaluation.tournaments.backtest.calibration_report import (
    MARKET_1X2,
    MARKET_BTTS,
    MARKET_CORNERS_OU_10_5,
    MARKET_OU_2_5,
    CalibrationReport,
)
from bip.evaluation.tournaments.backtest.lock_gate import (
    DEFAULT_COVERAGE_THRESHOLD,
    LockDecision,
    PredictorGateVerdict,
    _worst,
    evaluate_lock,
)

# ── synthetic report builder ──────────────────────────────────────────────────


def _report(
    predictor: str,
    market: str,
    *,
    classwise_ece: float = 0.03,
    brier: float = 0.18,
    bin_fill_passes: bool = True,
    bin_fill_pct: float = 0.85,
    n_samples: int = 200,
) -> CalibrationReport:
    """Build a CalibrationReport with controlled metric values for status testing.

    The report's `gate_status()` is what the aggregator uses, so we feed
    inputs that produce specific statuses without going through
    `from_predictions` (which would require synthesizing actual prob/outcome
    streams). Class names match Phase 1 conventions.
    """
    is_multi = market == MARKET_1X2
    class_names = ("home", "draw", "away") if is_multi else ("no", "yes")
    per_class = (classwise_ece,) * len(class_names)
    # For 1X2, multiclass_brier_score returns sum-over-K; brier_for_gate divides
    # by K. Provide ``brier`` already in per-class-averaged scale (matches gate
    # ceilings 0.21/0.20) and the report computes brier_for_gate accordingly:
    # for multiclass, brier = brier_for_gate * K, so we set brier = brier * K.
    raw_brier = brier * len(class_names) if is_multi else brier
    return CalibrationReport(
        predictor_name=predictor,
        market=market,
        class_names=class_names,
        n_samples=n_samples,
        n_bins=20,
        classwise_ece=classwise_ece,
        per_class_ece=per_class,
        bin_fill_pct=bin_fill_pct,
        bin_fill_passes=bin_fill_passes,
        brier=raw_brier,
        log_loss_value=0.5,
    )


# ── _worst ────────────────────────────────────────────────────────────────────


class TestWorstStatus:
    def test_pass_only(self):
        assert _worst(["pass", "pass"]) == "pass"

    def test_marginal_dominates_pass(self):
        assert _worst(["pass", "marginal"]) == "marginal"

    def test_unreliable_dominates_marginal(self):
        assert _worst(["marginal", "unreliable-bins"]) == "unreliable-bins"

    def test_below_gate_dominates_all(self):
        assert _worst(["pass", "marginal", "unreliable-bins", "below-gate"]) == "below-gate"

    def test_empty_returns_pass(self):
        assert _worst([]) == "pass"

    def test_unknown_status_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _worst(["pass", "invalid-status"])


# ── happy path: all predictors pass ──────────────────────────────────────────


class TestAllPass:
    def test_single_predictor_all_markets_pass(self):
        reports = [
            _report("bivariate_poisson", MARKET_1X2, classwise_ece=0.02, brier=0.18),
            _report("bivariate_poisson", MARKET_BTTS, classwise_ece=0.02, brier=0.18),
            _report("bivariate_poisson", MARKET_OU_2_5, classwise_ece=0.02, brier=0.18),
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=("euro_2024",),
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "pass"
        assert decision.passes_lock
        assert decision.coverage_pct == 1.0
        assert len(decision.predictor_verdicts) == 1
        verdict = decision.predictor_verdicts[0]
        assert verdict.overall_status == "pass"
        assert verdict.n_markets == 3
        assert all(s == "pass" for s in verdict.market_statuses.values())

    def test_multiple_predictors_all_pass(self):
        reports = [
            _report("bivariate_poisson", MARKET_1X2),
            _report("independent_poisson", MARKET_1X2),
            _report("elo_logistic", MARKET_1X2),
            _report("corners_poisson", MARKET_CORNERS_OU_10_5, brier=0.16),
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "pass"
        assert len(decision.predictor_verdicts) == 4
        # Verdicts sorted alphabetically by predictor_name
        names = [v.predictor_name for v in decision.predictor_verdicts]
        assert names == sorted(names)


DEFAULT_HELD_OUT = ("copa_2024", "euro_2024")


# ── per-market mixed statuses (worst-case propagation) ───────────────────────


class TestMixedStatuses:
    def test_one_market_marginal_one_pass_yields_marginal(self):
        reports = [
            _report("bv", MARKET_1X2, classwise_ece=0.02),
            _report("bv", MARKET_BTTS, classwise_ece=0.052),  # marginal (≤ 0.055)
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "marginal"
        assert decision.passes_lock  # marginal still proceeds

    def test_one_market_below_gate_blocks_lock(self):
        reports = [
            _report("bv", MARKET_1X2, classwise_ece=0.02),
            _report("bv", MARKET_BTTS, classwise_ece=0.10),  # below gate
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "below-gate"
        assert not decision.passes_lock

    def test_unreliable_bins_blocks_when_not_catastrophic(self):
        reports = [
            _report(
                "bv", MARKET_1X2,
                classwise_ece=0.02,
                bin_fill_passes=False,
                bin_fill_pct=0.4,  # well below threshold
                brier=0.18,  # not catastrophic
            ),
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "unreliable-bins"
        assert not decision.passes_lock

    def test_catastrophic_brier_with_unreliable_bins_still_below_gate(self):
        """Per CalibrationReport: catastrophic Brier (≥ 2× ceiling) overrides bin-fill."""
        reports = [
            _report(
                "bv", MARKET_BTTS,
                classwise_ece=0.30,
                brier=0.45,  # 2.25× the 0.20 ceiling → catastrophic
                bin_fill_passes=False,
            ),
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "below-gate"


# ── per-predictor isolation (one bad predictor doesn't sink another) ─────────


class TestPerPredictorIsolation:
    def test_bad_predictor_does_not_clobber_good_predictor_verdict(self):
        reports = [
            _report("bv", MARKET_1X2, classwise_ece=0.02),
            _report("bv", MARKET_BTTS, classwise_ece=0.02),
            _report("ip", MARKET_1X2, classwise_ece=0.10),  # below gate
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        # Aggregate is below-gate (one predictor failed)
        assert decision.calibration_status == "below-gate"
        # But the GOOD predictor's verdict is preserved as 'pass'
        bv_verdict = next(v for v in decision.predictor_verdicts if v.predictor_name == "bv")
        ip_verdict = next(v for v in decision.predictor_verdicts if v.predictor_name == "ip")
        assert bv_verdict.overall_status == "pass"
        assert ip_verdict.overall_status == "below-gate"
        # bv would still be usable post-tournament
        assert bv_verdict.passes_lock()
        assert not ip_verdict.passes_lock()


# ── escape hatches ───────────────────────────────────────────────────────────


class TestEscapeHatches:
    def test_structural_only_short_circuits_everything(self):
        decision = evaluate_lock(
            reports=[],  # ignored
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=0,  # ignored
            n_fixtures_with_predictions=0,
            structural_only=True,
        )
        assert decision.calibration_status == "structural-only"
        assert decision.passes_lock  # structural-only proceeds
        assert decision.predictor_verdicts == ()
        assert any("Layer-1 ship" in note for note in decision.operator_overrides)

    def test_no_reports_when_empty_and_not_structural_only(self):
        decision = evaluate_lock(
            reports=[],
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
        )
        assert decision.calibration_status == "no-reports"
        assert not decision.passes_lock

    def test_coverage_below_threshold_yields_coverage_partial(self):
        reports = [_report("bv", MARKET_1X2, classwise_ece=0.02)]  # would pass otherwise
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=80,  # 80% < 90% threshold
        )
        assert decision.calibration_status == "coverage-partial"
        assert not decision.passes_lock
        assert decision.coverage_pct == pytest.approx(0.80)

    def test_coverage_at_threshold_does_not_trigger_partial(self):
        reports = [_report("bv", MARKET_1X2, classwise_ece=0.02)]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=90,  # exactly threshold
        )
        assert decision.calibration_status == "pass"

    def test_zero_total_fixtures_yields_zero_coverage_and_partial(self):
        reports = [_report("bv", MARKET_1X2)]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=0,
            n_fixtures_with_predictions=0,
        )
        assert decision.coverage_pct == 0.0
        assert decision.calibration_status == "coverage-partial"

    def test_custom_coverage_threshold_respected(self):
        """Operator may relax threshold for less critical audits."""
        reports = [_report("bv", MARKET_1X2, classwise_ece=0.02)]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=80,
            coverage_threshold=0.75,  # 80% > 75% → passes
        )
        assert decision.calibration_status == "pass"


# ── operator override (allow_below_gate) ─────────────────────────────────────


class TestAllowBelowGateOverride:
    def test_allow_below_gate_promotes_aggregate_to_marginal(self):
        reports = [
            _report("bv", MARKET_1X2, classwise_ece=0.02),
            _report("bv", MARKET_BTTS, classwise_ece=0.10),  # below gate
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
            allow_below_gate=True,
        )
        # Aggregate moves to marginal (proceeds), but the per-predictor verdict
        # still records below-gate so the lock JSON shows the breakdown.
        assert decision.calibration_status == "marginal"
        assert decision.passes_lock
        verdict = decision.predictor_verdicts[0]
        assert verdict.overall_status == "below-gate"  # honest individual record
        assert any("allow_below_gate" in note for note in decision.operator_overrides)

    def test_allow_below_gate_no_op_when_no_below_gate(self):
        """If aggregate would already pass, the override doesn't lie about it."""
        reports = [_report("bv", MARKET_1X2, classwise_ece=0.02)]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
            allow_below_gate=True,
        )
        assert decision.calibration_status == "pass"
        assert decision.operator_overrides == ()  # no override recorded


# ── persistence ──────────────────────────────────────────────────────────────


class TestLockDecisionRoundTrip:
    def test_to_json_from_json_round_trip(self, tmp_path):
        reports = [
            _report("bv", MARKET_1X2, classwise_ece=0.02),
            _report("bv", MARKET_BTTS, classwise_ece=0.04),
        ]
        decision = evaluate_lock(
            reports,
            held_out_tournaments=DEFAULT_HELD_OUT,
            n_fixtures_total=100,
            n_fixtures_with_predictions=100,
            git_sha="abc1234",
        )
        path = tmp_path / "lock_decision.json"
        decision.to_json(path)
        loaded = LockDecision.from_json(path)
        assert loaded.calibration_status == decision.calibration_status
        assert loaded.git_sha == "abc1234"
        assert loaded.held_out_tournaments == DEFAULT_HELD_OUT
        assert len(loaded.predictor_verdicts) == 1


# ── PredictorGateVerdict shape ──────────────────────────────────────────────


class TestPredictorGateVerdict:
    def test_passes_lock_at_pass(self):
        v = PredictorGateVerdict(
            predictor_name="bv",
            n_markets=2,
            market_statuses={MARKET_1X2: "pass", MARKET_BTTS: "pass"},
            market_brier_for_gate={MARKET_1X2: 0.18, MARKET_BTTS: 0.18},
            market_classwise_ece={MARKET_1X2: 0.02, MARKET_BTTS: 0.02},
            overall_status="pass",
        )
        assert v.passes_lock()

    def test_passes_lock_at_marginal(self):
        v = PredictorGateVerdict(
            predictor_name="bv",
            n_markets=1,
            market_statuses={MARKET_1X2: "marginal"},
            market_brier_for_gate={MARKET_1X2: 0.21},
            market_classwise_ece={MARKET_1X2: 0.052},
            overall_status="marginal",
        )
        assert v.passes_lock()

    def test_fails_lock_at_below_gate(self):
        v = PredictorGateVerdict(
            predictor_name="bv",
            n_markets=1,
            market_statuses={MARKET_1X2: "below-gate"},
            market_brier_for_gate={MARKET_1X2: 0.30},
            market_classwise_ece={MARKET_1X2: 0.10},
            overall_status="below-gate",
        )
        assert not v.passes_lock()

    def test_fails_lock_at_unreliable_bins(self):
        v = PredictorGateVerdict(
            predictor_name="bv",
            n_markets=1,
            market_statuses={MARKET_1X2: "unreliable-bins"},
            market_brier_for_gate={MARKET_1X2: 0.18},
            market_classwise_ece={MARKET_1X2: 0.02},
            overall_status="unreliable-bins",
        )
        assert not v.passes_lock()


# ── default coverage threshold matches operator expectation ──────────────────


def test_default_coverage_threshold_matches_spike_doc():
    """Spike §7: ≥90% of WC2026 group-stage matches must have predictions."""
    assert DEFAULT_COVERAGE_THRESHOLD == pytest.approx(0.90)
