"""Bootstrap-CI gate tests — Sprint 1 Ola B.

Synthetic-data validation that the gate correctly distinguishes:
  - perfect predictor → small Brier CI, PASS
  - random predictor → large Brier CI, FAIL
  - profitable picks → ROI CI above 0, PASS-eligible
  - unprofitable picks → ROI CI below 0, FAIL/SHADOW boundary
"""

from __future__ import annotations

import numpy as np
import pytest

from bip.models.ligas.gate import (
    DEFAULT_POLICY,
    GatePolicy,
    GateVerdict,
    bootstrap_brier_ci,
    bootstrap_roi_ci,
    brier_score_multiclass,
    evaluate_gate,
)


# ──────────────────────────────────────────────────────────────────────
# Brier score core
# ──────────────────────────────────────────────────────────────────────


class TestBrierScore:
    def test_perfect_predictor_zero(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_proba = np.zeros((5, 3))
        y_proba[np.arange(5), y_true] = 1.0
        assert brier_score_multiclass(y_true, y_proba) == 0.0

    def test_uniform_predictor_two_thirds(self):
        # Multiclass Brier for uniform 1/3 over 3 classes on any label:
        # sum (p - onehot)^2 = (1/3-1)^2 + (1/3)^2 + (1/3)^2
        #                    = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        y_true = np.array([0, 1, 2])
        y_proba = np.full((3, 3), 1 / 3)
        assert abs(brier_score_multiclass(y_true, y_proba) - 2 / 3) < 1e-10

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score_multiclass(np.array([0, 1]), np.zeros((3, 3)))


# ──────────────────────────────────────────────────────────────────────
# Brier CI
# ──────────────────────────────────────────────────────────────────────


class TestBootstrapBrierCI:
    def test_perfect_predictor_ci_tight_around_zero(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=200)
        y_proba = np.zeros((200, 3))
        y_proba[np.arange(200), y_true] = 1.0
        lo, point, hi = bootstrap_brier_ci(y_true, y_proba, n_bootstrap=500, seed=0)
        assert point == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_random_predictor_ci_around_two_thirds(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=500)
        y_proba = np.full((500, 3), 1 / 3)
        lo, point, hi = bootstrap_brier_ci(y_true, y_proba, n_bootstrap=500, seed=0)
        assert 0.6 < lo
        assert hi < 0.75
        assert abs(point - 2 / 3) < 1e-6

    def test_empty_returns_nan(self):
        lo, point, hi = bootstrap_brier_ci(
            np.array([], dtype=int), np.zeros((0, 3)), n_bootstrap=10
        )
        assert np.isnan(lo) and np.isnan(point) and np.isnan(hi)

    def test_ci_lower_bound_le_point_le_upper(self):
        rng = np.random.default_rng(1)
        n = 100
        y_true = rng.integers(0, 3, size=n)
        y_proba = rng.dirichlet(np.ones(3), size=n)
        lo, point, hi = bootstrap_brier_ci(y_true, y_proba, n_bootstrap=300, seed=0)
        assert lo <= point <= hi


# ──────────────────────────────────────────────────────────────────────
# ROI CI
# ──────────────────────────────────────────────────────────────────────


class TestBootstrapROICI:
    def test_profitable_constant_picks(self):
        # 100 picks of +0.10 each → ROI = +0.10 with tight CI
        picks = [0.10] * 100
        lo, point, hi = bootstrap_roi_ci(picks, n_bootstrap=500, seed=0)
        assert abs(point - 0.10) < 1e-9
        # Zero variance → CI bounds equal the point
        assert abs(lo - 0.10) < 1e-9
        assert abs(hi - 0.10) < 1e-9

    def test_loss_picks(self):
        picks = [-1.0] * 50 + [0.95] * 30  # 50 losses, 30 wins at 1.95
        lo, point, hi = bootstrap_roi_ci(picks, n_bootstrap=500, seed=0)
        # Point: (50*-1 + 30*0.95) / 80 = -21.5/80 = -0.26875
        assert abs(point - (-0.26875)) < 1e-9
        # CI upper should be below 0 — strongly unprofitable
        assert hi < 0

    def test_empty_returns_nan(self):
        lo, point, hi = bootstrap_roi_ci([], n_bootstrap=10)
        assert np.isnan(lo) and np.isnan(point) and np.isnan(hi)

    def test_ci_ordering(self):
        rng = np.random.default_rng(2)
        picks = rng.normal(loc=0.02, scale=0.5, size=200)
        lo, point, hi = bootstrap_roi_ci(picks, n_bootstrap=500, seed=0)
        assert lo <= point <= hi


# ──────────────────────────────────────────────────────────────────────
# Gate evaluation
# ──────────────────────────────────────────────────────────────────────


class TestEvaluateGate:
    def test_pass_when_brier_ok_and_roi_lower_above_threshold(self):
        d = evaluate_gate(
            brier_ci=(0.18, 0.20, 0.22),
            roi_ci=(0.04, 0.06, 0.08),
            n_picks=200,
        )
        assert d.verdict == GateVerdict.PASS

    def test_shadow_when_brier_ok_but_roi_lower_below_pass_threshold(self):
        d = evaluate_gate(
            brier_ci=(0.18, 0.20, 0.22),
            roi_ci=(0.0, 0.02, 0.04),
            n_picks=200,
        )
        # roi_lo=0 is below roi_lower_for_pass=0.03 but above
        # roi_lower_for_shadow=-0.005, so SHADOW
        assert d.verdict == GateVerdict.SHADOW

    def test_fail_when_roi_lower_significantly_negative(self):
        d = evaluate_gate(
            brier_ci=(0.18, 0.20, 0.22),
            roi_ci=(-0.10, -0.05, -0.01),
            n_picks=200,
        )
        assert d.verdict == GateVerdict.FAIL

    def test_fail_when_brier_too_high(self):
        d = evaluate_gate(
            brier_ci=(0.28, 0.30, 0.32),
            roi_ci=(0.05, 0.07, 0.10),
            n_picks=200,
        )
        # roi looks great but calibration is poor → can't PASS; SHADOW
        # is still allowed since roi_lower > shadow threshold
        assert d.verdict == GateVerdict.SHADOW

    def test_shadow_when_no_picks_but_brier_ok(self):
        d = evaluate_gate(
            brier_ci=(0.18, 0.20, 0.22),
            roi_ci=None,
            n_picks=0,
        )
        assert d.verdict == GateVerdict.SHADOW

    def test_fail_when_no_picks_and_poor_calibration(self):
        d = evaluate_gate(
            brier_ci=(0.28, 0.30, 0.32),
            roi_ci=None,
            n_picks=0,
        )
        assert d.verdict == GateVerdict.FAIL

    def test_fail_when_n_picks_below_shadow_minimum(self):
        d = evaluate_gate(
            brier_ci=(0.18, 0.20, 0.22),
            roi_ci=(0.0, 0.02, 0.04),
            n_picks=10,  # below min_picks_for_shadow=30
        )
        assert d.verdict == GateVerdict.FAIL

    def test_custom_policy_can_relax_thresholds(self):
        # Relaxed policy that PASSes anything not significantly negative
        relaxed = GatePolicy(
            brier_upper_for_pass=0.30,
            roi_lower_for_pass=-0.01,
            roi_lower_for_shadow=-0.05,
            min_picks_for_pass=10,
            min_picks_for_shadow=5,
        )
        d = evaluate_gate(
            brier_ci=(0.20, 0.22, 0.25),
            roi_ci=(0.0, 0.02, 0.05),
            n_picks=15,
            policy=relaxed,
        )
        assert d.verdict == GateVerdict.PASS

    def test_decision_records_brier_and_roi_cis(self):
        brier = (0.18, 0.20, 0.22)
        roi = (0.04, 0.06, 0.08)
        d = evaluate_gate(brier_ci=brier, roi_ci=roi, n_picks=200)
        assert d.brier_ci == brier
        assert d.roi_ci == roi
        assert d.n_picks == 200
        assert len(d.reasons) >= 1


class TestDefaultPolicy:
    def test_default_thresholds_match_plan(self):
        p = DEFAULT_POLICY
        assert p.brier_upper_for_pass == 0.25
        assert p.roi_lower_for_pass == 0.03
        assert p.roi_lower_for_shadow == -0.005
        assert p.min_picks_for_pass == 100
        assert p.min_picks_for_shadow == 30
