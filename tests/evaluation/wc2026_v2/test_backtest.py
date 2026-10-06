"""Ola 5 — walk-forward backtest + bootstrap CI tests.

Layer-1 unit tests for the helpers: bootstrap_ci correctness,
_brier_1x2 / _brier_binary value boundaries, ECE on known distributions.

Layer-2: end-to-end walk-forward run on real corpus. Marked as slow but
must complete within ~5 s so it can stay in the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.tournaments.wc2026_v2.backtest import (
    _brier_1x2,
    _brier_binary,
    _ece_binary,
    bootstrap_ci,
)

# ─────────────────────────────────────────────────────────────────
# Brier — lock-convention (divide by K=3 for multiclass)
# ─────────────────────────────────────────────────────────────────


class TestBrier1x2:
    """1X2 Brier uses the lock convention: (1/K)·Σ_k (p_k − 1{y=k})^2 with K=3.

    Reference values (computed by hand):
    - perfect (1, 0, 0) with outcome=0 → 0
    - uniform (1/3, 1/3, 1/3) outcome=0 → (4/9 + 1/9 + 1/9) / 3 = 6/27 = 2/9 ≈ 0.2222
    - worst-case (0, 0, 1) outcome=0 → (1 + 0 + 1) / 3 = 2/3 ≈ 0.6667
    """

    def test_perfect_prediction_zero(self) -> None:
        assert _brier_1x2(1.0, 0.0, 0.0, 0) == pytest.approx(0.0, abs=1e-9)
        assert _brier_1x2(0.0, 1.0, 0.0, 1) == pytest.approx(0.0, abs=1e-9)
        assert _brier_1x2(0.0, 0.0, 1.0, 2) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_against_each_outcome(self) -> None:
        for outcome in (0, 1, 2):
            v = _brier_1x2(1.0 / 3, 1.0 / 3, 1.0 / 3, outcome)
            assert v == pytest.approx(2.0 / 9.0, abs=1e-9)

    def test_worst_case_when_certain_wrong(self) -> None:
        assert _brier_1x2(0.0, 0.0, 1.0, 0) == pytest.approx(2.0 / 3.0, abs=1e-9)


class TestBrierBinary:
    @pytest.mark.parametrize(
        ("p", "y", "expected"),
        [
            (0.0, 0, 0.0),
            (1.0, 1, 0.0),
            (0.5, 0, 0.25),
            (0.5, 1, 0.25),
            (0.8, 1, 0.04),
            (0.8, 0, 0.64),
        ],
    )
    def test_known_values(self, p: float, y: int, expected: float) -> None:
        assert _brier_binary(p, y) == pytest.approx(expected, abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# ECE
# ─────────────────────────────────────────────────────────────────


class TestECEBinary:
    def test_perfectly_calibrated_zero_ece(self) -> None:
        rng = np.random.default_rng(0)
        n = 5_000
        probs = rng.uniform(0.0, 1.0, n)
        outcomes = (rng.uniform(size=n) < probs).astype(int)
        # Perfectly calibrated random data should have ECE close to 0 with
        # enough samples per bin — allow small tolerance.
        ece = _ece_binary(probs, outcomes, n_bins=10)
        assert ece < 0.04

    def test_extremely_overconfident_high_ece(self) -> None:
        # All predicted 0.9, all actual 0.1. Single bin → ECE = |0.9 - 0.1| = 0.8.
        probs = np.full(100, 0.9)
        outcomes = (np.arange(100) < 10).astype(int)
        ece = _ece_binary(probs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.8, abs=1e-9)

    def test_empty_bin_skipped(self) -> None:
        # Only probabilities in 0.0–0.1 bucket. Should not crash.
        probs = np.full(50, 0.05)
        outcomes = np.zeros(50, dtype=int)
        ece = _ece_binary(probs, outcomes, n_bins=10)
        assert ece == pytest.approx(0.05, abs=1e-9)


# ─────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────


class TestBootstrap:
    def test_constant_vector_ci_collapses_to_point(self) -> None:
        v = np.full(100, 0.25)
        ci = bootstrap_ci(v, n_resamples=200)
        assert ci.point == pytest.approx(0.25, abs=1e-9)
        # Standard error 0; lower=upper=point.
        assert ci.lower == pytest.approx(0.25, abs=1e-9)
        assert ci.upper == pytest.approx(0.25, abs=1e-9)

    def test_ci_contains_point(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.normal(0.5, 0.1, 1_000)
        ci = bootstrap_ci(v, n_resamples=500, seed=0)
        assert ci.lower < ci.point < ci.upper

    def test_empty_returns_nan(self) -> None:
        ci = bootstrap_ci(np.array([]), n_resamples=200)
        assert np.isnan(ci.point)
        assert np.isnan(ci.lower)
        assert np.isnan(ci.upper)

    def test_seeded_reproducible(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.normal(0.5, 0.1, 200)
        ci1 = bootstrap_ci(v, n_resamples=500, seed=42)
        ci2 = bootstrap_ci(v, n_resamples=500, seed=42)
        assert ci1 == ci2


# ─────────────────────────────────────────────────────────────────
# Layer-2 — full walk-forward smoke
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestWalkForwardSmoke:
    """End-to-end backtest on the real corpus. Slow (~3s) but must run
    in default suite so we catch regressions on every commit."""

    def test_walk_forward_runs_clean(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.backtest import (
            run_walk_forward_backtest,
        )
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        result = run_walk_forward_backtest(
            train=split.train,
            calibration=split.calibration,
            heldout=split.heldout,
            n_bootstrap=200,  # smaller for test speed
            seed=42,
        )
        # We expect 199 evaluated matches across the 4 hold-outs.
        assert result.n_matches == 199
        assert len(result.per_tournament) == 4
        # Brier should be plausibly close to the lock baseline (0.2156)
        # — anywhere in [0.15, 0.30] is sane. If we get << 0.15 or >> 0.30
        # something is wrong.
        assert 0.15 < result.overall_brier_1x2.point < 0.30
        # Bootstrap CI must straddle the point.
        ci = result.overall_brier_1x2
        assert ci.lower < ci.point < ci.upper
        # ECE in a plausible range.
        assert 0.0 <= result.overall_ece_1x2 < 0.30

    def test_ablation_no_calibration_runs(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.backtest import (
            run_walk_forward_backtest,
        )
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        result = run_walk_forward_backtest(
            train=split.train,
            calibration=split.calibration,
            heldout=split.heldout,
            n_bootstrap=200,
            seed=42,
            use_beta_calibration=False,
        )
        assert result.n_matches == 199
        # No calibration → markets are uncalibrated raw probabilities;
        # still must satisfy Brier in the plausible range.
        assert 0.15 < result.overall_brier_1x2.point < 0.30
