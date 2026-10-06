"""Ola 2 — BetaCalibrator tests.

Boundary-case parametrized :
- identity-ish probabilities → fit recovers near (1, 1, 0)
- known asymmetric miscalibration → ECE drops post-fit
- monotonic shape preserved on uniform probability grid
- degenerate / small samples fall back to identity
- serialize/deserialize round-trip
"""

from __future__ import annotations

import numpy as np
import pytest

from bip.evaluation.tournaments.wc2026_v2.beta_calibrator import BetaCalibrator


def _expected_calibration_error(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width binned ECE on a 1-D probability vector."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:  # close the right edge
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        if not mask.any():
            continue
        conf = float(probs[mask].mean())
        acc = float(outcomes[mask].mean())
        ece += abs(conf - acc) * mask.sum() / n
    return ece


class TestIdentityRegime:
    """When raw probs are already well-calibrated, beta should learn ≈ (1, 1, 0)."""

    def test_well_calibrated_data_recovers_identity(self) -> None:
        rng = np.random.default_rng(0)
        n = 5_000
        p = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(size=n) < p).astype(int)
        cal = BetaCalibrator().fit(p, y)
        dump = cal.to_dict()
        # a, b should each be in a moderate range around 1; c should be small.
        assert 0.5 < dump["a"][1] < 1.5
        assert 0.5 < dump["b"][1] < 1.5
        assert abs(dump["c"][1]) < 0.5

    def test_identity_transform_preserves_probabilities(self) -> None:
        """If we feed the identity params directly via from_dict, the output
        must equal the input probabilities exactly."""
        cal = BetaCalibrator.from_dict(
            {"kind": "beta", "n_classes": 2, "a": [1.0, 1.0], "b": [1.0, 1.0], "c": [0.0, 0.0]}
        )
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        out = cal.transform(probs)
        np.testing.assert_allclose(out, probs, atol=1e-6)


class TestECEReduction:
    """Synthetic asymmetric miscalibration → BetaCalibrator should bring it
    closer to identity, ideally below the input ECE."""

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_ece_drops_after_fit_on_overconfident_data(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 5_000
        # Construct overconfident probabilities: true p ~ Uniform(0.2, 0.8) but
        # the predictor reports σ(2·logit(p)), which sharpens the distribution.
        from scipy.special import expit, logit

        true_p = rng.uniform(0.2, 0.8, n)
        raw_p = expit(2.0 * logit(true_p))  # over-confident
        y = (rng.uniform(size=n) < true_p).astype(int)
        pre_ece = _expected_calibration_error(raw_p, y)
        post = BetaCalibrator().fit_transform(raw_p, y)
        post_ece = _expected_calibration_error(post, y)
        assert post_ece < pre_ece, f"expected ECE improvement; got {pre_ece} → {post_ece}"
        # Allow a generous absolute target: post-fit ECE should be small.
        assert post_ece < 0.05

    def test_monotone_on_uniform_grid(self) -> None:
        """Post-fit transform must preserve order on a uniform input grid —
        if Beta calibration violates monotonicity it's degenerate."""
        rng = np.random.default_rng(0)
        n = 4_000
        p = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(size=n) < p).astype(int)
        cal = BetaCalibrator().fit(p, y)
        grid = np.linspace(0.05, 0.95, 50)
        out = cal.transform(grid)
        # Beta calibration is monotonic in p iff a > 0 and b > 0; if either
        # is non-positive (rare/degenerate), this test will fail loudly.
        diffs = np.diff(out)
        assert (diffs >= -1e-9).all(), f"non-monotone output: min diff = {diffs.min()}"


class TestMulticlass:
    def test_three_class_fit_preserves_simplex(self) -> None:
        rng = np.random.default_rng(0)
        n = 3_000
        true_logits = rng.normal(size=(n, 3))
        true_logits[:, 0] += 0.5  # class-0 slightly favoured
        # softmax
        z = true_logits - true_logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        probs = ez / ez.sum(axis=1, keepdims=True)
        # Generate outcomes from these probabilities.
        outcomes = np.array([rng.choice(3, p=row) for row in probs])
        cal = BetaCalibrator()
        post = cal.fit_transform(probs, outcomes)
        # Each row sums to 1.
        np.testing.assert_allclose(post.sum(axis=1), np.ones(n), atol=1e-6)
        # All probabilities in [0, 1].
        assert (post >= 0.0).all() and (post <= 1.0).all()


class TestFallback:
    def test_falls_back_to_identity_below_min_samples(self) -> None:
        """With only 5 positives, fit must return identity (1, 1, 0)."""
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10)
        y = np.zeros_like(p, dtype=int)
        y[:5] = 1
        cal = BetaCalibrator(min_samples_per_class=50).fit(p, y)
        dump = cal.to_dict()
        # Both classes hit the fallback since neither has 50+ positives.
        assert dump["a"][1] == 1.0
        assert dump["b"][1] == 1.0
        assert dump["c"][1] == 0.0


class TestErrorPaths:
    def test_transform_before_fit_raises(self) -> None:
        cal = BetaCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_length_mismatch_raises(self) -> None:
        cal = BetaCalibrator()
        with pytest.raises(ValueError, match="length mismatch"):
            cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_wrong_n_classes_at_transform_raises(self) -> None:
        cal = BetaCalibrator().fit(np.array([0.1, 0.9] * 100), np.array([0, 1] * 100))
        # Fit on binary; try to transform 3-class matrix.
        with pytest.raises(ValueError, match="columns"):
            cal.transform(np.array([[0.3, 0.4, 0.3]] * 5))


class TestSerialization:
    def test_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        n = 1_000
        p = rng.uniform(0.01, 0.99, n)
        y = (rng.uniform(size=n) < p).astype(int)
        cal = BetaCalibrator().fit(p, y)
        dump = cal.to_dict()
        restored = BetaCalibrator.from_dict(dump)
        # Identical transforms on the same input.
        grid = np.linspace(0.05, 0.95, 20)
        np.testing.assert_allclose(cal.transform(grid), restored.transform(grid), atol=1e-12)

    def test_wrong_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            BetaCalibrator.from_dict(
                {"kind": "logistic", "n_classes": 2, "a": [1, 1], "b": [1, 1], "c": [0, 0]}
            )
