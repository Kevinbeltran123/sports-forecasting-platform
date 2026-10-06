"""Tests for stack_profiles."""

from __future__ import annotations

import pytest

from bip.evaluation.live.stack_profiles import (
    PROFILES, StackProfile, get_profile, list_profiles,
)


class TestProfileRegistry:
    def test_four_profiles_registered(self):
        assert {"tier_1_only", "balanced", "aggressive", "low_variance"} <= set(
            PROFILES.keys()
        )

    def test_each_profile_has_required_fields(self):
        for name, p in PROFILES.items():
            assert isinstance(p, StackProfile)
            assert p.name == name
            assert p.description
            assert p.expected_roi_band
            assert isinstance(p.use_per_market, bool)

    def test_aggressive_has_per_market(self):
        p = PROFILES["aggressive"]
        assert p.use_per_market is True
        assert "per_market" in p.calibrator_path

    def test_tier_1_only_has_no_calibrator(self):
        p = PROFILES["tier_1_only"]
        assert p.calibrator_path is None

    def test_balanced_uses_global_calibrator(self):
        p = PROFILES["balanced"]
        assert p.use_per_market is False
        assert "isotonic" in p.calibrator_path

    def test_aggressive_mean_higher_than_balanced(self):
        # Documented invariant from CV results
        assert PROFILES["aggressive"].expected_roi_mean > PROFILES["balanced"].expected_roi_mean

    def test_low_variance_lower_std_than_aggressive(self):
        assert PROFILES["low_variance"].expected_roi_std < PROFILES["aggressive"].expected_roi_std


class TestGetProfile:
    def test_returns_profile_by_name(self):
        p = get_profile("aggressive")
        assert p.name == "aggressive"

    def test_raises_on_unknown_name(self):
        with pytest.raises(KeyError, match="Unknown stack profile"):
            get_profile("nonexistent")


class TestListProfiles:
    def test_returns_string_with_all_profile_names(self):
        text = list_profiles()
        for name in PROFILES.keys():
            assert name in text


class TestBuild:
    def test_tier_1_only_builds_without_calibrator(self):
        det = PROFILES["tier_1_only"].build()
        assert det.calibrator is None
        # Tier 1.4 haircut stays ON (no calibrator)
        assert det.high_prob_haircut_threshold < 1.0

    def test_aggressive_build_requires_calibrator_file(self, tmp_path):
        # The Day-1 calibrator path is the default; we can't depend on
        # it for tests. Use calibrator_path_override.
        import numpy as np
        from bip.evaluation.live.calibration import (
            IsotonicProbabilityCalibrator, PerMarketCalibrator,
        )
        rng = np.random.default_rng(0)
        probs = np.linspace(0.1, 0.9, 200)
        outcomes = (rng.uniform(size=200) < probs).astype(float)
        markets = ["m_a"] * 100 + ["m_b"] * 100
        cal = PerMarketCalibrator.fit(probs, outcomes, markets,
                                      min_samples_per_market=50)
        path = tmp_path / "per_market.json"
        cal.save_json(path)
        det = PROFILES["aggressive"].build(
            calibrator_path_override=str(path),
        )
        assert det.calibrator is not None
        # Haircut auto-disabled when calibrator loaded
        assert det.high_prob_haircut_threshold > 1.0

    def test_extra_kwargs_propagate(self, tmp_path):
        det = PROFILES["tier_1_only"].build(min_edge_pct=5.0)
        assert det.min_edge_pct == 5.0
