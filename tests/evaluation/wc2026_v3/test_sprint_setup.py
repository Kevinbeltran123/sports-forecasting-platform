"""Ola 0 smoke test — v3 sprint scaffolding + v2 module surface invariant.

This test verifies two things:
1. The v3 test directory is wired correctly (conftest fixtures resolve).
2. The wc2026_v2 module surface is intact and importable — v3 extends v2
   via toggles, so any v2 import regression would block all subsequent
   v3 olas.
"""

from __future__ import annotations

import pytest


def test_baseline_v1_fixture_has_canonical_values(baseline_v1) -> None:
    """Lock_v1 reference numbers match internal research notes

    These are the numbers every v3 ablation must compare against. If they
    drift, update conftest.py AND the source doc in the same commit.
    """
    assert baseline_v1.brier_1x2 == pytest.approx(0.2156)
    assert baseline_v1.brier_1x2_ci_lower < baseline_v1.brier_1x2
    assert baseline_v1.brier_1x2 < baseline_v1.brier_1x2_ci_upper
    assert baseline_v1.n_heldout == 199
    assert baseline_v1.n_calibration == 115
    assert baseline_v1.afcon_2023_brier == pytest.approx(0.2271)
    assert baseline_v1.copa_2024_brier == pytest.approx(0.2021)


def test_quality_gates_fixture_matches_v2_plan(quality_gates) -> None:
    """v3 gates are inherited verbatim from v2 PLAN.md §Quality gate."""
    assert quality_gates["brier_ci_upper_max"] == 0.21
    assert quality_gates["ece_max"] == 0.05
    assert quality_gates["brier_improvement_min"] == 0.005
    assert quality_gates["tournament_pass_count_min"] == 2
    assert quality_gates["tournament_pass_count_total"] == 4


def test_wc2026_v2_module_surface_intact() -> None:
    """v3 extends v2 via toggles; v2 module imports must stay green.

    Imports the four public entry points v3 will consume. Any
    ImportError here means a v2 refactor broke the v3 contract surface
    and must be fixed before any v3 ola proceeds.
    """
    from bip.evaluation.tournaments.wc2026_v2.pipeline import fit_v2_pipeline
    from bip.evaluation.tournaments.wc2026_v2.v2_predictor import (
        CalibratedDIBPPredictor,
    )
    from bip.evaluation.tournaments.wc2026_v2.weighted_strength import (
        WeightedMLEResult,
    )
    from bip.evaluation.tournaments.wc2026_v2.dibp import DIBPParams

    # The objects exist and are callable / instantiable types.
    assert callable(fit_v2_pipeline)
    assert isinstance(CalibratedDIBPPredictor, type)
    assert isinstance(WeightedMLEResult, type)
    assert isinstance(DIBPParams, type)


def test_wc2026_v2_pipeline_has_expected_toggles() -> None:
    """v3 will add a `use_market_value` toggle to fit_v2_pipeline.

    This test guards the existing 3 toggles so a v3 PR that touches the
    signature is forced to confirm backward-compat with v2 behavior.
    """
    import inspect

    from bip.evaluation.tournaments.wc2026_v2.pipeline import fit_v2_pipeline

    sig = inspect.signature(fit_v2_pipeline)
    params = set(sig.parameters)
    expected_toggles = {"use_dibp", "use_beta_calibration", "use_match_importance"}
    missing = expected_toggles - params
    assert not missing, f"v2 pipeline missing expected toggles: {missing}"
