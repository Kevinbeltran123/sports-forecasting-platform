"""Shared fixtures + canonical baseline constants for v3 sprint tests.

Baseline numbers are inherited verbatim from the v2 spike results
(internal research notes, FAIL verdict commit f500783) and
serve as the reference lock_v1 baseline that any v3 ablation must
compare against.

Constants are intentionally hard-coded here (not re-computed at test
time) to keep this conftest dependency-free and fast. The single source
of truth is internal research notes; if numbers are
re-measured, update both files in the same commit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class BaselineV1:
    """Lock_v1 reference numbers on the immovable n=199 hold-out.

    Source: internal research notes §3 (lock convention,
    1X2 Brier divided by K=3). v2 spike re-measured these for the
    'baseline (all off)' ablation cell and reported the values below.
    """

    brier_1x2: float
    brier_1x2_ci_lower: float
    brier_1x2_ci_upper: float
    ece_1x2: float
    n_heldout: int
    n_calibration: int
    n_train: int

    afcon_2023_brier: float
    copa_2024_brier: float

    bootstrap_seed: int = 42
    bootstrap_n_resamples: int = 2000


_BASELINE_V1 = BaselineV1(
    brier_1x2=0.2156,
    brier_1x2_ci_lower=0.2027,
    brier_1x2_ci_upper=0.2290,
    ece_1x2=0.1074,
    n_heldout=199,
    n_calibration=115,
    n_train=49_058,
    afcon_2023_brier=0.2271,
    copa_2024_brier=0.2021,
)


@pytest.fixture(scope="session")
def baseline_v1() -> BaselineV1:
    """Lock_v1 reference numbers on the n=199 hold-out (session-scoped)."""
    return _BASELINE_V1


@pytest.fixture(scope="session")
def quality_gates() -> dict[str, float]:
    """v3 quality gates (inherited verbatim from v2 PLAN.md §Quality gate).

    Any v3 candidate must clear all 5 of these to PASS lock_v3 registration.
    Gate 3 alone qualifies for SHADOW-only emission.
    """
    return {
        "brier_ci_upper_max": 0.21,
        "ece_max": 0.05,
        "brier_improvement_min": 0.005,
        "tournament_pass_count_min": 2,
        "tournament_pass_count_total": 4,
    }
