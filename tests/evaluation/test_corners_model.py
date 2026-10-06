"""Tests for CornersPoissonModel and CornersDistribution."""

from __future__ import annotations

import pytest

from bip.evaluation.tournaments.predictors.corners_poisson import (
    CornersDistribution,
    CornersPoissonModel,
)
from tests.evaluation.conftest import make_blended_rates


@pytest.fixture
def model() -> CornersPoissonModel:
    return CornersPoissonModel()


@pytest.fixture
def dist(model: CornersPoissonModel) -> CornersDistribution:
    home = make_blended_rates(team_id=1, corners=6.5)
    away = make_blended_rates(team_id=2, corners=4.5)
    return model.predict(home, away)


# ── Probability invariants ────────────────────────────────────────────────────


def test_team_superiority_sums_to_one(dist: CornersDistribution) -> None:
    total = dist.p_home_more + dist.p_away_more + dist.p_equal_corners
    assert abs(total - 1.0) < 1e-4


def test_over_under_complement(dist: CornersDistribution) -> None:
    for over, under in [
        (dist.p_over_8_5, dist.p_under_8_5),
        (dist.p_over_9_5, dist.p_under_9_5),
        (dist.p_over_10_5, dist.p_under_10_5),
        (dist.p_over_11_5, dist.p_under_11_5),
    ]:
        assert abs(over + under - 1.0) < 1e-6


def test_over_markets_descend(dist: CornersDistribution) -> None:
    assert dist.p_over_8_5 > dist.p_over_9_5 > dist.p_over_10_5 > dist.p_over_11_5


def test_ahc_minus_1_5_plus_complement(dist: CornersDistribution) -> None:
    # p_home_ahc_plus_1_5 = 1 - p_away_ahc_minus_1_5
    assert abs(dist.p_home_ahc_plus_1_5 - (1.0 - dist.p_away_ahc_minus_1_5)) < 1e-6


def test_all_probs_in_unit_interval(dist: CornersDistribution) -> None:
    for name, val in vars(dist).items():
        if name.startswith("p_"):
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"


# ── λ monotonicity ────────────────────────────────────────────────────────────


def test_higher_lambda_means_higher_over(model: CornersPoissonModel) -> None:
    low = model.predict(
        make_blended_rates(team_id=1, corners=3.0),
        make_blended_rates(team_id=2, corners=3.0),
    )
    high = model.predict(
        make_blended_rates(team_id=1, corners=7.0),
        make_blended_rates(team_id=2, corners=7.0),
    )
    assert high.p_over_10_5 > low.p_over_10_5


def test_team_with_higher_lambda_more_likely_to_win_corners(
    model: CornersPoissonModel,
) -> None:
    dist = model.predict(
        make_blended_rates(team_id=1, corners=8.0),
        make_blended_rates(team_id=2, corners=4.0),
    )
    assert dist.p_home_more > dist.p_away_more


# ── First half ────────────────────────────────────────────────────────────────


def test_first_half_lambdas_are_fraction_of_full(dist: CornersDistribution) -> None:
    from bip.evaluation.tournaments.predictors.corners_poisson import FIRST_HALF_FRACTION

    assert abs(dist.lambda_home_fh - dist.lambda_home * FIRST_HALF_FRACTION) < 1e-9
    assert abs(dist.lambda_away_fh - dist.lambda_away * FIRST_HALF_FRACTION) < 1e-9


def test_fh_over_5_5_less_than_4_5(dist: CornersDistribution) -> None:
    assert dist.p_total_fh_over_4_5 > dist.p_total_fh_over_5_5


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_very_low_lambda_clipped_to_minimum(model: CornersPoissonModel) -> None:
    from bip.evaluation.tournaments.predictors.corners_poisson import MIN_LAMBDA

    dist = model.predict(
        make_blended_rates(team_id=1, corners=0.0),
        make_blended_rates(team_id=2, corners=0.0),
    )
    assert dist.lambda_home == MIN_LAMBDA
    assert dist.lambda_away == MIN_LAMBDA


def test_lambda_total_equals_sum(dist: CornersDistribution) -> None:
    assert abs(dist.lambda_total - (dist.lambda_home + dist.lambda_away)) < 1e-9
