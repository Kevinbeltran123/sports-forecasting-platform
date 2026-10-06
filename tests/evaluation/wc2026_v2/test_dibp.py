"""Ola 3 — DIBP grid + market-derivation + fitter tests.

Layer-1 coverage:
- DIBPParams boundary validation
- π_diag = 0 collapses to plain Bivariate Poisson (regression)
- π_diag = 1 puts all mass on the diagonal (extreme)
- Grid sums to 1.0 after renormalisation
- Market probabilities (1X2, BTTS, OU2.5) sum correctly
- Synthetic recovery of (ρ, π_diag, θ_diag) via fit_dibp

Layer-2 (real data): smoke fit on calibration corpus (WC2018 + Euro2020).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from bip.evaluation.tournaments.predictors.bivariate_poisson import (
    _bivariate_poisson_grid,
)
from bip.evaluation.tournaments.wc2026_v2.dibp import (
    DIBPParams,
    compute_dibp_grid,
    markets_from_grid,
)
from bip.evaluation.tournaments.wc2026_v2.dibp_fitter import fit_dibp
from bip.evaluation.tournaments.wc2026_v2.weighted_strength import WeightedMLEFitter

# ─────────────────────────────────────────────────────────────────
# DIBPParams boundaries
# ─────────────────────────────────────────────────────────────────


class TestDIBPParamsValidation:
    @pytest.mark.parametrize("rho", [-0.5, -0.3, 0.0, 0.3, 0.5])
    def test_rho_in_bounds_ok(self, rho: float) -> None:
        DIBPParams(rho=rho, pi_diag=0.1, theta_diag=0.3)

    @pytest.mark.parametrize("rho", [-0.51, 0.51, 1.0])
    def test_rho_out_of_bounds_raises(self, rho: float) -> None:
        with pytest.raises(ValueError, match="rho"):
            DIBPParams(rho=rho, pi_diag=0.1, theta_diag=0.3)

    @pytest.mark.parametrize("pi", [0.0, 0.5, 1.0])
    def test_pi_in_bounds_ok(self, pi: float) -> None:
        DIBPParams(rho=0.0, pi_diag=pi, theta_diag=0.3)

    @pytest.mark.parametrize("pi", [-0.01, 1.01, 1.5])
    def test_pi_out_of_bounds_raises(self, pi: float) -> None:
        with pytest.raises(ValueError, match="pi_diag"):
            DIBPParams(rho=0.0, pi_diag=pi, theta_diag=0.3)

    @pytest.mark.parametrize("theta", [0.0, 0.5, 0.999])
    def test_theta_in_bounds_ok(self, theta: float) -> None:
        DIBPParams(rho=0.0, pi_diag=0.1, theta_diag=theta)

    @pytest.mark.parametrize("theta", [-0.01, 1.0, 1.5])
    def test_theta_out_of_bounds_raises(self, theta: float) -> None:
        with pytest.raises(ValueError, match="theta_diag"):
            DIBPParams(rho=0.0, pi_diag=0.1, theta_diag=theta)


# ─────────────────────────────────────────────────────────────────
# DIBP grid: shape, sum, collapse to BP at π=0
# ─────────────────────────────────────────────────────────────────


class TestDIBPGrid:
    def test_grid_shape(self) -> None:
        g = compute_dibp_grid(1.5, 1.2, DIBPParams(0.0, 0.0, 0.5), max_goals=6)
        assert g.shape == (7, 7)

    def test_grid_sums_to_one(self) -> None:
        g = compute_dibp_grid(1.5, 1.2, DIBPParams(0.05, 0.15, 0.4), max_goals=8)
        assert g.sum() == pytest.approx(1.0, abs=1e-9)

    def test_pi_zero_collapses_to_plain_bp(self) -> None:
        """At π_diag = 0, DIBP must reduce to the renormalised plain BP grid."""
        mu_h, mu_a, rho = 1.6, 1.2, 0.04
        max_goals = 8
        params = DIBPParams(rho=rho, pi_diag=0.0, theta_diag=0.5)
        dibp_grid = compute_dibp_grid(mu_h, mu_a, params, max_goals=max_goals)

        # Recompute reference plain BP grid (then renormalise to match DIBP).
        import math

        lam12 = rho * math.sqrt(mu_h * mu_a)
        lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
        lam1 = mu_h - lam12
        lam2 = mu_a - lam12
        bp_ref = _bivariate_poisson_grid(lam1, lam2, lam12, max_goals)
        bp_ref /= bp_ref.sum()

        np.testing.assert_allclose(dibp_grid, bp_ref, atol=1e-8)

    def test_pi_one_pushes_all_mass_onto_diagonal(self) -> None:
        """At π_diag = 1, off-diagonal cells must be zero."""
        params = DIBPParams(rho=0.0, pi_diag=1.0, theta_diag=0.4)
        g = compute_dibp_grid(1.5, 1.2, params, max_goals=8)
        off_diag_total = g.sum() - np.trace(g)
        assert off_diag_total == pytest.approx(0.0, abs=1e-9)
        # Diagonal mass should be ≈ 1.
        assert np.trace(g) == pytest.approx(1.0, abs=1e-9)

    def test_pi_positive_strictly_increases_diagonal_mass(self) -> None:
        """Holding θ fixed, increasing π should raise P(draw)."""
        mu_h, mu_a = 1.5, 1.2
        diag_at_pi_0 = np.trace(
            compute_dibp_grid(mu_h, mu_a, DIBPParams(0.04, 0.0, 0.3), max_goals=8)
        )
        diag_at_pi_03 = np.trace(
            compute_dibp_grid(mu_h, mu_a, DIBPParams(0.04, 0.3, 0.3), max_goals=8)
        )
        assert diag_at_pi_03 > diag_at_pi_0

    def test_nonnegative_probabilities(self) -> None:
        g = compute_dibp_grid(2.0, 1.0, DIBPParams(0.2, 0.2, 0.3), max_goals=8)
        assert (g >= 0).all()
        assert (g <= 1).all()


# ─────────────────────────────────────────────────────────────────
# Markets derivation
# ─────────────────────────────────────────────────────────────────


class TestMarkets:
    def test_1x2_sums_to_one(self) -> None:
        g = compute_dibp_grid(1.5, 1.2, DIBPParams(0.04, 0.1, 0.3), max_goals=8)
        m = markets_from_grid(g)
        assert m.p_home_win + m.p_draw + m.p_away_win == pytest.approx(1.0, abs=1e-9)

    def test_over_under_sums_to_one(self) -> None:
        g = compute_dibp_grid(1.5, 1.2, DIBPParams(0.04, 0.1, 0.3), max_goals=8)
        m = markets_from_grid(g)
        assert m.p_over_2_5 + m.p_under_2_5 == pytest.approx(1.0, abs=1e-9)

    def test_higher_mu_h_increases_home_win_prob(self) -> None:
        params = DIBPParams(0.04, 0.1, 0.3)
        weak = markets_from_grid(compute_dibp_grid(0.8, 1.5, params))
        strong = markets_from_grid(compute_dibp_grid(2.5, 1.5, params))
        assert strong.p_home_win > weak.p_home_win

    def test_higher_mu_total_increases_over(self) -> None:
        params = DIBPParams(0.04, 0.1, 0.3)
        low = markets_from_grid(compute_dibp_grid(0.6, 0.5, params))
        high = markets_from_grid(compute_dibp_grid(2.4, 2.0, params))
        assert high.p_over_2_5 > low.p_over_2_5

    def test_expected_total_matches_summation(self) -> None:
        params = DIBPParams(0.0, 0.0, 0.3)  # π=0 → pure BP
        mu_h, mu_a = 1.6, 1.4
        g = compute_dibp_grid(mu_h, mu_a, params, max_goals=8)
        m = markets_from_grid(g)
        # For plain BP (ρ=0), E[X+Y] ≈ μ_h + μ_a; allow grid-truncation drift.
        assert m.expected_total_goals == pytest.approx(mu_h + mu_a, abs=0.05)


# ─────────────────────────────────────────────────────────────────
# Fitter — synthetic recovery + smoke fit on real calibration corpus
# ─────────────────────────────────────────────────────────────────


def _synthetic_dibp_corpus(
    truth: DIBPParams,
    n_teams: int = 10,
    n_matches_per_pair: int = 8,
    seed: int = 31,
) -> tuple[pl.DataFrame, WeightedMLEFitter]:
    """Build a synthetic match corpus drawn from a DIBP family with known
    parameters, plus a fitted strength prior to feed into ``fit_dibp``.

    Returns (matches_df, strength_prior_fitter_result).
    """
    rng = np.random.default_rng(seed)
    teams = [f"team_{i:02d}" for i in range(n_teams)]
    attacks = rng.normal(scale=0.3, size=n_teams)
    attacks -= attacks.mean()
    defenses = rng.normal(scale=0.3, size=n_teams)
    defenses -= defenses.mean()

    # First build the martj42-style training frame so we can fit the prior.
    train_rows: list[dict[str, object]] = []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i == j:
                continue
            mu_h_truth = float(np.exp(0.30 + attacks[i] + defenses[j] + 0.20))
            mu_a_truth = float(np.exp(0.30 + attacks[j] + defenses[i]))
            for _ in range(4):
                train_rows.append(
                    {
                        "date": date(2025, 1, 1),
                        "home_team": h,
                        "away_team": a,
                        "home_score": int(rng.poisson(mu_h_truth)),
                        "away_score": int(rng.poisson(mu_a_truth)),
                        "tournament": "Friendly",
                        "neutral": False,
                    }
                )
    train_df = pl.DataFrame(train_rows)
    prior_result = WeightedMLEFitter(min_appearances=5).fit(
        train_df, reference_date=date(2026, 5, 23)
    )

    # Now build the calibration matches drawn from the actual DIBP family
    # so the fitter has a target.
    rows: list[dict[str, object]] = []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if i == j:
                continue
            mu_h, mu_a = prior_result.predict_lambdas(h, a)
            grid = compute_dibp_grid(mu_h, mu_a, truth, max_goals=8)
            flat = grid.flatten()
            for _ in range(n_matches_per_pair):
                idx = rng.choice(len(flat), p=flat)
                x, y = divmod(int(idx), grid.shape[1])
                rows.append(
                    {
                        "match_id": len(rows),
                        "tournament_slug": "synthetic",
                        "match_date": date(2025, 6, 1),
                        "home_team": h,
                        "away_team": a,
                        "home_goals": int(x),
                        "away_goals": int(y),
                        "home_corners": 0,
                        "away_corners": 0,
                        "home_xg": float(mu_h),
                        "away_xg": float(mu_a),
                        "home_shots": 0,
                        "away_shots": 0,
                    }
                )
    return pl.DataFrame(rows), prior_result


class TestDIBPFitter:
    """Synthetic recovery: generate from truth → fit → recover within
    tolerance. We don't insist on exact recovery — the fitter is bounded
    by sample noise + grid truncation."""

    def test_recovers_pi_diag_near_truth(self) -> None:
        truth = DIBPParams(rho=0.04, pi_diag=0.20, theta_diag=0.30)
        matches, prior = _synthetic_dibp_corpus(truth, n_teams=8, n_matches_per_pair=10)
        result = fit_dibp(matches, prior, max_goals=8, max_iter=200)
        # n_used should match the corpus minus any rows where teams missed.
        assert result.n_matches_used > 200
        # π_diag should be recovered to within 0.10.
        assert abs(result.params.pi_diag - truth.pi_diag) < 0.10

    def test_zero_pi_truth_recovers_near_zero(self) -> None:
        truth = DIBPParams(rho=0.04, pi_diag=0.0, theta_diag=0.50)
        matches, prior = _synthetic_dibp_corpus(truth, n_teams=8, n_matches_per_pair=10)
        result = fit_dibp(matches, prior, max_iter=200)
        # π_diag near zero (within 0.10 of the boundary).
        assert result.params.pi_diag < 0.15

    def test_no_overlap_raises(self) -> None:
        truth = DIBPParams(rho=0.04, pi_diag=0.2, theta_diag=0.3)
        matches, prior = _synthetic_dibp_corpus(truth, n_teams=6)
        # Rename teams so they don't intersect the prior.
        matches = matches.with_columns(
            (pl.col("home_team") + "_zzz").alias("home_team"),
            (pl.col("away_team") + "_zzz").alias("away_team"),
        )
        with pytest.raises(ValueError, match="No matches"):
            fit_dibp(matches, prior)


# ─────────────────────────────────────────────────────────────────
# Layer-2 — smoke fit on the real calibration corpus
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestRealCalibrationCorpusSmoke:
    """Layer-2: fit on the actual WC2018 + Euro2020 calibration corpus."""

    def test_fit_on_real_calibration_corpus(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        # Strength prior on training only (excludes the four hold-outs).
        recent_train = split.train.filter(pl.col("date") >= pl.lit(date(2010, 1, 1)))
        prior = WeightedMLEFitter(min_appearances=5).fit(
            recent_train, reference_date=date(2026, 5, 23)
        )
        result = fit_dibp(split.calibration, prior, max_iter=100)

        # Calibration corpus has 115 matches; most teams should be in the prior.
        assert result.n_matches_used >= 100
        # The fit must converge under L-BFGS-B bounds.
        assert result.converged
        # All three params inside their valid ranges. π_diag may legitimately
        # converge to 0 on this corpus (= "DIBP doesn't help here"); when π=0
        # the θ_diag value is irrelevant and may sit anywhere in [0, 0.95].
        assert 0.0 <= result.params.pi_diag <= 0.95
        assert 0.0 <= result.params.theta_diag <= 0.95
        assert 0.0 <= result.params.rho <= 0.5
