"""Tests for scripts/run_phase5_backtest.py — Phase 5 Layer-2 runner.

Layer-1 tests: BayesianPoissonPredictor on synthetic snapshots, Poisson
score-grid → market-prob derivation, snapshot conversion, end-to-end
plumbing with stub StatsBomb outcomes.

Layer-2 tests: real backtest over the 6 men's tournaments — runs only
when the StatsBomb Parquet has been seeded.
"""

from __future__ import annotations

from datetime import date

import pytest

from bip.evaluation.tournaments.live.competition_weights import CompetitionType
from scripts.run_phase5_backtest import (
    BayesianPoissonPredictor,
    _poisson_score_grid_to_prediction,
    _tournament_to_competition,
    make_snapshots,
)
from scripts.seed_statsbomb_tournaments import (
    DEFAULT_OUTCOMES_PARQUET,
    StatsBombMatchOutcome,
)

# ── Poisson grid → market probabilities ──────────────────────────────────────


class TestPoissonGrid:
    def test_low_lambda_predicts_low_scoring(self):
        """Both teams λ=0.5 → most matches end 0-0 → low P(over 2.5).

        Math: total ~ Poisson(1.0), P(total ≥ 3) ≈ 0.080.
        """
        pred = _poisson_score_grid_to_prediction(0.5, 0.5)
        assert pred.p_over_2_5 < 0.10
        assert pred.p_btts < 0.30  # rare BTTS at low rates
        assert pred.p_draw > pred.p_home_win
        assert pred.p_draw > pred.p_away_win

    def test_high_lambda_predicts_high_scoring(self):
        """Both teams λ=2.5 → high-scoring expected."""
        pred = _poisson_score_grid_to_prediction(2.5, 2.5)
        assert pred.p_over_2_5 > 0.70
        assert pred.p_btts > 0.60

    def test_asymmetric_lambdas_favour_higher_team(self):
        """λ_home=2.5 vs λ_away=0.5 → home win much more likely."""
        pred = _poisson_score_grid_to_prediction(2.5, 0.5)
        assert pred.p_home_win > 0.70
        assert pred.p_away_win < 0.10

    def test_probabilities_sum_to_one(self):
        pred = _poisson_score_grid_to_prediction(1.5, 1.2)
        assert pytest.approx(
            pred.p_home_win + pred.p_draw + pred.p_away_win, abs=1e-6
        ) == 1.0

    def test_zero_lambda_handled_gracefully(self):
        """λ=0 should yield a valid prediction (no inf/nan)."""
        pred = _poisson_score_grid_to_prediction(1e-3, 1e-3)
        # All probabilities finite
        assert 0 <= pred.p_home_win <= 1
        assert 0 <= pred.p_btts <= 1
        assert 0 <= pred.p_over_2_5 <= 1


# ── Snapshot conversion ──────────────────────────────────────────────────────


def _outcome(
    match_id: int,
    home: str,
    away: str,
    hg: int,
    ag: int,
    hc: int = 5,
    ac: int = 5,
    d: date = date(2024, 7, 10),
    slug: str = "copa_2024",
) -> StatsBombMatchOutcome:
    return StatsBombMatchOutcome(
        match_id=match_id,
        tournament_slug=slug,
        match_date=d,
        home_team=home,
        away_team=away,
        home_goals=hg,
        away_goals=ag,
        home_corners=hc,
        away_corners=ac,
    )


class TestMakeSnapshots:
    def test_sorts_by_match_date(self):
        outcomes = [
            _outcome(1, "A", "B", 1, 0, d=date(2024, 7, 10)),
            _outcome(2, "C", "D", 0, 0, d=date(2024, 6, 15)),
            _outcome(3, "E", "F", 2, 1, d=date(2024, 7, 1)),
        ]
        snapshots, _ = make_snapshots(outcomes)
        dates = [s.match_date for s in snapshots]
        assert dates == sorted(dates)

    def test_same_team_same_id_across_matches(self):
        outcomes = [
            _outcome(1, "Argentina", "Brazil", 1, 0),
            _outcome(2, "Brazil", "Uruguay", 2, 0),
        ]
        snapshots, team_id_map = make_snapshots(outcomes)
        # Brazil appears in both; should keep same id
        assert team_id_map["Brazil"] == snapshots[0].away_team_id
        assert team_id_map["Brazil"] == snapshots[1].home_team_id

    def test_propagates_observed_outcomes(self):
        outcomes = [_outcome(1, "A", "B", 2, 1, hc=4, ac=7)]
        snapshots, _ = make_snapshots(outcomes)
        snap = snapshots[0]
        assert snap.observed_1x2 == 0  # home win
        assert snap.observed_total_goals == 3
        assert snap.observed_btts == 1
        assert snap.observed_home_goals == 2
        assert snap.observed_away_goals == 1
        assert snap.observed_home_corners == 4
        assert snap.observed_away_corners == 7

    def test_team_id_map_is_deterministic_insertion_order(self):
        outcomes = [
            _outcome(1, "Z", "A", 0, 0),
            _outcome(2, "Z", "B", 0, 0),
            _outcome(3, "C", "A", 0, 0),
        ]
        _, team_id_map = make_snapshots(outcomes)
        # Sorted by date (all same) → insertion is in input order
        # First seen: Z → 0, A → 1, then B → 2, then C → 3
        # (But actually outcomes get sorted by date first, which is the same date,
        # so their relative order is preserved as Python sort is stable)
        assert team_id_map["Z"] == 0
        assert team_id_map["A"] == 1
        assert team_id_map["B"] == 2
        assert team_id_map["C"] == 3


# ── Tournament → competition mapping ─────────────────────────────────────────


class TestTournamentToCompetition:
    def test_world_cup_maps_to_wc(self):
        assert _tournament_to_competition("wc_2018") == CompetitionType.WORLD_CUP
        assert _tournament_to_competition("wc_2022") == CompetitionType.WORLD_CUP

    def test_continental_maps_to_confederation(self):
        assert _tournament_to_competition("euro_2024") == CompetitionType.CONFEDERATION
        assert _tournament_to_competition("copa_2024") == CompetitionType.CONFEDERATION
        assert _tournament_to_competition("afcon_2023") == CompetitionType.CONFEDERATION


# ── BayesianPoissonPredictor ─────────────────────────────────────────────────


class TestBayesianBivariatePoissonPredictor:
    def test_rho_zero_yields_independent_poisson_grid(self):
        """ρ=0 collapses to Independent Poisson (Bivariate is a strict generalization)."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import (
            BayesianBivariatePoissonPredictor,
            BayesianPoissonPredictor,
        )

        snap = BacktestSnapshot(
            match_id="m1", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=1, observed_total_goals=0, observed_btts=0,
            observed_home_goals=0, observed_away_goals=0,
            match_date="2018-06-14",
        )
        ind = BayesianPoissonPredictor()
        biv = BayesianBivariatePoissonPredictor(rho=0.0)
        ind_pred = ind.predict_fixture(snap)
        biv_pred = biv.predict_fixture(snap)
        assert biv_pred.p_home_win == pytest.approx(ind_pred.p_home_win, abs=1e-6)
        assert biv_pred.p_draw == pytest.approx(ind_pred.p_draw, abs=1e-6)
        assert biv_pred.p_btts == pytest.approx(ind_pred.p_btts, abs=1e-6)

    def test_positive_rho_inflates_draw_probability(self):
        """ρ>0 should yield higher P(draw) than ρ=0 — corrects Independent
        Poisson's draw underestimation per SYNTHESIS Conclusion 2."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import BayesianBivariatePoissonPredictor

        snap = BacktestSnapshot(
            match_id="m1", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=1, observed_total_goals=0, observed_btts=0,
            observed_home_goals=0, observed_away_goals=0,
            match_date="2018-06-14",
        )
        rho_0 = BayesianBivariatePoissonPredictor(rho=0.0).predict_fixture(snap)
        rho_high = BayesianBivariatePoissonPredictor(rho=0.20).predict_fixture(snap)
        assert rho_high.p_draw > rho_0.p_draw

    def test_invalid_rho_rejected(self):
        from scripts.run_phase5_backtest import BayesianBivariatePoissonPredictor

        with pytest.raises(ValueError, match="rho must be"):
            BayesianBivariatePoissonPredictor(rho=0.6)
        with pytest.raises(ValueError, match="rho must be"):
            BayesianBivariatePoissonPredictor(rho=-0.6)


class TestPredictorFactory:
    def test_independent_kind(self):
        from scripts.run_phase5_backtest import (
            BayesianPoissonPredictor,
            _make_predictor,
        )

        p = _make_predictor("independent", sigma_s_per_day=0.0001)
        assert isinstance(p, BayesianPoissonPredictor)

    def test_bivariate_kind_uses_rho(self):
        from scripts.run_phase5_backtest import (
            BayesianBivariatePoissonPredictor,
            _make_predictor,
        )

        p = _make_predictor("bivariate", sigma_s_per_day=0.0001, rho=0.10)
        assert isinstance(p, BayesianBivariatePoissonPredictor)
        assert p.rho == 0.10

    def test_unknown_kind_raises(self):
        from scripts.run_phase5_backtest import _make_predictor

        with pytest.raises(ValueError, match="Unknown predictor_kind"):
            _make_predictor("nonexistent", sigma_s_per_day=0.0001)


# ── Train-time ρ tuner (Layer-1 — synthetic snapshots) ───────────────────────


class TestBivariateLogPmf:
    def test_rho_zero_matches_independent_poisson_log_pmf(self):
        """ρ=0 → joint log-PMF == log P(X=h)·P(Y=a) (independent)."""
        import math

        from scripts.run_phase5_backtest import _bivariate_log_pmf, _poisson_pmf

        mu_h, mu_a = 1.5, 1.2
        for h in (0, 1, 2, 3):
            for a in (0, 1, 2, 3):
                joint = _bivariate_log_pmf(h, a, mu_h, mu_a, rho=0.0)
                expected = math.log(_poisson_pmf(h, mu_h) * _poisson_pmf(a, mu_a))
                assert joint == pytest.approx(expected, abs=1e-6)

    def test_positive_rho_inflates_diagonal_probability(self):
        """ρ>0 should raise log P(h, h) — concentrates mass on the diagonal."""
        from scripts.run_phase5_backtest import _bivariate_log_pmf

        mu_h, mu_a = 1.5, 1.5
        zero = _bivariate_log_pmf(1, 1, mu_h, mu_a, rho=0.0)
        positive = _bivariate_log_pmf(1, 1, mu_h, mu_a, rho=0.20)
        assert positive > zero

    def test_extreme_score_returns_finite_log(self):
        """Score above MAX_GOALS_GRID truncates to grid edge; result stays finite."""
        import math

        from scripts.run_phase5_backtest import _bivariate_log_pmf

        # Even very low probability still finite (truncated to grid edge)
        v = _bivariate_log_pmf(20, 20, 0.5, 0.5, rho=0.0)
        assert math.isfinite(v)

    def test_zero_probability_cell_floored(self):
        """When the grid cell is identically 0 (e.g., μ=0), log returns the floor
        rather than -inf, so averaging stays well-behaved."""
        import math

        from scripts.run_phase5_backtest import _MIN_BIVARIATE_LOG_PROB, _bivariate_log_pmf

        # μ=0 → grid[k>0, k>0] is exactly 0 — floor kicks in
        v = _bivariate_log_pmf(2, 2, mu_h=0.0, mu_a=0.0, rho=0.0)
        assert math.isfinite(v)
        assert v == pytest.approx(_MIN_BIVARIATE_LOG_PROB, abs=1e-9)


class TestTuneRhoWalkforward:
    def test_excludes_held_out_tournaments(self):
        """All snapshots in held-out tournaments → tuner refuses (no train data)."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_rho_walkforward

        snaps = [
            BacktestSnapshot(
                match_id=f"m{i}", tournament="copa_2024",
                home_team_id=i % 4, away_team_id=(i + 1) % 4,
                observed_1x2=0, observed_total_goals=2, observed_btts=1,
                observed_home_goals=1, observed_away_goals=1,
                match_date=f"2024-06-{i + 1:02d}",
            )
            for i in range(8)
        ]
        with pytest.raises(RuntimeError, match="No train snapshots"):
            tune_rho_walkforward(
                snaps, [0.0, 0.04],
                held_out_tournaments=("copa_2024",),
            )

    def test_returns_score_per_candidate(self):
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_rho_walkforward

        # Mix train (wc_2018) and held-out (copa_2024)
        train_snaps = [
            BacktestSnapshot(
                match_id=f"t{i}", tournament="wc_2018",
                home_team_id=i % 4, away_team_id=(i + 1) % 4,
                observed_1x2=0, observed_total_goals=2, observed_btts=1,
                observed_home_goals=1, observed_away_goals=1,
                match_date=f"2018-06-{(i % 28) + 1:02d}",
            )
            for i in range(20)
        ]
        held_snaps = [
            BacktestSnapshot(
                match_id=f"h{i}", tournament="copa_2024",
                home_team_id=i, away_team_id=i + 1,
                observed_1x2=0, observed_total_goals=2, observed_btts=1,
                observed_home_goals=1, observed_away_goals=1,
                match_date=f"2024-06-{(i % 28) + 1:02d}",
            )
            for i in range(8)
        ]
        candidates = [0.0, 0.04, 0.10]
        best, scores = tune_rho_walkforward(
            train_snaps + held_snaps, candidates,
            held_out_tournaments=("copa_2024",),
        )
        assert set(scores.keys()) == set(candidates)
        assert best in candidates
        assert all(isinstance(v, float) for v in scores.values())

    def test_picks_higher_rho_on_diagonal_heavy_synthetic_data(self):
        """If train data has many 1-1, 2-2 score lines, ρ>0 should win the
        sweep — diagonal mass is exactly what positive ρ adds.

        Construct 30 fixtures across 6 teams where ~80% end on the diagonal
        (1-1 or 2-2). Independent Poisson under-predicts these; positive ρ
        should yield higher avg joint log-likelihood.
        """
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_rho_walkforward

        snaps = []
        for i in range(30):
            home_id = i % 6
            away_id = (i + 1) % 6
            # 80% diagonal (1-1 or 2-2), 20% non-diagonal (2-1 / 1-2)
            if i % 5 == 0:
                hg, ag = (2, 1) if i % 2 == 0 else (1, 2)
            else:
                hg, ag = (1, 1) if i % 2 == 0 else (2, 2)
            snaps.append(BacktestSnapshot(
                match_id=f"m{i}", tournament="wc_2018",
                home_team_id=home_id, away_team_id=away_id,
                observed_1x2=0 if hg > ag else (1 if hg == ag else 2),
                observed_total_goals=hg + ag,
                observed_btts=int(hg > 0 and ag > 0),
                observed_home_goals=hg, observed_away_goals=ag,
                match_date=f"2018-06-{(i % 28) + 1:02d}",
            ))
        best, scores = tune_rho_walkforward(
            snaps, [0.0, 0.10, 0.25],
            held_out_tournaments=("copa_2024",),
        )
        # On diagonal-heavy data, ρ=0 should be strictly worse than ρ>0
        assert scores[0.10] > scores[0.0]
        assert scores[0.25] > scores[0.0]
        assert best in (0.10, 0.25)

    def test_empty_candidates_rejected(self):
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_rho_walkforward

        snaps = [BacktestSnapshot(
            match_id="m0", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=0, observed_total_goals=1, observed_btts=0,
            observed_home_goals=1, observed_away_goals=0,
            match_date="2018-06-14",
        )]
        with pytest.raises(ValueError, match="rho_candidates must not be empty"):
            tune_rho_walkforward(snaps, [])

    def test_empty_snapshots_rejected(self):
        from scripts.run_phase5_backtest import tune_rho_walkforward

        with pytest.raises(ValueError, match="snapshots must not be empty"):
            tune_rho_walkforward([], [0.0, 0.04])


# ── BayesianBivariateXGPredictor + α tuner (Fase C+D) ───────────────────────


class TestBayesianBivariateXGPredictor:
    def test_alpha_one_matches_goals_only_bivariate_at_cold_start(self):
        """At α=1 with no observations yet, should match goals-only Bivariate."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import (
            BayesianBivariatePoissonPredictor,
            BayesianBivariateXGPredictor,
        )

        snap = BacktestSnapshot(
            match_id="m1", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=0, observed_total_goals=2, observed_btts=1,
            observed_home_goals=1, observed_away_goals=1,
            observed_home_xg=2.0, observed_away_xg=0.5,
            match_date="2018-06-14",
        )
        biv = BayesianBivariatePoissonPredictor(rho=0.10).predict_fixture(snap)
        xg_alpha_1 = BayesianBivariateXGPredictor(rho=0.10, alpha=1.0).predict_fixture(snap)
        assert biv.p_home_win == pytest.approx(xg_alpha_1.p_home_win, abs=1e-9)
        assert biv.p_draw == pytest.approx(xg_alpha_1.p_draw, abs=1e-9)
        assert biv.p_btts == pytest.approx(xg_alpha_1.p_btts, abs=1e-9)

    def test_invalid_alpha_rejected(self):
        from scripts.run_phase5_backtest import BayesianBivariateXGPredictor

        with pytest.raises(ValueError, match="alpha must be"):
            BayesianBivariateXGPredictor(alpha=-0.1)
        with pytest.raises(ValueError, match="alpha must be"):
            BayesianBivariateXGPredictor(alpha=1.1)

    def test_xg_observation_shifts_predictions(self):
        """A team that played one match with xG > goals should have its
        next prediction lifted vs a team where xG = goals."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import BayesianBivariateXGPredictor

        # Team 0 played 1-0 with home_xg=2.5 (massively dominated)
        m1 = BacktestSnapshot(
            match_id="m1", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=0, observed_total_goals=1, observed_btts=0,
            observed_home_goals=1, observed_away_goals=0,
            observed_home_xg=2.5, observed_away_xg=0.4,
            match_date="2018-06-14",
        )
        # Same team plays again next round
        m2 = BacktestSnapshot(
            match_id="m2", tournament="wc_2018",
            home_team_id=0, away_team_id=2,
            observed_1x2=0, observed_total_goals=2, observed_btts=1,
            observed_home_goals=1, observed_away_goals=1,
            observed_home_xg=1.5, observed_away_xg=0.7,
            match_date="2018-06-19",
        )

        # Goals-only predictor: only sees the 1-0 result
        p_goals = BayesianBivariateXGPredictor(rho=0.04, alpha=1.0)
        p_goals.predict_fixture(m1)
        pred_goals = p_goals.predict_fixture(m2)

        # xG-blended predictor: sees the 2.5 xG, should predict more confidently
        # for team 0
        p_xg = BayesianBivariateXGPredictor(rho=0.04, alpha=0.0)  # all xG
        p_xg.predict_fixture(m1)
        pred_xg = p_xg.predict_fixture(m2)

        # xG-only predictor should weigh team 0 more strongly than goals-only
        # (because xG=2.5 >> goals=1 for that match)
        assert pred_xg.p_home_win > pred_goals.p_home_win


class TestTuneAlphaWalkforward:
    def test_returns_score_per_candidate(self):
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_alpha_walkforward

        train_snaps = [
            BacktestSnapshot(
                match_id=f"t{i}", tournament="wc_2018",
                home_team_id=i % 4, away_team_id=(i + 1) % 4,
                observed_1x2=0, observed_total_goals=2, observed_btts=1,
                observed_home_goals=1, observed_away_goals=1,
                observed_home_xg=1.5 + (i % 3) * 0.3,
                observed_away_xg=1.0 + (i % 2) * 0.4,
                match_date=f"2018-06-{(i % 28) + 1:02d}",
            )
            for i in range(20)
        ]
        candidates = [0.0, 0.5, 1.0]
        best, scores = tune_alpha_walkforward(
            train_snaps, candidates, rho=0.04,
        )
        assert set(scores.keys()) == set(candidates)
        assert best in candidates

    def test_invalid_alpha_in_grid_rejected(self):
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_alpha_walkforward

        snap = BacktestSnapshot(
            match_id="m0", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=0, observed_total_goals=1, observed_btts=0,
            observed_home_goals=1, observed_away_goals=0,
            observed_home_xg=1.0, observed_away_xg=0.5,
            match_date="2018-06-14",
        )
        with pytest.raises(ValueError, match="alpha_candidates must be"):
            tune_alpha_walkforward([snap], [0.5, 1.5])

    def test_excludes_held_out(self):
        """Snapshots only from held_out tournaments → tuner refuses."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import tune_alpha_walkforward

        snaps = [
            BacktestSnapshot(
                match_id=f"m{i}", tournament="copa_2024",
                home_team_id=i % 4, away_team_id=(i + 1) % 4,
                observed_1x2=0, observed_total_goals=1, observed_btts=0,
                observed_home_goals=1, observed_away_goals=0,
                observed_home_xg=1.0, observed_away_xg=0.5,
                match_date=f"2024-06-{i + 1:02d}",
            )
            for i in range(8)
        ]
        with pytest.raises(RuntimeError, match="No train snapshots"):
            tune_alpha_walkforward(snaps, [0.5], held_out_tournaments=("copa_2024",))


# ── Rolling-origin cross-validation (Option 3 / Fase F) ─────────────────────


class TestBootstrapCI:
    def test_bootstrap_returns_mean_and_ci_around_truth(self):
        import numpy as np

        from scripts.run_phase5_backtest import _bootstrap_ci

        # Distribution centered on 0.5 with known SE
        rng = np.random.RandomState(0)
        v = rng.normal(0.5, 0.1, 200)
        m, lo, hi = _bootstrap_ci(v, n_bootstrap=500, seed=0)
        # Mean should match sample mean
        assert m == pytest.approx(float(v.mean()), abs=1e-6)
        # CI should bracket the mean
        assert lo < m < hi
        # CI width should be reasonable for n=200, sigma=0.1 → SE ≈ 0.007 →
        # 95% CI ≈ ±0.014. Generous bound: width < 0.05.
        assert hi - lo < 0.05

    def test_empty_input_returns_zeros(self):
        import numpy as np

        from scripts.run_phase5_backtest import _bootstrap_ci

        m, lo, hi = _bootstrap_ci(np.asarray([]))
        assert (m, lo, hi) == (0.0, 0.0, 0.0)


class TestBrierPerSample:
    def test_perfect_prediction_yields_zero_brier(self):
        """Probability 1.0 on the correct outcome → Brier = 0."""
        import numpy as np

        from scripts.run_phase5_backtest import _brier_per_sample_1x2

        # Predict home win with prob 1.0; observed home win
        b = _brier_per_sample_1x2(
            np.asarray([1.0]), np.asarray([0.0]), np.asarray([0.0]),
            np.asarray([0]),
        )
        assert b[0] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_prediction_brier_value(self):
        """Uniform 1/3 over 3 classes — Brier = (2/3)²/3 = 0.148... per match."""
        import numpy as np

        from scripts.run_phase5_backtest import _brier_per_sample_1x2

        n = 100
        b = _brier_per_sample_1x2(
            np.full(n, 1 / 3), np.full(n, 1 / 3), np.full(n, 1 / 3),
            np.zeros(n, dtype=int),
        )
        # Each match: ((1/3-1)² + (1/3)² + (1/3)²) / 3 = (4/9 + 1/9 + 1/9) / 3 = 2/9
        assert b[0] == pytest.approx(2 / 9, abs=1e-6)

    def test_binary_brier_squared_error(self):
        import numpy as np

        from scripts.run_phase5_backtest import _brier_per_sample_binary

        b = _brier_per_sample_binary(
            np.asarray([0.7, 0.2, 1.0]), np.asarray([1, 0, 1]),
        )
        # (0.7-1)² = 0.09, (0.2-0)² = 0.04, (1.0-1)² = 0.0
        np.testing.assert_allclose(b, [0.09, 0.04, 0.0], atol=1e-9)


class TestRollingOriginCV:
    def test_default_folds_are_chronologically_valid(self):
        from scripts.run_phase5_backtest import DEFAULT_ROLLING_ORIGIN_FOLDS

        # Sanity: each fold's held-out tournament must come AFTER all train ones
        # (chronologically). We don't have actual dates here, but the slugs
        # encode order: WC2018 < Euro2020 < WC2022 < AFCON2023 < Copa2024 < Euro2024
        chronology = ["wc_2018", "euro_2020", "wc_2022", "afcon_2023", "copa_2024", "euro_2024"]
        for train, held in DEFAULT_ROLLING_ORIGIN_FOLDS:
            held_idx = chronology.index(held)
            for t in train:
                assert chronology.index(t) < held_idx, (
                    f"Fold violates chronology: {t} >= {held}"
                )

    def test_three_folds_each_growing(self):
        from scripts.run_phase5_backtest import DEFAULT_ROLLING_ORIGIN_FOLDS

        assert len(DEFAULT_ROLLING_ORIGIN_FOLDS) == 3
        # Each subsequent fold has a larger training set
        for i in range(1, len(DEFAULT_ROLLING_ORIGIN_FOLDS)):
            prev_train = set(DEFAULT_ROLLING_ORIGIN_FOLDS[i - 1][0])
            curr_train = set(DEFAULT_ROLLING_ORIGIN_FOLDS[i][0])
            assert prev_train.issubset(curr_train)

    def test_synthetic_run_produces_aggregate_metrics(self):
        """Smoke: build a synthetic snapshot set spanning the 6 tournaments
        and confirm rolling-origin CV runs end-to-end."""
        from scripts.backtest_int_tournaments import BacktestSnapshot
        from scripts.run_phase5_backtest import (
            DEFAULT_ROLLING_ORIGIN_FOLDS,
            run_rolling_origin_cv,
        )

        snaps = []
        slugs_dates = [
            ("wc_2018", "2018-06-14"),
            ("euro_2020", "2021-06-11"),
            ("wc_2022", "2022-11-20"),
            ("afcon_2023", "2024-01-13"),
            ("copa_2024", "2024-06-20"),
            ("euro_2024", "2024-06-14"),
        ]
        # 30 matches per tournament; rotating team ids
        match_idx = 0
        for slug, base_date in slugs_dates:
            for i in range(30):
                hg = i % 3
                ag = (i + 1) % 3
                snaps.append(BacktestSnapshot(
                    match_id=f"{slug}_{i}", tournament=slug,
                    home_team_id=i % 8, away_team_id=(i + 1) % 8,
                    observed_1x2=0 if hg > ag else (1 if hg == ag else 2),
                    observed_total_goals=hg + ag,
                    observed_btts=int(hg > 0 and ag > 0),
                    observed_home_goals=hg, observed_away_goals=ag,
                    observed_home_xg=1.0 + (i % 4) * 0.3,
                    observed_away_xg=0.8 + (i % 3) * 0.4,
                    match_date=f"{base_date[:7]}-{(i % 28) + 1:02d}",
                ))
                match_idx += 1

        result = run_rolling_origin_cv(
            snaps, DEFAULT_ROLLING_ORIGIN_FOLDS,
            rho=0.04, alpha=0.50,
            n_bootstrap=100,
        )

        # 3 folds
        assert len(result.folds) == 3
        # Total test count = afcon_2023 + copa_2024 + euro_2024 sizes
        assert result.n_total_test == 30 * 3
        # Each fold has growing train set
        for i in range(1, 3):
            assert result.folds[i].n_train > result.folds[i - 1].n_train
        # Aggregate metrics present for all 3 markets × 2 variants
        from bip.evaluation.tournaments.backtest.calibration_report import (
            MARKET_1X2,
            MARKET_BTTS,
            MARKET_OU_2_5,
        )
        for market in (MARKET_1X2, MARKET_BTTS, MARKET_OU_2_5):
            for variant in ("raw", "cal"):
                m = result.aggregate_metrics[(market, variant)]
                # Bootstrap CI bounds the mean
                assert m["ci_low"] <= m["brier"] <= m["ci_high"]
                # Brier in valid range
                assert 0.0 <= m["brier"] <= 1.0
                # ECE in valid range
                assert 0.0 <= m["ece"] <= 1.0


class TestBayesianPoissonPredictor:
    def test_first_match_uses_priors(self):
        """Cold-start: with both teams unseen, prediction uses cohort priors
        (λ_home ≈ λ_away ≈ prior_goals → balanced 1X2 with high draw)."""
        from scripts.backtest_int_tournaments import BacktestSnapshot

        predictor = BayesianPoissonPredictor(prior_goals=1.30)
        snap = BacktestSnapshot(
            match_id="m1",
            tournament="wc_2018",
            home_team_id=0,
            away_team_id=1,
            observed_1x2=1,
            observed_total_goals=0,
            observed_btts=0,
            observed_home_goals=0,
            observed_away_goals=0,
            match_date="2018-06-14",
        )
        pred = predictor.predict_fixture(snap)
        # Symmetric priors → balanced home/away win probability
        assert abs(pred.p_home_win - pred.p_away_win) < 0.02
        # Probabilities valid
        assert 0 <= pred.p_btts <= 1
        assert 0 <= pred.p_over_2_5 <= 1

    def test_state_advances_after_prediction(self):
        """Predicting + observing one fixture should leave state with
        non-zero observations for both teams."""
        from scripts.backtest_int_tournaments import BacktestSnapshot

        predictor = BayesianPoissonPredictor()
        snap = BacktestSnapshot(
            match_id="m1",
            tournament="wc_2018",
            home_team_id=0,
            away_team_id=1,
            observed_1x2=0,
            observed_total_goals=3,
            observed_btts=1,
            observed_home_goals=2,
            observed_away_goals=1,
            match_date="2018-06-14",
        )
        predictor.predict_fixture(snap)
        # State should reflect the observation
        assert predictor._state.team_states[0].n_matches_played == 1
        assert predictor._state.team_states[1].n_matches_played == 1
        # obs_weight_total scaled by competition weight (WC = 4×)
        assert predictor._state.team_states[0].obs_weight_total == 4.0

    def test_two_fixtures_chain_state(self):
        """Same team appearing in multiple fixtures accumulates observations."""
        from scripts.backtest_int_tournaments import BacktestSnapshot

        predictor = BayesianPoissonPredictor()
        snap1 = BacktestSnapshot(
            match_id="m1", tournament="wc_2018",
            home_team_id=0, away_team_id=1,
            observed_1x2=0, observed_total_goals=3, observed_btts=1,
            observed_home_goals=2, observed_away_goals=1,
            match_date="2018-06-14",
        )
        snap2 = BacktestSnapshot(
            match_id="m2", tournament="wc_2018",
            home_team_id=0, away_team_id=2,
            observed_1x2=2, observed_total_goals=1, observed_btts=0,
            observed_home_goals=0, observed_away_goals=1,
            match_date="2018-06-19",
        )
        predictor.predict_fixture(snap1)
        predictor.predict_fixture(snap2)
        # Team 0 played twice
        assert predictor._state.team_states[0].n_matches_played == 2
        # Team 1 played once, team 2 played once
        assert predictor._state.team_states[1].n_matches_played == 1
        assert predictor._state.team_states[2].n_matches_played == 1


# ── Layer-2: real-data integration ───────────────────────────────────────────


def _real_data_available() -> bool:
    return DEFAULT_OUTCOMES_PARQUET.exists()


@pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Run `uv run python scripts/seed_statsbomb_tournaments.py` first "
        "to seed the StatsBomb match-outcomes Parquet."
    ),
)
class TestRealBacktestLayer2:
    def test_full_backtest_emits_lock_decision(self, tmp_path):
        from scripts.run_phase5_backtest import run_phase5_backtest

        out = tmp_path / "lock_decision.json"
        decision, reports = run_phase5_backtest(output_path=out)

        # Pipeline ran end-to-end
        assert decision.calibration_status in (
            "pass",
            "marginal",
            "below-gate",
            "unreliable-bins",
        )
        assert decision.n_fixtures_with_predictions > 250
        assert out.exists()

        # Reports cover the expected markets (1X2 + BTTS + OU 2.5)
        markets = {r.market for r in reports}
        from bip.evaluation.tournaments.backtest.calibration_report import (
            MARKET_1X2,
            MARKET_BTTS,
            MARKET_OU_2_5,
        )
        assert MARKET_1X2 in markets
        assert MARKET_BTTS in markets
        assert MARKET_OU_2_5 in markets

    def test_lock_decision_is_loadable(self, tmp_path):
        from bip.evaluation.tournaments.backtest.lock_gate import LockDecision
        from scripts.run_phase5_backtest import run_phase5_backtest

        out = tmp_path / "lock_decision.json"
        run_phase5_backtest(output_path=out)
        loaded = LockDecision.from_json(out)
        assert loaded.held_out_tournaments == ("copa_2024", "euro_2024")

    def test_calibrated_backtest_emits_two_predictor_verdicts(self, tmp_path):
        """Phase 2 ∘ Phase 5: side-by-side raw vs calibrated."""
        from scripts.run_phase5_backtest import run_phase5_backtest_calibrated

        out = tmp_path / "lock_decision.json"
        decision, reports = run_phase5_backtest_calibrated(output_path=out)

        # Two predictors: raw + logit-calibrated, both with 3 markets each
        names = {v.predictor_name for v in decision.predictor_verdicts}
        assert "bayesian_independent_poisson_raw" in names
        assert "bayesian_independent_poisson_logit_calibrated" in names
        # Held-out scope: ~83 fixtures (Copa 2024 + Euro 2024)
        assert decision.n_fixtures_with_predictions > 50
        assert decision.n_fixtures_with_predictions < 100

    def test_calibrated_backtest_calibrator_helps_at_least_one_market(self, tmp_path):
        """The post-hoc calibrator should improve ECE on at least one
        market vs the raw baseline. (Football data + small test set
        means improvements are mixed; we only assert that the calibrator
        is not strictly worse on every market.)"""
        from scripts.run_phase5_backtest import run_phase5_backtest_calibrated

        decision, _ = run_phase5_backtest_calibrated(output_path=None)
        raw_v = next(
            v for v in decision.predictor_verdicts
            if v.predictor_name == "bayesian_independent_poisson_raw"
        )
        cal_v = next(
            v for v in decision.predictor_verdicts
            if v.predictor_name == "bayesian_independent_poisson_logit_calibrated"
        )
        # At least one market should have lower ECE after calibration
        improved = sum(
            1 for m in raw_v.market_classwise_ece
            if cal_v.market_classwise_ece.get(m, 1.0) < raw_v.market_classwise_ece[m]
        )
        assert improved >= 1, (
            f"Calibrator did not improve any market. raw={raw_v.market_classwise_ece}, "
            f"cal={cal_v.market_classwise_ece}"
        )
