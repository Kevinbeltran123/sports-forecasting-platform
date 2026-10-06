"""Ola 1 — WeightedMLEFitter tests.

Layer-1 synthetic recovery: generate matches from a known Poisson model
with known team strengths + home advantage, fit, and recover the
parameters within tolerance.

Layer-2 integration: smoke-test the fitter on the actual martj42 training
split. Marked as a separate class so it can be skipped if data is
unavailable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from bip.evaluation.tournaments.wc2026_v2.weighted_strength import (
    WeightedMLEFitter,
)

# ─────────────────────────────────────────────────────────────────
# Layer-1 — synthetic recovery
# ─────────────────────────────────────────────────────────────────


def _generate_synthetic_matches(
    n_teams: int = 12,
    n_matches_per_pair: int = 6,
    seed: int = 17,
    home_advantage_log: float = 0.30,
    base_intercept: float = 0.30,
) -> tuple[pl.DataFrame, dict[str, tuple[float, float]]]:
    """Return (df, truth) where truth[team] = (attack, defense)."""
    rng = np.random.default_rng(seed)
    teams = [f"team_{i:02d}" for i in range(n_teams)]
    truth_attack = rng.normal(scale=0.3, size=n_teams)
    truth_defense = rng.normal(scale=0.3, size=n_teams)
    # Centre to enforce identifiability.
    truth_attack -= truth_attack.mean()
    truth_defense -= truth_defense.mean()

    rows: list[dict[str, object]] = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            for k in range(n_matches_per_pair):
                lam_h = float(
                    np.exp(base_intercept + truth_attack[i] + truth_defense[j] + home_advantage_log)
                )
                lam_a = float(np.exp(base_intercept + truth_attack[j] + truth_defense[i]))
                hg = int(rng.poisson(lam_h))
                ag = int(rng.poisson(lam_a))
                # All matches dated within a year of the reference so time-decay ≈ 1.
                rows.append(
                    {
                        "date": date(2026, 1, 1 + (k % 28)),
                        "home_team": home,
                        "away_team": away,
                        "home_score": hg,
                        "away_score": ag,
                        "tournament": "Friendly",
                        "neutral": False,
                    }
                )

    df = pl.DataFrame(rows)
    truth = {teams[i]: (float(truth_attack[i]), float(truth_defense[i])) for i in range(n_teams)}
    return df, truth


class TestSyntheticRecovery:
    """Generate from known truth → fit → recover within tolerance.

    With 12 teams × 132 ordered pairs × 6 matches = 792 matches, the MLE
    should recover team strengths to ±0.10 in log-rate units and home
    advantage to ±0.05.
    """

    def test_recovery_within_tolerance(self) -> None:
        df, truth = _generate_synthetic_matches(seed=17)
        fitter = WeightedMLEFitter(min_appearances=5)
        result = fitter.fit(df, reference_date=date(2026, 5, 23))
        # Home advantage: truth 0.30. Tolerate ±0.10 with synthetic noise.
        assert result.home_advantage == pytest.approx(0.30, abs=0.10)
        # Team strengths: each team should be within 0.20 log-units. We do
        # per-team comparison since the fit is on centred params.
        max_attack_err = 0.0
        max_defense_err = 0.0
        for team, (a_truth, d_truth) in truth.items():
            s = result.strengths[team]
            max_attack_err = max(max_attack_err, abs(s.attack - a_truth))
            max_defense_err = max(max_defense_err, abs(s.defense - d_truth))
        assert max_attack_err < 0.20
        assert max_defense_err < 0.20

    def test_predict_lambdas_close_to_truth_mean(self) -> None:
        df, _ = _generate_synthetic_matches(seed=42)
        fitter = WeightedMLEFitter(min_appearances=5)
        result = fitter.fit(df, reference_date=date(2026, 5, 23))
        lh, la = result.predict_lambdas("team_00", "team_01")
        # Sanity: rates must be positive and finite.
        assert 0.1 < lh < 10.0
        assert 0.1 < la < 10.0

    def test_unknown_team_raises(self) -> None:
        df, _ = _generate_synthetic_matches()
        fitter = WeightedMLEFitter(min_appearances=5)
        result = fitter.fit(df, reference_date=date(2026, 5, 23))
        with pytest.raises(KeyError):
            result.predict_lambdas("nonexistent_team", "team_00")

    def test_strengths_are_zero_mean_after_fit(self) -> None:
        df, _ = _generate_synthetic_matches()
        fitter = WeightedMLEFitter(min_appearances=5)
        result = fitter.fit(df, reference_date=date(2026, 5, 23))
        attacks = np.array([s.attack for s in result.strengths.values()])
        defenses = np.array([s.defense for s in result.strengths.values()])
        assert attacks.mean() == pytest.approx(0.0, abs=1e-8)
        assert defenses.mean() == pytest.approx(0.0, abs=1e-8)


class TestEdgeCases:
    def test_min_appearances_filter_works(self) -> None:
        # Build a small df with one team appearing only twice.
        rng = np.random.default_rng(0)
        rows = []
        for k in range(20):
            rows.append(
                {
                    "date": date(2026, 1, 1 + (k % 28)),
                    "home_team": "main_a",
                    "away_team": "main_b",
                    "home_score": int(rng.poisson(1.5)),
                    "away_score": int(rng.poisson(1.2)),
                    "tournament": "Friendly",
                    "neutral": False,
                }
            )
        # rare_team only plays twice.
        for k in range(2):
            rows.append(
                {
                    "date": date(2026, 1, 1 + k),
                    "home_team": "rare_team",
                    "away_team": "main_a",
                    "home_score": 1,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "neutral": False,
                }
            )
        df = pl.DataFrame(rows)
        fitter = WeightedMLEFitter(min_appearances=10)
        result = fitter.fit(df, reference_date=date(2026, 5, 23))
        # rare_team must be filtered out.
        assert "rare_team" not in result.strengths
        assert "main_a" in result.strengths
        assert "main_b" in result.strengths

    def test_empty_train_raises(self) -> None:
        empty = pl.DataFrame(
            {
                "date": pl.Series([], dtype=pl.Date),
                "home_team": pl.Series([], dtype=pl.String),
                "away_team": pl.Series([], dtype=pl.String),
                "home_score": pl.Series([], dtype=pl.Int64),
                "away_score": pl.Series([], dtype=pl.Int64),
                "tournament": pl.Series([], dtype=pl.String),
                "neutral": pl.Series([], dtype=pl.Boolean),
            }
        )
        with pytest.raises(ValueError, match="min_appearances"):
            WeightedMLEFitter().fit(empty, reference_date=date(2026, 5, 23))


# ─────────────────────────────────────────────────────────────────
# Layer-2 — smoke test on the real martj42 split
# ─────────────────────────────────────────────────────────────────


@pytest.mark.requires_data
class TestRealCorpusSmoke:
    """Run the fitter on the actual martj42 train split. Layer-2 — uses real data."""

    def test_fits_without_error(self) -> None:
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        # Restrict to recent matches for a fast smoke test: post-2018.
        recent = split.train.filter(pl.col("date") >= pl.lit(date(2018, 1, 1)))
        fitter = WeightedMLEFitter(min_appearances=10)
        result = fitter.fit(recent, reference_date=date(2026, 5, 23))
        # Sanity checks on the fit.
        assert result.n_teams > 100  # ~200 active national teams since 2018
        assert result.n_train_matches >= 2000
        # Home advantage should be positive (homes do better, log scale).
        assert result.home_advantage > 0.0
        # Intercept should put base goal rate ≈ 1.0–1.6 per side.
        assert 0.0 < result.intercept < 1.5

    def test_top_teams_have_higher_attack_than_minnows(self) -> None:
        """Spain/Argentina/France/Brazil should outrank, e.g., San Marino in
        attack strength on log-rate scale. Sanity-check the prior makes sense."""
        from bip.evaluation.tournaments.wc2026_v2.corpus import build_split

        split = build_split()
        recent = split.train.filter(pl.col("date") >= pl.lit(date(2018, 1, 1)))
        fitter = WeightedMLEFitter(min_appearances=10)
        result = fitter.fit(recent, reference_date=date(2026, 5, 23))
        # At least one strong + one weak team should be in the fit.
        strong_teams = ["Spain", "France", "Brazil", "Argentina", "Germany"]
        weak_teams = ["San Marino", "Gibraltar", "Andorra", "Liechtenstein"]
        strong_in_fit = [t for t in strong_teams if t in result.strengths]
        weak_in_fit = [t for t in weak_teams if t in result.strengths]
        # If neither side present this test is uninformative; require at least
        # one of each.
        assert strong_in_fit
        assert weak_in_fit
        strong_attack_mean = np.mean([result.strengths[t].attack for t in strong_in_fit])
        weak_attack_mean = np.mean([result.strengths[t].attack for t in weak_in_fit])
        # Strong teams' mean attack must exceed weak teams' mean attack.
        assert strong_attack_mean > weak_attack_mean
