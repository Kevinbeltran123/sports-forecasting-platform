"""Weighted-MLE team strength prior — Dixon-Coles parameterisation.

Implements Ley, Van de Wiele & Van Eetvelde (2019), §2.3:

    log λ_h = α_attack[home] + β_defense[away] + γ_home
    log λ_a = α_attack[away] + β_defense[home]

fit by maximum likelihood with per-match weights from
``match_importance.compute_match_weight``. The likelihood treats home and
away goals as independent Poisson conditioned on (α, β, γ); correlation is
restored downstream by the Bivariate / DIBP predictor (Ola 3).

Identifiability: α and β are shifted so that ``mean(α) = mean(β) = 0`` after
fit. γ then absorbs the global home advantage on log-rate scale.

Why a custom MLE and not ``sklearn.PoissonRegressor``: ``PoissonRegressor``
requires building a 2N+1-wide dense design matrix; with 49 058 train rows
and ~250 teams that's a 98 116 × 501 matrix (~50 MB float64) for what is
fundamentally a sparse problem. Direct gradient-based MLE on the natural
parameterisation is ~3× faster and ~50× less memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from scipy.optimize import minimize

from .match_importance import DEFAULT_HALF_LIFE_DAYS, compute_match_weight


@dataclass(frozen=True)
class TeamStrength:
    """Fitted attack / defense parameters for one team (log-rate scale)."""

    team: str
    attack: float
    defense: float


@dataclass(frozen=True)
class WeightedMLEResult:
    """Output of WeightedMLEFitter.fit().

    - ``strengths``: dict team_name → TeamStrength (mean-zero after fit)
    - ``home_advantage``: γ in the parameterisation above (log-rate units)
    - ``intercept``: global log-mean goal rate; useful for sanity checks
    - ``reference_date``: as supplied; matches with date > reference are
      clamped to weight 1.0 in fit
    - ``n_train_matches``: rows actually used (≥ ``min_appearances``-filtered)
    - ``n_teams``: distinct teams in the strengths table
    """

    strengths: dict[str, TeamStrength]
    home_advantage: float
    intercept: float
    reference_date: date
    n_train_matches: int
    n_teams: int

    def predict_lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Predict (λ_home, λ_away) for a prospective fixture.

        Raises KeyError if either team is unknown — callers should fall
        back to a cold-start cohort prior in that case (out of scope for
        this module).
        """

        h = self.strengths[home_team]
        a = self.strengths[away_team]
        log_lambda_h = self.intercept + h.attack + a.defense + self.home_advantage
        log_lambda_a = self.intercept + a.attack + h.defense
        return float(np.exp(log_lambda_h)), float(np.exp(log_lambda_a))


def _build_team_index(df: pl.DataFrame, min_appearances: int) -> dict[str, int]:
    """Return team → index map for teams with ≥ min_appearances matches.

    A team needs both home and away rows; we union home_team + away_team
    counts and threshold on the total.
    """

    counts = (
        pl.concat(
            [
                df.select(pl.col("home_team").alias("team")),
                df.select(pl.col("away_team").alias("team")),
            ]
        )
        .group_by("team")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") >= min_appearances)
        .sort("team")
    )
    return {row["team"]: i for i, row in enumerate(counts.iter_rows(named=True))}


class WeightedMLEFitter:
    """Fit team attack/defense + home advantage via weighted Poisson MLE.

    Usage:
        fitter = WeightedMLEFitter()
        result = fitter.fit(martj42_train_df, reference_date=date(2026, 5, 23))
        lh, la = result.predict_lambdas("Spain", "France")
    """

    def __init__(
        self,
        min_appearances: int = 10,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        max_iter: int = 1000,
        tol: float = 1e-6,
        ridge: float = 1e-4,
    ) -> None:
        self.min_appearances = min_appearances
        self.half_life_days = half_life_days
        self.max_iter = max_iter
        self.tol = tol
        # Tiny ridge on attack/defense to kill the (intercept, mean(attack),
        # mean(defense)) shift degeneracy. Without it L-BFGS-B sees a flat
        # direction and converges slowly / not at all on rank-deficient runs.
        self.ridge = ridge

    def fit(self, df: pl.DataFrame, reference_date: date) -> WeightedMLEResult:
        team_index = _build_team_index(df, self.min_appearances)
        if not team_index:
            raise ValueError("No teams cleared min_appearances threshold; check the input frame.")
        n_teams = len(team_index)

        filtered = df.filter(
            pl.col("home_team").is_in(list(team_index.keys()))
            & pl.col("away_team").is_in(list(team_index.keys()))
        )

        weights = np.array(
            [
                compute_match_weight(
                    row["date"], row["tournament"], reference_date, self.half_life_days
                )
                for row in filtered.iter_rows(named=True)
            ],
            dtype=np.float64,
        )
        # Defensive: drop zero-weight rows (e.g. very ancient + low-importance).
        valid = weights > 0
        filtered = filtered.filter(pl.Series(values=valid))
        weights = weights[valid]

        home_idx = np.array(
            [team_index[t] for t in filtered["home_team"].to_list()], dtype=np.int64
        )
        away_idx = np.array(
            [team_index[t] for t in filtered["away_team"].to_list()], dtype=np.int64
        )
        home_goals = filtered["home_score"].to_numpy().astype(np.float64)
        away_goals = filtered["away_score"].to_numpy().astype(np.float64)

        # Parameter vector layout:
        #   [intercept, gamma_home, attack_0..n-1, defense_0..n-1]
        # Total length: 2 + 2n.
        n_params = 2 + 2 * n_teams

        def unpack(theta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
            intercept = float(theta[0])
            gamma = float(theta[1])
            attack = theta[2 : 2 + n_teams]
            defense = theta[2 + n_teams :]
            return intercept, gamma, attack, defense

        def neg_log_likelihood(theta: np.ndarray) -> float:
            intercept, gamma, attack, defense = unpack(theta)
            log_lambda_h = intercept + attack[home_idx] + defense[away_idx] + gamma
            log_lambda_a = intercept + attack[away_idx] + defense[home_idx]
            lambda_h = np.exp(log_lambda_h)
            lambda_a = np.exp(log_lambda_a)
            # Weighted Poisson log-likelihood (constant -log(k!) terms dropped;
            # they're irrelevant to the optimum). Ridge penalty kills the
            # shift degeneracy between intercept and mean(attack)/mean(defense).
            ll = weights * (
                home_goals * log_lambda_h - lambda_h + away_goals * log_lambda_a - lambda_a
            )
            penalty = 0.5 * self.ridge * float((attack * attack).sum() + (defense * defense).sum())
            return -float(ll.sum()) + penalty

        def grad(theta: np.ndarray) -> np.ndarray:
            intercept, gamma, attack, defense = unpack(theta)
            log_lambda_h = intercept + attack[home_idx] + defense[away_idx] + gamma
            log_lambda_a = intercept + attack[away_idx] + defense[home_idx]
            lambda_h = np.exp(log_lambda_h)
            lambda_a = np.exp(log_lambda_a)
            r_h = weights * (home_goals - lambda_h)
            r_a = weights * (away_goals - lambda_a)
            g = np.zeros(n_params, dtype=np.float64)
            g[0] = -float(r_h.sum() + r_a.sum())  # intercept
            g[1] = -float(r_h.sum())  # gamma_home (only home log_lambda has γ)
            # Attack: home rows contribute to attack[home_idx]; away rows contribute
            # to attack[away_idx].
            np.add.at(g, 2 + home_idx, -r_h)
            np.add.at(g, 2 + away_idx, -r_a)
            # Defense: home rows feed defense[away_idx]; away rows feed
            # defense[home_idx].
            np.add.at(g, 2 + n_teams + away_idx, -r_h)
            np.add.at(g, 2 + n_teams + home_idx, -r_a)
            # Ridge gradient on attack/defense (skips intercept and gamma).
            g[2 : 2 + n_teams] += self.ridge * attack
            g[2 + n_teams :] += self.ridge * defense
            return g

        # Initial guess: intercept = log(mean goals), zeros elsewhere.
        mean_goals = float((home_goals.sum() + away_goals.sum()) / (2 * len(home_idx)))
        theta0 = np.zeros(n_params, dtype=np.float64)
        theta0[0] = np.log(max(mean_goals, 0.5))

        result = minimize(
            neg_log_likelihood,
            theta0,
            jac=grad,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        if not result.success:
            raise RuntimeError(f"WeightedMLEFitter convergence failed: {result.message}")

        intercept, gamma, attack, defense = unpack(result.x)
        # Centre attack and defense so identifiability is explicit.
        attack -= attack.mean()
        defense -= defense.mean()

        index_to_team = {v: k for k, v in team_index.items()}
        strengths = {
            index_to_team[i]: TeamStrength(
                team=index_to_team[i], attack=float(attack[i]), defense=float(defense[i])
            )
            for i in range(n_teams)
        }

        return WeightedMLEResult(
            strengths=strengths,
            home_advantage=gamma,
            intercept=intercept,
            reference_date=reference_date,
            n_train_matches=int(weights.size),
            n_teams=n_teams,
        )
