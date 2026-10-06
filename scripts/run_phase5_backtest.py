"""Phase 5 Layer-2 backtest runner — real-data lock decision.

Walks the 6 men's StatsBomb tournaments chronologically, predicts each
fixture using a stateful Bayesian predictor (built on
``BayesianUpdater`` from Phase 3), scores against observed outcomes via
``CalibrationAudit``, and emits a single ``LockDecision`` artifact
consumable by Phase 6's lock JSON emitter.

The predictor is the simplest defensible reference implementation:
maintain per-team Poisson rates via the Phase 3 Bayesian updater,
predict the next match's goal distribution as the product of two
Poissons (independence assumption — Bivariate Poisson would be the
upgrade), derive 1X2 / BTTS / OU 2.5 / Corners O/U from the score grid.
The walk-forward respects causality: state is updated only AFTER
scoring each fixture.

This is the test the spike was built for. The output ``LockDecision``
either passes the hard gates (classwise-ECE ≤ 5%, Brier ≤ 0.21 / 0.20)
— locking real WC2026 predictions becomes safe — or fails honestly
with a per-predictor / per-market breakdown that operator can interrogate.

Run::

    uv run python scripts/run_phase5_backtest.py \\
        --output data/cache/lock_decision.json

Prerequisites: ``uv run python scripts/seed_statsbomb_tournaments.py``
(populates ``data/cache/statsbomb/match_outcomes.parquet``).

NOT invoked by the scheduler.
"""

from __future__ import annotations

# Allow running as `python scripts/run_phase5_backtest.py` (sibling-script imports)
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse  # noqa: E402
import math  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import date  # noqa: E402

import numpy as np  # noqa: E402

from bip.evaluation.tournaments.backtest.calibration_audit import (  # noqa: E402
    CalibrationAudit,
)
from bip.evaluation.tournaments.backtest.calibration_report import (  # noqa: E402
    MARKET_1X2,
    MARKET_BTTS,
    MARKET_OU_2_5,
    CalibrationReport,
)
from bip.evaluation.tournaments.backtest.lock_gate import LockDecision  # noqa: E402
from bip.evaluation.tournaments.calibration.logit_calibrator import (  # noqa: E402
    LogisticLogitCalibrator,
)
from bip.evaluation.tournaments.live.bayesian_updater import (  # noqa: E402
    DEFAULT_SIGMA_S_PER_DAY,
    BayesianUpdater,
    TournamentMatchResult,
)
from bip.evaluation.tournaments.live.competition_weights import (  # noqa: E402
    CompetitionType,
)
from bip.evaluation.tournaments.live.state import (  # noqa: E402
    DEFAULT_N_PRIOR,
    TeamLiveState,
    TournamentLiveState,
)
from bip.evaluation.tournaments.predictors.bivariate_poisson import (  # noqa: E402
    DEFAULT_RHO as BIVARIATE_DEFAULT_RHO,
)
from bip.evaluation.tournaments.predictors.bivariate_poisson import (  # noqa: E402
    _bivariate_poisson_grid,
)

# Default ρ search grid — covers the 0.04 (Dixon-Coles) to 0.32 region. The
# 2026-05-08 held-out sweep favored 0.20 with a known test-set contamination
# caveat; this train-time tuner is the de-contamination pass.
DEFAULT_RHO_GRID: tuple[float, ...] = (
    0.00, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32,
)

# Default α search grid for xG blending — 1.0 = goals-only (baseline,
# equivalent to BayesianBivariatePoissonPredictor), 0.0 = xG-only. The
# operator memory recommended α=0.35 (35% goals, 65% xG); we expand the
# grid to surface what training actually prefers.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (
    0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00,
)
from scripts.backtest_int_tournaments import (  # noqa: E402
    DEFAULT_HELD_OUT_TOURNAMENTS,
    BacktestSnapshot,
    FixturePrediction,
    aggregate_to_lock_decision,
    run_backtest_layer1,
)
from scripts.seed_statsbomb_tournaments import (  # noqa: E402
    DEFAULT_OUTCOMES_PARQUET as DEFAULT_STATSBOMB_PARQUET,
)
from scripts.seed_statsbomb_tournaments import (  # noqa: E402
    StatsBombMatchOutcome,
    load_outcomes_from_parquet,
)

DEFAULT_LOCK_OUTPUT = Path("data/cache/lock_decision.json")
MAX_GOALS_GRID = 8  # truncation point for Poisson grid (P(X≥9 | λ=2.5) ≈ 5e-6)

# Football-specific calibration audit defaults. Walsh & Joshi 2024 used
# n_bins=20 / min_bin_fill=0.8 on NBA — but international football
# probabilities concentrate in 0.40-0.60 (over-3 outcomes are rare;
# decisive results have probabilities clustered narrowly). Empirically
# verified on 314 matches: with n_bins=20 only 8-10 of 20 bins fill,
# triggering 'unreliable-bins' on every market regardless of actual
# calibration quality. n_bins=10 spreads the same probability mass
# across half the bins (each averaging more samples → less noise),
# and min_bin_fill=0.5 matches the realistic fill rate. Commit message
# c247b25 has the full sweep results.
FOOTBALL_DEFAULT_N_BINS = 10
FOOTBALL_DEFAULT_MIN_BIN_FILL = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot conversion: StatsBomb → BacktestSnapshot
# ─────────────────────────────────────────────────────────────────────────────


def make_snapshots(
    outcomes: list[StatsBombMatchOutcome],
) -> tuple[list[BacktestSnapshot], dict[str, int]]:
    """Convert StatsBomb outcomes to BacktestSnapshots (chronologically sorted).

    Returns ``(snapshots, team_id_map)`` where ``team_id_map`` is a
    deterministic name → int mapping built in insertion order. The same
    team appearing in multiple tournaments keeps the same id, so Bayesian
    state persists naturally across the walk-forward.
    """
    sorted_outcomes = sorted(outcomes, key=lambda o: o.match_date)
    team_id_map: dict[str, int] = {}

    def _id(name: str) -> int:
        if name not in team_id_map:
            team_id_map[name] = len(team_id_map)
        return team_id_map[name]

    snapshots = [
        BacktestSnapshot(
            match_id=str(o.match_id),
            tournament=o.tournament_slug,
            home_team_id=_id(o.home_team),
            away_team_id=_id(o.away_team),
            observed_1x2=o.outcome_1x2,
            observed_total_goals=o.total_goals,
            observed_btts=o.btts,
            observed_home_goals=o.home_goals,
            observed_away_goals=o.away_goals,
            observed_home_corners=o.home_corners,
            observed_away_corners=o.away_corners,
            observed_home_xg=o.home_xg,
            observed_away_xg=o.away_xg,
            match_date=o.match_date.isoformat(),
        )
        for o in sorted_outcomes
    ]
    return snapshots, team_id_map


# ─────────────────────────────────────────────────────────────────────────────
# BayesianPoissonPredictor — stateful, walk-forward
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BayesianPoissonPredictor:
    """Independent-Poisson predictor over Bayesian per-team rates.

    For each fixture, picks current λ_for/λ_against from the live state,
    composes match-level rates as ``(home.for + away.against) / 2`` and
    symmetric for away, builds a truncated Poisson product grid, and
    derives 1X2 / BTTS / OU 2.5 / Corners O/U. After predicting, observes
    the actual outcome and advances state via ``within_tournament_step``.
    """

    name: str = "bayesian_poisson_intl"
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY
    prior_goals: float = 1.30  # cohort baseline 2010-2024 international avg
    prior_corners: float = 5.0  # rough international avg per team per match
    n_prior: int = DEFAULT_N_PRIOR

    def __post_init__(self) -> None:
        self._state = TournamentLiveState(tournament_slug="phase5_backtest")
        self._updater = BayesianUpdater(sigma_s_per_day=self.sigma_s_per_day)
        self._registered: set[int] = set()
        self._last_match_date: date | None = None

    # ── team registration ────────────────────────────────────────────────

    def _ensure_team(self, team_id: int) -> None:
        if team_id in self._registered:
            return
        self._state.team_states[team_id] = TeamLiveState(
            team_id=team_id,
            prior_goals_for=self.prior_goals,
            prior_goals_against=self.prior_goals,
            prior_corners_for=self.prior_corners,
            prior_corners_against=self.prior_corners,
            prior_shots_for=12.0,
            prior_sot_for=4.0,
            n_prior=self.n_prior,
        )
        self._registered.add(team_id)

    # ── prediction ───────────────────────────────────────────────────────

    def predict_fixture(self, snap: BacktestSnapshot) -> FixturePrediction:
        self._ensure_team(snap.home_team_id)
        self._ensure_team(snap.away_team_id)

        # Apply between-window decay if we know the gap to the previous match
        if snap.match_date is not None:
            current = date.fromisoformat(snap.match_date)
            if self._last_match_date is not None:
                gap = (current - self._last_match_date).days
                if gap > 0:
                    self._updater.between_window_step(self._state, days_elapsed=gap)
            self._last_match_date = current

        home_state = self._state.team_states[snap.home_team_id]
        away_state = self._state.team_states[snap.away_team_id]

        # Match-level expected goals — symmetric averaging keeps the
        # match expectation at the cohort baseline when both teams are
        # average. (Premier-league-style calibrators with home-field
        # advantage would add a multiplicative term here; international
        # tournaments are predominantly neutral venues, so we omit HFA.)
        lam_home = max(
            (home_state.lambda_goals_for + away_state.lambda_goals_against) / 2.0,
            1e-3,
        )
        lam_away = max(
            (away_state.lambda_goals_for + home_state.lambda_goals_against) / 2.0,
            1e-3,
        )

        prediction = _poisson_score_grid_to_prediction(lam_home, lam_away)

        # Observe AFTER predicting — walk-forward causality.
        self._observe(snap)

        return prediction

    def _observe(self, snap: BacktestSnapshot) -> None:
        if snap.observed_home_goals is None or snap.observed_away_goals is None:
            return  # nothing to update with; predict-only mode
        result = TournamentMatchResult(
            match_id=snap.match_id,
            home_team_id=snap.home_team_id,
            away_team_id=snap.away_team_id,
            home_goals=snap.observed_home_goals,
            away_goals=snap.observed_away_goals,
            home_corners=snap.observed_home_corners,
            away_corners=snap.observed_away_corners,
            competition=_tournament_to_competition(snap.tournament),
        )
        self._updater.within_tournament_step(self._state, result)


def _tournament_to_competition(slug: str) -> CompetitionType:
    """Map StatsBomb tournament slug to CompetitionType for weighting.

    All 6 covered tournaments are top-tier confederations / WC, so the
    distinction is between ``WORLD_CUP`` (4×) and ``CONFEDERATION`` (3×).
    """
    if slug.startswith("wc_"):
        return CompetitionType.WORLD_CUP
    return CompetitionType.CONFEDERATION


# ─────────────────────────────────────────────────────────────────────────────
# BayesianBivariatePoissonPredictor — Karlis-Ntzoufras bivariate grid
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BayesianBivariatePoissonPredictor:
    """Bivariate-Poisson predictor over Bayesian per-team rates.

    Same state machinery as ``BayesianPoissonPredictor`` (Phase 3
    ``BayesianUpdater``), but the score grid is built with the Karlis &
    Ntzoufras (2003) Bivariate Poisson distribution — ``λ_3 = ρ · √(μ_h
    · μ_a)`` shared component captures positive home/away goal correlation.
    Independent Poisson is the special case ρ = 0; default ρ = 0.04
    (Dixon-Coles 1997 empirical estimate).

    Per SYNTHESIS Conclusion 2: Independent Poisson systematically
    underestimates draws (ΔG = 0). Bivariate Poisson with ρ > 0 corrects
    this — empirically 24-37% draw rate ceiling cited by HIGFormer is
    closer reachable. This predictor is the structural sharpness fix the
    Phase 5 Layer-2 verdict identified as the bottleneck (LogisticLogitCalibrator
    can re-shape but not improve discrimination).
    """

    name: str = "bayesian_bivariate_poisson_intl"
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY
    rho: float = BIVARIATE_DEFAULT_RHO  # 0.04 default (Dixon-Coles)
    prior_goals: float = 1.30
    prior_corners: float = 5.0
    n_prior: int = DEFAULT_N_PRIOR

    def __post_init__(self) -> None:
        if not -0.5 <= self.rho <= 0.5:
            raise ValueError(f"rho must be in [-0.5, 0.5], got {self.rho}")
        self._state = TournamentLiveState(tournament_slug="phase5_backtest_biv")
        self._updater = BayesianUpdater(sigma_s_per_day=self.sigma_s_per_day)
        self._registered: set[int] = set()
        self._last_match_date: date | None = None

    def _ensure_team(self, team_id: int) -> None:
        if team_id in self._registered:
            return
        self._state.team_states[team_id] = TeamLiveState(
            team_id=team_id,
            prior_goals_for=self.prior_goals,
            prior_goals_against=self.prior_goals,
            prior_corners_for=self.prior_corners,
            prior_corners_against=self.prior_corners,
            prior_shots_for=12.0,
            prior_sot_for=4.0,
            n_prior=self.n_prior,
        )
        self._registered.add(team_id)

    def predict_fixture(self, snap: BacktestSnapshot) -> FixturePrediction:
        self._ensure_team(snap.home_team_id)
        self._ensure_team(snap.away_team_id)

        if snap.match_date is not None:
            current = date.fromisoformat(snap.match_date)
            if self._last_match_date is not None:
                gap = (current - self._last_match_date).days
                if gap > 0:
                    self._updater.between_window_step(self._state, days_elapsed=gap)
            self._last_match_date = current

        home_state = self._state.team_states[snap.home_team_id]
        away_state = self._state.team_states[snap.away_team_id]

        mu_h = max(
            (home_state.lambda_goals_for + away_state.lambda_goals_against) / 2.0,
            1e-3,
        )
        mu_a = max(
            (away_state.lambda_goals_for + home_state.lambda_goals_against) / 2.0,
            1e-3,
        )

        # λ_3 = ρ · √(μ_h · μ_a) clamped so λ_1, λ_2 stay non-negative.
        lam12 = self.rho * math.sqrt(mu_h * mu_a)
        lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
        lam1 = mu_h - lam12
        lam2 = mu_a - lam12

        grid = _bivariate_poisson_grid(lam1, lam2, lam12, MAX_GOALS_GRID)
        prediction = _grid_to_prediction(grid)

        # Observe AFTER predicting — walk-forward causality.
        if snap.observed_home_goals is not None and snap.observed_away_goals is not None:
            result = TournamentMatchResult(
                match_id=snap.match_id,
                home_team_id=snap.home_team_id,
                away_team_id=snap.away_team_id,
                home_goals=snap.observed_home_goals,
                away_goals=snap.observed_away_goals,
                home_corners=snap.observed_home_corners,
                away_corners=snap.observed_away_corners,
                competition=_tournament_to_competition(snap.tournament),
            )
            self._updater.within_tournament_step(self._state, result)

        return prediction


# ─────────────────────────────────────────────────────────────────────────────
# BayesianBivariateXGPredictor — Bivariate Poisson with xG-blended rates
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BayesianBivariateXGPredictor:
    """Bivariate Poisson over an xG-blended Bayesian rate per team.

    Composes two ideas:

    1. **xG as forward-information signal** — a team that "should have
       scored 2.5 but only got 1" carries different next-match
       expectation than a team that scored 1 from 0.8 xG. The Bayesian
       updater accumulates a parallel xG rate alongside goals (see
       `TeamLiveState.lambda_xg_for/against`).
    2. **Bivariate Poisson over those blended rates** — same ρ-tunable
       joint distribution as `BayesianBivariatePoissonPredictor`,
       capturing positive home/away goal correlation Independent Poisson
       systematically underestimates.

    The match-level rate fed to the Bivariate grid is:

        μ_h = α · μ_h_goals + (1-α) · μ_h_xg
        μ_a = α · μ_a_goals + (1-α) · μ_a_xg

    where μ_*_goals = (home.lambda_goals_for + away.lambda_goals_against) / 2
    (existing pattern) and μ_*_xg the analogous xG blend.

    α = 1.0 reduces to goals-only (matches `BayesianBivariatePoissonPredictor`
    exactly when ρ matches). α = 0.0 is xG-only. Any α ∈ (0,1) is a convex
    combination — the Bayesian update on goals AND xG is preserved either
    way, so the state always reflects both observations.

    Cold start: xG priors default to goals priors (see
    `TeamLiveState.__post_init__`), so before any matches the prediction
    is independent of α.
    """

    name: str = "bayesian_bivariate_xg_intl"
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY
    rho: float = BIVARIATE_DEFAULT_RHO
    alpha: float = 0.50  # blend weight on goals; 1-alpha on xG
    prior_goals: float = 1.30
    prior_corners: float = 5.0
    n_prior: int = DEFAULT_N_PRIOR

    def __post_init__(self) -> None:
        if not -0.5 <= self.rho <= 0.5:
            raise ValueError(f"rho must be in [-0.5, 0.5], got {self.rho}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        self._state = TournamentLiveState(tournament_slug="phase5_backtest_xg")
        self._updater = BayesianUpdater(sigma_s_per_day=self.sigma_s_per_day)
        self._registered: set[int] = set()
        self._last_match_date: date | None = None

    def _ensure_team(self, team_id: int) -> None:
        if team_id in self._registered:
            return
        self._state.team_states[team_id] = TeamLiveState(
            team_id=team_id,
            prior_goals_for=self.prior_goals,
            prior_goals_against=self.prior_goals,
            prior_corners_for=self.prior_corners,
            prior_corners_against=self.prior_corners,
            prior_shots_for=12.0,
            prior_sot_for=4.0,
            n_prior=self.n_prior,
        )
        self._registered.add(team_id)

    def predict_fixture(self, snap: BacktestSnapshot) -> FixturePrediction:
        self._ensure_team(snap.home_team_id)
        self._ensure_team(snap.away_team_id)

        if snap.match_date is not None:
            current = date.fromisoformat(snap.match_date)
            if self._last_match_date is not None:
                gap = (current - self._last_match_date).days
                if gap > 0:
                    self._updater.between_window_step(self._state, days_elapsed=gap)
            self._last_match_date = current

        home_state = self._state.team_states[snap.home_team_id]
        away_state = self._state.team_states[snap.away_team_id]

        # Goals-only blend (existing pattern)
        mu_h_goals = (
            home_state.lambda_goals_for + away_state.lambda_goals_against
        ) / 2.0
        mu_a_goals = (
            away_state.lambda_goals_for + home_state.lambda_goals_against
        ) / 2.0

        # xG blend (parallel rate from Phase B updater)
        mu_h_xg = (
            home_state.lambda_xg_for + away_state.lambda_xg_against
        ) / 2.0
        mu_a_xg = (
            away_state.lambda_xg_for + home_state.lambda_xg_against
        ) / 2.0

        # Convex combination — α=1 is goals-only, α=0 is xG-only.
        mu_h = max(self.alpha * mu_h_goals + (1.0 - self.alpha) * mu_h_xg, 1e-3)
        mu_a = max(self.alpha * mu_a_goals + (1.0 - self.alpha) * mu_a_xg, 1e-3)

        # Bivariate Poisson grid (same as BayesianBivariatePoissonPredictor)
        lam12 = self.rho * math.sqrt(mu_h * mu_a)
        lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
        lam1 = mu_h - lam12
        lam2 = mu_a - lam12

        grid = _bivariate_poisson_grid(lam1, lam2, lam12, MAX_GOALS_GRID)
        prediction = _grid_to_prediction(grid)

        # Observe AFTER predicting — walk-forward causality. Pass xG so the
        # state's xG rate also updates.
        if snap.observed_home_goals is not None and snap.observed_away_goals is not None:
            result = TournamentMatchResult(
                match_id=snap.match_id,
                home_team_id=snap.home_team_id,
                away_team_id=snap.away_team_id,
                home_goals=snap.observed_home_goals,
                away_goals=snap.observed_away_goals,
                home_corners=snap.observed_home_corners,
                away_corners=snap.observed_away_corners,
                home_xg=snap.observed_home_xg,
                away_xg=snap.observed_away_xg,
                competition=_tournament_to_competition(snap.tournament),
            )
            self._updater.within_tournament_step(self._state, result)

        return prediction


def _grid_to_prediction(grid: np.ndarray) -> FixturePrediction:
    """Convert a 2D score-grid into FixturePrediction market probabilities.

    Shared logic between Independent and Bivariate Poisson predictors —
    both produce the same shape grid; only the construction differs.
    """
    total = float(grid.sum())
    if total > 0:
        grid = grid / total

    p_home = float(np.sum(np.tril(grid, k=-1)))
    p_draw = float(np.sum(np.diag(grid)))
    p_away = float(np.sum(np.triu(grid, k=1)))
    s = p_home + p_draw + p_away
    if s > 0:
        p_home /= s
        p_draw /= s
        p_away /= s

    p_h0 = float(grid[0, :].sum())
    p_a0 = float(grid[:, 0].sum())
    p_00 = float(grid[0, 0])
    p_btts = max(0.0, min(1.0, 1.0 - p_h0 - p_a0 + p_00))

    over25 = 0.0
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            if i + j >= 3:
                over25 += grid[i, j]
    p_over_2_5 = max(0.0, min(1.0, over25))

    return FixturePrediction(
        p_home_win=p_home,
        p_draw=p_draw,
        p_away_win=p_away,
        p_btts=p_btts,
        p_over_2_5=p_over_2_5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Poisson score grid → market probabilities
# ─────────────────────────────────────────────────────────────────────────────


def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k | λ) = λ^k · e^-λ / k!."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _poisson_score_grid_to_prediction(
    lam_home: float, lam_away: float, *, max_goals: int = MAX_GOALS_GRID
) -> FixturePrediction:
    """Build a truncated 2D Poisson product grid and derive market probs."""
    home_pmf = np.array([_poisson_pmf(k, lam_home) for k in range(max_goals + 1)])
    away_pmf = np.array([_poisson_pmf(k, lam_away) for k in range(max_goals + 1)])
    grid = np.outer(home_pmf, away_pmf)  # grid[i, j] = P(home=i, away=j)
    total = float(grid.sum())
    if total > 0:
        grid = grid / total

    # 1X2
    p_home = float(np.sum(np.tril(grid, k=-1)))  # i > j
    p_draw = float(np.sum(np.diag(grid)))
    p_away = float(np.sum(np.triu(grid, k=1)))
    s = p_home + p_draw + p_away
    if s > 0:
        p_home /= s
        p_draw /= s
        p_away /= s

    # BTTS = 1 - P(home=0) - P(away=0) + P(0,0)
    p_h0 = float(grid[0, :].sum())
    p_a0 = float(grid[:, 0].sum())
    p_00 = float(grid[0, 0])
    p_btts = max(0.0, min(1.0, 1.0 - p_h0 - p_a0 + p_00))

    # OU 2.5 — total ≥ 3
    over25 = 0.0
    n = grid.shape[0]
    for i in range(n):
        for j in range(n):
            if i + j >= 3:
                over25 += grid[i, j]
    p_over_2_5 = max(0.0, min(1.0, over25))

    return FixturePrediction(
        p_home_win=p_home,
        p_draw=p_draw,
        p_away_win=p_away,
        p_btts=p_btts,
        p_over_2_5=p_over_2_5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: load → predict → aggregate → emit
# ─────────────────────────────────────────────────────────────────────────────


def _make_predictor(
    predictor_kind: str,
    *,
    sigma_s_per_day: float,
    rho: float = BIVARIATE_DEFAULT_RHO,
    alpha: float = 0.50,
):
    """Instantiate the requested predictor by name.

    ``predictor_kind`` must be one of ``'independent'``, ``'bivariate'``,
    or ``'xg_blended'``. ``alpha`` is consumed only for xg_blended.
    """
    if predictor_kind == "independent":
        return BayesianPoissonPredictor(sigma_s_per_day=sigma_s_per_day)
    if predictor_kind == "bivariate":
        return BayesianBivariatePoissonPredictor(
            sigma_s_per_day=sigma_s_per_day, rho=rho,
        )
    if predictor_kind == "xg_blended":
        return BayesianBivariateXGPredictor(
            sigma_s_per_day=sigma_s_per_day, rho=rho, alpha=alpha,
        )
    raise ValueError(
        f"Unknown predictor_kind {predictor_kind!r}. "
        f"Valid: 'independent', 'bivariate', 'xg_blended'."
    )


def run_phase5_backtest(
    outcomes_parquet: Path = DEFAULT_STATSBOMB_PARQUET,
    *,
    held_out_tournaments: tuple[str, ...] = DEFAULT_HELD_OUT_TOURNAMENTS,
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY,
    output_path: Path | None = DEFAULT_LOCK_OUTPUT,
    git_sha: str | None = None,
    allow_below_gate: bool = False,
    n_bins: int | None = None,
    min_bin_fill: float | None = None,
    predictor_kind: str = "independent",
    rho: float = BIVARIATE_DEFAULT_RHO,
    alpha: float = 0.50,
) -> tuple[LockDecision, list[CalibrationReport]]:
    """Main entry: load StatsBomb outcomes, run backtest, emit LockDecision.

    Returns the LockDecision plus the list of CalibrationReports for
    transparency / further introspection. ``n_bins`` and ``min_bin_fill``
    default to football-specific values (10 / 0.5) — see the
    ``FOOTBALL_DEFAULT_*`` constants for rationale. Override to use
    Walsh & Joshi 2024's NBA defaults (20 / 0.8) if calibration is
    being checked on a non-football dataset.

    ``predictor_kind`` selects between Independent Poisson (default,
    backward-compatible) and Bivariate Poisson (corrects draw
    underestimation per SYNTHESIS Conclusion 2).
    """
    if n_bins is None:
        n_bins = FOOTBALL_DEFAULT_N_BINS
    if min_bin_fill is None:
        min_bin_fill = FOOTBALL_DEFAULT_MIN_BIN_FILL
    outcomes = load_outcomes_from_parquet(outcomes_parquet)
    if not outcomes:
        raise RuntimeError(
            f"No outcomes found in {outcomes_parquet}. "
            f"Run scripts/seed_statsbomb_tournaments.py first."
        )

    snapshots, _ = make_snapshots(outcomes)
    predictor = _make_predictor(
        predictor_kind, sigma_s_per_day=sigma_s_per_day, rho=rho, alpha=alpha,
    )

    reports = run_backtest_layer1(
        snapshots, [predictor],
        git_sha=git_sha,
        n_bins=n_bins,
        min_bin_fill=min_bin_fill,
    )
    decision = aggregate_to_lock_decision(
        reports,
        n_fixtures_total=len(snapshots),
        n_fixtures_with_predictions=len(snapshots),
        held_out_tournaments=held_out_tournaments,
        allow_below_gate=allow_below_gate,
        git_sha=git_sha,
        output_path=output_path,
    )
    return decision, reports


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated backtest (LogisticLogitCalibrator post-hoc)
# ─────────────────────────────────────────────────────────────────────────────


def run_phase5_backtest_calibrated(
    outcomes_parquet: Path = DEFAULT_STATSBOMB_PARQUET,
    *,
    held_out_tournaments: tuple[str, ...] = DEFAULT_HELD_OUT_TOURNAMENTS,
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY,
    output_path: Path | None = DEFAULT_LOCK_OUTPUT,
    git_sha: str | None = None,
    allow_below_gate: bool = False,
    n_bins: int | None = None,
    min_bin_fill: float | None = None,
    predictor_kind: str = "independent",
    rho: float = BIVARIATE_DEFAULT_RHO,
    alpha: float = 0.50,
) -> tuple[LockDecision, list[CalibrationReport]]:
    """Phase 2 ∘ Phase 5 — apply LogisticLogitCalibrator post-hoc.

    Two-pass pipeline:

    1. Run ``BayesianPoissonPredictor`` on every fixture chronologically
       (state advances naturally), collecting raw probabilities per market.
    2. Split predictions by tournament: training = all *not* in
       ``held_out_tournaments``; test = the held-out set.
    3. Fit ``LogisticLogitCalibrator`` per market on training raw probs +
       outcomes (1X2 multiclass, BTTS / OU 2.5 binary).
    4. Apply the fitted calibrators to test raw probs.
    5. Record TWO predictor verdicts on the test set — raw vs calibrated —
       so the operator can see the empirical lift (or its absence).

    The split-by-tournament approach respects causality: held-out
    fixtures (Copa 2024 + Euro 2024) are *only* scored, never used to
    fit the calibrator. Coverage in ``LockDecision`` reports the test-set
    fixture count, not the training fixtures.
    """
    if n_bins is None:
        n_bins = FOOTBALL_DEFAULT_N_BINS
    if min_bin_fill is None:
        min_bin_fill = FOOTBALL_DEFAULT_MIN_BIN_FILL

    outcomes = load_outcomes_from_parquet(outcomes_parquet)
    if not outcomes:
        raise RuntimeError(
            f"No outcomes found in {outcomes_parquet}. "
            f"Run scripts/seed_statsbomb_tournaments.py first."
        )
    snapshots, _ = make_snapshots(outcomes)
    predictor = _make_predictor(
        predictor_kind, sigma_s_per_day=sigma_s_per_day, rho=rho, alpha=alpha,
    )

    # ── Pass 1: predict every fixture, capture raw probs ───────────────
    raw_predictions: list[tuple[BacktestSnapshot, FixturePrediction]] = []
    for snap in snapshots:
        pred = predictor.predict_fixture(snap)
        raw_predictions.append((snap, pred))

    held_out = set(held_out_tournaments)
    train_pairs = [(s, p) for s, p in raw_predictions if s.tournament not in held_out]
    test_pairs = [(s, p) for s, p in raw_predictions if s.tournament in held_out]

    if not train_pairs or not test_pairs:
        raise RuntimeError(
            f"Empty train/test split (train={len(train_pairs)}, "
            f"test={len(test_pairs)}). Held-out: {held_out_tournaments}"
        )

    # ── Pass 2: fit per-market calibrators on training set ─────────────
    cal_1x2 = _fit_1x2_calibrator(train_pairs)
    cal_btts = _fit_binary_calibrator(
        [p.p_btts for _, p in train_pairs if p.p_btts is not None],
        [s.observed_btts for s, p in train_pairs if p.p_btts is not None],
    )
    cal_ou25 = _fit_binary_calibrator(
        [p.p_over_2_5 for _, p in train_pairs if p.p_over_2_5 is not None],
        [
            int(s.observed_total_goals > 2)
            for s, p in train_pairs if p.p_over_2_5 is not None
        ],
    )

    # ── Pass 3: score test set both raw + calibrated ───────────────────
    raw_name = f"bayesian_{predictor_kind}_poisson_raw"
    cal_name = f"bayesian_{predictor_kind}_poisson_logit_calibrated"
    raw_audit = CalibrationAudit(
        predictor_name=raw_name,
        n_bins=n_bins, min_bin_fill=min_bin_fill, git_sha=git_sha,
    )
    cal_audit = CalibrationAudit(
        predictor_name=cal_name,
        n_bins=n_bins, min_bin_fill=min_bin_fill, git_sha=git_sha,
    )

    for snap, pred in test_pairs:
        raw_probs_1x2 = [pred.p_home_win, pred.p_draw, pred.p_away_win]
        cal_probs_1x2 = (
            cal_1x2.transform(np.asarray([raw_probs_1x2]))[0].tolist()
            if cal_1x2 is not None else raw_probs_1x2
        )

        raw_audit.record_1x2(
            p_home=raw_probs_1x2[0], p_draw=raw_probs_1x2[1],
            p_away=raw_probs_1x2[2], outcome=snap.observed_1x2,
        )
        cal_audit.record_1x2(
            p_home=cal_probs_1x2[0], p_draw=cal_probs_1x2[1],
            p_away=cal_probs_1x2[2], outcome=snap.observed_1x2,
        )

        if pred.p_btts is not None:
            raw_audit.record_binary(
                MARKET_BTTS, p_yes=pred.p_btts, outcome=snap.observed_btts,
            )
            p_cal = (
                float(cal_btts.transform([pred.p_btts])[0])
                if cal_btts is not None else pred.p_btts
            )
            cal_audit.record_binary(
                MARKET_BTTS, p_yes=p_cal, outcome=snap.observed_btts,
            )

        if pred.p_over_2_5 is not None:
            obs_over = int(snap.observed_total_goals > 2)
            raw_audit.record_binary(
                MARKET_OU_2_5, p_yes=pred.p_over_2_5, outcome=obs_over,
            )
            p_cal = (
                float(cal_ou25.transform([pred.p_over_2_5])[0])
                if cal_ou25 is not None else pred.p_over_2_5
            )
            cal_audit.record_binary(
                MARKET_OU_2_5, p_yes=p_cal, outcome=obs_over,
            )

    reports = raw_audit.compute_reports() + cal_audit.compute_reports()

    decision = aggregate_to_lock_decision(
        reports,
        n_fixtures_total=len(test_pairs),
        n_fixtures_with_predictions=len(test_pairs),
        held_out_tournaments=held_out_tournaments,
        allow_below_gate=allow_below_gate,
        git_sha=git_sha,
        output_path=output_path,
    )
    return decision, reports


def _fit_1x2_calibrator(
    pairs: list[tuple[BacktestSnapshot, FixturePrediction]],
) -> LogisticLogitCalibrator | None:
    """Fit a 3-class calibrator on training 1X2 raw probs."""
    if not pairs:
        return None
    probs = np.asarray(
        [[p.p_home_win, p.p_draw, p.p_away_win] for _, p in pairs]
    )
    outcomes = np.asarray([s.observed_1x2 for s, _ in pairs])
    cal = LogisticLogitCalibrator()
    try:
        cal.fit(probs, outcomes)
    except Exception:
        return None
    return cal


def _fit_binary_calibrator(
    probs: list[float],
    outcomes: list[int],
) -> LogisticLogitCalibrator | None:
    """Fit a 2-class calibrator on training binary raw probs."""
    if not probs:
        return None
    cal = LogisticLogitCalibrator()
    try:
        cal.fit(np.asarray(probs), np.asarray(outcomes))
    except Exception:
        return None
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# Train-time ρ tuner (Bivariate Poisson)
# ─────────────────────────────────────────────────────────────────────────────


# Floor for log-probabilities so a single 0-probability cell can't dominate
# the average via -inf (e.g., extreme score above MAX_GOALS_GRID after
# truncation). 1e-12 ≈ -27.6 nats — ~10× worse than a genuine 9-goal upset.
_MIN_BIVARIATE_LOG_PROB = math.log(1e-12)


def _bivariate_log_pmf(
    home_goals: int,
    away_goals: int,
    mu_h: float,
    mu_a: float,
    rho: float,
    *,
    max_n: int = MAX_GOALS_GRID,
) -> float:
    """Joint log P(X=home_goals, Y=away_goals | μ_h, μ_a, ρ) — Bivariate Poisson.

    Required because the *marginals* of a Bivariate Poisson are still
    Poisson(μ), so scoring marginal log-PMFs is invariant to ρ. Only the
    joint cell carries ρ-dependent information. See Karlis & Ntzoufras
    (2003) §2.

    The decomposition mirrors ``BayesianBivariatePoissonPredictor``:
    ``λ_3 = ρ · √(μ_h · μ_a)`` clamped so ``λ_1, λ_2 ≥ 0``.
    """
    lam12 = rho * math.sqrt(mu_h * mu_a)
    lam12 = max(0.0, min(lam12, min(mu_h, mu_a) - 1e-9))
    lam1 = mu_h - lam12
    lam2 = mu_a - lam12

    grid = _bivariate_poisson_grid(lam1, lam2, lam12, max_n)
    h = min(home_goals, max_n)
    a = min(away_goals, max_n)
    p = float(grid[h, a])
    if p <= 0.0:
        return _MIN_BIVARIATE_LOG_PROB
    return math.log(p)


def tune_rho_walkforward(
    snapshots: list[BacktestSnapshot],
    rho_candidates: list[float] | tuple[float, ...] = DEFAULT_RHO_GRID,
    *,
    held_out_tournaments: tuple[str, ...] = DEFAULT_HELD_OUT_TOURNAMENTS,
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY,
    prior_goals: float = 1.30,
    prior_corners: float = 5.0,
    n_prior: int = DEFAULT_N_PRIOR,
) -> tuple[float, dict[float, float]]:
    """Held-criterion-C ρ sweep on training matches only — defuses overfit.

    Mirrors ``calibrate_sigma_s_walkforward`` in
    ``scripts/seed_international_history.py``: for each candidate ρ, builds
    a fresh ``TournamentLiveState`` over all teams seen in training, walks
    matches in chronological order, scores each fixture's observed
    (home_goals, away_goals) via the JOINT Bivariate Poisson log-PMF, and
    advances state via ``within_tournament_step`` with competition
    weighting. Optimal ρ maximizes average per-match log-likelihood.

    Snapshots whose tournament is in ``held_out_tournaments`` are excluded
    completely — the tuner never peeks at test data, which is the whole
    point of this pass over the 2026-05-08 held-out sweep.

    Returns ``(best_rho, scores)`` where ``scores[ρ]`` is the average
    per-match (NOT per-side) joint log-likelihood under that ρ.
    """
    if not rho_candidates:
        raise ValueError("rho_candidates must not be empty")
    if not snapshots:
        raise ValueError("snapshots must not be empty")

    held_out = set(held_out_tournaments)
    train = [s for s in snapshots if s.tournament not in held_out]
    if not train:
        raise RuntimeError(
            f"No train snapshots after filtering held_out={held_out_tournaments}. "
            f"All snapshots belong to held-out tournaments."
        )

    # Order matters — Held's criterion C is one-step-ahead predictive.
    train = sorted(train, key=lambda s: s.match_date or "")

    # Stable team_id assignment via insertion order; matches the
    # BayesianBivariatePoissonPredictor lazy-registration pattern.
    teams_seen: dict[int, bool] = {}
    for s in train:
        teams_seen[s.home_team_id] = True
        teams_seen[s.away_team_id] = True

    scores: dict[float, float] = {}
    for rho in rho_candidates:
        state = TournamentLiveState(tournament_slug="phase5_rho_tune")
        for tid in teams_seen:
            state.team_states[tid] = TeamLiveState(
                team_id=tid,
                prior_goals_for=prior_goals,
                prior_goals_against=prior_goals,
                prior_corners_for=prior_corners,
                prior_corners_against=prior_corners,
                prior_shots_for=12.0,
                prior_sot_for=4.0,
                n_prior=n_prior,
            )
        updater = BayesianUpdater(sigma_s_per_day=sigma_s_per_day)
        last_match_date: date | None = None

        log_lik_total = 0.0
        n_scored = 0

        for snap in train:
            if snap.match_date is not None:
                current = date.fromisoformat(snap.match_date)
                if last_match_date is not None:
                    gap = (current - last_match_date).days
                    if gap > 0:
                        updater.between_window_step(state, days_elapsed=gap)
                last_match_date = current

            home_state = state.team_states[snap.home_team_id]
            away_state = state.team_states[snap.away_team_id]
            mu_h = max(
                (home_state.lambda_goals_for + away_state.lambda_goals_against) / 2.0,
                1e-3,
            )
            mu_a = max(
                (away_state.lambda_goals_for + home_state.lambda_goals_against) / 2.0,
                1e-3,
            )

            if snap.observed_home_goals is None or snap.observed_away_goals is None:
                continue

            log_lik_total += _bivariate_log_pmf(
                snap.observed_home_goals,
                snap.observed_away_goals,
                mu_h, mu_a, rho,
            )
            n_scored += 1

            # Update state for next iteration — preserves walk-forward causality.
            result = TournamentMatchResult(
                match_id=snap.match_id,
                home_team_id=snap.home_team_id,
                away_team_id=snap.away_team_id,
                home_goals=snap.observed_home_goals,
                away_goals=snap.observed_away_goals,
                home_corners=snap.observed_home_corners,
                away_corners=snap.observed_away_corners,
                competition=_tournament_to_competition(snap.tournament),
            )
            updater.within_tournament_step(state, result)

        scores[rho] = log_lik_total / n_scored if n_scored else float("-inf")

    best_rho = max(scores, key=lambda r: scores[r])
    return best_rho, scores


# ─────────────────────────────────────────────────────────────────────────────
# Train-time α tuner (xG-blend weight)
# ─────────────────────────────────────────────────────────────────────────────


def tune_alpha_walkforward(
    snapshots: list[BacktestSnapshot],
    alpha_candidates: list[float] | tuple[float, ...] = DEFAULT_ALPHA_GRID,
    *,
    rho: float = BIVARIATE_DEFAULT_RHO,
    held_out_tournaments: tuple[str, ...] = DEFAULT_HELD_OUT_TOURNAMENTS,
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY,
    prior_goals: float = 1.30,
    prior_corners: float = 5.0,
    n_prior: int = DEFAULT_N_PRIOR,
) -> tuple[float, dict[float, float]]:
    """Walk-forward α sweep — train-time only, no held-out peek.

    Same Held-criterion-C pattern as ``tune_rho_walkforward``: for each
    candidate α ∈ [0, 1], walks the training tournaments chronologically
    with a fresh ``BayesianBivariateXGPredictor(rho=rho, alpha=α)``,
    scores observed (home_goals, away_goals) via JOINT Bivariate Poisson
    log-PMF, returns argmax.

    The xG signal enters via the predictor's blended μ; α=1 collapses to
    goals-only Bivariate (matches `BayesianBivariatePoissonPredictor`),
    α=0 is xG-only. The Bayesian state always updates BOTH goals and xG
    rates regardless of α — α only controls how predictions blend the two.

    Snapshots without xG are silently skipped during scoring (they still
    update goals state). For the StatsBomb cache as of 2026-05-09 every
    match has xG, so this is a defensive guard.
    """
    if not alpha_candidates:
        raise ValueError("alpha_candidates must not be empty")
    if not snapshots:
        raise ValueError("snapshots must not be empty")
    for a in alpha_candidates:
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"alpha_candidates must be in [0, 1], got {a}")

    held_out = set(held_out_tournaments)
    train = [s for s in snapshots if s.tournament not in held_out]
    if not train:
        raise RuntimeError(
            f"No train snapshots after filtering held_out={held_out_tournaments}"
        )
    train = sorted(train, key=lambda s: s.match_date or "")

    scores: dict[float, float] = {}
    for alpha in alpha_candidates:
        predictor = BayesianBivariateXGPredictor(
            sigma_s_per_day=sigma_s_per_day,
            rho=rho, alpha=alpha,
            prior_goals=prior_goals, prior_corners=prior_corners,
            n_prior=n_prior,
        )
        log_lik_total = 0.0
        n_scored = 0

        for snap in train:
            # We need the (μ_h, μ_a) the predictor WOULD use, before it
            # observes. The predictor's `predict_fixture` advances state
            # internally, so we replicate the read-then-update dance to
            # capture the pre-observation rates for joint log-PMF scoring.
            predictor._ensure_team(snap.home_team_id)
            predictor._ensure_team(snap.away_team_id)

            if snap.match_date is not None:
                current = date.fromisoformat(snap.match_date)
                if predictor._last_match_date is not None:
                    gap = (current - predictor._last_match_date).days
                    if gap > 0:
                        predictor._updater.between_window_step(
                            predictor._state, days_elapsed=gap,
                        )
                predictor._last_match_date = current

            home_state = predictor._state.team_states[snap.home_team_id]
            away_state = predictor._state.team_states[snap.away_team_id]

            mu_h_goals = (
                home_state.lambda_goals_for + away_state.lambda_goals_against
            ) / 2.0
            mu_a_goals = (
                away_state.lambda_goals_for + home_state.lambda_goals_against
            ) / 2.0
            mu_h_xg = (
                home_state.lambda_xg_for + away_state.lambda_xg_against
            ) / 2.0
            mu_a_xg = (
                away_state.lambda_xg_for + home_state.lambda_xg_against
            ) / 2.0
            mu_h = max(alpha * mu_h_goals + (1.0 - alpha) * mu_h_xg, 1e-3)
            mu_a = max(alpha * mu_a_goals + (1.0 - alpha) * mu_a_xg, 1e-3)

            if (
                snap.observed_home_goals is None
                or snap.observed_away_goals is None
            ):
                continue

            log_lik_total += _bivariate_log_pmf(
                snap.observed_home_goals,
                snap.observed_away_goals,
                mu_h, mu_a, rho,
            )
            n_scored += 1

            # Update state for next iteration (preserves walk-forward)
            result = TournamentMatchResult(
                match_id=snap.match_id,
                home_team_id=snap.home_team_id,
                away_team_id=snap.away_team_id,
                home_goals=snap.observed_home_goals,
                away_goals=snap.observed_away_goals,
                home_corners=snap.observed_home_corners,
                away_corners=snap.observed_away_corners,
                home_xg=snap.observed_home_xg,
                away_xg=snap.observed_away_xg,
                competition=_tournament_to_competition(snap.tournament),
            )
            predictor._updater.within_tournament_step(predictor._state, result)

        scores[alpha] = log_lik_total / n_scored if n_scored else float("-inf")

    best_alpha = max(scores, key=lambda a: scores[a])
    return best_alpha, scores


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-origin cross-validation (proper time-series evaluation)
# ─────────────────────────────────────────────────────────────────────────────


# Default folds — chronologically valid: each fold trains on tournaments
# that ended BEFORE the held-out tournament started. Random K-fold would
# break causality (a 2024 match training a 2018 prediction). Hyndman &
# Athanasopoulos (2018) call this rolling-origin evaluation; it is the
# only valid CV scheme for autoregressive/walk-forward predictors.
#
# Tournament chronology (StatsBomb cache as of 2026-05-09):
#   WC 2018  → Jun-Jul 2018
#   Euro 2020→ Jun-Jul 2021 (delayed)
#   WC 2022  → Nov-Dec 2022
#   AFCON23  → Jan-Feb 2024
#   Copa 24  → Jun-Jul 2024
#   Euro 24  → Jun-Jul 2024
#
# Each fold's training set has at least 179 matches (>3× n_prior=10), enough
# for the LogisticLogitCalibrator's MLE to converge stably.
DEFAULT_ROLLING_ORIGIN_FOLDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("wc_2018", "euro_2020", "wc_2022"), "afcon_2023"),
    (("wc_2018", "euro_2020", "wc_2022", "afcon_2023"), "copa_2024"),
    (("wc_2018", "euro_2020", "wc_2022", "afcon_2023", "copa_2024"), "euro_2024"),
)


@dataclass(frozen=True)
class FoldMetrics:
    """Per-fold calibration metrics for one predictor variant (raw or calibrated)."""

    market: str
    ece: float
    brier: float
    n_samples: int


@dataclass(frozen=True)
class FoldResult:
    """Outcome of one rolling-origin fold."""

    fold_index: int
    train_tournaments: tuple[str, ...]
    held_out_tournament: str
    n_train: int
    n_test: int
    raw_metrics: tuple[FoldMetrics, ...]
    calibrated_metrics: tuple[FoldMetrics, ...]


@dataclass(frozen=True)
class RollingOriginCVResult:
    """Aggregate result across all folds, with bootstrap CIs."""

    folds: tuple[FoldResult, ...]
    n_total_test: int
    # Aggregate metrics — keyed by (market, variant) where variant is 'raw' or 'cal'.
    aggregate_metrics: dict[tuple[str, str], dict[str, float]]


def _bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap mean + 95% CI for an i.i.d. score sample.

    Returns ``(mean, ci_low, ci_high)``. For Brier scores (one per match)
    this is exactly what we want — Brier is i.i.d. across matches under
    the standard calibration assumption.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    alpha = (1 - confidence) / 2
    return (
        float(values.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def _brier_per_sample_1x2(
    p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray, outcome: np.ndarray,
) -> np.ndarray:
    """Per-match 3-class sum-Brier divided by K=3 to align with the gate scale."""
    one_hot = np.zeros((len(outcome), 3))
    one_hot[np.arange(len(outcome)), outcome] = 1.0
    probs = np.column_stack([p_home, p_draw, p_away])
    return np.sum((probs - one_hot) ** 2, axis=1) / 3.0


def _brier_per_sample_binary(p_yes: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    """Per-match binary Brier (already in [0,1]; matches gate scale directly)."""
    return (p_yes - outcome.astype(float)) ** 2


def run_rolling_origin_cv(
    snapshots: list[BacktestSnapshot],
    folds_definition: tuple[tuple[tuple[str, ...], str], ...] = DEFAULT_ROLLING_ORIGIN_FOLDS,
    *,
    sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY,
    rho: float = BIVARIATE_DEFAULT_RHO,
    alpha: float = 0.50,
    predictor_kind: str = "xg_blended",
    n_bins: int | None = None,
    min_bin_fill: float | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> RollingOriginCVResult:
    """Rolling-origin CV: per-fold walk-forward + calibrator + aggregate metrics.

    For each fold (train_slugs, held_out_slug):

    1. Filter snapshots to those in train_slugs ∪ {held_out_slug}, sorted
       chronologically.
    2. Build a fresh predictor; walk-forward predicts EVERY snapshot
       (state advances naturally by chronology — train and held-out
       interleave seamlessly because the predictor doesn't know which is
       which until we partition by tournament_slug afterwards).
    3. Split predictions: train_pairs = those in train_slugs; test_pairs
       = those in held_out_slug.
    4. Fit ``LogisticLogitCalibrator`` on train_pairs (one per market).
    5. Score test_pairs both raw and calibrated → per-match Brier vectors.
    6. Compute fold-level ECE on test_pairs.

    After all folds, concatenate per-match Brier vectors across folds
    (each match scored exactly once across all folds) and compute
    aggregate Brier mean + bootstrap 95% CI. Aggregate ECE is the
    sample-weighted average of fold ECEs (proxy for global ECE on
    aggregated probabilities — exact would require recomputing on the
    full pooled vector).
    """
    if n_bins is None:
        n_bins = FOOTBALL_DEFAULT_N_BINS
    if min_bin_fill is None:
        min_bin_fill = FOOTBALL_DEFAULT_MIN_BIN_FILL

    folds: list[FoldResult] = []

    # Per-match Brier accumulators across all folds (each match exactly once)
    raw_brier_1x2: list[np.ndarray] = []
    cal_brier_1x2: list[np.ndarray] = []
    raw_brier_btts: list[np.ndarray] = []
    cal_brier_btts: list[np.ndarray] = []
    raw_brier_ou25: list[np.ndarray] = []
    cal_brier_ou25: list[np.ndarray] = []

    # Per-fold ECE values (sample-weighted aggregation)
    raw_ece_1x2: list[tuple[float, int]] = []
    cal_ece_1x2: list[tuple[float, int]] = []
    raw_ece_btts: list[tuple[float, int]] = []
    cal_ece_btts: list[tuple[float, int]] = []
    raw_ece_ou25: list[tuple[float, int]] = []
    cal_ece_ou25: list[tuple[float, int]] = []

    for fold_idx, (train_slugs, held_out_slug) in enumerate(folds_definition):
        relevant_slugs = set(train_slugs) | {held_out_slug}
        fold_snaps = sorted(
            (s for s in snapshots if s.tournament in relevant_slugs),
            key=lambda s: s.match_date or "",
        )

        predictor = _make_predictor(
            predictor_kind, sigma_s_per_day=sigma_s_per_day,
            rho=rho, alpha=alpha,
        )

        raw_predictions: list[tuple[BacktestSnapshot, FixturePrediction]] = []
        for snap in fold_snaps:
            pred = predictor.predict_fixture(snap)
            raw_predictions.append((snap, pred))

        train_pairs = [
            (s, p) for s, p in raw_predictions if s.tournament in train_slugs
        ]
        test_pairs = [
            (s, p) for s, p in raw_predictions if s.tournament == held_out_slug
        ]

        if not train_pairs or not test_pairs:
            raise RuntimeError(
                f"Fold {fold_idx}: empty split (train={len(train_pairs)}, "
                f"test={len(test_pairs)})"
            )

        # Fit calibrators on training set
        cal_1x2 = _fit_1x2_calibrator(train_pairs)
        cal_btts = _fit_binary_calibrator(
            [p.p_btts for _, p in train_pairs if p.p_btts is not None],
            [s.observed_btts for s, p in train_pairs if p.p_btts is not None],
        )
        cal_ou25 = _fit_binary_calibrator(
            [p.p_over_2_5 for _, p in train_pairs if p.p_over_2_5 is not None],
            [
                int(s.observed_total_goals > 2)
                for s, p in train_pairs if p.p_over_2_5 is not None
            ],
        )

        # Score test set raw + calibrated
        raw_audit = CalibrationAudit(
            predictor_name=f"fold{fold_idx}_raw",
            n_bins=n_bins, min_bin_fill=min_bin_fill,
        )
        cal_audit = CalibrationAudit(
            predictor_name=f"fold{fold_idx}_cal",
            n_bins=n_bins, min_bin_fill=min_bin_fill,
        )

        # Per-match Brier vectors for bootstrap
        p_home_raw, p_draw_raw, p_away_raw = [], [], []
        p_home_cal, p_draw_cal, p_away_cal = [], [], []
        outcomes_1x2 = []
        p_btts_raw, p_btts_cal, outcomes_btts = [], [], []
        p_ou_raw, p_ou_cal, outcomes_ou = [], [], []

        for snap, pred in test_pairs:
            raw_1x2 = [pred.p_home_win, pred.p_draw, pred.p_away_win]
            cal_1x2_p = (
                cal_1x2.transform(np.asarray([raw_1x2]))[0].tolist()
                if cal_1x2 is not None else raw_1x2
            )
            raw_audit.record_1x2(
                p_home=raw_1x2[0], p_draw=raw_1x2[1], p_away=raw_1x2[2],
                outcome=snap.observed_1x2,
            )
            cal_audit.record_1x2(
                p_home=cal_1x2_p[0], p_draw=cal_1x2_p[1], p_away=cal_1x2_p[2],
                outcome=snap.observed_1x2,
            )
            p_home_raw.append(raw_1x2[0])
            p_draw_raw.append(raw_1x2[1])
            p_away_raw.append(raw_1x2[2])
            p_home_cal.append(cal_1x2_p[0])
            p_draw_cal.append(cal_1x2_p[1])
            p_away_cal.append(cal_1x2_p[2])
            outcomes_1x2.append(snap.observed_1x2)

            if pred.p_btts is not None:
                p_btts_cal_v = (
                    float(cal_btts.transform([pred.p_btts])[0])
                    if cal_btts is not None else pred.p_btts
                )
                raw_audit.record_binary(
                    MARKET_BTTS, p_yes=pred.p_btts, outcome=snap.observed_btts,
                )
                cal_audit.record_binary(
                    MARKET_BTTS, p_yes=p_btts_cal_v, outcome=snap.observed_btts,
                )
                p_btts_raw.append(pred.p_btts)
                p_btts_cal.append(p_btts_cal_v)
                outcomes_btts.append(snap.observed_btts)

            if pred.p_over_2_5 is not None:
                obs_over = int(snap.observed_total_goals > 2)
                p_ou_cal_v = (
                    float(cal_ou25.transform([pred.p_over_2_5])[0])
                    if cal_ou25 is not None else pred.p_over_2_5
                )
                raw_audit.record_binary(
                    MARKET_OU_2_5, p_yes=pred.p_over_2_5, outcome=obs_over,
                )
                cal_audit.record_binary(
                    MARKET_OU_2_5, p_yes=p_ou_cal_v, outcome=obs_over,
                )
                p_ou_raw.append(pred.p_over_2_5)
                p_ou_cal.append(p_ou_cal_v)
                outcomes_ou.append(obs_over)

        # Compute per-match Brier vectors
        outcome_arr = np.asarray(outcomes_1x2)
        b_raw_1x2 = _brier_per_sample_1x2(
            np.asarray(p_home_raw), np.asarray(p_draw_raw),
            np.asarray(p_away_raw), outcome_arr,
        )
        b_cal_1x2 = _brier_per_sample_1x2(
            np.asarray(p_home_cal), np.asarray(p_draw_cal),
            np.asarray(p_away_cal), outcome_arr,
        )
        raw_brier_1x2.append(b_raw_1x2)
        cal_brier_1x2.append(b_cal_1x2)

        if outcomes_btts:
            o_btts = np.asarray(outcomes_btts)
            raw_brier_btts.append(
                _brier_per_sample_binary(np.asarray(p_btts_raw), o_btts)
            )
            cal_brier_btts.append(
                _brier_per_sample_binary(np.asarray(p_btts_cal), o_btts)
            )

        if outcomes_ou:
            o_ou = np.asarray(outcomes_ou)
            raw_brier_ou25.append(
                _brier_per_sample_binary(np.asarray(p_ou_raw), o_ou)
            )
            cal_brier_ou25.append(
                _brier_per_sample_binary(np.asarray(p_ou_cal), o_ou)
            )

        # Pull fold-level ECE from the audit reports
        raw_reports = {r.market: r for r in raw_audit.compute_reports()}
        cal_reports = {r.market: r for r in cal_audit.compute_reports()}

        def _ece_for(reports_dict: dict, market: str) -> float:
            return float(reports_dict[market].classwise_ece) if market in reports_dict else 0.0

        n_test = len(test_pairs)
        raw_ece_1x2.append((_ece_for(raw_reports, MARKET_1X2), n_test))
        cal_ece_1x2.append((_ece_for(cal_reports, MARKET_1X2), n_test))
        raw_ece_btts.append((_ece_for(raw_reports, MARKET_BTTS), len(outcomes_btts)))
        cal_ece_btts.append((_ece_for(cal_reports, MARKET_BTTS), len(outcomes_btts)))
        raw_ece_ou25.append((_ece_for(raw_reports, MARKET_OU_2_5), len(outcomes_ou)))
        cal_ece_ou25.append((_ece_for(cal_reports, MARKET_OU_2_5), len(outcomes_ou)))

        folds.append(FoldResult(
            fold_index=fold_idx,
            train_tournaments=tuple(train_slugs),
            held_out_tournament=held_out_slug,
            n_train=len(train_pairs),
            n_test=n_test,
            raw_metrics=tuple([
                FoldMetrics(MARKET_1X2, _ece_for(raw_reports, MARKET_1X2),
                            float(b_raw_1x2.mean()), n_test),
                FoldMetrics(MARKET_BTTS, _ece_for(raw_reports, MARKET_BTTS),
                            float(raw_brier_btts[-1].mean()) if outcomes_btts else 0.0,
                            len(outcomes_btts)),
                FoldMetrics(MARKET_OU_2_5, _ece_for(raw_reports, MARKET_OU_2_5),
                            float(raw_brier_ou25[-1].mean()) if outcomes_ou else 0.0,
                            len(outcomes_ou)),
            ]),
            calibrated_metrics=tuple([
                FoldMetrics(MARKET_1X2, _ece_for(cal_reports, MARKET_1X2),
                            float(b_cal_1x2.mean()), n_test),
                FoldMetrics(MARKET_BTTS, _ece_for(cal_reports, MARKET_BTTS),
                            float(cal_brier_btts[-1].mean()) if outcomes_btts else 0.0,
                            len(outcomes_btts)),
                FoldMetrics(MARKET_OU_2_5, _ece_for(cal_reports, MARKET_OU_2_5),
                            float(cal_brier_ou25[-1].mean()) if outcomes_ou else 0.0,
                            len(outcomes_ou)),
            ]),
        ))

    # Aggregate Brier across folds via concatenation (each match once)
    def _agg_brier(brier_list: list[np.ndarray]) -> dict[str, float]:
        if not brier_list:
            return {"brier": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        all_b = np.concatenate(brier_list)
        mean, lo, hi = _bootstrap_ci(all_b, n_bootstrap=n_bootstrap, seed=seed)
        return {"brier": mean, "ci_low": lo, "ci_high": hi, "n": int(len(all_b))}

    # Aggregate ECE via sample-weighted average of per-fold ECEs
    def _agg_ece(ece_list: list[tuple[float, int]]) -> float:
        total_n = sum(n for _, n in ece_list)
        if total_n == 0:
            return 0.0
        return sum(e * n for e, n in ece_list) / total_n

    aggregate: dict[tuple[str, str], dict[str, float]] = {}
    aggregate[(MARKET_1X2, "raw")] = {**_agg_brier(raw_brier_1x2), "ece": _agg_ece(raw_ece_1x2)}
    aggregate[(MARKET_1X2, "cal")] = {**_agg_brier(cal_brier_1x2), "ece": _agg_ece(cal_ece_1x2)}
    aggregate[(MARKET_BTTS, "raw")] = {**_agg_brier(raw_brier_btts), "ece": _agg_ece(raw_ece_btts)}
    aggregate[(MARKET_BTTS, "cal")] = {**_agg_brier(cal_brier_btts), "ece": _agg_ece(cal_ece_btts)}
    aggregate[(MARKET_OU_2_5, "raw")] = {**_agg_brier(raw_brier_ou25), "ece": _agg_ece(raw_ece_ou25)}
    aggregate[(MARKET_OU_2_5, "cal")] = {**_agg_brier(cal_brier_ou25), "ece": _agg_ece(cal_ece_ou25)}

    n_total_test = sum(f.n_test for f in folds)
    return RollingOriginCVResult(
        folds=tuple(folds),
        n_total_test=n_total_test,
        aggregate_metrics=aggregate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outcomes-parquet", type=Path, default=DEFAULT_STATSBOMB_PARQUET,
        help="StatsBomb match-outcomes Parquet (from seed_statsbomb_tournaments.py)",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_LOCK_OUTPUT)
    p.add_argument("--sigma-s", type=float, default=DEFAULT_SIGMA_S_PER_DAY)
    p.add_argument("--allow-below-gate", action="store_true")
    p.add_argument(
        "--n-bins", type=int, default=None,
        help="Override calibration audit n_bins (default 20). Football "
        "probabilities concentrate in 0.40-0.60; n_bins=10 is more honest.",
    )
    p.add_argument(
        "--min-bin-fill", type=float, default=None,
        help="Override calibration audit min_bin_fill (default 0.80). "
        "Walsh & Joshi 2024 used 0.8 for NBA; football needs ~0.5.",
    )
    p.add_argument(
        "--apply-calibrator", action="store_true",
        help="Apply LogisticLogitCalibrator post-hoc (Phase 2). Trains "
        "per-market calibrators on non-held-out tournaments and applies "
        "them to held-out (Copa 2024 + Euro 2024). Emits side-by-side "
        "raw vs calibrated verdicts in the LockDecision.",
    )
    p.add_argument(
        "--predictor", default="independent",
        choices=("independent", "bivariate", "xg_blended"),
        help="Goal-distribution predictor. 'independent' = Independent "
        "Poisson (default, backward-compat). 'bivariate' = Bivariate "
        "Poisson with shared λ_3 correlation (corrects draw "
        "underestimation per SYNTHESIS Conclusion 2). 'xg_blended' = "
        "Bivariate Poisson over an α-weighted blend of goals-rate and "
        "xG-rate per team — adds forward-information signal that goals "
        "alone miss (Phase 5 Layer-2 Option 2).",
    )
    p.add_argument(
        "--rho", type=float, default=BIVARIATE_DEFAULT_RHO,
        help="Correlation parameter for Bivariate Poisson. Dixon-Coles "
        "1997 estimate is 0.04. Range [-0.5, 0.5]; ρ=0 collapses to "
        "Independent Poisson. Consumed by --predictor=bivariate or "
        "--predictor=xg_blended. Ignored when --auto-tune-rho is set.",
    )
    p.add_argument(
        "--auto-tune-rho", action="store_true",
        help="Sweep ρ on training tournaments only (excludes held-out via "
        "DEFAULT_HELD_OUT_TOURNAMENTS), pick the ρ that maximizes joint "
        "Bivariate Poisson log-likelihood, and use that ρ for the held-out "
        "backtest. Defuses the test-set-contamination caveat from the "
        "2026-05-08 held-out sweep. Forces --predictor=bivariate when no "
        "other --predictor is set.",
    )
    p.add_argument(
        "--rho-grid", type=float, nargs="+", default=None,
        help="Custom ρ candidates (space-separated). Defaults to "
        f"{list(DEFAULT_RHO_GRID)}.",
    )
    p.add_argument(
        "--alpha", type=float, default=0.50,
        help="xG-blend weight for --predictor=xg_blended. α=1 → goals-only "
        "(equivalent to --predictor=bivariate); α=0 → xG-only; "
        "α∈(0,1) is convex blend. Ignored when --auto-tune-alpha is set.",
    )
    p.add_argument(
        "--auto-tune-alpha", action="store_true",
        help="Sweep α on training tournaments only, pick the α that "
        "maximizes joint Bivariate Poisson log-likelihood, and use that α "
        "for the held-out backtest. Forces --predictor=xg_blended.",
    )
    p.add_argument(
        "--alpha-grid", type=float, nargs="+", default=None,
        help="Custom α candidates. Defaults to "
        f"{list(DEFAULT_ALPHA_GRID)}.",
    )
    p.add_argument(
        "--rolling-origin-cv", action="store_true",
        help="Run rolling-origin cross-validation across 3 chronologically-"
        "valid folds (Hyndman & Athanasopoulos 2018). Each fold trains a "
        "calibrator on tournaments that finished BEFORE the held-out — "
        "valid for walk-forward predictors. Reports aggregate Brier with "
        "bootstrap 95% CI plus per-fold breakdown. Replaces the single "
        "held-out backtest path.",
    )
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap iterations for Brier CIs (default 1000). Only "
        "consumed with --rolling-origin-cv.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"[phase5] loading outcomes from {args.outcomes_parquet}", file=sys.stderr)

    rho = args.rho
    alpha = args.alpha
    predictor_kind = args.predictor

    snapshots = None  # Lazy-load if any tuner activates

    if args.auto_tune_rho:
        # Force at least bivariate; preserve xg_blended if operator combined.
        if predictor_kind == "independent":
            predictor_kind = "bivariate"
        outcomes = load_outcomes_from_parquet(args.outcomes_parquet)
        if not outcomes:
            raise RuntimeError(
                f"No outcomes found in {args.outcomes_parquet}. "
                f"Run scripts/seed_statsbomb_tournaments.py first."
            )
        snapshots, _ = make_snapshots(outcomes)
        grid = list(args.rho_grid) if args.rho_grid is not None else list(DEFAULT_RHO_GRID)

        print(f"[phase5] tuning ρ on training tournaments (excluding "
              f"{', '.join(DEFAULT_HELD_OUT_TOURNAMENTS)})", file=sys.stderr)
        print(f"[phase5] candidate ρ grid: {grid}", file=sys.stderr)

        best_rho, scores = tune_rho_walkforward(
            snapshots, grid, sigma_s_per_day=args.sigma_s,
        )

        print()
        print("=" * 72)
        print(f"ρ TUNING — train-time avg joint log-likelihood per match")
        print("=" * 72)
        for rho_cand in sorted(scores):
            marker = " ←" if rho_cand == best_rho else ""
            print(f"  ρ = {rho_cand:.3f}   loglik = {scores[rho_cand]:.6f}{marker}")
        print()
        print(f"best ρ_train = {best_rho:.3f}")
        print()

        rho = best_rho

    if args.auto_tune_alpha:
        predictor_kind = "xg_blended"
        if snapshots is None:
            outcomes = load_outcomes_from_parquet(args.outcomes_parquet)
            if not outcomes:
                raise RuntimeError(
                    f"No outcomes found in {args.outcomes_parquet}. "
                    f"Run scripts/seed_statsbomb_tournaments.py first."
                )
            snapshots, _ = make_snapshots(outcomes)
        a_grid = list(args.alpha_grid) if args.alpha_grid is not None else list(DEFAULT_ALPHA_GRID)

        print(f"[phase5] tuning α on training tournaments at ρ={rho:.3f} "
              f"(excluding {', '.join(DEFAULT_HELD_OUT_TOURNAMENTS)})",
              file=sys.stderr)
        print(f"[phase5] candidate α grid: {a_grid}", file=sys.stderr)

        best_alpha, a_scores = tune_alpha_walkforward(
            snapshots, a_grid, rho=rho, sigma_s_per_day=args.sigma_s,
        )

        print()
        print("=" * 72)
        print(f"α TUNING — train-time avg joint log-likelihood per match (ρ={rho:.3f})")
        print("=" * 72)
        for a_cand in sorted(a_scores):
            marker = " ←" if a_cand == best_alpha else ""
            print(f"  α = {a_cand:.2f}   loglik = {a_scores[a_cand]:.6f}{marker}")
        print()
        print(f"best α_train = {best_alpha:.2f}  (1 = goals-only, 0 = xG-only)")
        print()

        alpha = best_alpha

    if args.rolling_origin_cv:
        if snapshots is None:
            outcomes = load_outcomes_from_parquet(args.outcomes_parquet)
            if not outcomes:
                raise RuntimeError(
                    f"No outcomes found in {args.outcomes_parquet}."
                )
            snapshots, _ = make_snapshots(outcomes)

        print(f"[phase5] running rolling-origin CV (3 folds, "
              f"predictor={predictor_kind}, ρ={rho:.3f}, α={alpha:.2f})",
              file=sys.stderr)

        cv_result = run_rolling_origin_cv(
            snapshots,
            sigma_s_per_day=args.sigma_s,
            rho=rho, alpha=alpha,
            predictor_kind=predictor_kind,
            n_bins=args.n_bins,
            min_bin_fill=args.min_bin_fill,
            n_bootstrap=args.n_bootstrap,
        )

        print()
        print("=" * 78)
        print("ROLLING-ORIGIN CROSS-VALIDATION — proper time-series evaluation")
        print("=" * 78)
        print(f"Predictor: {predictor_kind} (ρ={rho:.3f}, α={alpha:.2f})")
        print(f"Total held-out matches across folds: {cv_result.n_total_test}")
        print()
        print("Per-fold breakdown:")
        for f in cv_result.folds:
            print(f"\n  Fold {f.fold_index + 1}: train={'+'.join(f.train_tournaments)} "
                  f"({f.n_train} matches) → held={f.held_out_tournament} ({f.n_test})")
            for raw, cal in zip(f.raw_metrics, f.calibrated_metrics):
                print(f"    {raw.market:30}  raw  ECE={raw.ece:.4f} Brier={raw.brier:.4f} "
                      f"|  cal  ECE={cal.ece:.4f} Brier={cal.brier:.4f}")

        print()
        print("=" * 78)
        print(f"AGGREGATE METRICS (n={cv_result.n_total_test} held-out, bootstrap 95% CI)")
        print("=" * 78)
        for market in (MARKET_1X2, MARKET_BTTS, MARKET_OU_2_5):
            for variant in ("raw", "cal"):
                m = cv_result.aggregate_metrics[(market, variant)]
                gate_brier = 0.21 if market == MARKET_1X2 else 0.20
                inside = "✓ within gate" if m["ci_high"] <= gate_brier else (
                    "≈ at gate" if m["brier"] <= gate_brier else "✗ above gate"
                )
                print(
                    f"  {market:30} {variant:4}  "
                    f"ECE={m['ece']:.4f}  "
                    f"Brier={m['brier']:.4f}  "
                    f"95% CI=[{m['ci_low']:.4f}, {m['ci_high']:.4f}]  "
                    f"({inside})"
                )

        # Don't write a LockDecision JSON — the rolling-origin verdict is
        # advisory; the lock decision still uses the spike-approved
        # held-out (Copa+Euro 2024). Operator interprets the CV result
        # alongside that for the final R-08 call.
        print()
        return

    runner = (
        run_phase5_backtest_calibrated if args.apply_calibrator
        else run_phase5_backtest
    )
    decision, reports = runner(
        args.outcomes_parquet,
        sigma_s_per_day=args.sigma_s,
        output_path=args.output,
        allow_below_gate=args.allow_below_gate,
        n_bins=args.n_bins,
        min_bin_fill=args.min_bin_fill,
        predictor_kind=predictor_kind,
        rho=rho,
        alpha=alpha,
    )

    print()
    print("=" * 72)
    print(f"PHASE 5 BACKTEST — calibration_status: {decision.calibration_status}")
    print("=" * 72)
    if predictor_kind == "bivariate":
        rho_tag = " tuned" if args.auto_tune_rho else ""
        print(f"Predictor: bivariate (ρ = {rho:.3f}{rho_tag})")
    elif predictor_kind == "xg_blended":
        rho_tag = " tuned" if args.auto_tune_rho else ""
        a_tag = " tuned" if args.auto_tune_alpha else ""
        print(
            f"Predictor: xg_blended (ρ = {rho:.3f}{rho_tag}, "
            f"α = {alpha:.2f}{a_tag})"
        )
    else:
        print(f"Predictor: {predictor_kind}")
    print(f"Fixtures scored: {decision.n_fixtures_with_predictions}")
    print(f"Coverage: {decision.coverage_pct:.1%}")
    print(f"Held-out tournaments: {', '.join(decision.held_out_tournaments)}")
    print(f"passes_lock = {decision.passes_lock}")
    print()
    for v in decision.predictor_verdicts:
        print(f"  predictor: {v.predictor_name}")
        print(f"    overall: {v.overall_status}")
        for market, status in sorted(v.market_statuses.items()):
            ece = v.market_classwise_ece[market]
            brier = v.market_brier_for_gate[market]
            print(
                f"    {market:30} {status:15} "
                f"ECE={ece:.4f} Brier={brier:.4f}"
            )
    print()
    if decision.operator_overrides:
        print("Operator overrides:")
        for note in decision.operator_overrides:
            print(f"  - {note}")
    if args.output:
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
