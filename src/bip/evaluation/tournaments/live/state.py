"""In-tournament state: per-team Bayesian λ estimates + player availability.

The state starts from pre-tournament priors (qualifying baselines) and
updates after each completed match. Properties compute the current blended
estimate on the fly — no need to store them separately.

Phase 3 (2026-05-15) extends per-team state with **explicit prior variance**
per metric. This unlocks the Held & Vollnhals (2005) random walk in
``BayesianUpdater.between_window_step``: variance grows by ``Δt · σ_s² · μ²``
between FIFA windows so that long gaps shrink the effective n_prior.

Backward compatibility: states saved before Phase 3 (no ``prior_var_*`` or
``obs_weight_total`` keys) load unchanged — ``__post_init__`` fills the
missing fields from the previous semantics so the existing tests keep passing.

JSON round-trip is supported via .save() / TournamentLiveState.load().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Effective sample size assigned to the pre-tournament prior.
# A value of 10 means: the prior is worth 10 tournament matches.
# After 3 group-stage matches, the tournament data has 30% weight.
DEFAULT_N_PRIOR = 10


@dataclass
class TeamLiveState:
    """Per-team mutable state updated after each tournament match."""

    team_id: int

    # Pre-tournament priors — set once at initialization, never mutated.
    prior_goals_for: float
    prior_goals_against: float
    prior_corners_for: float
    prior_corners_against: float
    prior_shots_for: float
    prior_sot_for: float
    # xG priors (Phase 5 Layer-2 Option-2). Default initialised to the
    # goals priors when None — the Gamma-Poisson updater on xG keeps a
    # second rate per team, blended with goals at predict time. Backward-
    # compat: states saved before xG-tracking deserialize fine because
    # `__post_init__` derives missing fields from goals priors.
    prior_xg_for: float | None = None
    prior_xg_against: float | None = None

    # Explicit Bayesian variance per prior (Phase 3 — Held random walk).
    # Default = mean / n_prior, which makes the Bayesian blend equivalent
    # to the pre-Phase-3 (n_prior, n_obs) weighted average. Variance grows
    # in BayesianUpdater.between_window_step proportional to elapsed days.
    # ``None`` defaults are filled by __post_init__ to keep callers terse
    # and JSON load backward-compatible.
    prior_var_goals_for: float | None = None
    prior_var_goals_against: float | None = None
    prior_var_corners_for: float | None = None
    prior_var_corners_against: float | None = None
    prior_var_shots_for: float | None = None
    prior_var_sot_for: float | None = None
    prior_var_xg_for: float | None = None
    prior_var_xg_against: float | None = None

    # Running sums of tournament observations (add to after each match).
    obs_goals_for: float = 0.0
    obs_goals_against: float = 0.0
    obs_corners_for: float = 0.0
    obs_corners_against: float = 0.0
    obs_shots_for: float = 0.0
    obs_sot_for: float = 0.0
    obs_xg_for: float = 0.0
    obs_xg_against: float = 0.0
    n_matches_played: int = 0

    # Sum of competition_weight over completed matches. With default weight
    # 1.0 this equals n_matches_played and the existing blend formula is
    # unchanged. With WC=4× weight, one WC match adds 4 units of evidence
    # to the blend while n_matches_played still increments by 1.
    obs_weight_total: float = 0.0

    # How many "equivalent matches" the prior is worth (legacy field; the
    # property ``n_prior_eff_*`` derives a metric-specific effective sample
    # size from the current variance, which is what the blend actually uses).
    n_prior: int = DEFAULT_N_PRIOR

    # Player availability (api_football_ids as int).
    injured_player_ids: list[int] = field(default_factory=list)
    suspended_player_ids: list[int] = field(default_factory=list)
    yellow_cards: dict[int, int] = field(default_factory=dict)  # id → count

    def __post_init__(self) -> None:
        # Initialize variances from priors so Bayesian blend matches pre-Phase-3
        # behaviour: n_prior_eff = mean / var = mean / (mean/n_prior) = n_prior.
        if self.prior_var_goals_for is None:
            self.prior_var_goals_for = self._init_var(self.prior_goals_for)
        if self.prior_var_goals_against is None:
            self.prior_var_goals_against = self._init_var(self.prior_goals_against)
        if self.prior_var_corners_for is None:
            self.prior_var_corners_for = self._init_var(self.prior_corners_for)
        if self.prior_var_corners_against is None:
            self.prior_var_corners_against = self._init_var(self.prior_corners_against)
        if self.prior_var_shots_for is None:
            self.prior_var_shots_for = self._init_var(self.prior_shots_for)
        if self.prior_var_sot_for is None:
            self.prior_var_sot_for = self._init_var(self.prior_sot_for)

        # xG priors default to goals priors (Poisson with same rate, before
        # any observations). This keeps xG-blended predictions identical to
        # goals-only at cold start. Phase 0.5 Understat could provide
        # team-specific xG priors in future, but for international teams the
        # international-cohort baseline is the same for both metrics.
        if self.prior_xg_for is None:
            self.prior_xg_for = self.prior_goals_for
        if self.prior_xg_against is None:
            self.prior_xg_against = self.prior_goals_against
        if self.prior_var_xg_for is None:
            self.prior_var_xg_for = self._init_var(self.prior_xg_for)
        if self.prior_var_xg_against is None:
            self.prior_var_xg_against = self._init_var(self.prior_xg_against)

        # Backward-compat: states saved before Phase 3 lack obs_weight_total.
        # Reconstruct it from n_matches_played (default weight 1.0 each).
        if self.obs_weight_total == 0.0 and self.n_matches_played > 0:
            self.obs_weight_total = float(self.n_matches_played)

    @staticmethod
    def _init_var(prior_mean: float) -> float:
        # For Poisson/rate-like quantities, var(λ̂) ≈ λ/n under mean-variance
        # equivalence. Floor at a small positive value so very small priors
        # don't produce var=0 (which would freeze the blend at the prior).
        return max(prior_mean, 1e-3) / DEFAULT_N_PRIOR

    # ── effective sample size (post-decay) ────────────────────────────

    @property
    def n_prior_eff_goals_for(self) -> float:
        return _n_prior_eff(self.prior_goals_for, self.prior_var_goals_for)

    @property
    def n_prior_eff_goals_against(self) -> float:
        return _n_prior_eff(self.prior_goals_against, self.prior_var_goals_against)

    @property
    def n_prior_eff_corners_for(self) -> float:
        return _n_prior_eff(self.prior_corners_for, self.prior_var_corners_for)

    @property
    def n_prior_eff_corners_against(self) -> float:
        return _n_prior_eff(self.prior_corners_against, self.prior_var_corners_against)

    @property
    def n_prior_eff_shots_for(self) -> float:
        return _n_prior_eff(self.prior_shots_for, self.prior_var_shots_for)

    @property
    def n_prior_eff_sot_for(self) -> float:
        return _n_prior_eff(self.prior_sot_for, self.prior_var_sot_for)

    @property
    def n_prior_eff_xg_for(self) -> float:
        return _n_prior_eff(self.prior_xg_for, self.prior_var_xg_for)

    @property
    def n_prior_eff_xg_against(self) -> float:
        return _n_prior_eff(self.prior_xg_against, self.prior_var_xg_against)

    # ── computed estimates ─────────────────────────────────────────

    @property
    def lambda_goals_for(self) -> float:
        return _blend(self.prior_goals_for, self.obs_goals_for,
                      self.obs_weight_total, self.n_prior_eff_goals_for)

    @property
    def lambda_goals_against(self) -> float:
        return _blend(self.prior_goals_against, self.obs_goals_against,
                      self.obs_weight_total, self.n_prior_eff_goals_against)

    @property
    def lambda_corners_for(self) -> float:
        return _blend(self.prior_corners_for, self.obs_corners_for,
                      self.obs_weight_total, self.n_prior_eff_corners_for)

    @property
    def lambda_corners_against(self) -> float:
        return _blend(self.prior_corners_against, self.obs_corners_against,
                      self.obs_weight_total, self.n_prior_eff_corners_against)

    @property
    def lambda_shots_for(self) -> float:
        return _blend(self.prior_shots_for, self.obs_shots_for,
                      self.obs_weight_total, self.n_prior_eff_shots_for)

    @property
    def lambda_sot_for(self) -> float:
        return _blend(self.prior_sot_for, self.obs_sot_for,
                      self.obs_weight_total, self.n_prior_eff_sot_for)

    @property
    def lambda_xg_for(self) -> float:
        return _blend(self.prior_xg_for, self.obs_xg_for,
                      self.obs_weight_total, self.n_prior_eff_xg_for)

    @property
    def lambda_xg_against(self) -> float:
        return _blend(self.prior_xg_against, self.obs_xg_against,
                      self.obs_weight_total, self.n_prior_eff_xg_against)

    def suspension_risk_ids(self) -> list[int]:
        """Players with exactly 1 yellow — one more triggers suspension."""
        return [pid for pid, cnt in self.yellow_cards.items() if cnt == 1]


@dataclass
class TournamentLiveState:
    """Full tournament state: one TeamLiveState per participating team."""

    tournament_slug: str
    team_states: dict[int, TeamLiveState] = field(default_factory=dict)
    completed_match_ids: list[str] = field(default_factory=list)

    def get_team(self, team_id: int) -> TeamLiveState | None:
        return self.team_states.get(team_id)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(_to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TournamentLiveState:
        raw = json.loads(path.read_text())
        return _from_dict(raw)


# ── Bayesian blend ────────────────────────────────────────────────────────────


def _blend(prior: float, obs_sum: float, n_obs: float, n_prior_eff: float) -> float:
    """Weighted average of pre-tournament prior and in-tournament observations.

    Formula: (n_prior_eff * λ_prior + obs_sum) / (n_prior_eff + n_obs)

    n_prior_eff is now metric-specific and derived from the current variance,
    so the blend automatically shrinks toward the observed mean when the
    between-window decay has inflated prior variance.

    When n_obs == 0 (no matches played yet), returns the prior.
    As n_obs grows, the tournament data takes over.
    """
    if n_obs == 0:
        return prior
    total_precision = n_prior_eff + n_obs
    if total_precision <= 0:
        return prior
    return (n_prior_eff * prior + obs_sum) / total_precision


def _n_prior_eff(prior_mean: float, prior_var: float | None) -> float:
    """Effective sample size: n_eff = mean / var (Poisson mean-variance link).

    As variance grows (between-window decay), n_eff shrinks → the blend
    weighs in-tournament observations more heavily. Floored at zero so that
    a fully-decayed prior simply yields the observed mean.
    """
    if prior_var is None or prior_var <= 0 or prior_mean <= 0:
        return 0.0
    return prior_mean / prior_var


# ── JSON serialization ────────────────────────────────────────────────────────


def _to_dict(state: TournamentLiveState) -> dict[str, Any]:
    return {
        "tournament_slug": state.tournament_slug,
        "completed_match_ids": state.completed_match_ids,
        "team_states": {
            str(tid): asdict(ts)
            for tid, ts in state.team_states.items()
        },
    }


def _from_dict(raw: dict[str, Any]) -> TournamentLiveState:
    team_states: dict[int, TeamLiveState] = {}
    for k, v in raw.get("team_states", {}).items():
        # JSON keys are always strings; convert back to int.
        v["yellow_cards"] = {int(yk): yv for yk, yv in v.get("yellow_cards", {}).items()}
        team_states[int(k)] = TeamLiveState(**v)
    return TournamentLiveState(
        tournament_slug=raw["tournament_slug"],
        team_states=team_states,
        completed_match_ids=raw.get("completed_match_ids", []),
    )
