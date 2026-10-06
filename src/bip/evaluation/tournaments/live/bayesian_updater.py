"""Bayesian updater: fold a completed match result into TournamentLiveState.

After every tournament match the operator calls
``BayesianUpdater.within_tournament_step()`` (or the legacy
``update()`` shim) to incorporate observed goals / corners / shots and
player events (injuries, bookings). The updated state then feeds the next
prediction cycle.

Phase 3 (2026-05-15) splits the single legacy ``update()`` into three
timescale-specific methods following SYNTHESIS Conclusion 5
(Glickman & Stern 1998, Held & Vollnhals 2005, Zou 2020):

- ``between_window_step``: random-walk decay applied **between FIFA windows**.
  Variance of each per-90 prior grows linearly with elapsed days
  (Held Eq. 4–5), modelled as fractional volatility so a single ``σ_s``
  works across goals/corners/shots: ``Δvar = days · σ_s² · prior_mean²``.
  Calibrated offline via Held criterion C on 2010–2024 international
  results (queued — see ``scripts/seed_international_history.py``).

- ``within_tournament_step``: applied **between matches inside a tournament**.
  Same Bayesian blend as the legacy ``update()``, but observations are now
  scaled by ``competition_weight`` (Held weighting hierarchy:
  WC = 4×, qualifier = 2×, friendly = 1×, etc. — see
  ``competition_weights.py``). The legacy ``update()`` is a shim that
  forwards to this method with weight 1.0 so existing callers keep working.

- ``within_match_step``: Gamma-Poisson conjugate update for **mid-match**
  refresh after a goal / red card / minute-N event. Treats the elapsed
  match window as a fractional Poisson observation (Zou 2020 closed form).
  Caller tracks the deltas; the updater is idempotent only on completed
  match_ids, not on partial-match snapshots.

The legacy ``update()`` remains as a backward-compatible shim — all 17
existing tests pass unchanged.

Yellow-card accumulation rules (WC2026):
- 2 yellow cards across group stage → suspended for first knockout match.
- At the group→knockout boundary the operator calls
  wipe_group_stage_bookings() to reset the count.
- Red card → immediate 1-match suspension (handled via suspended_player_ids).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bip.evaluation.tournaments.live.competition_weights import (
    CompetitionType,
    weight_for,
)
from bip.evaluation.tournaments.live.state import TournamentLiveState

SUSPENSION_THRESHOLD = 2  # yellow cards needed to trigger suspension

# Default σ_s — fractional volatility per day for between-window decay.
#
# Calibrated 2026-05-08 via Held criterion C walk-forward sweep over 14,502
# international matches (2010–2024) from the martj42/international_results
# CC0 dataset. See ``scripts/seed_international_history.py`` for the
# reproducible pipeline; full sweep results in the spike doc decisions log.
#
# Empirical finding: σ_s ∈ [0, 1e-4] all yield log-lik ≈ -1.5993 per goal
# observation; σ_s ≥ 0.001 degrades by ~0.003 per observation; σ_s = 0.005
# (the pre-calibration default) was 0.015 worse. The data says international
# match priors are stable enough that random-walk noise hurts more than it
# helps — international teams play ~10-15 matches/year with frequent
# observation, so the prior weight is dominated by data after ~30 matches
# regardless of σ. The mechanism is preserved at non-zero (defensible
# against unusually long gaps) but minimised in magnitude.
DEFAULT_SIGMA_S_PER_DAY = 0.0001


@dataclass(frozen=True)
class TournamentMatchResult:
    """Operator-supplied result for one completed tournament match.

    Stats marked Optional (corners, shots, sot) are set to None when not
    available from the data source — the updater skips those fields gracefully.

    Phase 3 adds ``competition`` so within_tournament_step can apply Held's
    weighting hierarchy. Defaults to FRIENDLY_FIFA (weight 1.0) so legacy
    callers without a competition tag preserve previous semantics.
    """

    match_id: str
    home_team_id: int
    away_team_id: int

    home_goals: int
    away_goals: int

    home_corners: int | None = None
    away_corners: int | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_sot: int | None = None
    away_sot: int | None = None
    # Expected goals (sum of shot.statsbomb_xg over the match). When present,
    # the updater accumulates a parallel xG rate used by xG-blended predictors.
    home_xg: float | None = None
    away_xg: float | None = None

    # Player events — api_football_ids.
    injured_home: tuple[int, ...] = field(default_factory=tuple)
    injured_away: tuple[int, ...] = field(default_factory=tuple)
    yellow_home: tuple[int, ...] = field(default_factory=tuple)
    yellow_away: tuple[int, ...] = field(default_factory=tuple)
    red_home: tuple[int, ...] = field(default_factory=tuple)
    red_away: tuple[int, ...] = field(default_factory=tuple)

    # Phase 3: competition tag determines the Bayesian weight on this
    # observation. None → treated as FRIENDLY_FIFA (weight 1.0) for
    # backward compatibility with pre-Phase-3 callers.
    competition: CompetitionType | None = None


class BayesianUpdater:
    """Mutates TournamentLiveState across three timescales.

    Usage::

        updater = BayesianUpdater()

        # Between FIFA windows (e.g., from June qualifier to October friendly):
        updater.between_window_step(state, days_elapsed=120)

        # After each completed tournament match:
        result = TournamentMatchResult(
            match_id="wc2026_grpA_001",
            ...,
            competition=CompetitionType.WORLD_CUP,  # weight 4.0
        )
        updater.within_tournament_step(state, result)

        # Optional: refresh λ mid-match (e.g., after a goal at minute 30):
        updater.within_match_step(
            state,
            team_id=10,
            minute=30.0,
            goals_for_delta=1,
            goals_against_delta=0,
        )

        state.save(Path("state/after_match_001.json"))

    Legacy callers using ``update()`` continue to work unchanged.
    """

    def __init__(self, sigma_s_per_day: float = DEFAULT_SIGMA_S_PER_DAY) -> None:
        # σ_s is stored on the updater instance so a calibrator can sweep it
        # across candidates without rebuilding callers. See Held criterion C
        # in scripts/seed_international_history.py (Layer-2, queued).
        if sigma_s_per_day < 0:
            raise ValueError("sigma_s_per_day must be non-negative")
        self.sigma_s_per_day = sigma_s_per_day

    # ── timescale 1: between FIFA windows (Held random walk) ────────────────

    def between_window_step(
        self,
        state: TournamentLiveState,
        days_elapsed: float,
        *,
        sigma_s_per_day: float | None = None,
    ) -> None:
        """Inflate prior variance to reflect time-decay between FIFA windows.

        Each per-90 rate evolves as a log-rate Gaussian random walk
        (Held & Vollnhals 2005, Eq. 4–5):

            log(λ_t) = log(λ_{t-1}) + ε,  ε ~ N(0, σ_s² · Δt)

        Applying the delta-method, ``Var(λ) ≈ λ² · σ_s² · Δt``. We
        accumulate that increment onto each ``prior_var_*`` field. The
        priors themselves stay at their last MAP estimates — only the
        confidence around them decays. This shrinks ``n_prior_eff`` so
        the next ``within_tournament_step`` weighs new observations more.

        No-op when days_elapsed ≤ 0 (state machine is monotonic in time).
        """
        if days_elapsed <= 0:
            return
        sigma = sigma_s_per_day if sigma_s_per_day is not None else self.sigma_s_per_day
        if sigma <= 0:
            return  # zero σ → frozen priors (useful for backtests)
        sigma_sq = sigma * sigma
        for ts in state.team_states.values():
            ts.prior_var_goals_for = (ts.prior_var_goals_for or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_goals_for ** 2)
            ts.prior_var_goals_against = (ts.prior_var_goals_against or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_goals_against ** 2)
            ts.prior_var_corners_for = (ts.prior_var_corners_for or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_corners_for ** 2)
            ts.prior_var_corners_against = (ts.prior_var_corners_against or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_corners_against ** 2)
            ts.prior_var_shots_for = (ts.prior_var_shots_for or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_shots_for ** 2)
            ts.prior_var_sot_for = (ts.prior_var_sot_for or 0.0) + \
                days_elapsed * sigma_sq * (ts.prior_sot_for ** 2)
            # xG decays with the same fractional volatility — international
            # team xG and goals share the same generative timescale (form
            # changes coach + roster, both metrics shift together).
            if ts.prior_xg_for is not None:
                ts.prior_var_xg_for = (ts.prior_var_xg_for or 0.0) + \
                    days_elapsed * sigma_sq * (ts.prior_xg_for ** 2)
            if ts.prior_xg_against is not None:
                ts.prior_var_xg_against = (ts.prior_var_xg_against or 0.0) + \
                    days_elapsed * sigma_sq * (ts.prior_xg_against ** 2)

    # ── timescale 2: between matches inside a tournament ────────────────────

    def within_tournament_step(
        self,
        state: TournamentLiveState,
        result: TournamentMatchResult,
    ) -> None:
        """Fold one completed match result into the live state.

        Idempotent on ``result.match_id``. Observations are scaled by the
        match's ``competition_weight`` so a WC group-stage match adds 4
        units of evidence to the Bayesian blend (vs 1 for a friendly).
        ``n_matches_played`` always increments by exactly 1 — only the
        evidential weight depends on competition.
        """
        if result.match_id in state.completed_match_ids:
            return

        weight = weight_for(result.competition)
        state.completed_match_ids.append(result.match_id)
        self._apply_match_result(state, result.home_team_id, result, is_home=True, weight=weight)
        self._apply_match_result(state, result.away_team_id, result, is_home=False, weight=weight)

    # ── timescale 3: mid-match Gamma-Poisson refresh ────────────────────────

    def within_match_step(
        self,
        state: TournamentLiveState,
        team_id: int,
        *,
        minutes_elapsed: float,
        goals_for_delta: int = 0,
        goals_against_delta: int = 0,
        corners_for_delta: int | None = None,
        corners_against_delta: int | None = None,
    ) -> None:
        """Gamma-Poisson conjugate update for an in-progress match.

        Treats the elapsed window as a fractional Poisson observation
        (Zou et al. 2020 closed-form):

            Prior:    λ ~ Gamma(α, β)         mean = α/β = prior, β = n_prior_eff
            Obs:      Y ~ Poisson(λ · f),     f = minutes_elapsed / 90
            Posterior:λ ~ Gamma(α + Y, β + f), mean = (α + Y) / (β + f)

        Caller passes **deltas since the previous call**, not absolute
        running totals — there is no idempotency on partial-match snapshots,
        so double-calling with the same Y causes double-counting. Pattern:
        invoke once after each goal / red card with the elapsed minutes
        and event counts since the previous invocation, not the running
        total since kickoff.

        Note: this updates ``obs_*`` and ``obs_weight_total`` directly. After
        full-time, the operator should call ``within_tournament_step()`` with
        the full result **only if** mid-match folds were not used — the
        two paths are mutually exclusive for a given match_id.
        """
        ts = state.team_states.get(team_id)
        if ts is None or minutes_elapsed <= 0:
            return
        fraction = min(minutes_elapsed / 90.0, 1.0)
        ts.obs_goals_for += float(goals_for_delta)
        ts.obs_goals_against += float(goals_against_delta)
        if corners_for_delta is not None:
            ts.obs_corners_for += float(corners_for_delta)
        if corners_against_delta is not None:
            ts.obs_corners_against += float(corners_against_delta)
        ts.obs_weight_total += fraction
        # n_matches_played is left untouched — partial-match folds are not
        # full matches; only within_tournament_step bumps the count.

    # ── backward-compat shim ────────────────────────────────────────────────

    def update(
        self,
        state: TournamentLiveState,
        result: TournamentMatchResult,
    ) -> None:
        """Legacy entry point — forwards to ``within_tournament_step``.

        Maintained for backward compatibility with pre-Phase-3 callers.
        Behaves identically when ``result.competition`` is None (weight 1.0).
        """
        self.within_tournament_step(state, result)

    def wipe_group_stage_bookings(self, state: TournamentLiveState) -> None:
        """Clear yellow card counts and suspensions at the group→knockout boundary.

        Call once, after all group-stage matches are processed, before
        generating round-of-16 predictions.
        """
        for ts in state.team_states.values():
            ts.yellow_cards.clear()
            ts.suspended_player_ids.clear()

    # ── private ──────────────────────────────────────────────────────────────

    def _apply_match_result(
        self,
        state: TournamentLiveState,
        team_id: int,
        result: TournamentMatchResult,
        *,
        is_home: bool,
        weight: float,
    ) -> None:
        ts = state.team_states.get(team_id)
        if ts is None:
            return  # team not registered in this state snapshot

        # Goals — scale observation by competition weight (Held 2005).
        scored = result.home_goals if is_home else result.away_goals
        conceded = result.away_goals if is_home else result.home_goals
        ts.obs_goals_for += scored * weight
        ts.obs_goals_against += conceded * weight

        # Corners (optional)
        corners_for = result.home_corners if is_home else result.away_corners
        corners_against = result.away_corners if is_home else result.home_corners
        if corners_for is not None and corners_against is not None:
            ts.obs_corners_for += corners_for * weight
            ts.obs_corners_against += corners_against * weight

        # Shots (optional)
        shots = result.home_shots if is_home else result.away_shots
        if shots is not None:
            ts.obs_shots_for += shots * weight

        # SoT (optional)
        sot = result.home_sot if is_home else result.away_sot
        if sot is not None:
            ts.obs_sot_for += sot * weight

        # xG (optional) — parallel rate updated alongside goals. Both for
        # this team's offense (xg_for) and conceded (xg_against from opp).
        xg_for = result.home_xg if is_home else result.away_xg
        xg_against = result.away_xg if is_home else result.home_xg
        if xg_for is not None:
            ts.obs_xg_for += float(xg_for) * weight
        if xg_against is not None:
            ts.obs_xg_against += float(xg_against) * weight

        ts.n_matches_played += 1
        ts.obs_weight_total += weight

        # ── Player events ────────────────────────────────────────────────────

        injured = result.injured_home if is_home else result.injured_away
        for pid in injured:
            if pid not in ts.injured_player_ids:
                ts.injured_player_ids.append(pid)

        # Red cards → 1-match suspension
        reds = result.red_home if is_home else result.red_away
        for pid in reds:
            if pid not in ts.suspended_player_ids:
                ts.suspended_player_ids.append(pid)

        # Yellow cards → accumulate, trigger suspension at threshold
        yellows = result.yellow_home if is_home else result.yellow_away
        for pid in yellows:
            ts.yellow_cards[pid] = ts.yellow_cards.get(pid, 0) + 1
            if ts.yellow_cards[pid] >= SUSPENSION_THRESHOLD:
                if pid not in ts.suspended_player_ids:
                    ts.suspended_player_ids.append(pid)
