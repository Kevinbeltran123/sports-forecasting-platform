"""Tests for BayesianUpdater and TournamentLiveState."""

from __future__ import annotations

import json

import pytest

from bip.evaluation.tournaments.live.bayesian_updater import (
    BayesianUpdater,
    TournamentMatchResult,
)
from bip.evaluation.tournaments.live.state import (
    DEFAULT_N_PRIOR,
    TeamLiveState,
    TournamentLiveState,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_team(team_id: int, goals_for: float = 1.8) -> TeamLiveState:
    return TeamLiveState(
        team_id=team_id,
        prior_goals_for=goals_for,
        prior_goals_against=1.2,
        prior_corners_for=5.0,
        prior_corners_against=4.5,
        prior_shots_for=12.0,
        prior_sot_for=4.0,
    )


def _make_state(*team_ids: int) -> TournamentLiveState:
    state = TournamentLiveState(tournament_slug="test_tournament")
    for tid in team_ids:
        state.team_states[tid] = _make_team(tid)
    return state


def _result(
    match_id: str = "m001",
    home: int = 1,
    away: int = 2,
    hg: int = 2,
    ag: int = 1,
    **kwargs,
) -> TournamentMatchResult:
    return TournamentMatchResult(
        match_id=match_id,
        home_team_id=home,
        away_team_id=away,
        home_goals=hg,
        away_goals=ag,
        **kwargs,
    )


# ── Prior passthrough ─────────────────────────────────────────────────────────


def test_lambda_equals_prior_before_any_match() -> None:
    state = _make_state(1)
    ts = state.team_states[1]
    assert ts.lambda_goals_for == pytest.approx(1.8)
    assert ts.n_matches_played == 0


# ── Bayesian blend after one match ───────────────────────────────────────────


def test_lambda_blends_toward_observation_after_one_match() -> None:
    state = _make_state(1, 2)
    updater = BayesianUpdater()
    updater.update(state, _result(hg=4, ag=0))

    ts1 = state.team_states[1]
    # λ_new = (10 * 1.8 + 4) / (10 + 1) ≈ 2.0
    expected = (DEFAULT_N_PRIOR * 1.8 + 4) / (DEFAULT_N_PRIOR + 1)
    assert ts1.lambda_goals_for == pytest.approx(expected)
    assert ts1.n_matches_played == 1


def test_lambda_goals_against_updated_correctly() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(hg=2, ag=3))

    ts1 = state.team_states[1]
    # home conceded 3
    expected = (DEFAULT_N_PRIOR * 1.2 + 3) / (DEFAULT_N_PRIOR + 1)
    assert ts1.lambda_goals_against == pytest.approx(expected)


# ── Idempotency ───────────────────────────────────────────────────────────────


def test_same_match_id_not_double_counted() -> None:
    state = _make_state(1, 2)
    updater = BayesianUpdater()
    result = _result(hg=2, ag=1)

    updater.update(state, result)
    updater.update(state, result)  # second call → no-op

    assert state.team_states[1].n_matches_played == 1
    assert len(state.completed_match_ids) == 1


# ── Corners and optional stats ────────────────────────────────────────────────


def test_corners_accumulated_when_provided() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(
        state,
        _result(home_corners=7, away_corners=4),
    )
    assert state.team_states[1].obs_corners_for == 7.0
    assert state.team_states[2].obs_corners_for == 4.0


def test_corners_not_updated_when_none() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result())  # no corners supplied
    assert state.team_states[1].obs_corners_for == 0.0


# ── Player events ─────────────────────────────────────────────────────────────


def test_injury_added_to_injured_list() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(injured_home=(99,)))
    assert 99 in state.team_states[1].injured_player_ids


def test_yellow_card_tracked() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(yellow_home=(10,)))
    assert state.team_states[1].yellow_cards[10] == 1
    assert 10 not in state.team_states[1].suspended_player_ids


def test_two_yellows_trigger_suspension() -> None:
    state = _make_state(1, 2)
    updater = BayesianUpdater()
    updater.update(state, _result(match_id="m1", yellow_home=(10,)))
    updater.update(state, _result(match_id="m2", yellow_home=(10,)))
    assert 10 in state.team_states[1].suspended_player_ids


def test_red_card_triggers_suspension_immediately() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(red_home=(77,)))
    assert 77 in state.team_states[1].suspended_player_ids


# ── Wipe group-stage bookings ─────────────────────────────────────────────────


def test_wipe_clears_yellows_and_suspensions() -> None:
    state = _make_state(1, 2)
    updater = BayesianUpdater()
    updater.update(state, _result(yellow_home=(10,), red_home=(20,)))
    updater.wipe_group_stage_bookings(state)

    ts = state.team_states[1]
    assert ts.yellow_cards == {}
    assert ts.suspended_player_ids == []


# ── Suspension risk ───────────────────────────────────────────────────────────


def test_suspension_risk_identifies_one_yellow_players() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(yellow_home=(10, 20)))
    risk = state.team_states[1].suspension_risk_ids()
    assert set(risk) == {10, 20}


# ── JSON round-trip ───────────────────────────────────────────────────────────


def test_save_load_round_trip(tmp_path) -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(hg=3, ag=1, yellow_home=(55,)))

    path = tmp_path / "state.json"
    state.save(path)
    loaded = TournamentLiveState.load(path)

    ts = loaded.team_states[1]
    assert ts.n_matches_played == 1
    assert ts.obs_goals_for == 3.0
    assert ts.yellow_cards == {55: 1}
    assert loaded.completed_match_ids == ["m001"]


def test_yellow_card_keys_are_ints_after_load(tmp_path) -> None:
    state = _make_state(1, 2)
    BayesianUpdater().update(state, _result(yellow_home=(42,)))

    path = tmp_path / "state.json"
    state.save(path)
    loaded = TournamentLiveState.load(path)

    keys = list(loaded.team_states[1].yellow_cards.keys())
    assert all(isinstance(k, int) for k in keys)


# ── Phase 3: between_window_step (Held random walk) ──────────────────────────


def test_between_window_zero_days_is_noop() -> None:
    state = _make_state(1, 2)
    var_before = state.team_states[1].prior_var_goals_for
    BayesianUpdater().between_window_step(state, days_elapsed=0)
    assert state.team_states[1].prior_var_goals_for == var_before


def test_between_window_negative_days_is_noop() -> None:
    state = _make_state(1, 2)
    var_before = state.team_states[1].prior_var_goals_for
    BayesianUpdater().between_window_step(state, days_elapsed=-30)
    assert state.team_states[1].prior_var_goals_for == var_before


def test_between_window_zero_sigma_is_noop() -> None:
    state = _make_state(1, 2)
    var_before = state.team_states[1].prior_var_goals_for
    BayesianUpdater(sigma_s_per_day=0.0).between_window_step(state, days_elapsed=120)
    assert state.team_states[1].prior_var_goals_for == var_before


def test_between_window_inflates_variance_linearly() -> None:
    """var_new = var_old + days · σ² · μ²  (Held log-rate random walk)."""
    state = _make_state(1)
    ts = state.team_states[1]
    sigma = 0.01
    days = 100
    expected_increment = days * (sigma ** 2) * (ts.prior_goals_for ** 2)
    var_before = ts.prior_var_goals_for

    BayesianUpdater(sigma_s_per_day=sigma).between_window_step(state, days_elapsed=days)
    assert ts.prior_var_goals_for == pytest.approx(var_before + expected_increment)


def test_between_window_decay_is_monotonic() -> None:
    """Repeated calls only grow variance (random walk is one-way in time)."""
    state = _make_state(1)
    updater = BayesianUpdater(sigma_s_per_day=0.005)

    var_track: list[float] = [state.team_states[1].prior_var_goals_for]
    for _ in range(5):
        updater.between_window_step(state, days_elapsed=30)
        var_track.append(state.team_states[1].prior_var_goals_for)

    # Strictly increasing
    assert all(b > a for a, b in zip(var_track, var_track[1:]))


def test_between_window_shrinks_effective_n_prior() -> None:
    """As variance grows, n_prior_eff = mean/var falls toward zero."""
    state = _make_state(1)
    ts = state.team_states[1]
    n_prior_eff_before = ts.n_prior_eff_goals_for

    BayesianUpdater(sigma_s_per_day=0.02).between_window_step(state, days_elapsed=365)
    assert ts.n_prior_eff_goals_for < n_prior_eff_before


def test_between_window_makes_blend_more_responsive() -> None:
    """After a long gap, a single observation moves λ further than without decay."""
    # State A: no decay, single match observed (4 goals)
    state_a = _make_state(1, 2)
    BayesianUpdater().within_tournament_step(state_a, _result(hg=4, ag=0))
    lam_a = state_a.team_states[1].lambda_goals_for

    # State B: 2-year gap (730 days), then same match
    state_b = _make_state(1, 2)
    updater = BayesianUpdater(sigma_s_per_day=0.01)
    updater.between_window_step(state_b, days_elapsed=730)
    updater.within_tournament_step(state_b, _result(hg=4, ag=0))
    lam_b = state_b.team_states[1].lambda_goals_for

    # B's λ should be closer to the observed 4 than A's (less prior weight)
    assert abs(lam_b - 4.0) < abs(lam_a - 4.0)


def test_between_window_decays_all_six_metrics() -> None:
    state = _make_state(1)
    ts = state.team_states[1]
    before = (
        ts.prior_var_goals_for, ts.prior_var_goals_against,
        ts.prior_var_corners_for, ts.prior_var_corners_against,
        ts.prior_var_shots_for, ts.prior_var_sot_for,
    )
    BayesianUpdater(sigma_s_per_day=0.01).between_window_step(state, days_elapsed=120)
    after = (
        ts.prior_var_goals_for, ts.prior_var_goals_against,
        ts.prior_var_corners_for, ts.prior_var_corners_against,
        ts.prior_var_shots_for, ts.prior_var_sot_for,
    )
    assert all(b > a for a, b in zip(before, after))


# ── Phase 3: within_tournament_step (competition weighting) ──────────────────


def test_within_tournament_step_no_competition_matches_legacy_update() -> None:
    """Default competition (None) → weight 1.0 → identical to legacy update()."""
    state_legacy = _make_state(1, 2)
    state_new = _make_state(1, 2)
    BayesianUpdater().update(state_legacy, _result(hg=4, ag=0))
    BayesianUpdater().within_tournament_step(state_new, _result(hg=4, ag=0))

    assert state_legacy.team_states[1].lambda_goals_for == \
           state_new.team_states[1].lambda_goals_for
    assert state_legacy.team_states[1].obs_weight_total == \
           state_new.team_states[1].obs_weight_total


def test_world_cup_match_weighs_4x_in_blend() -> None:
    """A WC match should add 4 units of obs_weight_total (Held weighting)."""
    from bip.evaluation.tournaments.live.competition_weights import CompetitionType

    state = _make_state(1, 2)
    BayesianUpdater().within_tournament_step(
        state,
        _result(hg=4, ag=0, competition=CompetitionType.WORLD_CUP),
    )
    ts = state.team_states[1]
    assert ts.obs_weight_total == pytest.approx(4.0)
    assert ts.n_matches_played == 1  # match count unchanged
    assert ts.obs_goals_for == pytest.approx(4 * 4.0)  # scaled by weight


def test_friendly_non_fifa_weighs_half() -> None:
    from bip.evaluation.tournaments.live.competition_weights import CompetitionType

    state = _make_state(1, 2)
    BayesianUpdater().within_tournament_step(
        state,
        _result(hg=2, ag=1, competition=CompetitionType.FRIENDLY_NON_FIFA),
    )
    ts = state.team_states[1]
    assert ts.obs_weight_total == pytest.approx(0.5)
    assert ts.obs_goals_for == pytest.approx(2 * 0.5)


def test_competition_string_value_accepted() -> None:
    state = _make_state(1, 2)
    BayesianUpdater().within_tournament_step(
        state,
        _result(hg=1, ag=0, competition="qualifier"),
    )
    assert state.team_states[1].obs_weight_total == pytest.approx(2.0)


def test_unknown_competition_raises() -> None:
    state = _make_state(1, 2)
    with pytest.raises(ValueError, match="Unknown competition type"):
        BayesianUpdater().within_tournament_step(
            state,
            _result(hg=1, ag=0, competition="champions_league"),
        )


# ── Phase 3: within_match_step (Gamma-Poisson) ──────────────────────────────


def test_within_match_zero_minutes_is_noop() -> None:
    state = _make_state(1)
    BayesianUpdater().within_match_step(
        state, team_id=1, minutes_elapsed=0, goals_for_delta=1,
    )
    ts = state.team_states[1]
    assert ts.obs_goals_for == 0.0
    assert ts.obs_weight_total == 0.0


def test_within_match_unknown_team_is_noop() -> None:
    state = _make_state(1)
    BayesianUpdater().within_match_step(
        state, team_id=999, minutes_elapsed=45, goals_for_delta=2,
    )
    assert state.team_states[1].obs_goals_for == 0.0


def test_within_match_fractional_blend_at_30_minutes() -> None:
    """Goal at minute 30 → fraction 1/3 added to obs_weight_total."""
    state = _make_state(1)
    BayesianUpdater().within_match_step(
        state, team_id=1, minutes_elapsed=30, goals_for_delta=1,
    )
    ts = state.team_states[1]
    assert ts.obs_goals_for == pytest.approx(1.0)
    assert ts.obs_weight_total == pytest.approx(30 / 90)
    assert ts.n_matches_played == 0  # partial match doesn't count as full


def test_within_match_gamma_poisson_posterior_mean() -> None:
    """λ_post = (n_prior_eff · μ_prior + Y) / (n_prior_eff + f) — Zou closed form."""
    state = _make_state(1)
    ts = state.team_states[1]
    n_eff = ts.n_prior_eff_goals_for
    BayesianUpdater().within_match_step(
        state, team_id=1, minutes_elapsed=45, goals_for_delta=2,
    )
    f = 45 / 90
    expected = (n_eff * 1.8 + 2.0) / (n_eff + f)
    assert ts.lambda_goals_for == pytest.approx(expected)


def test_within_match_caps_at_full_match() -> None:
    """minutes_elapsed > 90 → fraction caps at 1.0 (no super-evidence)."""
    state = _make_state(1)
    BayesianUpdater().within_match_step(
        state, team_id=1, minutes_elapsed=120, goals_for_delta=3,
    )
    assert state.team_states[1].obs_weight_total == pytest.approx(1.0)


def test_within_match_corners_optional_delta() -> None:
    state = _make_state(1)
    BayesianUpdater().within_match_step(
        state, team_id=1, minutes_elapsed=30,
        goals_for_delta=0, corners_for_delta=3,
    )
    ts = state.team_states[1]
    assert ts.obs_corners_for == pytest.approx(3.0)


# ── Phase 3: state persistence (variance survives JSON round-trip) ───────────


def test_prior_var_survives_round_trip(tmp_path) -> None:
    state = _make_state(1, 2)
    BayesianUpdater(sigma_s_per_day=0.01).between_window_step(state, days_elapsed=200)
    decayed_var = state.team_states[1].prior_var_goals_for

    path = tmp_path / "state.json"
    state.save(path)
    loaded = TournamentLiveState.load(path)
    assert loaded.team_states[1].prior_var_goals_for == pytest.approx(decayed_var)


def test_legacy_state_json_loads_with_phase3_defaults(tmp_path) -> None:
    """Old JSON saves (no prior_var_*, no obs_weight_total) load with sane defaults."""
    legacy_payload = {
        "tournament_slug": "old_tournament",
        "completed_match_ids": ["m1"],
        "team_states": {
            "1": {
                "team_id": 1,
                "prior_goals_for": 1.8,
                "prior_goals_against": 1.2,
                "prior_corners_for": 5.0,
                "prior_corners_against": 4.5,
                "prior_shots_for": 12.0,
                "prior_sot_for": 4.0,
                "obs_goals_for": 2.0,
                "obs_goals_against": 1.0,
                "obs_corners_for": 0.0,
                "obs_corners_against": 0.0,
                "obs_shots_for": 0.0,
                "obs_sot_for": 0.0,
                "n_matches_played": 1,
                "n_prior": 10,
                "injured_player_ids": [],
                "suspended_player_ids": [],
                "yellow_cards": {},
            },
        },
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_payload))
    loaded = TournamentLiveState.load(path)

    ts = loaded.team_states[1]
    # Variance defaults applied
    assert ts.prior_var_goals_for is not None
    assert ts.prior_var_goals_for == pytest.approx(1.8 / 10)
    # obs_weight_total reconstructed from n_matches_played
    assert ts.obs_weight_total == pytest.approx(1.0)
    # Blend still works with legacy semantics
    expected_lambda = (10 * 1.8 + 2.0) / (10 + 1.0)
    assert ts.lambda_goals_for == pytest.approx(expected_lambda)


# ── Sanity: sigma constructor validation ─────────────────────────────────────


def test_negative_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BayesianUpdater(sigma_s_per_day=-0.001)
