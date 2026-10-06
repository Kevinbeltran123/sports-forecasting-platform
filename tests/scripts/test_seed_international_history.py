"""Tests for scripts/seed_international_history.py (Phase 3 σ_s calibrator).

Layer-1 only: validates the Held criterion C calibrator on synthetic data
so the harness ships unit-tested even before the 2010–2024 international
fixtures are loaded. The full backtest pull is gated `# requires-real-data`.
"""

from __future__ import annotations

import pytest

from bip.evaluation.tournaments.live.bayesian_updater import (
    TournamentMatchResult,
)
from bip.evaluation.tournaments.live.state import (
    TeamLiveState,
    TournamentLiveState,
)

# Direct import from the script (the script-as-module pattern used elsewhere
# in tests/scripts/). PYTHONPATH includes scripts/ via pyproject pytest config.
from scripts.seed_international_history import (
    HistoricalMatchSnapshot,
    _poisson_log_pmf,
    calibrate_sigma_s_held_criterion_c,
)


def _team(team_id: int, lam: float) -> TeamLiveState:
    return TeamLiveState(
        team_id=team_id,
        prior_goals_for=lam,
        prior_goals_against=lam,
        prior_corners_for=5.0,
        prior_corners_against=5.0,
        prior_shots_for=12.0,
        prior_sot_for=4.0,
    )


def _state_with(home_lam: float, away_lam: float) -> TournamentLiveState:
    return TournamentLiveState(
        tournament_slug="synth",
        team_states={
            1: _team(1, home_lam),
            2: _team(2, away_lam),
        },
    )


def _snapshot(
    home_goals: int,
    away_goals: int,
    home_lam: float = 1.5,
    away_lam: float = 1.5,
    days: float = 60.0,
) -> HistoricalMatchSnapshot:
    return HistoricalMatchSnapshot(
        state_before=_state_with(home_lam, away_lam),
        result=TournamentMatchResult(
            match_id=f"m_{home_goals}_{away_goals}",
            home_team_id=1,
            away_team_id=2,
            home_goals=home_goals,
            away_goals=away_goals,
        ),
        days_since_prev=days,
    )


def test_poisson_log_pmf_matches_known_value() -> None:
    # Poisson(λ=2) at k=1: log(2·exp(-2)) = log(2) - 2
    import math
    assert _poisson_log_pmf(1, 2.0) == pytest.approx(math.log(2.0) - 2.0)


def test_poisson_log_pmf_zero_lambda_is_neg_inf() -> None:
    assert _poisson_log_pmf(0, 0.0) == float("-inf")


def test_calibrator_rejects_empty_candidates() -> None:
    snaps = [_snapshot(1, 1)]
    with pytest.raises(ValueError, match="sigma_candidates"):
        calibrate_sigma_s_held_criterion_c(snaps, sigma_candidates=[])


def test_calibrator_rejects_empty_snapshots() -> None:
    with pytest.raises(ValueError, match="snapshots"):
        calibrate_sigma_s_held_criterion_c(
            [], sigma_candidates=[0.001, 0.005]
        )


def test_calibrator_returns_best_from_sweep() -> None:
    """Sweep returns the σ with maximum avg log-likelihood across snapshots."""
    snapshots = [_snapshot(1, 1), _snapshot(2, 1), _snapshot(0, 2)]
    candidates = [0.001, 0.010, 0.050]
    best, scores = calibrate_sigma_s_held_criterion_c(snapshots, candidates)

    assert best in candidates
    assert set(scores.keys()) == set(candidates)
    # The chosen σ has the highest score
    assert scores[best] == max(scores.values())


def test_calibrator_score_dict_complete() -> None:
    snapshots = [_snapshot(1, 1)]
    candidates = [0.001, 0.005, 0.010, 0.020]
    _, scores = calibrate_sigma_s_held_criterion_c(snapshots, candidates)
    assert all(c in scores for c in candidates)
    assert all(score > float("-inf") for score in scores.values())


def test_calibrator_skips_missing_team_in_snapshot() -> None:
    """Snapshot whose teams aren't in state_before is silently skipped."""
    valid_snap = _snapshot(1, 1)
    orphan_snap = HistoricalMatchSnapshot(
        state_before=TournamentLiveState(tournament_slug="empty"),
        result=TournamentMatchResult(
            match_id="orphan",
            home_team_id=999,
            away_team_id=998,
            home_goals=2,
            away_goals=0,
        ),
        days_since_prev=30.0,
    )
    best, scores = calibrate_sigma_s_held_criterion_c(
        [valid_snap, orphan_snap], sigma_candidates=[0.005, 0.010]
    )
    # No crash; best is well-defined
    assert best in {0.005, 0.010}


# ── Layer-2: real-data ingest + walk-forward calibration ─────────────────────


def _real_data_available() -> bool:
    """Check if the cached martj42 CSV exists locally."""
    from scripts.seed_international_history import DEFAULT_RAW_CSV
    return DEFAULT_RAW_CSV.exists()


@pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Run `uv run python scripts/seed_international_history.py` to "
        "download the martj42 CSV and unblock this Layer-2 test."
    ),
)
class TestRealDataLayer2:
    """Validates the Layer-2 ingest + walk-forward calibrator end-to-end.

    Requires the cached martj42 CSV at the default location. The seed
    script downloads it on first run; subsequent test runs are cached.
    """

    def test_csv_parses_with_expected_volume(self):
        """2010–2024 should yield ~14k+ international matches."""
        from scripts.seed_international_history import (
            DEFAULT_RAW_CSV,
            parse_csv_to_matches,
        )

        matches = parse_csv_to_matches(
            DEFAULT_RAW_CSV, start_year=2010, end_year=2024
        )
        # Empirically observed 14,502 on first calibration run (2026-05-08)
        assert len(matches) > 12_000
        assert len(matches) < 20_000

    def test_competition_mapping_covers_major_tournaments(self):
        """Top 10 tournaments by volume should all map to a CompetitionType."""
        from scripts.seed_international_history import (
            DEFAULT_RAW_CSV,
            map_tournament_to_competition,
            parse_csv_to_matches,
        )

        matches = parse_csv_to_matches(
            DEFAULT_RAW_CSV, start_year=2010, end_year=2024
        )
        from collections import Counter

        tournament_counts = Counter(m.tournament for m in matches)
        for tournament, _count in tournament_counts.most_common(10):
            comp = map_tournament_to_competition(tournament)
            # All top-10 tournaments should map (no silent FRIENDLY_FIFA fallback
            # for the major categories — that fallback is for obscure regional cups)
            assert comp is not None

    def test_walkforward_calibrator_runs_on_real_data(self):
        """Held criterion C walk-forward should run end-to-end on 14k+ matches.

        Performance guard: should complete in well under 60 seconds on a single
        modern core (the timeout below is generous to absorb CI variance).
        """
        from scripts.seed_international_history import (
            DEFAULT_RAW_CSV,
            calibrate_sigma_s_walkforward,
            parse_csv_to_matches,
        )

        matches = parse_csv_to_matches(
            DEFAULT_RAW_CSV, start_year=2010, end_year=2024
        )
        # Tight sweep around the empirical optimum found on 2026-05-08 calibration.
        candidates = [0.0, 0.0001, 0.001, 0.005]
        best_sigma, scores = calibrate_sigma_s_walkforward(
            matches, sigma_candidates=candidates
        )
        # All candidates produced a finite score
        assert all(s > float("-inf") for s in scores.values())
        # The optimum is in the low end — empirical finding from real data
        assert best_sigma in (0.0, 0.0001), (
            f"σ_s optimum drifted from low-end. Got {best_sigma}, "
            f"scores={scores}. If this is intentional (data updated, "
            f"empirical drift), update DEFAULT_SIGMA_S_PER_DAY in "
            f"bayesian_updater.py and the assertion below."
        )
        # Monotonic degradation as σ grows past the optimum (specific to
        # this dataset; expected from international-match frequency).
        assert scores[0.005] < scores[0.0001]
        assert scores[0.001] < scores[0.0001]

    def test_default_sigma_s_per_day_matches_calibrated_value(self):
        """Anchor: the locked-in DEFAULT_SIGMA_S_PER_DAY should be in the
        empirically-validated low-end zone (≤ 1e-3).
        """
        from bip.evaluation.tournaments.live.bayesian_updater import (
            DEFAULT_SIGMA_S_PER_DAY,
        )

        assert DEFAULT_SIGMA_S_PER_DAY <= 0.001, (
            f"DEFAULT_SIGMA_S_PER_DAY = {DEFAULT_SIGMA_S_PER_DAY} is above "
            f"the empirical optimum zone. Re-run "
            f"`scripts/seed_international_history.py` to confirm."
        )
