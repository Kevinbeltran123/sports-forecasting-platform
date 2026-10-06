"""International tournament history seed — Phase 3 Layer-2 (σ_s calibration).

Downloads 2010–2024 international football results from
``martj42/international_results`` (CC0 public-domain CSV, 49k+ matches),
filters to the calibration window, and calibrates the
``BayesianUpdater.between_window_step`` σ_s parameter via Held criterion C
(one-step-ahead predictive log-likelihood maximization).

Why martj42 instead of API-Football: the σ_s calibration only needs
``(date, home_team, away_team, home_goals, away_goals, tournament)``. The
martj42 CSV provides exactly that for every men's international match
since 1872 — including all WC qualifiers, Euros qualifiers, AFCON,
nations leagues, and friendlies — at zero cost and with no rate-limit
concerns. API-Football's value (team IDs, lineups) is not consumed by
the Held criterion. See research investigation logged in
internal planning notes decisions log.

Source attribution: ``https://github.com/martj42/international_results``
(CC0; data updated through 2026-06-27 as of last verification).

Run end-to-end::

    uv run python scripts/seed_international_history.py \\
        --start-year 2010 --end-year 2024 \\
        --output data/cache/international_history.parquet

This will (1) download the CSV, (2) filter to the date range, (3) write
a Parquet snapshot at ``--output``, and (4) run the σ_s sweep, printing
the optimal value and per-candidate log-likelihood scores.

NOT invoked by the scheduler.
"""

from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from bip.evaluation.tournaments.live.bayesian_updater import (
    BayesianUpdater,
    TournamentMatchResult,
)
from bip.evaluation.tournaments.live.competition_weights import CompetitionType
from bip.evaluation.tournaments.live.state import (
    DEFAULT_N_PRIOR,
    TeamLiveState,
    TournamentLiveState,
)

# ─────────────────────────────────────────────────────────────────────────────
# Source: martj42/international_results (CC0)
# ─────────────────────────────────────────────────────────────────────────────

MARTJ42_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

DEFAULT_DATA_DIR = Path("data/cache")
DEFAULT_RAW_CSV = DEFAULT_DATA_DIR / "martj42_international_results.csv"
DEFAULT_PARQUET = DEFAULT_DATA_DIR / "international_history.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Tournament-name → CompetitionType mapping
# ─────────────────────────────────────────────────────────────────────────────
#
# martj42 tournament names → our CompetitionType enum (Held weighting hierarchy).
# Anything not listed here defaults to FRIENDLY_FIFA (weight 1.0); regional
# minor cups contribute as noise to the σ_s estimate, which is fine.

_TOURNAMENT_TO_COMPETITION: dict[str, CompetitionType] = {
    # Top-tier global / continental
    "FIFA World Cup": CompetitionType.WORLD_CUP,
    "FIFA World Cup qualification": CompetitionType.QUALIFIER,
    "UEFA Euro": CompetitionType.CONFEDERATION,
    "UEFA Euro qualification": CompetitionType.QUALIFIER,
    "Copa América": CompetitionType.CONFEDERATION,
    "African Cup of Nations": CompetitionType.CONFEDERATION,
    "African Cup of Nations qualification": CompetitionType.QUALIFIER,
    "AFC Asian Cup": CompetitionType.CONFEDERATION,
    "AFC Asian Cup qualification": CompetitionType.QUALIFIER,
    "Gold Cup": CompetitionType.CONFEDERATION,
    "Gold Cup qualification": CompetitionType.QUALIFIER,
    "Confederations Cup": CompetitionType.CONFEDERATION,
    # Nations leagues — competitive between members of a confederation,
    # weighted as confederation-tier per Held's "competitive intensity" logic
    "UEFA Nations League": CompetitionType.CONFEDERATION,
    "CONCACAF Nations League": CompetitionType.CONFEDERATION,
    "CONCACAF Nations League qualification": CompetitionType.QUALIFIER,
    # Regional minor cups → friendly-tier signal
    "Friendly": CompetitionType.FRIENDLY_FIFA,
    "FIFA Series": CompetitionType.FRIENDLY_FIFA,
    "ASEAN Championship": CompetitionType.FRIENDLY_FIFA,
    "Gulf Cup": CompetitionType.FRIENDLY_FIFA,
    "Oceania Nations Cup": CompetitionType.CONFEDERATION,
    "EAFF Championship": CompetitionType.FRIENDLY_FIFA,
    "EAFF Championship qualification": CompetitionType.FRIENDLY_FIFA,
    "King's Cup": CompetitionType.FRIENDLY_NON_FIFA,
    "Merdeka Tournament": CompetitionType.FRIENDLY_NON_FIFA,
    "Baltic Cup": CompetitionType.FRIENDLY_NON_FIFA,
}


def map_tournament_to_competition(tournament: str) -> CompetitionType:
    """Map a martj42 tournament string to our CompetitionType enum.

    Unknown tournaments fall back to FRIENDLY_FIFA (weight 1.0) — we'd
    rather underweight an obscure regional cup than crash the ingest.
    """
    return _TOURNAMENT_TO_COMPETITION.get(tournament, CompetitionType.FRIENDLY_FIFA)


# ─────────────────────────────────────────────────────────────────────────────
# Match record — Layer-2 working dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MartJ42Match:
    """One row from the martj42 CSV, normalized to our calibration shape."""

    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    tournament: str
    competition: CompetitionType
    neutral: bool


# ─────────────────────────────────────────────────────────────────────────────
# CSV download + parse
# ─────────────────────────────────────────────────────────────────────────────


def download_martj42_csv(
    cache_path: Path = DEFAULT_RAW_CSV,
    *,
    force_refresh: bool = False,
) -> Path:
    """Download the martj42 international_results CSV (cache-first)."""
    if cache_path.exists() and not force_refresh:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[seed] downloading {MARTJ42_CSV_URL} → {cache_path}", file=sys.stderr)
    urllib.request.urlretrieve(MARTJ42_CSV_URL, cache_path)
    return cache_path


def parse_csv_to_matches(
    csv_path: Path,
    *,
    start_year: int = 2010,
    end_year: int = 2024,
) -> list[MartJ42Match]:
    """Parse the CSV and filter to ``[start_year, end_year]`` (inclusive).

    Skips rows where either score is NA (unplayed fixtures — including
    future WC2026 matches that martj42 includes for completeness).
    """
    import csv

    out: list[MartJ42Match] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except (ValueError, KeyError):
                continue
            if d.year < start_year or d.year > end_year:
                continue
            if row.get("home_score") in ("", "NA", None):
                continue
            if row.get("away_score") in ("", "NA", None):
                continue
            try:
                home_goals = int(row["home_score"])
                away_goals = int(row["away_score"])
            except ValueError:
                continue
            tournament = row["tournament"]
            out.append(
                MartJ42Match(
                    match_date=d,
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    home_goals=home_goals,
                    away_goals=away_goals,
                    tournament=tournament,
                    competition=map_tournament_to_competition(tournament),
                    neutral=row.get("neutral", "FALSE").strip().upper() == "TRUE",
                )
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Held criterion C — σ_s calibration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HistoricalMatchSnapshot:
    """One historical fixture + the state snapshot just before it.

    ``state_before`` is the TournamentLiveState that the updater would have
    held immediately before this match. ``days_since_prev`` is the gap
    between this match and the previous international match for either team.
    Used by the Layer-1 synthetic test path (``calibrate_sigma_s_held_criterion_c``).
    Layer-2 walk-forward (``calibrate_sigma_s_walkforward``) builds state
    from scratch per σ candidate to avoid cross-candidate contamination.
    """

    state_before: TournamentLiveState
    result: TournamentMatchResult
    days_since_prev: float


def calibrate_sigma_s_held_criterion_c(
    snapshots: Iterable[HistoricalMatchSnapshot],
    sigma_candidates: list[float],
) -> tuple[float, dict[float, float]]:
    """Layer-1 synthetic-snapshot calibrator (kept for unit-test compatibility).

    This entry point is what the existing
    ``test_calibrator_returns_best_from_sweep`` style tests exercise.
    For real data prefer ``calibrate_sigma_s_walkforward``, which builds
    state from scratch per σ candidate (correct walk-forward semantics).
    """
    if not sigma_candidates:
        raise ValueError("sigma_candidates must not be empty")

    snapshots_list = list(snapshots)
    if not snapshots_list:
        raise ValueError("snapshots must not be empty")

    scores: dict[float, float] = {}
    for sigma in sigma_candidates:
        log_lik_total = 0.0
        n_scored = 0
        updater = BayesianUpdater(sigma_s_per_day=sigma)
        for snap in snapshots_list:
            state = snap.state_before
            updater.between_window_step(state, days_elapsed=snap.days_since_prev)
            home_ts = state.team_states.get(snap.result.home_team_id)
            away_ts = state.team_states.get(snap.result.away_team_id)
            if home_ts is None or away_ts is None:
                continue
            lam_home = max(home_ts.lambda_goals_for, 1e-6)
            lam_away = max(away_ts.lambda_goals_for, 1e-6)
            log_lik_total += _poisson_log_pmf(snap.result.home_goals, lam_home)
            log_lik_total += _poisson_log_pmf(snap.result.away_goals, lam_away)
            n_scored += 2
        scores[sigma] = log_lik_total / n_scored if n_scored else float("-inf")

    best_sigma = max(scores, key=lambda s: scores[s])
    return best_sigma, scores


def calibrate_sigma_s_walkforward(
    matches: list[MartJ42Match],
    sigma_candidates: list[float],
    *,
    prior_goals_for: float = 1.30,
    prior_goals_against: float = 1.30,
    n_prior: int = DEFAULT_N_PRIOR,
) -> tuple[float, dict[float, float]]:
    """Real-data Held criterion C: walk-forward σ_s sweep.

    For each σ candidate, builds a fresh ``TournamentLiveState`` over the
    full set of teams encountered, then walks the matches in chronological
    order:

    1. Apply ``between_window_step`` with the per-team gap since previous match.
    2. Score predicted goals (lambda before update) against actual via
       Poisson log-PMF.
    3. Apply ``within_tournament_step`` with the match's competition_weight
       so the next prediction uses an updated state.

    The optimal σ_s is the value maximizing the average log-likelihood
    across all (match, side) observations. Defaults align with the
    historical international-match goal-rate baseline (~1.3 per side per
    match across 2010–2024 — check via ``np.mean`` on the data if needed).
    """
    if not sigma_candidates:
        raise ValueError("sigma_candidates must not be empty")
    if not matches:
        raise ValueError("matches must not be empty")

    # Order matters — Held's criterion is one-step-ahead predictive.
    matches = sorted(matches, key=lambda m: m.match_date)

    # Stable team_id assignment via insertion order (deterministic).
    teams_seen: dict[str, int] = {}
    for m in matches:
        teams_seen.setdefault(m.home_team, len(teams_seen))
        teams_seen.setdefault(m.away_team, len(teams_seen))

    scores: dict[float, float] = {}
    for sigma in sigma_candidates:
        state = _build_initial_state(
            teams_seen,
            prior_goals_for=prior_goals_for,
            prior_goals_against=prior_goals_against,
            n_prior=n_prior,
        )
        updater = BayesianUpdater(sigma_s_per_day=sigma)
        last_match_date_per_team: dict[int, date] = {}

        log_lik_total = 0.0
        n_scored = 0
        match_idx = 0

        for m in matches:
            home_id = teams_seen[m.home_team]
            away_id = teams_seen[m.away_team]

            # Days since each team's last match — between_window_step is
            # global, so we use the MIN gap across home/away (most recent
            # signal). Could also use per-team decay but global is simpler.
            prev_dates = [
                d for d in (
                    last_match_date_per_team.get(home_id),
                    last_match_date_per_team.get(away_id),
                )
                if d is not None
            ]
            days_elapsed = (
                (m.match_date - max(prev_dates)).days
                if prev_dates
                else 0
            )

            if days_elapsed > 0:
                updater.between_window_step(state, days_elapsed=days_elapsed)

            home_ts = state.team_states[home_id]
            away_ts = state.team_states[away_id]
            lam_home = max(home_ts.lambda_goals_for, 1e-6)
            lam_away = max(away_ts.lambda_goals_for, 1e-6)
            log_lik_total += _poisson_log_pmf(m.home_goals, lam_home)
            log_lik_total += _poisson_log_pmf(m.away_goals, lam_away)
            n_scored += 2

            # Apply update for the next iteration.
            result = TournamentMatchResult(
                match_id=f"m_{match_idx}",
                home_team_id=home_id,
                away_team_id=away_id,
                home_goals=m.home_goals,
                away_goals=m.away_goals,
                competition=m.competition,
            )
            updater.within_tournament_step(state, result)
            last_match_date_per_team[home_id] = m.match_date
            last_match_date_per_team[away_id] = m.match_date
            match_idx += 1

        scores[sigma] = log_lik_total / n_scored if n_scored else float("-inf")

    best_sigma = max(scores, key=lambda s: scores[s])
    return best_sigma, scores


def _build_initial_state(
    teams: dict[str, int],
    *,
    prior_goals_for: float,
    prior_goals_against: float,
    n_prior: int,
) -> TournamentLiveState:
    """Construct a fresh state with uniform priors per team."""
    state = TournamentLiveState(tournament_slug="historical_2010_2024")
    for _name, tid in teams.items():
        state.team_states[tid] = TeamLiveState(
            team_id=tid,
            prior_goals_for=prior_goals_for,
            prior_goals_against=prior_goals_against,
            prior_corners_for=5.0,
            prior_corners_against=5.0,
            prior_shots_for=12.0,
            prior_sot_for=4.0,
            n_prior=n_prior,
        )
    return state


def _poisson_log_pmf(k: int, lam: float) -> float:
    """log P(K=k | λ) = k·log(λ) - λ - log(k!)."""
    if lam <= 0:
        return float("-inf")
    return k * math.log(lam) - lam - math.lgamma(k + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Parquet writer (optional persistence)
# ─────────────────────────────────────────────────────────────────────────────


def write_matches_to_parquet(matches: list[MartJ42Match], output: Path) -> None:
    """Persist the filtered match list to a Parquet snapshot."""
    import polars as pl

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "match_date": [m.match_date for m in matches],
            "home_team": [m.home_team for m in matches],
            "away_team": [m.away_team for m in matches],
            "home_goals": [m.home_goals for m in matches],
            "away_goals": [m.away_goals for m in matches],
            "tournament": [m.tournament for m in matches],
            "competition": [m.competition.value for m in matches],
            "neutral": [m.neutral for m in matches],
        }
    )
    df.write_parquet(output)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument(
        "--csv-cache",
        type=Path,
        default=DEFAULT_RAW_CSV,
        help="Where to cache the downloaded martj42 CSV",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Parquet snapshot of filtered matches",
    )
    p.add_argument(
        "--sigma-sweep",
        nargs="+",
        type=float,
        default=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050],
        help="σ_s candidates for Held criterion C (per-day fractional volatility)",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download the CSV even if cached",
    )
    p.add_argument(
        "--prior-goals",
        type=float,
        default=1.30,
        help="Initial prior goals/match per team (default tuned for 2010–2024 baseline)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    csv_path = download_martj42_csv(args.csv_cache, force_refresh=args.force_refresh)
    matches = parse_csv_to_matches(
        csv_path, start_year=args.start_year, end_year=args.end_year
    )
    if not matches:
        print(
            f"[seed] No matches found in {args.start_year}..{args.end_year}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[seed] {len(matches)} matches in {args.start_year}..{args.end_year}",
        file=sys.stderr,
    )

    write_matches_to_parquet(matches, args.output)
    print(f"[seed] wrote {args.output}", file=sys.stderr)

    print("[sigma-sweep] running Held criterion C walk-forward...", file=sys.stderr)
    best_sigma, scores = calibrate_sigma_s_walkforward(
        matches,
        sigma_candidates=list(args.sigma_sweep),
        prior_goals_for=args.prior_goals,
        prior_goals_against=args.prior_goals,
    )

    print()
    print("σ_s sweep results (avg one-step-ahead log-likelihood per goal observation)")
    print("-" * 70)
    for sigma in sorted(scores):
        marker = "  *" if sigma == best_sigma else "   "
        print(f"{marker} σ={sigma:<8} log-lik={scores[sigma]:+.6f}")
    print("-" * 70)
    print(f"OPTIMAL σ_s = {best_sigma}")
    print(
        "Update DEFAULT_SIGMA_S_PER_DAY in "
        "src/bip/evaluation/tournaments/live/bayesian_updater.py "
        "if this value differs from the current default."
    )


if __name__ == "__main__":
    main()
