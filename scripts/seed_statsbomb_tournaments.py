"""StatsBomb open-data tournament ingest — Phase 5 Layer-2.

Downloads match-level outcomes (goals, corners, BTTS) from the 6 modern
men's international tournaments StatsBomb has released free:

- FIFA World Cup 2018  (comp=43,   season=3)
- FIFA World Cup 2022  (comp=43,   season=106)
- UEFA Euro 2020       (comp=55,   season=43)
- UEFA Euro 2024       (comp=55,   season=282)
- Copa América 2024    (comp=223,  season=282)
- AFCON 2023           (comp=1267, season=107)

Per spike doc Phase 5 Layer-2: this is the real-data substrate for the
pre-WC2026 lock backtest. Held-out set (operator-approved 2026-05-08):
``copa_2024 + euro_2024`` are the most recent and closest in style to
WC2026; the other four serve as training data for the backtest's walk-
forward state.

Cache structure (under ``data/cache/statsbomb/``):

    statsbomb/
        matches/{comp_id}_{season_id}.json     — match-list per tournament
        events/{match_id}.json                  — event stream per match
        match_outcomes.parquet                  — aggregated match-level table

License: StatsBomb open data is **CC BY-NC 4.0** — non-commercial use only.
This calibration backtest is internal model validation, not redistribution.

Source attribution: ``https://github.com/statsbomb/open-data``

Run end-to-end::

    uv run python scripts/seed_statsbomb_tournaments.py

This will (1) download all 6 tournament match lists, (2) download every
match's event JSON (≈370 matches), (3) parse corners/goals per team,
and (4) write ``data/cache/statsbomb/match_outcomes.parquet``.

NOT invoked by the scheduler.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# StatsBomb open-data structure
# ─────────────────────────────────────────────────────────────────────────────

STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

DEFAULT_CACHE_DIR = Path("data/cache/statsbomb")
DEFAULT_OUTCOMES_PARQUET = DEFAULT_CACHE_DIR / "match_outcomes.parquet"


@dataclass(frozen=True)
class StatsBombTournament:
    """One competition × season pair from the open-data set."""

    competition_id: int
    season_id: int
    slug: str  # canonical ID we use internally (e.g., 'wc_2018', 'euro_2024')
    name: str  # human-readable label

    @property
    def matches_url(self) -> str:
        return f"{STATSBOMB_BASE_URL}/matches/{self.competition_id}/{self.season_id}.json"

    def events_url(self, match_id: int) -> str:
        return f"{STATSBOMB_BASE_URL}/events/{match_id}.json"


# Six modern men's international tournaments. Slugs match what the
# `LockDecision.held_out_tournaments` field expects and what the spike
# doc references.
SUPPORTED_TOURNAMENTS: tuple[StatsBombTournament, ...] = (
    StatsBombTournament(43, 3, "wc_2018", "FIFA World Cup 2018"),
    StatsBombTournament(43, 106, "wc_2022", "FIFA World Cup 2022"),
    StatsBombTournament(55, 43, "euro_2020", "UEFA Euro 2020"),
    StatsBombTournament(55, 282, "euro_2024", "UEFA Euro 2024"),
    StatsBombTournament(223, 282, "copa_2024", "Copa América 2024"),
    StatsBombTournament(1267, 107, "afcon_2023", "AFCON 2023"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Match outcome — the ingest target shape
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StatsBombMatchOutcome:
    """One historical match aggregated to match-level outcomes.

    This is the canonical handoff to the backtest harness: every field
    needed to score 1X2 / BTTS / O2.5 / Corners O/U calibration is here.
    """

    match_id: int
    tournament_slug: str
    match_date: date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_corners: int
    away_corners: int
    # Expected goals — sum of `shot.statsbomb_xg` per team across the match.
    # Forward-information signal: a team that "should have scored 2.5 but
    # only scored 1" carries different next-match info than a team that
    # scored 1 from 0.8 xG. None when no shots were observed (rare; usually
    # extreme defensive matches).
    home_xg: float | None = None
    away_xg: float | None = None
    home_shots: int | None = None
    away_shots: int | None = None

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def total_corners(self) -> int:
        return self.home_corners + self.away_corners

    @property
    def btts(self) -> int:
        """Both teams to score outcome (0/1)."""
        return int(self.home_goals > 0 and self.away_goals > 0)

    @property
    def outcome_1x2(self) -> int:
        """0 = home win, 1 = draw, 2 = away win."""
        if self.home_goals > self.away_goals:
            return 0
        if self.home_goals == self.away_goals:
            return 1
        return 2


# ─────────────────────────────────────────────────────────────────────────────
# Download helpers (cache-first; resumable)
# ─────────────────────────────────────────────────────────────────────────────


def _download_json(
    url: str, cache_path: Path, *, force_refresh: bool = False
) -> Path:
    if cache_path.exists() and not force_refresh:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def download_match_list(
    tournament: StatsBombTournament,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> list[dict]:
    """Download (or load cached) match list for one tournament."""
    name = f"{tournament.competition_id}_{tournament.season_id}.json"
    path = _download_json(
        tournament.matches_url,
        cache_dir / "matches" / name,
        force_refresh=force_refresh,
    )
    return json.loads(path.read_text())


def download_events(
    match_id: int,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> list[dict]:
    """Download (or load cached) event stream for one match."""
    path = _download_json(
        f"{STATSBOMB_BASE_URL}/events/{match_id}.json",
        cache_dir / "events" / f"{match_id}.json",
        force_refresh=force_refresh,
    )
    return json.loads(path.read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Event → match-outcome aggregation
# ─────────────────────────────────────────────────────────────────────────────


def count_corners_per_team(events: list[dict]) -> dict[str, int]:
    """Count corner kicks per team from an event stream.

    StatsBomb encodes corners as ``type.name == 'Pass'`` events with
    ``pass.type.name == 'Corner'``. Each corner kick appears exactly once.
    """
    counts: dict[str, int] = {}
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Pass":
            continue
        if ((ev.get("pass") or {}).get("type") or {}).get("name") != "Corner":
            continue
        team = (ev.get("team") or {}).get("name")
        if team:
            counts[team] = counts.get(team, 0) + 1
    return counts


def aggregate_xg_per_team(events: list[dict]) -> tuple[dict[str, float], dict[str, int]]:
    """Sum ``shot.statsbomb_xg`` per team and count shots.

    StatsBomb encodes shot events with ``type.name == 'Shot'`` and the
    expected-goal value at ``shot.statsbomb_xg``. Returns
    ``(xg_per_team, shot_count_per_team)``.

    Shots without xG (rare — usually own goals or missing data) contribute
    to the shot count but not to xG. Penalty shootouts in event data are
    already excluded by StatsBomb's ``period <= 4`` convention.
    """
    xg: dict[str, float] = {}
    shots: dict[str, int] = {}
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Shot":
            continue
        # Skip penalty-shootout events (period 5).
        if ev.get("period", 0) > 4:
            continue
        team = (ev.get("team") or {}).get("name")
        if not team:
            continue
        shots[team] = shots.get(team, 0) + 1
        x = (ev.get("shot") or {}).get("statsbomb_xg")
        if x is not None:
            xg[team] = xg.get(team, 0.0) + float(x)
    return xg, shots


def aggregate_match_to_outcome(
    match_meta: dict,
    events: list[dict],
    tournament_slug: str,
) -> StatsBombMatchOutcome:
    """Combine match metadata + events into a calibration-ready outcome."""
    home_name = match_meta["home_team"]["home_team_name"]
    away_name = match_meta["away_team"]["away_team_name"]
    corners = count_corners_per_team(events)
    xg, shots = aggregate_xg_per_team(events)
    return StatsBombMatchOutcome(
        match_id=int(match_meta["match_id"]),
        tournament_slug=tournament_slug,
        match_date=datetime.strptime(match_meta["match_date"], "%Y-%m-%d").date(),
        home_team=home_name,
        away_team=away_name,
        home_goals=int(match_meta["home_score"]),
        away_goals=int(match_meta["away_score"]),
        home_corners=corners.get(home_name, 0),
        away_corners=corners.get(away_name, 0),
        home_xg=xg.get(home_name),
        away_xg=xg.get(away_name),
        home_shots=shots.get(home_name),
        away_shots=shots.get(away_name),
    )


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end ingest
# ─────────────────────────────────────────────────────────────────────────────


def ingest_all_tournaments(
    tournaments: Iterable[StatsBombTournament] = SUPPORTED_TOURNAMENTS,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
    progress: bool = True,
) -> list[StatsBombMatchOutcome]:
    """Download and aggregate every match across the given tournaments."""
    out: list[StatsBombMatchOutcome] = []
    for t in tournaments:
        if progress:
            print(
                f"[statsbomb] {t.name} "
                f"(comp={t.competition_id}, season={t.season_id})",
                file=sys.stderr,
            )
        matches = download_match_list(t, cache_dir=cache_dir, force_refresh=force_refresh)
        for i, m in enumerate(matches, start=1):
            try:
                events = download_events(
                    int(m["match_id"]), cache_dir=cache_dir, force_refresh=force_refresh
                )
            except Exception as exc:
                print(f"[statsbomb]   {m['match_id']}: failed ({exc})", file=sys.stderr)
                continue
            try:
                out.append(aggregate_match_to_outcome(m, events, t.slug))
            except (KeyError, ValueError) as exc:
                print(f"[statsbomb]   {m['match_id']}: skip ({exc})", file=sys.stderr)
                continue
            if progress and i % 10 == 0:
                print(f"[statsbomb]   {i}/{len(matches)}", file=sys.stderr)
    return out


def write_outcomes_to_parquet(
    outcomes: list[StatsBombMatchOutcome],
    output: Path = DEFAULT_OUTCOMES_PARQUET,
) -> None:
    import polars as pl

    output.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "match_id": [o.match_id for o in outcomes],
            "tournament_slug": [o.tournament_slug for o in outcomes],
            "match_date": [o.match_date for o in outcomes],
            "home_team": [o.home_team for o in outcomes],
            "away_team": [o.away_team for o in outcomes],
            "home_goals": [o.home_goals for o in outcomes],
            "away_goals": [o.away_goals for o in outcomes],
            "home_corners": [o.home_corners for o in outcomes],
            "away_corners": [o.away_corners for o in outcomes],
            "home_xg": [o.home_xg for o in outcomes],
            "away_xg": [o.away_xg for o in outcomes],
            "home_shots": [o.home_shots for o in outcomes],
            "away_shots": [o.away_shots for o in outcomes],
        }
    )
    df.write_parquet(output)


def load_outcomes_from_parquet(
    path: Path = DEFAULT_OUTCOMES_PARQUET,
) -> list[StatsBombMatchOutcome]:
    """Read back the cached outcomes Parquet (avoid re-downloading).

    Tolerates older Parquets without xG columns — those rows return None
    for xG / shots fields. Predictors that need xG should fall back to
    a goals-only path when these are absent.
    """
    import polars as pl

    df = pl.read_parquet(path)
    has_xg = "home_xg" in df.columns
    return [
        StatsBombMatchOutcome(
            match_id=int(row["match_id"]),
            tournament_slug=str(row["tournament_slug"]),
            match_date=row["match_date"],
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
            home_corners=int(row["home_corners"]),
            away_corners=int(row["away_corners"]),
            home_xg=(float(row["home_xg"]) if has_xg and row["home_xg"] is not None else None),
            away_xg=(float(row["away_xg"]) if has_xg and row["away_xg"] is not None else None),
            home_shots=(int(row["home_shots"]) if has_xg and row["home_shots"] is not None else None),
            away_shots=(int(row["away_shots"]) if has_xg and row["away_shots"] is not None else None),
        )
        for row in df.iter_rows(named=True)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Where to cache downloaded JSON",
    )
    p.add_argument(
        "--output", type=Path, default=DEFAULT_OUTCOMES_PARQUET,
        help="Parquet output for match-level outcomes",
    )
    p.add_argument(
        "--tournaments", nargs="+", default=None,
        help="Subset of tournament slugs to ingest (default: all 6)",
    )
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    selected: tuple[StatsBombTournament, ...]
    if args.tournaments:
        wanted = set(args.tournaments)
        selected = tuple(t for t in SUPPORTED_TOURNAMENTS if t.slug in wanted)
        if not selected:
            valid = [t.slug for t in SUPPORTED_TOURNAMENTS]
            print(f"No tournaments matched. Valid: {valid}", file=sys.stderr)
            sys.exit(1)
    else:
        selected = SUPPORTED_TOURNAMENTS
    outcomes = ingest_all_tournaments(
        selected, cache_dir=args.cache_dir, force_refresh=args.force_refresh
    )
    print(f"[statsbomb] {len(outcomes)} match outcomes total", file=sys.stderr)
    write_outcomes_to_parquet(outcomes, args.output)
    print(f"[statsbomb] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
