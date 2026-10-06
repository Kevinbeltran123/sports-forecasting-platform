"""Football feature engineering -- Polars-based.

CRITICAL (DATA-05): computed_at must be set to the time features are computed,
NOT the fixture kickoff time. Every historical lookup MUST filter:
    historical.filter(pl.col("kickoff_utc") < fixture.kickoff_utc)

Phase 2: Full feature set (~40-60 features) across:
  - rolling form (goals scored/conceded, wins) over windows 3/5/10
  - ELO ratings via penaltyblog.ratings.Elo
  - H2H aggregates
  - rest days + matchday (matchday passed externally)
  - Dixon-Coles attack/defense ratings (pre-fitted externally, passed in)
  - odds signals (implied probability, line movement)
  - motivation context (top-4 race, relegation, dead rubber)
  - pressing/tactical (shots, possession, set-piece rate)

T-05-03 / T-02-03-01: the caller is responsible for passing ``historical_matches``
pre-filtered to prior data. As a defense-in-depth, every internal filter also
applies ``pl.col("kickoff_utc") < cutoff``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import structlog

from bip.sports import FeatureMatrix, FixtureData

logger = structlog.get_logger(__name__)

ROLLING_WINDOWS = [3, 5, 10]


class FeatureEngineer:
    """Builds point-in-time correct feature matrices from API-Football data.

    Phase 1: Basic availability features (stats present, lineups confirmed).
    Phase 2 (ML Core): Full ~40-60 feature set covering form, H2H, Dixon-Coles
    ratings, ELO, odds signals, motivation, pressing/tactical.
    """

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def build_features_for_fixture(
        self,
        fixture: FixtureData,
        raw_stats: dict,
        raw_lineups: dict,
        computed_at: datetime | None = None,
        *,
        historical_matches: pl.DataFrame | None = None,
        raw_h2h: dict | None = None,  # noqa: ARG002 -- reserved for future H2H enrichment
        raw_odds: dict | None = None,
        raw_standings: dict | None = None,
        elo_state: Any | None = None,
        dc_attack: dict[str, float] | None = None,
        dc_defence: dict[str, float] | None = None,
    ) -> FeatureMatrix:
        """Build feature matrix from raw API responses + historical corpus.

        Args:
            fixture: Fixture being analyzed.
            raw_stats: Response from GET /fixtures/statistics.
            raw_lineups: Response from GET /fixtures/lineups.
            computed_at: Point-in-time timestamp. Defaults to now (UTC).
            historical_matches: Polars DataFrame with columns kickoff_utc,
                home_team, away_team, home_goals, away_goals. Caller must
                ensure this contains ALL history; this method filters by
                ``kickoff_utc < fixture.kickoff_utc`` per lookup.
            raw_h2h: Optional dict — reserved for Phase 2+ H2H augmentation.
            raw_odds: Optional dict with "opening" / "current" sub-dicts of
                1/X/2 decimal odds. Produces odds-signal features.
            raw_standings: Optional dict with a "standings" list of per-team
                rank snapshots; produces motivation features.
            elo_state: Optional pre-fitted ``penaltyblog.ratings.Elo`` instance
                snapshotted to before ``fixture.kickoff_utc``.
            dc_attack / dc_defence: Dicts of Dixon-Coles attack/defence rating
                per team, pre-fitted externally.

        Returns:
            FeatureMatrix with computed_at set to the passed-in (or current) time.
        """
        if computed_at is None:
            computed_at = datetime.now(UTC)

        features: dict[str, float] = {}

        # Basic availability (Phase 1 carryover)
        features["stats_available"] = (
            1.0 if len(raw_stats.get("response", [])) > 0 else 0.0
        )
        features["lineups_confirmed"] = (
            1.0 if len(raw_lineups.get("response", [])) == 2 else 0.0
        )

        # Rolling form + rest days + H2H (require historical corpus)
        if historical_matches is not None and len(historical_matches) > 0:
            features.update(
                self._rolling_form(
                    team=fixture.home_team,
                    historical=historical_matches,
                    cutoff=fixture.kickoff_utc,
                    side="home",
                    windows=ROLLING_WINDOWS,
                )
            )
            features.update(
                self._rolling_form(
                    team=fixture.away_team,
                    historical=historical_matches,
                    cutoff=fixture.kickoff_utc,
                    side="away",
                    windows=ROLLING_WINDOWS,
                )
            )
            features.update(
                self._rest_days(
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    historical=historical_matches,
                    cutoff=fixture.kickoff_utc,
                )
            )
            features.update(
                self._h2h_features(
                    home=fixture.home_team,
                    away=fixture.away_team,
                    historical=historical_matches,
                    cutoff=fixture.kickoff_utc,
                )
            )

        # ELO snapshot (caller passes state snapshotted to before kickoff)
        if elo_state is not None:
            features.update(
                self._elo_snapshot(
                    elo=elo_state,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                )
            )

        # Dixon-Coles attack/defence (pre-fitted dict lookups)
        if dc_attack is not None and dc_defence is not None:
            features.update(
                self._dc_features(
                    home=fixture.home_team,
                    away=fixture.away_team,
                    attack=dc_attack,
                    defence=dc_defence,
                )
            )

        # Odds signals (opening vs current movement, implied probabilities)
        if raw_odds is not None:
            features.update(self._odds_signals(raw_odds))

        # Pressing / tactical (from raw_stats — safe when response empty)
        features.update(
            self._pressing_tactical(
                raw_stats=raw_stats,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
            )
        )

        # Motivation context (from league-table standings snapshot)
        if raw_standings is not None:
            features.update(
                self._motivation_features(
                    home=fixture.home_team,
                    away=fixture.away_team,
                    raw_standings=raw_standings,
                    cutoff=fixture.kickoff_utc,
                )
            )

        logger.info(
            "features_built",
            fixture_id=fixture.fixture_id,
            feature_count=len(features),
            computed_at=computed_at.isoformat(),
        )

        return FeatureMatrix(
            fixture_id=fixture.fixture_id,
            sport=fixture.sport,
            league=fixture.league,
            computed_at=computed_at,
            features=features,
            kickoff_utc=fixture.kickoff_utc,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
        )

    def to_parquet_row(
        self,
        fm: FeatureMatrix,
        matchday: int,
        season: str,
    ) -> pl.DataFrame:
        """Convert FeatureMatrix to a Polars DataFrame row for ParquetStore.

        Includes all 4 partition columns required by ParquetStore:
        sport, league, season, matchday.

        Args:
            fm: Computed feature matrix.
            matchday: Matchday number for Hive partition key.
            season: Season string (e.g., "2025-2026") for Hive partition key.

        Returns:
            Single-row Polars DataFrame ready for ParquetStore.write_features().
        """
        row: dict[str, list] = {
            "fixture_id": [fm.fixture_id],
            "sport": [fm.sport],
            "league": [fm.league],
            "season": [season],
            "matchday": [matchday],
            "computed_at": [fm.computed_at.isoformat()],
            "feature_schema_version": [2],   # D-08: Phase 1 data is implicit v1 (absent column)
        }
        for key, val in fm.features.items():
            row[key] = [val]
        return pl.DataFrame(row)

    # -----------------------------------------------------------------
    # Private — feature groups
    # -----------------------------------------------------------------

    def _rolling_form(
        self,
        team: str,
        historical: pl.DataFrame,
        cutoff: datetime,
        side: str,
        windows: list[int],
    ) -> dict[str, float]:
        """Rolling form: goals for / against, wins over last N matches.

        T-02-03-01: every row used satisfies ``kickoff_utc < cutoff``.

        historical DataFrame must contain columns:
            kickoff_utc (datetime), home_team, away_team, home_goals, away_goals

        Returns 3*len(windows) keys per side: goals_for_N, goals_against_N, wins_N.
        """
        out: dict[str, float] = {}
        team_rows = historical.filter(
            (pl.col("kickoff_utc") < cutoff)
            & (
                (pl.col("home_team") == team)
                | (pl.col("away_team") == team)
            )
        ).sort("kickoff_utc", descending=True)

        for n in windows:
            last_n = team_rows.head(n)
            if len(last_n) == 0:
                out[f"form_{side}_goals_for_{n}"] = 0.0
                out[f"form_{side}_goals_against_{n}"] = 0.0
                out[f"form_{side}_wins_{n}"] = 0.0
                continue
            # Compute per-row goals-for / goals-against for this team
            enriched = last_n.with_columns(
                pl.when(pl.col("home_team") == team)
                .then(pl.col("home_goals"))
                .otherwise(pl.col("away_goals"))
                .alias("gf"),
                pl.when(pl.col("home_team") == team)
                .then(pl.col("away_goals"))
                .otherwise(pl.col("home_goals"))
                .alias("ga"),
            )
            goals_for = float(enriched["gf"].sum())
            goals_against = float(enriched["ga"].sum())
            wins = float((enriched["gf"] > enriched["ga"]).sum())
            out[f"form_{side}_goals_for_{n}"] = goals_for
            out[f"form_{side}_goals_against_{n}"] = goals_against
            out[f"form_{side}_wins_{n}"] = wins

        return out

    def _elo_snapshot(
        self,
        elo: Any,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """Snapshot ELO ratings — caller MUST pass an Elo state whose updates
        only include matches finishing before ``fixture.kickoff_utc`` (Pitfall 5).

        Uses ``penaltyblog.ratings.Elo.get_team_rating`` — not hand-rolled.
        """
        home = float(elo.get_team_rating(home_team))
        away = float(elo.get_team_rating(away_team))
        return {
            "elo_home": home,
            "elo_away": away,
            "elo_diff": home - away,
        }

    def _h2h_features(
        self,
        home: str,
        away: str,
        historical: pl.DataFrame,
        cutoff: datetime,
    ) -> dict[str, float]:
        """Head-to-head aggregates, temporal-safe (kickoff_utc < cutoff)."""
        meetings = historical.filter(
            (pl.col("kickoff_utc") < cutoff)
            & (
                ((pl.col("home_team") == home) & (pl.col("away_team") == away))
                | ((pl.col("home_team") == away) & (pl.col("away_team") == home))
            )
        ).sort("kickoff_utc", descending=True)

        if len(meetings) == 0:
            return {
                "h2h_total_meetings": 0.0,
                "h2h_home_wins": 0.0,
                "h2h_avg_goals": 0.0,
                "h2h_days_since_last": 9999.0,
            }

        enriched = meetings.with_columns(
            (
                (
                    (pl.col("home_team") == home)
                    & (pl.col("home_goals") > pl.col("away_goals"))
                )
                | (
                    (pl.col("away_team") == home)
                    & (pl.col("away_goals") > pl.col("home_goals"))
                )
            )
            .cast(pl.Int64)
            .alias("home_win")
        )
        home_wins = float(enriched["home_win"].sum())
        total_goals = float(
            meetings["home_goals"].sum() + meetings["away_goals"].sum()
        )
        last_kickoff = meetings["kickoff_utc"][0]
        if isinstance(last_kickoff, datetime) and last_kickoff.tzinfo is None:
            last_kickoff = last_kickoff.replace(tzinfo=UTC)
        cutoff_utc = cutoff if cutoff.tzinfo is not None else cutoff.replace(tzinfo=UTC)
        days_since = float((cutoff_utc - last_kickoff).total_seconds() / 86400.0)

        return {
            "h2h_total_meetings": float(len(meetings)),
            "h2h_home_wins": home_wins,
            "h2h_avg_goals": total_goals / float(len(meetings)),
            "h2h_days_since_last": days_since,
        }

    def _rest_days(
        self,
        home_team: str,
        away_team: str,
        historical: pl.DataFrame,
        cutoff: datetime,
    ) -> dict[str, float]:
        """Days since each team's last fixture (kickoff_utc < cutoff)."""
        cutoff_utc = cutoff if cutoff.tzinfo is not None else cutoff.replace(tzinfo=UTC)

        def last_game_days(team: str) -> float:
            team_rows = historical.filter(
                (pl.col("kickoff_utc") < cutoff)
                & (
                    (pl.col("home_team") == team)
                    | (pl.col("away_team") == team)
                )
            ).sort("kickoff_utc", descending=True)
            if len(team_rows) == 0:
                return 14.0  # sensible default (bye week)
            last = team_rows["kickoff_utc"][0]
            if isinstance(last, datetime) and last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            return float((cutoff_utc - last).total_seconds() / 86400.0)

        return {
            "rest_days_home": last_game_days(home_team),
            "rest_days_away": last_game_days(away_team),
        }

    def _dc_features(
        self,
        home: str,
        away: str,
        attack: dict[str, float],
        defence: dict[str, float],
    ) -> dict[str, float]:
        """Dixon-Coles attack/defence features from pre-fitted penaltyblog model."""
        a_home = float(attack.get(home, 0.0))
        a_away = float(attack.get(away, 0.0))
        d_home = float(defence.get(home, 0.0))
        d_away = float(defence.get(away, 0.0))
        return {
            "dc_attack_home": a_home,
            "dc_attack_away": a_away,
            "dc_defence_home": d_home,
            "dc_defence_away": d_away,
            "dc_attack_diff": a_home - a_away,
            "dc_defence_diff": d_home - d_away,
        }

    def _odds_signals(self, raw_odds: dict) -> dict[str, float]:
        """Extract implied probability (opening & current) plus line movement.

        Expects raw_odds shape::

            {"opening": {"1": <decimal>, "X": <decimal>, "2": <decimal>},
             "current": {"1": <decimal>, "X": <decimal>, "2": <decimal>}}

        Missing keys default to 0 (implied probability 0, movement 0) so this
        never raises on malformed input (T-02-03-03).
        """
        out: dict[str, float] = {}
        opening = raw_odds.get("opening", {}) or {}
        current = raw_odds.get("current", {}) or {}
        for key in ("1", "X", "2"):
            try:
                o = float(opening.get(key) or 0.0)
            except (TypeError, ValueError):
                o = 0.0
            try:
                c = float(current.get(key) or 0.0)
            except (TypeError, ValueError):
                c = 0.0
            out[f"odds_open_implied_{key}"] = (1.0 / o) if o > 0 else 0.0
            out[f"odds_curr_implied_{key}"] = (1.0 / c) if c > 0 else 0.0
            out[f"odds_movement_{key}"] = (c - o) if (o > 0 and c > 0) else 0.0
        return out

    def _pressing_tactical(
        self,
        raw_stats: dict,
        home_team: str,
        away_team: str,
    ) -> dict[str, float]:
        """Shots, possession, set-piece rate from API-Football /statistics.

        Safe when raw_stats["response"] is empty or malformed (T-02-03-03).
        """
        out: dict[str, float] = {
            "shots_home": 0.0,
            "shots_away": 0.0,
            "possession_home": 0.0,
            "possession_away": 0.0,
            "set_pieces_home": 0.0,
            "set_pieces_away": 0.0,
        }
        # WR-05: API-Football's statistics list often contains multiple
        # entries whose type strings match the previous substring matchers
        # ("Total Shots" + "Shots insidebox", "Corner Kicks" + a hypothetical
        # "Corner Kicks Conceded"). The earlier ``in`` matcher accepted both
        # and the LAST one won — order-dependent on API response shape.
        # Switch to exact-name matching against an explicit whitelist so
        # the lookup is deterministic and robust to new "*-like" stat
        # names appearing in future API responses.
        _SHOT_TOTAL = "total shots"
        _POSSESSION = "ball possession"
        _CORNERS = "corner kicks"
        for item in raw_stats.get("response", []) or []:
            team_name = (item.get("team") or {}).get("name", "")
            if team_name == home_team:
                side = "home"
            elif team_name == away_team:
                side = "away"
            else:
                continue
            for stat in item.get("statistics", []) or []:
                name = (stat.get("type") or "").strip().lower()
                val = stat.get("value")
                if val is None:
                    continue
                try:
                    num = float(str(val).rstrip("%"))
                except (TypeError, ValueError):
                    continue
                if name == _SHOT_TOTAL:
                    out[f"shots_{side}"] = num
                elif name == _POSSESSION:
                    out[f"possession_{side}"] = num
                elif name == _CORNERS:
                    out[f"set_pieces_{side}"] = num
        return out

    def _motivation_features(
        self,
        home: str,
        away: str,
        raw_standings: dict,
        cutoff: datetime,  # noqa: ARG002 -- caller passes snapshot valid at cutoff
    ) -> dict[str, float]:
        """Top-4 race, relegation, dead-rubber flags from league-table state.

        The caller is responsible for providing a table-state snapshot taken
        BEFORE ``cutoff`` (Pitfall: table state must not include the fixture
        being predicted).

        Expects raw_standings shape::

            {"standings": [
                {"team": str, "rank": int, "points": int,
                 "played": int, "total_games": int},
                ...
            ]}
        """
        rows = raw_standings.get("standings", []) or []
        by_team: dict[str, dict] = {r["team"]: r for r in rows if "team" in r}
        total_games = int(rows[0].get("total_games", 38)) if rows else 38

        def mot(team: str, is_home: bool) -> dict[str, float]:
            side = "home" if is_home else "away"
            r = by_team.get(team, {})
            try:
                rank = int(r.get("rank", 99))
            except (TypeError, ValueError):
                rank = 99
            try:
                played = int(r.get("played", 0))
            except (TypeError, ValueError):
                played = 0
            games_remaining = max(0, total_games - played)
            # Top-4 race contender: loosely anyone within top-6
            top4 = 1.0 if 1 <= rank <= 6 else 0.0
            # Relegation battle: bottom quartile
            relegation = 1.0 if rank >= max(15, total_games // 2 - 2) else 0.0
            return {
                f"motivation_top4_{side}": top4,
                f"motivation_relegation_{side}": relegation,
                f"motivation_games_remaining_{side}": float(games_remaining),
            }

        out: dict[str, float] = {}
        out.update(mot(home, is_home=True))
        out.update(mot(away, is_home=False))

        games_rem_home = out.get("motivation_games_remaining_home", 38.0)
        games_rem_away = out.get("motivation_games_remaining_away", 38.0)
        neither_race = (
            out.get("motivation_top4_home", 0.0) == 0.0
            and out.get("motivation_top4_away", 0.0) == 0.0
            and out.get("motivation_relegation_home", 0.0) == 0.0
            and out.get("motivation_relegation_away", 0.0) == 0.0
        )
        out["motivation_dead_rubber"] = (
            1.0 if (neither_race and games_rem_home <= 5 and games_rem_away <= 5) else 0.0
        )
        return out
