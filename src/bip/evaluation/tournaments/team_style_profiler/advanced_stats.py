"""Advanced team stats from API-Football /fixtures/players.

Sidecar to the basic TSV — does NOT mutate tsv_schema. Aggregates
player-level stats per fixture to team-level, then bootstraps mean + 95% CI
across all profiled fixtures of a team (same methodology as TSV).

Fields added beyond basic TSV:
  - player_rating_avg (weighted by minutes)
  - passes_total, passes_accuracy, key_passes
  - tackles_total, blocks_total, interceptions_total
  - duels_total, duels_won_rate
  - dribbles_attempts, dribbles_success_rate
  - fouls_committed (already in TSV but re-computed for consistency)
  - penalties_won, penalties_committed
  - gk_saves (sum of all GK saves)
  - offsides_against_team (offsides committed by THIS team — different from
    offsides_against in TSV which means committed by OPPONENT)
"""
from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

ADVANCED_SCHEMA_VERSION = "1.0.0"


class AdvancedMatchStats(BaseModel):
    """Per-match advanced stats aggregated from /fixtures/players."""

    model_config = ConfigDict(frozen=True)

    fixture_id: int
    team_id: int

    # Aggregates (sum across all rostered players, weighted where noted)
    player_rating_weighted_avg: float | None  # by minutes; None if no ratings
    n_players_with_rating: int

    passes_total: int
    passes_accuracy_avg: float | None  # weighted by passes_total
    key_passes: int

    tackles_total: int
    blocks_total: int
    interceptions_total: int

    duels_total: int
    duels_won: int

    dribbles_attempts: int
    dribbles_success: int

    fouls_committed: int
    fouls_drawn: int

    penalties_won: int
    penalties_scored: int
    penalties_missed: int
    penalties_committed: int

    gk_saves: int

    offsides_committed: int  # offsides BY this team (not against)

    @property
    def duels_won_rate(self) -> float | None:
        return (self.duels_won / self.duels_total) if self.duels_total > 0 else None

    @property
    def dribbles_success_rate(self) -> float | None:
        return (self.dribbles_success / self.dribbles_attempts) if self.dribbles_attempts > 0 else None


class TeamAdvancedProfile(BaseModel):
    """Aggregated advanced profile for a team (mean + bootstrap CI across matches)."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = ADVANCED_SCHEMA_VERSION
    team_name: str
    api_football_team_id: int
    confederation: str

    n_matches: int
    last_updated: datetime

    player_rating_avg: DistributionStat
    passes_per_match: DistributionStat
    passes_accuracy_pct: DistributionStat
    key_passes_per_match: DistributionStat
    tackles_per_match: DistributionStat
    blocks_per_match: DistributionStat
    interceptions_per_match: DistributionStat
    duels_per_match: DistributionStat
    duels_won_rate: DistributionStat
    dribbles_attempts_per_match: DistributionStat
    dribbles_success_rate: DistributionStat
    fouls_committed_per_match: DistributionStat
    penalties_won_per_match: DistributionStat
    gk_saves_per_match: DistributionStat
    offsides_committed_per_match: DistributionStat


def _safe_int(x: Any) -> int:
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _safe_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        s = str(x).rstrip("%")
        return float(s)
    except (TypeError, ValueError):
        return None


def extract_advanced_team_stats(
    players_response: list[dict],
    team_id: int,
    fixture_id: int,
) -> AdvancedMatchStats:
    """Aggregate API-Football /fixtures/players response for a single team.

    ``players_response`` is the ``response`` array from the API: a list with
    one entry per team, each containing ``team`` + ``players``.
    """
    team_block = None
    for block in players_response:
        if (block.get("team") or {}).get("id") == team_id:
            team_block = block
            break
    if team_block is None:
        # No data for this team — return zeroed stats
        return AdvancedMatchStats(
            fixture_id=fixture_id, team_id=team_id,
            player_rating_weighted_avg=None, n_players_with_rating=0,
            passes_total=0, passes_accuracy_avg=None, key_passes=0,
            tackles_total=0, blocks_total=0, interceptions_total=0,
            duels_total=0, duels_won=0,
            dribbles_attempts=0, dribbles_success=0,
            fouls_committed=0, fouls_drawn=0,
            penalties_won=0, penalties_scored=0, penalties_missed=0,
            penalties_committed=0, gk_saves=0, offsides_committed=0,
        )

    rating_weighted_sum = 0.0
    rating_weight_total = 0
    n_with_rating = 0

    passes_total = 0
    passes_accuracy_weighted_sum = 0.0
    passes_weight_total = 0
    key_passes = 0

    tackles_total = 0
    blocks_total = 0
    interceptions_total = 0

    duels_total = 0
    duels_won = 0

    dribbles_attempts = 0
    dribbles_success = 0

    fouls_committed = 0
    fouls_drawn = 0

    pens_won = 0
    pens_scored = 0
    pens_missed = 0
    pens_committed = 0

    gk_saves = 0
    offsides_committed = 0

    for player in team_block.get("players", []):
        stats_list = player.get("statistics") or []
        if not stats_list:
            continue
        stats = stats_list[0]
        games = stats.get("games") or {}
        minutes = _safe_int(games.get("minutes"))

        rating = _safe_float(games.get("rating"))
        if rating is not None and minutes > 0:
            rating_weighted_sum += rating * minutes
            rating_weight_total += minutes
            n_with_rating += 1

        ps = stats.get("passes") or {}
        p_total = _safe_int(ps.get("total"))
        passes_total += p_total
        p_acc = _safe_float(ps.get("accuracy"))
        if p_acc is not None and p_total > 0:
            passes_accuracy_weighted_sum += p_acc * p_total
            passes_weight_total += p_total
        key_passes += _safe_int(ps.get("key"))

        tk = stats.get("tackles") or {}
        tackles_total += _safe_int(tk.get("total"))
        blocks_total += _safe_int(tk.get("blocks"))
        interceptions_total += _safe_int(tk.get("interceptions"))

        duels = stats.get("duels") or {}
        duels_total += _safe_int(duels.get("total"))
        duels_won += _safe_int(duels.get("won"))

        dr = stats.get("dribbles") or {}
        dribbles_attempts += _safe_int(dr.get("attempts"))
        dribbles_success += _safe_int(dr.get("success"))

        fouls = stats.get("fouls") or {}
        fouls_committed += _safe_int(fouls.get("committed"))
        fouls_drawn += _safe_int(fouls.get("drawn"))

        pen = stats.get("penalty") or {}
        pens_won += _safe_int(pen.get("won"))
        pens_scored += _safe_int(pen.get("scored"))
        pens_missed += _safe_int(pen.get("missed"))
        pens_committed += _safe_int(pen.get("commited"))  # API typo

        g = stats.get("goals") or {}
        gk_saves += _safe_int(g.get("saves"))

        offsides_committed += _safe_int(stats.get("offsides"))

    rating_avg = (rating_weighted_sum / rating_weight_total) if rating_weight_total > 0 else None
    passes_acc_avg = (passes_accuracy_weighted_sum / passes_weight_total) if passes_weight_total > 0 else None

    return AdvancedMatchStats(
        fixture_id=fixture_id, team_id=team_id,
        player_rating_weighted_avg=rating_avg, n_players_with_rating=n_with_rating,
        passes_total=passes_total, passes_accuracy_avg=passes_acc_avg, key_passes=key_passes,
        tackles_total=tackles_total, blocks_total=blocks_total, interceptions_total=interceptions_total,
        duels_total=duels_total, duels_won=duels_won,
        dribbles_attempts=dribbles_attempts, dribbles_success=dribbles_success,
        fouls_committed=fouls_committed, fouls_drawn=fouls_drawn,
        penalties_won=pens_won, penalties_scored=pens_scored, penalties_missed=pens_missed,
        penalties_committed=pens_committed, gk_saves=gk_saves,
        offsides_committed=offsides_committed,
    )


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> DistributionStat:
    """Mean + 95% bootstrap CI. Skips None values."""
    clean = [v for v in values if v is not None]
    n = len(clean)
    if n == 0:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    arr = np.array(clean, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    return DistributionStat(
        mean=float(arr.mean()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n=n,
    )


def aggregate_advanced_profile(
    team_name: str,
    api_football_team_id: int,
    confederation: str,
    matches: list[AdvancedMatchStats],
    last_updated: datetime,
) -> TeamAdvancedProfile:
    """Bootstrap mean+CI for each dimension across matches."""
    if not matches:
        raise ValueError(f"No matches for {team_name}")

    return TeamAdvancedProfile(
        team_name=team_name,
        api_football_team_id=api_football_team_id,
        confederation=confederation,
        n_matches=len(matches),
        last_updated=last_updated,
        player_rating_avg=_bootstrap_ci([m.player_rating_weighted_avg for m in matches]),
        passes_per_match=_bootstrap_ci([float(m.passes_total) for m in matches]),
        passes_accuracy_pct=_bootstrap_ci([m.passes_accuracy_avg for m in matches]),
        key_passes_per_match=_bootstrap_ci([float(m.key_passes) for m in matches]),
        tackles_per_match=_bootstrap_ci([float(m.tackles_total) for m in matches]),
        blocks_per_match=_bootstrap_ci([float(m.blocks_total) for m in matches]),
        interceptions_per_match=_bootstrap_ci([float(m.interceptions_total) for m in matches]),
        duels_per_match=_bootstrap_ci([float(m.duels_total) for m in matches]),
        duels_won_rate=_bootstrap_ci([m.duels_won_rate for m in matches]),
        dribbles_attempts_per_match=_bootstrap_ci([float(m.dribbles_attempts) for m in matches]),
        dribbles_success_rate=_bootstrap_ci([m.dribbles_success_rate for m in matches]),
        fouls_committed_per_match=_bootstrap_ci([float(m.fouls_committed) for m in matches]),
        penalties_won_per_match=_bootstrap_ci([float(m.penalties_won) for m in matches]),
        gk_saves_per_match=_bootstrap_ci([float(m.gk_saves) for m in matches]),
        offsides_committed_per_match=_bootstrap_ci([float(m.offsides_committed) for m in matches]),
    )
