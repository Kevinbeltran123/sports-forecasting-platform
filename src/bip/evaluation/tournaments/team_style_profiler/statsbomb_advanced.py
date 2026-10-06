"""StatsBomb-derived advanced team stats for WC2026 teams.

Computes from raw events JSON in data/cache/statsbomb/events:
  - xG / xGA per match (sum of statsbomb_xg over shots)
  - Big chances = shots with statsbomb_xg > 0.3
  - Penalty xG (separate from open-play)
  - Set-piece xG (free kicks + corners + throw-ins indirect)
  - Open-play xG
  - Shots under pressure rate
  - PPDA proxy = opponent passes / (our pressures + tackles + interceptions in opp third)
  - Conversion rate (goals / xG ratio)
  - Goalkeeper post-shot xG conceded vs goals (PSxG-G proxy)

Sidecar to basic TSV + API-Football advanced. Sourced from 314 cached matches
across WC18/22, Euro20/24, AFCON23, Copa24.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

SB_SCHEMA_VERSION = "1.0.0"

# Team name mapping: TSV canonical -> StatsBomb name
SB_NAME_OF = {
    "Cape Verde": "Cape Verde Islands",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "USA": "United States",
}


class SBMatchAggregate(BaseModel):
    """Per-match StatsBomb aggregate for one team."""

    model_config = ConfigDict(frozen=True)

    match_id: int
    competition: str
    season: str
    team: str
    opponent: str
    is_home: bool

    # Shooting
    shots_total: int
    shots_on_target: int
    xg_total: float
    xg_open_play: float
    xg_set_piece: float
    xg_penalty: float
    big_chances: int  # xG > 0.3
    goals: int
    shots_under_pressure: int

    # Defense (against)
    shots_against: int
    xg_against: float
    big_chances_against: int
    goals_against: int

    # Possession-related
    passes_total: int
    passes_into_box: int
    pressures: int
    pressures_in_attacking_third: int

    # Defensive actions in opp third (for PPDA)
    defensive_actions_in_opp_third: int  # pressures + tackles + interceptions
    opp_passes_total: int


class TeamStatsBombProfile(BaseModel):
    """Aggregated StatsBomb profile for a team."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = SB_SCHEMA_VERSION
    team_name: str
    confederation: str
    n_matches: int
    competitions_covered: list[str]
    last_updated: datetime

    xg_for_per_match: DistributionStat
    xg_against_per_match: DistributionStat
    xg_open_play_per_match: DistributionStat
    xg_set_piece_per_match: DistributionStat
    xg_penalty_per_match: DistributionStat
    big_chances_per_match: DistributionStat
    big_chances_against_per_match: DistributionStat
    shots_per_match: DistributionStat
    shots_against_per_match: DistributionStat
    pressures_per_match: DistributionStat
    ppda: DistributionStat
    """Passes per defensive action — LOW = high pressing intensity."""
    conversion_rate: DistributionStat
    """Goals / xG ratio. >1 = clinical, <1 = wasteful."""
    set_piece_xg_share: DistributionStat
    """Fraction of total xG from set pieces."""


def _norm_team_name(canonical: str) -> str:
    return SB_NAME_OF.get(canonical, canonical)


def _reverse_norm(sb_name: str) -> str:
    for canon, sb in SB_NAME_OF.items():
        if sb == sb_name:
            return canon
    return sb_name


def aggregate_match_for_team(
    events: list[dict],
    team_name: str,
    match_meta: dict,
) -> SBMatchAggregate | None:
    """Compute team-level aggregates from raw StatsBomb events."""
    home = match_meta["home_team"]["home_team_name"]
    away = match_meta["away_team"]["away_team_name"]
    if team_name not in {home, away}:
        return None
    opp = away if team_name == home else home
    is_home = team_name == home

    shots_total = 0
    shots_on_target = 0
    xg_total = 0.0
    xg_open = 0.0
    xg_sp = 0.0
    xg_pen = 0.0
    big_chances = 0
    goals = 0
    shots_under_pressure = 0

    shots_against = 0
    xg_against = 0.0
    big_chances_against = 0
    goals_against = 0

    passes_total = 0
    passes_into_box = 0
    pressures = 0
    pressures_atk3 = 0
    def_actions_opp3 = 0

    opp_passes_total = 0

    # Field length in StatsBomb = 120; attacking third = x >= 80
    for ev in events:
        etype = (ev.get("type") or {}).get("name", "")
        e_team = (ev.get("team") or {}).get("name", "")
        is_our_event = e_team == team_name

        if etype == "Shot":
            shot = ev.get("shot") or {}
            xg = float(shot.get("statsbomb_xg") or 0.0)
            shot_type = (shot.get("type") or {}).get("name", "")
            outcome = (shot.get("outcome") or {}).get("name", "")
            if is_our_event:
                shots_total += 1
                xg_total += xg
                if outcome == "Goal":
                    goals += 1
                if outcome in {"Saved", "Goal", "Saved To Post"}:
                    shots_on_target += 1
                if xg > 0.3:
                    big_chances += 1
                if ev.get("under_pressure"):
                    shots_under_pressure += 1
                if shot_type == "Open Play":
                    xg_open += xg
                elif shot_type == "Penalty":
                    xg_pen += xg
                else:
                    xg_sp += xg
            else:
                shots_against += 1
                xg_against += xg
                if outcome == "Goal":
                    goals_against += 1
                if xg > 0.3:
                    big_chances_against += 1

        elif etype == "Pass":
            if is_our_event:
                passes_total += 1
                p = ev.get("pass") or {}
                end = p.get("end_location") or []
                if len(end) >= 2 and end[0] >= 102 and 18 <= end[1] <= 62:
                    passes_into_box += 1
            else:
                opp_passes_total += 1

        elif etype == "Pressure":
            if is_our_event:
                pressures += 1
                loc = ev.get("location") or []
                if len(loc) >= 1 and loc[0] >= 80:
                    pressures_atk3 += 1
                    def_actions_opp3 += 1

        elif etype in {"Duel", "Interception", "Block"}:
            # Tackles register as Duel type=Tackle in StatsBomb
            if is_our_event:
                loc = ev.get("location") or []
                if len(loc) >= 1 and loc[0] >= 80:
                    def_actions_opp3 += 1

    return SBMatchAggregate(
        match_id=match_meta["match_id"],
        competition=match_meta["competition"]["competition_name"],
        season=match_meta["season"]["season_name"],
        team=team_name,
        opponent=opp,
        is_home=is_home,
        shots_total=shots_total,
        shots_on_target=shots_on_target,
        xg_total=round(xg_total, 4),
        xg_open_play=round(xg_open, 4),
        xg_set_piece=round(xg_sp, 4),
        xg_penalty=round(xg_pen, 4),
        big_chances=big_chances,
        goals=goals,
        shots_under_pressure=shots_under_pressure,
        shots_against=shots_against,
        xg_against=round(xg_against, 4),
        big_chances_against=big_chances_against,
        goals_against=goals_against,
        passes_total=passes_total,
        passes_into_box=passes_into_box,
        pressures=pressures,
        pressures_in_attacking_third=pressures_atk3,
        defensive_actions_in_opp_third=def_actions_opp3,
        opp_passes_total=opp_passes_total,
    )


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 42) -> DistributionStat:
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


def aggregate_team_profile(
    team_name: str,
    confederation: str,
    matches: list[SBMatchAggregate],
    last_updated: datetime,
) -> TeamStatsBombProfile:
    if not matches:
        raise ValueError(f"No matches for {team_name}")

    comps = sorted(set(f"{m.competition} {m.season}" for m in matches))

    ppda_per_match = []
    for m in matches:
        if m.defensive_actions_in_opp_third > 0:
            ppda_per_match.append(m.opp_passes_total / m.defensive_actions_in_opp_third)

    conv_per_match = []
    for m in matches:
        if m.xg_total > 0.05:
            conv_per_match.append(m.goals / m.xg_total)

    sp_share_per_match = []
    for m in matches:
        if m.xg_total > 0:
            sp_share_per_match.append((m.xg_set_piece + m.xg_penalty) / m.xg_total)

    return TeamStatsBombProfile(
        team_name=team_name,
        confederation=confederation,
        n_matches=len(matches),
        competitions_covered=comps,
        last_updated=last_updated,
        xg_for_per_match=_bootstrap_ci([m.xg_total for m in matches]),
        xg_against_per_match=_bootstrap_ci([m.xg_against for m in matches]),
        xg_open_play_per_match=_bootstrap_ci([m.xg_open_play for m in matches]),
        xg_set_piece_per_match=_bootstrap_ci([m.xg_set_piece for m in matches]),
        xg_penalty_per_match=_bootstrap_ci([m.xg_penalty for m in matches]),
        big_chances_per_match=_bootstrap_ci([float(m.big_chances) for m in matches]),
        big_chances_against_per_match=_bootstrap_ci([float(m.big_chances_against) for m in matches]),
        shots_per_match=_bootstrap_ci([float(m.shots_total) for m in matches]),
        shots_against_per_match=_bootstrap_ci([float(m.shots_against) for m in matches]),
        pressures_per_match=_bootstrap_ci([float(m.pressures) for m in matches]),
        ppda=_bootstrap_ci(ppda_per_match),
        conversion_rate=_bootstrap_ci(conv_per_match),
        set_piece_xg_share=_bootstrap_ci(sp_share_per_match),
    )


def build_match_index(matches_dir: Path) -> tuple[dict[int, dict], dict[str, list[int]]]:
    """Return (match_id -> match_meta, team_name -> [match_ids])."""
    match_index: dict[int, dict] = {}
    team_to_matches: dict[str, list[int]] = defaultdict(list)
    for f in matches_dir.glob("*.json"):
        for m in json.loads(f.read_text()):
            mid = m["match_id"]
            match_index[mid] = m
            team_to_matches[m["home_team"]["home_team_name"]].append(mid)
            team_to_matches[m["away_team"]["away_team_name"]].append(mid)
    return match_index, dict(team_to_matches)
