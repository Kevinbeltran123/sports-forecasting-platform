"""Advanced team metrics mined from the cached StatsBomb events.

These are signals an elite analyst reads that the earlier extractors did NOT
capture — and they need NO new scraping, just a deeper pass over the 314 cached
matches:

  field_tilt          — share of final-third passes+carries (territorial control)
  gk_goals_prevented  — xG of shots-on-target faced minus goals conceded (keeper
                        shot-stopping; + = saves above expectation)
  game-state xG split — fraction of attacking xG generated while leading / level /
                        trailing (match-shape: front-runner vs chaser vs game-manager)

Sidecar to tactical_identity. Pure per-match extractor separated from aggregation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

_ON_TARGET = {"Saved", "Goal", "Saved To Post"}
_FINAL_THIRD_X = 80.0  # StatsBomb pitch length 120; attacking third starts at x=80
_DEF_ACTIONS = {"Interception", "Block", "Clearance"}  # + Tackle duels
_LONG_PASS_X = 15.0    # forward delta for a pass to count "direct/long"


@dataclass(frozen=True)
class MatchAdvanced:
    """One team's advanced tallies for a single match."""

    team: str
    ft_for: int          # final-third passes+carries (this team)
    ft_against: int      # ... opponent
    xgot_faced: float    # xG of opponent shots on target
    goals_conceded: int
    xg_leading: float    # team's attacking xG while ahead
    xg_level: float
    xg_trailing: float
    line_height: float   # mean x of defensive actions (high = high line/press)
    directness: float    # share of passes played long/forward (>=15y)


@dataclass(frozen=True)
class AdvancedTeamProfile:
    team: str
    n_matches: int
    field_tilt: DistributionStat              # mean share in [0,1]
    gk_goals_prevented_per_match: DistributionStat
    line_height: DistributionStat             # mean def-action x (high=high line)
    directness: DistributionStat              # long-pass share (high=direct)
    xg_share_leading: float                   # fraction of total attacking xG
    xg_share_level: float
    xg_share_trailing: float


def _x(ev: dict, key: str = "location") -> float | None:
    loc = ev.get(key) or []
    return loc[0] if len(loc) >= 1 else None


def _goal_timeline(events: list[dict]) -> list[tuple[int, str]]:
    """(minute, scoring_team) for all goals — open play + own goals."""
    goals: list[tuple[int, str]] = []
    for e in events:
        t = (e.get("type") or {}).get("name")
        team = (e.get("team") or {}).get("name", "")
        if t == "Shot" and (e.get("shot") or {}).get("outcome", {}).get("name") == "Goal":
            goals.append((e.get("minute", 0), team))
        elif t == "Own Goal For":          # credited to the benefiting team
            goals.append((e.get("minute", 0), team))
    return sorted(goals)


def _state(team: str, opp: str, minute: int, goals: list[tuple[int, str]]) -> str:
    """team's game state strictly BEFORE `minute`."""
    gf = sum(1 for m, t in goals if m < minute and t == team)
    ga = sum(1 for m, t in goals if m < minute and t == opp)
    return "leading" if gf > ga else "trailing" if gf < ga else "level"


def extract_match(events: list[dict], home: str, away: str) -> list[MatchAdvanced]:
    """Both teams' advanced tallies for one match."""
    goals = _goal_timeline(events)
    acc = {home: _blank(), away: _blank()}
    opp_of = {home: away, away: home}
    for e in events:
        t = (e.get("type") or {}).get("name")
        team = (e.get("team") or {}).get("name", "")
        if team not in acc:
            continue
        if t == "Pass":
            if (_x(e) or 0) >= _FINAL_THIRD_X:
                acc[team]["ft"] += 1
                acc[opp_of[team]]["ft_opp"] += 1
            end = (e.get("pass") or {}).get("end_location") or []
            loc = e.get("location") or []
            if len(end) >= 1 and len(loc) >= 1:
                acc[team]["pass_n"] += 1
                if (end[0] - loc[0]) >= _LONG_PASS_X:
                    acc[team]["pass_long"] += 1
        elif t == "Carry":
            if (_x(e) or 0) >= _FINAL_THIRD_X:
                acc[team]["ft"] += 1
                acc[opp_of[team]]["ft_opp"] += 1
        elif t in _DEF_ACTIONS or (t == "Duel" and (e.get("duel") or {}).get("type", {}).get("name") == "Tackle"):
            if (_x(e) is not None):
                acc[team]["def_x"].append(_x(e))
        elif t == "Shot":
            shot = e.get("shot") or {}
            xg = float(shot.get("statsbomb_xg") or 0.0)
            outcome = (shot.get("outcome") or {}).get("name")
            # game-state attacking xG (this team)
            st = _state(team, opp_of[team], e.get("minute", 0), goals)
            acc[team]["xg" + {"leading": "l", "level": "v", "trailing": "t"}[st]] += xg
            # opponent's keeper faces this shot if on target
            if outcome in _ON_TARGET:
                acc[opp_of[team]]["xgot"] += xg
            if outcome == "Goal":
                acc[opp_of[team]]["gc"] += 1
    return [MatchAdvanced(
                team=tm, ft_for=d["ft"], ft_against=d["ft_opp"],
                xgot_faced=d["xgot"], goals_conceded=d["gc"],
                xg_leading=d["xgl"], xg_level=d["xgv"], xg_trailing=d["xgt"],
                line_height=(sum(d["def_x"]) / len(d["def_x"]) if d["def_x"] else 0.0),
                directness=(d["pass_long"] / d["pass_n"] if d["pass_n"] else 0.0))
            for tm, d in acc.items()]


def _blank() -> dict:
    return dict(ft=0, ft_opp=0, xgot=0.0, gc=0, xgl=0.0, xgv=0.0, xgt=0.0,
                def_x=[], pass_n=0, pass_long=0)


def _bootstrap(values: list[float], n_boot: int = 2000, seed: int = 42) -> DistributionStat:
    n = len(values)
    if n == 0:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return DistributionStat(mean=float(arr.mean()),
                            ci_low=float(np.percentile(boots, 2.5)),
                            ci_high=float(np.percentile(boots, 97.5)), n=n)


def aggregate_team(team: str, matches: list[MatchAdvanced]) -> AdvancedTeamProfile:
    tilts, prevented, heights, directs = [], [], [], []
    xgl = xgv = xgt = 0.0
    for m in matches:
        den = m.ft_for + m.ft_against
        if den > 0:
            tilts.append(m.ft_for / den)
        prevented.append(m.xgot_faced - m.goals_conceded)
        if m.line_height > 0:
            heights.append(m.line_height)
        if m.directness > 0:
            directs.append(m.directness)
        xgl += m.xg_leading
        xgv += m.xg_level
        xgt += m.xg_trailing
    total = xgl + xgv + xgt
    share = (lambda v: v / total if total > 0 else 0.0)
    return AdvancedTeamProfile(
        team=team, n_matches=len(matches),
        field_tilt=_bootstrap(tilts),
        gk_goals_prevented_per_match=_bootstrap(prevented),
        line_height=_bootstrap(heights),
        directness=_bootstrap(directs),
        xg_share_leading=share(xgl), xg_share_level=share(xgv),
        xg_share_trailing=share(xgt),
    )


def build_profiles(per_match_teams: list[list[MatchAdvanced]]) -> dict[str, AdvancedTeamProfile]:
    by_team: dict[str, list[MatchAdvanced]] = defaultdict(list)
    for match in per_match_teams:
        for ma in match:
            by_team[ma.team].append(ma)
    return {t: aggregate_team(t, ms) for t, ms in by_team.items()}
