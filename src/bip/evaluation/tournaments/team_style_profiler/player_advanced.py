"""Per-player advanced metrics from cached StatsBomb events — progression,
creation (SCA/GCA) and defence (the weak-link inputs).

Complements player_props (shots/fouls/cards) with what an opposition report
actually needs:
  prog_passes / prog_carries  — ball progression (who drives play)
  sca / gca                   — shot-/goal-creating actions (creativity)
  tackles / interceptions / clearances / recoveries — defensive volume
  dribbled_past               — times beaten 1v1 (the slow-CB weak-link signal)
  aerial_won / aerial_lost    — aerial reliability (set-piece / target weak-link)

Reuses player_props' minutes parsing + per-90 bootstrap. 100% offline.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from bip.evaluation.tournaments.team_style_profiler.player_props import (
    _bootstrap_rate,
    _confidence_for,
    PropConfidence,
    parse_player_minutes,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

_PROGRESSIVE_X = 10.0     # yards toward goal for a pass/carry to count progressive
_SCA_LOOKBACK = 2         # offensive actions before a shot credited as SCA
_OFFENSIVE = {"Pass", "Carry", "Dribble", "Foul Won"}

_METRICS = ("prog_pass", "prog_carry", "sca", "gca", "tackles", "interceptions",
            "blocks", "clearances", "recoveries", "dribbled_past",
            "aerial_won", "aerial_lost")


@dataclass(frozen=True)
class PlayerAdvancedProfile:
    player_id: int
    player_name: str
    team: str
    position: str
    n_matches: int
    confidence: PropConfidence

    prog_passes_per90: DistributionStat
    prog_carries_per90: DistributionStat
    sca_per90: DistributionStat
    gca_per90: DistributionStat
    tackles_per90: DistributionStat
    interceptions_per90: DistributionStat
    clearances_per90: DistributionStat
    recoveries_per90: DistributionStat
    dribbled_past_per90: DistributionStat
    aerial_won_per90: DistributionStat
    aerial_lost_per90: DistributionStat

    @property
    def aerial_win_rate(self) -> float:
        tot = self.aerial_won_per90.mean + self.aerial_lost_per90.mean
        return self.aerial_won_per90.mean / tot if tot > 0 else 0.0


def _fwd(ev: dict, end_key: str) -> bool:
    loc = ev.get("location") or []
    end = (ev.get(end_key) or {}).get("end_location") if end_key in ("pass", "carry") else None
    if end is None or len(loc) < 1 or len(end) < 1:
        return False
    return (end[0] - loc[0]) >= _PROGRESSIVE_X


def extract_match_advanced(
    events: list[dict],
) -> tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]]]:
    """Per-player advanced counts + metadata for one match."""
    counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    meta: dict[int, tuple[str, str, str]] = {}

    # metadata from Starting XI
    for e in events:
        if (e.get("type") or {}).get("name") == "Starting XI":
            for slot in (e.get("tactics") or {}).get("lineup", []):
                p = slot["player"]
                meta[p["id"]] = (p["name"], (e.get("team") or {}).get("name", ""),
                                 (slot.get("position") or {}).get("name", "?"))

    for e in events:
        t = (e.get("type") or {}).get("name")
        pid = (e.get("player") or {}).get("id")
        if pid is None:
            continue
        meta.setdefault(pid, ((e.get("player") or {}).get("name", str(pid)),
                              (e.get("team") or {}).get("name", ""), "?"))
        if t == "Pass":
            if _fwd(e, "pass"):
                counts[pid]["prog_pass"] += 1
        elif t == "Carry":
            if _fwd(e, "carry"):
                counts[pid]["prog_carry"] += 1
        elif t == "Duel":
            dt = (e.get("duel") or {}).get("type", {}).get("name")
            if dt == "Tackle":
                counts[pid]["tackles"] += 1
            elif dt == "Aerial Lost":
                counts[pid]["aerial_lost"] += 1
        elif t == "Interception":
            counts[pid]["interceptions"] += 1
        elif t == "Block":
            counts[pid]["blocks"] += 1
        elif t == "Clearance":
            counts[pid]["clearances"] += 1
        elif t == "Ball Recovery":
            counts[pid]["recoveries"] += 1
        elif t == "Dribbled Past":
            counts[pid]["dribbled_past"] += 1
        # aerial won: flag on Clearance/Pass/Miscontrol/Shot
        for k in ("clearance", "pass", "miscontrol", "shot"):
            if (e.get(k) or {}).get("aerial_won"):
                counts[pid]["aerial_won"] += 1
                break

    _credit_sca(events, counts)
    return counts, meta


def _credit_sca(events: list[dict], counts: dict[int, dict[str, float]]) -> None:
    """Credit the last `_SCA_LOOKBACK` offensive actions before each shot."""
    by_possession: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        poss = e.get("possession")
        if poss is not None:
            by_possession[poss].append(e)
    for poss_events in by_possession.values():
        poss_events.sort(key=lambda e: e.get("index", 0))
        for i, e in enumerate(poss_events):
            if (e.get("type") or {}).get("name") != "Shot":
                continue
            shooting_team = (e.get("team") or {}).get("name")
            is_goal = (e.get("shot") or {}).get("outcome", {}).get("name") == "Goal"
            credited: set[int] = set()
            for prev in reversed(poss_events[:i]):
                if (prev.get("type") or {}).get("name") not in _OFFENSIVE:
                    continue
                if (prev.get("team") or {}).get("name") != shooting_team:
                    continue
                ppid = (prev.get("player") or {}).get("id")
                if ppid is None or ppid in credited:
                    continue
                counts[ppid]["sca"] += 1
                if is_goal:
                    counts[ppid]["gca"] += 1
                credited.add(ppid)
                if len(credited) >= _SCA_LOOKBACK:
                    break


def build_advanced_profiles(
    per_match: list[tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]], dict[int, float]]],
) -> list[PlayerAdvancedProfile]:
    samples: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {m: [] for m in _METRICS})
    name_of, team_of = {}, {}
    pos_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for counts, meta, minutes in per_match:
        for pid, mins in minutes.items():
            if mins < 20.0:
                continue
            c = counts.get(pid, {})
            for m in _METRICS:
                samples[pid][m].append((float(c.get(m, 0.0)), mins))
            if pid in meta:
                nm, tm, pos = meta[pid]
                name_of[pid], team_of[pid] = nm, tm
                if pos != "?":
                    pos_counts[pid][pos] += 1

    out: list[PlayerAdvancedProfile] = []
    for pid, ms in samples.items():
        n = len(ms["prog_pass"])
        if n == 0:
            continue
        pos = max(pos_counts[pid], key=pos_counts[pid].get) if pos_counts[pid] else "?"
        boot = {m: _bootstrap_rate(ms[m]) for m in _METRICS}
        out.append(PlayerAdvancedProfile(
            player_id=pid, player_name=name_of.get(pid, str(pid)),
            team=team_of.get(pid, ""), position=pos, n_matches=n,
            confidence=_confidence_for(n),
            prog_passes_per90=boot["prog_pass"], prog_carries_per90=boot["prog_carry"],
            sca_per90=boot["sca"], gca_per90=boot["gca"],
            tackles_per90=boot["tackles"], interceptions_per90=boot["interceptions"],
            clearances_per90=boot["clearances"], recoveries_per90=boot["recoveries"],
            dribbled_past_per90=boot["dribbled_past"],
            aerial_won_per90=boot["aerial_won"], aerial_lost_per90=boot["aerial_lost"],
        ))
    return out


def weak_links(profiles: list[PlayerAdvancedProfile], top_n: int = 3) -> list[dict]:
    """Defenders/midfielders most beaten 1v1 or weakest in the air — the
    structural vulnerabilities an opposition report targets."""
    defs = [p for p in profiles
            if p.confidence in ("green", "yellow")
            and any(k in p.position for k in ("Back", "Defen", "Center Mid", "Wing Back"))]
    out = []
    for p in sorted(defs, key=lambda x: -x.dribbled_past_per90.mean)[:top_n]:
        if p.dribbled_past_per90.mean <= 0:
            continue
        out.append({"player": p.player_name, "position": p.position,
                    "reason": f"beaten {p.dribbled_past_per90.mean:.1f}×/90 1v1",
                    "kind": "pace/1v1"})
    for p in sorted(defs, key=lambda x: x.aerial_win_rate):
        if p.aerial_won_per90.mean + p.aerial_lost_per90.mean < 1.5:
            continue
        out.append({"player": p.player_name, "position": p.position,
                    "reason": f"aerial win rate {p.aerial_win_rate:.0%}",
                    "kind": "aerial"})
        if len([o for o in out if o["kind"] == "aerial"]) >= top_n:
            break
    return out
