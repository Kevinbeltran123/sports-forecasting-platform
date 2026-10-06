"""Player Prop Profiler — per-player prop-relevant rates from StatsBomb events.

PHILOSOPHY (operator directive 2026-05-30):
  "Quizá se podría abrir un nuevo enfoque para player props."

Sibling of tactical_identity, one level down: from team style to the individual
prop board. Computes per-90 rates (with bootstrap CIs + sample-size confidence)
for the markets where edge actually lives, ranked softest-first:

  fouls_committed, fouls_drawn,
  yellow_cards                    <- SOFT markets (referee-driven, under-modeled)
  shots, shots_on_target          <- medium
  xg / anytime-scorer, assists    <- hardest (saturated, juiced)

100% offline from the 314 cached StatsBomb matches (40 WC2026 teams). The 8
StatsBomb-less teams have no player profiles (honest — not fabricated). Prop
ODDS are out of budget, so this is a model-now / price-manually-later track.

Minutes are parsed from Starting XI + Substitution + red-card events so per-90
rates are not distorted by substitute cameos.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

# Players carry fewer matches than teams — looser thresholds than the TSV flag.
N_GREEN_MIN = 8
N_YELLOW_MIN = 4
# Sub cameos below this minute count are excluded from per-90 (rate noise).
MIN_MINUTES_FOR_RATE = 20.0

PropConfidence = Literal["green", "yellow", "red"]
PropSource = Literal["statsbomb", "espn", "api_football"]

_RED_CARDS = {"Red Card", "Second Yellow"}
_YELLOW_CARDS = {"Yellow Card", "Second Yellow"}
_ON_TARGET = {"Goal", "Saved", "Saved To Post"}


@dataclass(frozen=True)
class PlayerPropProfile:
    """Per-player prop rates (per-90, bootstrapped) across StatsBomb matches."""

    player_id: int
    player_name: str
    team: str
    position: str  # from Starting XI; "?" if only ever a sub
    n_matches: int
    minutes_total: float
    confidence: PropConfidence

    shots_per90: DistributionStat
    shots_on_target_per90: DistributionStat
    xg_per90: DistributionStat
    goals_per90: DistributionStat
    fouls_committed_per90: DistributionStat
    fouls_drawn_per90: DistributionStat  # fouls suffered (StatsBomb "Foul Won")
    yellow_cards_per90: DistributionStat
    key_passes_per90: DistributionStat
    assists_per90: DistributionStat

    penalties_taken: int  # count across sample (set-piece duty signal)

    # "statsbomb": real minutes + xG. "espn": minutes unknown (each appearance
    # treated as 90' → per-90 with full-match assumption, conservative for subs)
    # and NO xG (anytime-scorer falls back to the goals rate).
    source: PropSource = "statsbomb"

    @property
    def p_anytime_scorer(self) -> float:
        """P(>=1 goal | plays 90'): 1 - exp(-rate). Uses xG when available
        (StatsBomb), else the goals rate (ESPN has no xG)."""
        rate = self.xg_per90.mean if self.xg_per90.mean > 0 else self.goals_per90.mean
        return float(1.0 - np.exp(-rate))


def _confidence_for(n: int) -> PropConfidence:
    if n >= N_GREEN_MIN:
        return "green"
    if n >= N_YELLOW_MIN:
        return "yellow"
    return "red"


def _match_end_minute(events: list[dict]) -> int:
    return max((e.get("minute", 0) for e in events), default=90)


def parse_player_minutes(events: list[dict]) -> dict[int, float]:
    """Minutes played per player_id from Starting XI + subs + red cards."""
    end = _match_end_minute(events)
    on: dict[int, float] = {}
    off: dict[int, float] = {}
    for e in events:
        t = (e.get("type") or {}).get("name")
        if t == "Starting XI":
            for slot in (e.get("tactics") or {}).get("lineup", []):
                on[slot["player"]["id"]] = 0.0
        elif t == "Substitution":
            pid = (e.get("player") or {}).get("id")
            if pid is not None:
                off[pid] = float(e.get("minute", end))
            rep = (e.get("substitution") or {}).get("replacement") or {}
            if rep.get("id") is not None:
                on[rep["id"]] = float(e.get("minute", end))
    # Red / second-yellow → player leaves at that minute.
    for e in events:
        card = (
            ((e.get("foul_committed") or {}).get("card") or {}).get("name")
            or ((e.get("bad_behaviour") or {}).get("card") or {}).get("name")
        )
        if card in _RED_CARDS:
            pid = (e.get("player") or {}).get("id")
            if pid is not None:
                off[pid] = min(off.get(pid, float(end)), float(e.get("minute", end)))
    return {pid: max(0.0, off.get(pid, float(end)) - on_m) for pid, on_m in on.items()}


def parse_player_counts(
    events: list[dict],
) -> tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]]]:
    """Per-player event counts + metadata (name, team, position) for one match."""
    counts: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    meta: dict[int, tuple[str, str, str]] = {}
    for e in events:
        t = (e.get("type") or {}).get("name")
        team = (e.get("team") or {}).get("name", "")
        if t == "Starting XI":
            for slot in (e.get("tactics") or {}).get("lineup", []):
                p = slot["player"]
                meta[p["id"]] = (p["name"], (e.get("team") or {}).get("name", ""),
                                 (slot.get("position") or {}).get("name", "?"))
        elif t == "Substitution":
            rep = (e.get("substitution") or {}).get("replacement") or {}
            if rep.get("id") is not None:
                meta.setdefault(rep["id"], (rep.get("name", ""), team, "?"))
        pid = (e.get("player") or {}).get("id")
        if pid is None:
            continue
        if t == "Shot":
            s = e.get("shot") or {}
            counts[pid]["shots"] += 1
            counts[pid]["xg"] += float(s.get("statsbomb_xg") or 0.0)
            outcome = (s.get("outcome") or {}).get("name")
            if outcome == "Goal":
                counts[pid]["goals"] += 1
            if outcome in _ON_TARGET:
                counts[pid]["sot"] += 1
            if (s.get("type") or {}).get("name") == "Penalty":
                counts[pid]["pens"] += 1
        elif t == "Foul Committed":
            counts[pid]["fouls"] += 1
            card = ((e.get("foul_committed") or {}).get("card") or {}).get("name")
            if card in _YELLOW_CARDS:
                counts[pid]["yellows"] += 1
        elif t == "Bad Behaviour":
            card = ((e.get("bad_behaviour") or {}).get("card") or {}).get("name")
            if card in _YELLOW_CARDS:
                counts[pid]["yellows"] += 1
        elif t == "Foul Won":
            counts[pid]["fouls_won"] += 1
        elif t == "Pass":
            p = e.get("pass") or {}
            if p.get("shot_assist"):
                counts[pid]["key_passes"] += 1
            if p.get("goal_assist"):
                counts[pid]["assists"] += 1
    return counts, meta


def _bootstrap_rate(
    pairs: list[tuple[float, float]], n_boot: int = 2000, seed: int = 42
) -> DistributionStat:
    """Per-90 rate with CI by resampling matches. pairs = [(count, minutes)]."""
    usable = [(c, m) for c, m in pairs if m >= MIN_MINUTES_FOR_RATE]
    n = len(usable)
    if n == 0:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    counts = np.array([c for c, _ in usable], dtype=float)
    mins = np.array([m for _, m in usable], dtype=float)
    total_min = mins.sum()
    mean = float(counts.sum() / total_min * 90.0) if total_min > 0 else 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    bc = counts[idx].sum(axis=1)
    bm = mins[idx].sum(axis=1)
    valid = bm > 0
    boots = bc[valid] / bm[valid] * 90.0
    return DistributionStat(
        mean=mean,
        ci_low=float(np.percentile(boots, 2.5)) if boots.size else mean,
        ci_high=float(np.percentile(boots, 97.5)) if boots.size else mean,
        n=n,
    )


_METRICS = ("shots", "sot", "xg", "goals", "fouls", "fouls_won", "yellows", "key_passes", "assists")


def build_player_profiles(
    per_match: list[tuple[dict[int, dict[str, float]], dict[int, tuple[str, str, str]], dict[int, float]]],
    source: PropSource = "statsbomb",
) -> list[PlayerPropProfile]:
    """Aggregate parsed per-match (counts, meta, minutes) into player profiles."""
    # player_id -> metric -> list[(count, minutes)]
    samples: dict[int, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: {m: [] for m in _METRICS}
    )
    minutes_total: dict[int, float] = defaultdict(float)
    pens_total: dict[int, int] = defaultdict(int)
    name_of: dict[int, str] = {}
    team_of: dict[int, str] = {}
    pos_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for counts, meta, minutes in per_match:
        for pid, mins in minutes.items():
            if mins < MIN_MINUTES_FOR_RATE:
                continue
            c = counts.get(pid, {})
            for m in _METRICS:
                samples[pid][m].append((float(c.get(m, 0.0)), mins))
            minutes_total[pid] += mins
            pens_total[pid] += int(c.get("pens", 0))
            if pid in meta:
                name, team, pos = meta[pid]
                name_of[pid] = name
                team_of[pid] = team
                if pos != "?":
                    pos_counts[pid][pos] += 1

    profiles: list[PlayerPropProfile] = []
    for pid, metric_samples in samples.items():
        n = len(metric_samples["shots"])  # matches with usable minutes
        if n == 0:
            continue
        pos = max(pos_counts[pid], key=pos_counts[pid].get) if pos_counts[pid] else "?"
        profiles.append(PlayerPropProfile(
            player_id=pid,
            player_name=name_of.get(pid, str(pid)),
            team=team_of.get(pid, ""),
            position=pos,
            n_matches=n,
            minutes_total=minutes_total[pid],
            confidence=_confidence_for(n),
            shots_per90=_bootstrap_rate(metric_samples["shots"]),
            shots_on_target_per90=_bootstrap_rate(metric_samples["sot"]),
            xg_per90=_bootstrap_rate(metric_samples["xg"]),
            goals_per90=_bootstrap_rate(metric_samples["goals"]),
            fouls_committed_per90=_bootstrap_rate(metric_samples["fouls"]),
            fouls_drawn_per90=_bootstrap_rate(metric_samples["fouls_won"]),
            yellow_cards_per90=_bootstrap_rate(metric_samples["yellows"]),
            key_passes_per90=_bootstrap_rate(metric_samples["key_passes"]),
            assists_per90=_bootstrap_rate(metric_samples["assists"]),
            penalties_taken=pens_total[pid],
            source=source,
        ))
    return profiles


@dataclass(frozen=True)
class PropCandidate:
    """One player flagged for a prop market, with the driving stat."""

    market: str            # e.g. "Fouls cometidas", "Disparos al arco", "Anytime scorer"
    player_name: str
    position: str
    stat: str              # human-readable driving number
    softness: int          # 1=softest (fouls/cards) ... 3=hardest (scorer)
    confidence: PropConfidence
    team: str = ""         # owning team (for cross-source injury/referee linking)
    flag: str = ""         # non-destructive annotation (e.g. "⛔ LESIONADO", availability)


def prop_board(
    profiles: list[PlayerPropProfile], top_n: int = 3
) -> list[PropCandidate]:
    """Rank a team's prop candidates softest-market-first (where edge lives).

    Markets ordered by book softness: fouls/cards (referee-driven, under-modeled)
    > shots/SoT > anytime scorer (saturated). Only green/yellow players (red =
    too few matches to trust the rate). The operator prices these with real odds.
    """
    usable = [p for p in profiles if p.confidence in ("green", "yellow")]

    def top(key, n=top_n):
        return sorted(usable, key=key, reverse=True)[:n]

    board: list[PropCandidate] = []
    # 1 — SOFTEST: fouls committed (referee-driven). DEF/MID skew.
    for p in top(lambda x: x.fouls_committed_per90.mean):
        if p.fouls_committed_per90.mean <= 0:
            continue
        board.append(PropCandidate(
            "Faltas cometidas (over)", p.player_name, p.position,
            f"{p.fouls_committed_per90.mean:.1f} faltas/90 (n={p.n_matches})",
            softness=1, confidence=p.confidence))
    # 1 — fouls drawn (faltas recibidas): under-modeled, feeds opponent-cards
    # and penalty markets (a high foul-drawer is fouled → rival yellows / pens).
    for p in top(lambda x: x.fouls_drawn_per90.mean):
        if p.fouls_drawn_per90.mean <= 0:
            continue
        board.append(PropCandidate(
            "Faltas recibidas (over)", p.player_name, p.position,
            f"{p.fouls_drawn_per90.mean:.1f} faltas recibidas/90 (n={p.n_matches})",
            softness=1, confidence=p.confidence))
    # 1 — cards: high foulers who also see yellows.
    for p in top(lambda x: x.yellow_cards_per90.mean):
        if p.yellow_cards_per90.mean <= 0:
            continue
        board.append(PropCandidate(
            "Tarjeta a jugador", p.player_name, p.position,
            f"{p.yellow_cards_per90.mean:.2f} amarillas/90, {p.fouls_committed_per90.mean:.1f} faltas/90",
            softness=1, confidence=p.confidence))
    # 2 — shots on target.
    for p in top(lambda x: x.shots_on_target_per90.mean):
        if p.shots_on_target_per90.mean <= 0:
            continue
        board.append(PropCandidate(
            "Disparos al arco (over)", p.player_name, p.position,
            f"{p.shots_on_target_per90.mean:.1f} SoT/90, {p.shots_per90.mean:.1f} disparos/90",
            softness=2, confidence=p.confidence))
    # 3 — HARDEST: anytime scorer (xG-driven; goals rate when xG absent/ESPN).
    def scorer_rate(x):
        return x.xg_per90.mean if x.xg_per90.mean > 0 else x.goals_per90.mean
    for p in top(scorer_rate):
        if scorer_rate(p) <= 0:
            continue
        pen = " · penaltis" if p.penalties_taken > 0 else ""
        driver = (f"xG {p.xg_per90.mean:.2f}/90" if p.xg_per90.mean > 0
                  else f"{p.goals_per90.mean:.2f} goles/90")
        board.append(PropCandidate(
            "Anytime scorer", p.player_name, p.position,
            f"P(gol|90')≈{p.p_anytime_scorer:.0%}, {driver}{pen}",
            softness=3, confidence=p.confidence))
    return board
