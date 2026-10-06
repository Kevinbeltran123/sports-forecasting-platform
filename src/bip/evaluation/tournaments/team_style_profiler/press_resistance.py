"""Press-resistance — how well a team builds under opponent pressure.

Complement to press_intensity (PPDA = how hard a team PRESSES): this is the
RECEIVING side — pass completion when under pressure. A press-fragile team (low
completion when pressed) coughs the ball up in dangerous areas; crossed with a
high-pressing opponent, that is the BTTS / over (transition-goals) signal. A
press-resistant team neutralises a high press.

100% offline from cached StatsBomb events. Corpus-only — the 8 WC2026 teams
without StatsBomb data have no press-resistance profile (honest, not fabricated).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.tsv_schema import DistributionStat

N_GREEN_MIN = 10
N_YELLOW_MIN = 5

# Empirical terciles of completion-under-pressure across corpus teams (n>=5),
# 2026-05-31, ~76 passes-under-pressure/match (see 40_build_press_resistance.py).
RESISTANT_MIN = 0.74       # p67 (~0.736): >= -> press-resistant
FRAGILE_MAX = 0.69         # p33 (~0.693): <= -> press-fragile

Resistance = Literal["press-resistant", "average", "press-fragile"]
ResConfidence = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class PressResistance:
    team: str
    n_matches: int
    confidence: ResConfidence
    passes_under_pressure_per_match: DistributionStat
    completion_under_pressure: DistributionStat   # rate in [0, 1]
    resistance: Resistance


def _confidence_for(n: int) -> ResConfidence:
    if n >= N_GREEN_MIN:
        return "green"
    if n >= N_YELLOW_MIN:
        return "yellow"
    return "red"


def _resistance(completion: float) -> Resistance:
    if completion >= RESISTANT_MIN:
        return "press-resistant"
    if completion <= FRAGILE_MAX:
        return "press-fragile"
    return "average"


def _bootstrap_mean(values: list[float], n_boot: int = 2000, seed: int = 42) -> DistributionStat:
    n = len(values)
    if n == 0:
        return DistributionStat(mean=0.0, ci_low=0.0, ci_high=0.0, n=0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return DistributionStat(
        mean=float(arr.mean()),
        ci_low=float(np.percentile(boots, 2.5)),
        ci_high=float(np.percentile(boots, 97.5)),
        n=n,
    )


def extract_match_press(events: list[dict]) -> dict[str, tuple[int, int]]:
    """Per-team (passes_under_pressure, completed_under_pressure) for one match.

    A StatsBomb pass with no `pass.outcome` is completed; an outcome
    (Incomplete / Out) marks a failed pass.
    """
    up: dict[str, int] = defaultdict(int)
    comp: dict[str, int] = defaultdict(int)
    for e in events:
        if (e.get("type") or {}).get("name") != "Pass" or not e.get("under_pressure"):
            continue
        team = (e.get("team") or {}).get("name", "")
        if not team:
            continue
        up[team] += 1
        if not (e.get("pass") or {}).get("outcome"):
            comp[team] += 1
    return {t: (up[t], comp[t]) for t in up}


def build_press_resistance(
    per_match: list[dict[str, tuple[int, int]]],
) -> dict[str, PressResistance]:
    """Aggregate per-match (team -> (passes_up, completed_up)) into profiles."""
    samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for match in per_match:
        for team, pair in match.items():
            samples[team].append(pair)

    out: dict[str, PressResistance] = {}
    for team, rows in samples.items():
        n = len(rows)
        rates = [c / u for u, c in rows if u > 0]
        if not rates:
            continue
        comp_ds = _bootstrap_mean(rates)
        up_ds = _bootstrap_mean([float(u) for u, _ in rows])
        out[team] = PressResistance(
            team=team, n_matches=n, confidence=_confidence_for(n),
            passes_under_pressure_per_match=up_ds,
            completion_under_pressure=comp_ds,
            resistance=_resistance(comp_ds.mean))
    return out


def press_resistance_note(
    presser_press_intensity: str, builder: PressResistance | None
) -> str | None:
    """Cross a high-pressing team with the OTHER team's build-up resistance.

    high press × press-fragile → turnovers in own half → transition chances
        (BTTS / over). high press × press-resistant → press neutralised.
    Only for green/yellow profiles; else None (small-sample honesty).
    """
    if builder is None or builder.confidence not in ("green", "yellow"):
        return None
    if presser_press_intensity != "high_press":
        return None
    rate = builder.completion_under_pressure.mean
    if builder.resistance == "press-fragile":
        return (f"{builder.team} frágil bajo presión ({rate:.0%} pase completado presionado) "
                f"vs presión alta rival → pérdidas en zona propia, transiciones → BTTS/Over.")
    if builder.resistance == "press-resistant":
        return (f"{builder.team} resistente a la presión ({rate:.0%} completado) → neutraliza "
                f"la presión alta rival → menos transiciones de lo que sugiere el estilo.")
    return None
