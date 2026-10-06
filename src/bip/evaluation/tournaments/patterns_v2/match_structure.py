"""Match structure → market family (G2, auditoría 2026-06-12).

EVIDENCE. EDA pre-registrada sobre el corpus StatsBomb (314 partidos, 6 torneos:
WC2018/22, Euro2020/24, Copa2024, AFCON2023), spike 50_match_structure.py.
Clasificador outcome-blind: |ΔElo| pre-partido (Elo as-of-date por replay
cronológico de martj42, MISMA aritmética y escala que el own_elo del SoS del
brief — sin anacronismo TM). Umbrales = terciles redondeados a 25, congelados.

| estructura (|ΔElo|)  | n   | P(fav>córn)        | tie | P(fav>tiros) | dogTT≤1 (grupos) |
|----------------------|-----|--------------------|-----|--------------|------------------|
| pick_em   (<100)     | 108 | 0.51 [0.42, 0.60]  | 9%  | 0.53         | 0.74             |
| favorito  (100-200)  | 102 | 0.59 [0.49, 0.68]  | 7%  | 0.62         | 0.77             |
| mismatch  (>=200)    | 104 | 0.78 [0.69, 0.85]  | 4%  | 0.83         | 0.82             |

Checks: monotonicidad córners+tiros OK; dirección consistente en los 6 torneos
(mismatch >=69% en todos); se mantiene en grupos (79%) y KO (74%); coherente con
P8 (favorite-corner-gap +2.4 grupos, TM>=1.5). NULLS honestos: O2.5 plano
(~0.42-0.45 en las 3 clases — la estructura NO informa totales) y tarjetas sin
señal (3.9/3.0/3.3 amarillas, no monótono; direccional no testeable sin lado).

WHY. Lección MATCH_RESULTS_LOG #6/#7: "equipo más córners" fue oro en
México-Sudáfrica (ΔElo 303 → mismatch, 78%) y habría perdido en Corea-Chequia
(pick'em, 51%, córners 5-4 al rival). El mismo mercado cambia de señal a moneda
según la estructura; la regla vivía solo en la cabeza del operador.

APPLICATION. Capa de GATE/contexto en el brief 43: imprime la clase y qué
familias de mercado tienen base-rate real en ella. NO emite señal a la síntesis
(estructura-Elo y P8-TM son el MISMO mecanismo de fuerza — emitir ambos sería el
doble-conteo que la auditoría G9 señala). Cross-check: señal de córners presente
en un pick'em → warning explícito de no-apostar-dirección.

LIMITATIONS. Base-rates de calibración (sin holdout; n_bucket≈100-108, CIs
mostrados — usar el CI bajo para decisiones). "favorito" (100-200) es zona gris:
59% córners con CI tocando 50% → tratar como débil. El overlay "autobús"
(identidad táctica del rival) NO está validado por separado: leerlo cualitativo.
Volumen (dogTT/O2.5/BTTS) medido SOLO en fase de grupos (los KO mezclan 120').
"""
from __future__ import annotations

from dataclasses import dataclass

# Umbrales congelados (terciles |ΔElo| del corpus, redondeados a 25 — spike 50).
PICK_EM_MAX_GAP = 100.0
FAVORITO_MAX_GAP = 200.0


@dataclass(frozen=True)
class StructureCell:
    """Base-rates empíricas de una clase de estructura (fav = mayor Elo pre-partido)."""

    n: int
    p_fav_more_corners: float
    ci_low: float
    ci_high: float
    p_corner_tie: float
    p_fav_more_shots: float
    corner_diff_mean: float
    xg_diff_mean: float
    p_fav_win: float
    n_group: int
    p_over25_group: float
    p_btts_group: float
    p_dog_tt_under15_group: float
    p_dog_tt_zero_group: float


STRUCTURE_TABLE: dict[str, StructureCell] = {
    # Pegado del output del spike 50 (2026-06-12) — NO editar a mano sin re-correrlo.
    "pick_em": StructureCell(
        n=108, p_fav_more_corners=0.509, ci_low=0.416, ci_high=0.602,
        p_corner_tie=0.093, p_fav_more_shots=0.528, corner_diff_mean=0.71,
        xg_diff_mean=0.13, p_fav_win=0.472, n_group=66, p_over25_group=0.424,
        p_btts_group=0.500, p_dog_tt_under15_group=0.742, p_dog_tt_zero_group=0.364,
    ),
    "favorito": StructureCell(
        n=102, p_fav_more_corners=0.588, ci_low=0.491, ci_high=0.679,
        p_corner_tie=0.069, p_fav_more_shots=0.618, corner_diff_mean=1.71,
        xg_diff_mean=0.44, p_fav_win=0.520, n_group=77, p_over25_group=0.455,
        p_btts_group=0.506, p_dog_tt_under15_group=0.766, p_dog_tt_zero_group=0.390,
    ),
    "mismatch": StructureCell(
        n=104, p_fav_more_corners=0.779, ci_low=0.690, ci_high=0.848,
        p_corner_tie=0.038, p_fav_more_shots=0.825, corner_diff_mean=3.07,
        xg_diff_mean=1.11, p_fav_win=0.625, n_group=85, p_over25_group=0.447,
        p_btts_group=0.376, p_dog_tt_under15_group=0.824, p_dog_tt_zero_group=0.506,
    ),
}

# Guía por mercado, derivada de la tabla con disciplina de CI (señal = CI excluye 0.50):
#   strong = CI bajo > 0.50 con margen; weak = punto > 0.50 pero CI lo toca; none = moneda.
CORNERS_DIRECTION_SIGNAL: dict[str, str] = {
    "pick_em": "none", "favorito": "weak", "mismatch": "strong",
}
SHOTS_DIRECTION_SIGNAL: dict[str, str] = {
    "pick_em": "none", "favorito": "weak", "mismatch": "strong",
}
# Nulls honestos (NO derivar picks de estructura en estos mercados):
#   O2.5 plano 0.42-0.45 en las 3 clases; tarjetas 3.9/3.0/3.3 no monótono.
TOTALS_FLAT_BY_STRUCTURE = True
CARDS_FLAT_BY_STRUCTURE = True


def classify_structure(elo_gap: float) -> str:
    """Clase de estructura por |ΔElo| pre-partido (umbrales congelados del spike 50)."""
    gap = abs(elo_gap)
    if gap < PICK_EM_MAX_GAP:
        return "pick_em"
    if gap < FAVORITO_MAX_GAP:
        return "favorito"
    return "mismatch"


@dataclass(frozen=True)
class StructureVerdict:
    structure: str          # pick_em | favorito | mismatch
    favorite: str           # nombre del equipo con mayor Elo
    underdog: str
    elo_gap: float
    cell: StructureCell
    corners_signal: str     # none | weak | strong
    shots_signal: str


def match_structure_verdict(
    home: str, away: str, elo_home: float | None, elo_away: float | None,
) -> StructureVerdict | None:
    """None si falta Elo de algún lado (el brief lo dice explícito, no silencia — G1)."""
    if elo_home is None or elo_away is None:
        return None
    fav, dog = (home, away) if elo_home >= elo_away else (away, home)
    structure = classify_structure(elo_home - elo_away)
    return StructureVerdict(
        structure=structure,
        favorite=fav,
        underdog=dog,
        elo_gap=abs(elo_home - elo_away),
        cell=STRUCTURE_TABLE[structure],
        corners_signal=CORNERS_DIRECTION_SIGNAL[structure],
        shots_signal=SHOTS_DIRECTION_SIGNAL[structure],
    )
