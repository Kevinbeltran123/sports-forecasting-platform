"""Transferencia amistoso→torneo POR MÉTRICA (G4, auditoría 2026-06-12).

EVIDENCE. Spike 51_friendlies_transfer_secondary.py, dos corpus triangulados:
- A (réplica exp 15): 27 selecciones, amistosos sep-nov 2022 → WC22 (pares 19-27;
  mediana 2 amistosos/equipo → atenuación declarada). Ancla replicada EXACTA:
  gf r=+0.54 / ga r=+0.04 (el hallazgo original de goles).
- B (moderno): 31-41 selecciones bajo DT actual 2023-26, amistosos vs competitivo
  (quali+torneos, con localía — limitación declarada), ≥3 partidos con dato por régimen.

| métrica    | r A (→WC) | r B    | Δ A (WC−amist)      | Δ B    | veredicto ranking |
|------------|-----------|--------|---------------------|--------|-------------------|
| shots      | +0.54     | +0.53  | +2.04 [+0.4,+3.9]   | +2.11  | TRANSFIERE        |
| sot        | +0.44     | +0.38  | +0.20 (ns)          | +0.82  | TRANSFIERE (borde)|
| possession | +0.36 (ρ.60)| +0.59| −6.3 [−14.3,−0.9]   | +5.4*  | TRANSFIERE        |
| goals_for  | +0.54     | +0.16  | −0.37 (ns)          | +0.31  | TRANSFIERE (A)    |
| goals_against| +0.04   | +0.15  | +0.76 [+0.4,+1.1]   | −0.24  | NO                |
| corners    | −0.05     | +0.35  | −0.07 (ns)          | +0.64  | NO (→WC)          |
| cards (TA) | −0.17     | +0.00  | **+1.07 [+0.6,+1.6]**| +0.22 | NO; SHIFT de régimen|
| fouls      | n/d       | −0.07  | n/d                 | +1.19  | NO                |

(* el signo de Δposesión difiere entre corpus → no congelar shift de posesión.)

WHY. El hallazgo de transferencia validado era SOLO de goles; córners/tarjetas/tiros
se usaban con mezcla amistoso+competitivo sin descuento medido. Resultado clave:
los CÓRNERS amistosos no dicen nada del torneo (r=−0.05 → la dirección de córners
sale de la ESTRUCTURA, match_structure/G2, no del promedio del equipo) y las
TARJETAS tienen un shift de régimen real: ~+1.07 TA/equipo (~+2/partido) al pasar
de amistoso a WC — el prior "Under tarjetas en amistoso" cuantificado.

APPLICATION. Brief 43 §3/§5: cuando el perfil de un equipo es amistoso-pesado
(≥50%), degradar explícitamente sus promedios de córners/TA y citar el shift; los
reads de tiros/posesión/ofensiva sobreviven el cambio de régimen.

LIMITATIONS. Corpus A con mediana 2 amistosos/equipo (atenuación: las r de A son
cotas inferiores); corpus B mezcla quali con localía. n=19-41 por celda. Las r son
cross-team (ranking), no garantizan calibración de nivel por equipo.
"""
from __future__ import annotations

from dataclasses import dataclass

# Shift de régimen amistoso→WC en tarjetas (corpus A, CI bootstrap 95%).
CARDS_WC_SHIFT_PER_TEAM = 1.07
CARDS_WC_SHIFT_CI = (0.63, 1.56)


@dataclass(frozen=True)
class TransferFinding:
    metric: str
    ranking: str        # "transfers" | "weak" | "none" — ¿el ranking cross-team transfiere?
    r_wc22: float | None      # corpus A (amistosos→WC22); None = no medible en A
    n_wc22: int | None
    r_modern: float           # corpus B (amistosos→competitivo, DT actual)
    n_modern: int
    note: str           # frase operativa para el brief


FRIENDLY_TRANSFER: dict[str, TransferFinding] = {
    # Pegado del output del spike 51 (2026-06-12) — NO editar a mano sin re-correrlo.
    "shots": TransferFinding(
        "shots", "transfers", 0.54, 22, 0.53, 35,
        "tiros transfieren (r≈0.53 en ambos corpus) — los reads de volumen de tiro sobreviven el régimen"),
    "sot": TransferFinding(
        "sot", "transfers", 0.44, 22, 0.38, 35,
        "SoT transfiere (r 0.38-0.44, borde) — usable con cautela"),
    "possession": TransferFinding(
        "possession", "transfers", 0.36, 19, 0.59, 36,
        "posesión transfiere (ρ 0.50-0.60) — la identidad de estilo es estable entre regímenes"),
    "goals_for": TransferFinding(
        "goals_for", "transfers", 0.54, 27, 0.16, 41,
        "ofensiva transfiere (r=+0.54 →WC, hallazgo original) pero el NIVEL baja en WC"),
    "goals_against": TransferFinding(
        "goals_against", "none", 0.04, 27, 0.15, 41,
        "defensa NO transfiere (r=+0.04) y conceden +0.76 más en WC — GA amistosa no vale nada"),
    "corners": TransferFinding(
        "corners", "none", -0.05, 21, 0.35, 37,
        "córners amistosos NO predicen torneo (r=−0.05 →WC; +0.35 débil moderno) — "
        "la dirección de córners sale de la ESTRUCTURA (§3c), no del promedio del equipo"),
    "cards": TransferFinding(
        "cards", "none", -0.17, 27, 0.00, 31,
        f"TA del equipo NO transfiere (árbitro/contexto manda) pero el RÉGIMEN sí: "
        f"+{CARDS_WC_SHIFT_PER_TEAM:.1f} TA/equipo amistoso→WC "
        f"[{CARDS_WC_SHIFT_CI[0]:.1f},{CARDS_WC_SHIFT_CI[1]:.1f}]"),
    "fouls": TransferFinding(
        "fouls", "none", None, None, -0.07, 36,
        "faltas NO transfieren (r=−0.07) y suben +1.2 en competitivo"),
}
