"""Confederation tilt — Iter 3 + Iter 4 findings.

EVIDENCE.
  Iter 3 (StatsBomb n=128 WC18+WC22): CONMEBOL ppm 1.74, UEFA 1.57, CAF 1.11,
    AFC 1.03, CONCACAF 0.83. CONMEBOL > UEFA at WC level.
  Iter 4 (martj42 n=192 modern WC >=2012): CONMEBOL ppm 1.80, UEFA 1.60,
    CONCACAF 1.00, CAF 0.98, AFC 0.84. Same ordering. Modern CONMEBOL vs UEFA
    head-to-head: 43.24% / 24.32% / 32.43% (CONMEBOL wins more).
  Time-stable hierarchy: ordering identical pre-2012 vs post-2012.

WHY. Lock_v1's strength prior over martj42 + StatsBomb blends decades of
matches. The Bradley-Terry equivalent under-weights CONMEBOL's recent
dominance in WC-specific matches. CONCACAF gets a slight over-rate
because (a) home-advantage in CONCACAF qualifying inflates ppm,
(b) USA/Mexico/Canada will host WC2026 and lock_v1 lacks host feature.

APPLICATION. Tilt the predictor-implied probability based on confederation
matchup. Magnitudes are calibrated to the observed empirical gap.

NOT a fixed delta — a multiplicative factor on the team's strength.
Caps applied to avoid runaway tilts on extreme matchups.
"""
from __future__ import annotations

from dataclasses import dataclass


# Empirical ppm per confederation in modern WC (>= 2012, martj42 n=192).
# Used as the relative scale for tilt magnitudes.
MODERN_WC_PPM: dict[str, float] = {
    "CONMEBOL": 1.797,
    "UEFA": 1.600,
    "CONCACAF": 1.000,
    "CAF": 0.981,
    "AFC": 0.837,
    "OFC": 0.500,  # very low n; effectively lowest
}

# Confederation tilt factor — multiplier on lock_v1's implied team-win
# probability. Calibrated such that:
#   - CONMEBOL gets a +5% lift (it's the dominant WC conf, under-priced by
#     UEFA-dominant priors)
#   - CONCACAF gets a -10% downgrade (combined with host-nation finding)
#   - AFC/CAF/OFC neutral-to-slight-downgrade
# The base value (UEFA = 1.00) is the reference.
CONF_TILT: dict[str, float] = {
    "CONMEBOL": 1.05,
    "UEFA": 1.00,
    "CAF": 0.97,
    "AFC": 0.95,
    "CONCACAF": 0.90,
    "OFC": 0.85,
    "UNK": 1.00,  # neutral when conf is missing
}

# Per-pair predictor reliability (Brier from walk-forward backtest n=199).
# Higher Brier => less reliable predictor => widen edge threshold.
# Brier_threshold * pair_uncertainty_factor = effective edge threshold.
PAIR_UNCERTAINTY: dict[frozenset[str], float] = {
    # Most reliable (Brier <= 0.15) — narrow threshold
    frozenset({"CONMEBOL", "UEFA"}): 1.00,
    frozenset({"CONMEBOL", "CAF"}): 1.00,
    # Moderate (0.15 < Brier <= 0.20)
    frozenset({"AFC", "UEFA"}): 1.10,
    # Worst (Brier > 0.20) — widen edge threshold materially
    frozenset({"CAF", "UEFA"}): 1.20,
    frozenset({"CONMEBOL", "AFC"}): 1.20,
    frozenset({"CONCACAF", "UEFA"}): 1.25,
    frozenset({"CAF", "AFC"}): 1.40,  # Brier 0.317 — worst observed
}


CONFEDERATION_BY_TEAM: dict[str, str] = {
    # Same canonical list as iter 3 — single source of truth.
    "France": "UEFA", "Germany": "UEFA", "England": "UEFA", "Spain": "UEFA",
    "Italy": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA",
    "Belgium": "UEFA", "Croatia": "UEFA", "Switzerland": "UEFA",
    "Denmark": "UEFA", "Sweden": "UEFA", "Poland": "UEFA",
    "Russia": "UEFA", "Wales": "UEFA", "Iceland": "UEFA",
    "Serbia": "UEFA", "Ukraine": "UEFA", "Austria": "UEFA",
    "Czech Republic": "UEFA", "Slovakia": "UEFA", "Romania": "UEFA",
    "Republic of Ireland": "UEFA", "Northern Ireland": "UEFA",
    "Hungary": "UEFA", "Turkey": "UEFA", "Greece": "UEFA",
    "Scotland": "UEFA", "Albania": "UEFA", "Finland": "UEFA",
    "North Macedonia": "UEFA", "Norway": "UEFA", "Slovenia": "UEFA",
    "Georgia": "UEFA",
    "Brazil": "CONMEBOL", "Argentina": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Bolivia": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
    "Tunisia": "CAF", "Ghana": "CAF", "Nigeria": "CAF",
    "Côte d'Ivoire": "CAF", "Ivory Coast": "CAF",
    "Cameroon": "CAF", "South Africa": "CAF", "Mali": "CAF",
    "Burkina Faso": "CAF", "DR Congo": "CAF", "Cape Verde": "CAF",
    "Equatorial Guinea": "CAF", "Gabon": "CAF",
    "Japan": "AFC", "South Korea": "AFC", "Korea Republic": "AFC",
    "Saudi Arabia": "AFC", "Iran": "AFC", "Australia": "AFC", "Qatar": "AFC",
    "USA": "CONCACAF", "United States": "CONCACAF", "Mexico": "CONCACAF",
    "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Panama": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "New Zealand": "OFC",
}


def confederation_of(team: str) -> str:
    return CONFEDERATION_BY_TEAM.get(team, "UNK")


@dataclass(frozen=True)
class ConfederationTiltVerdict:
    home_team: str
    away_team: str
    home_conf: str
    away_conf: str
    home_tilt: float
    """Multiplier on home team's implied win prob."""
    away_tilt: float
    is_cross_conf: bool
    edge_threshold_multiplier: float
    """Multiplier on the configured edge threshold."""
    rationale: str


def confederation_tilt_verdict(
    home_team: str, away_team: str
) -> ConfederationTiltVerdict:
    """Returns confederation-based tilts and edge-threshold widening.

    Use case:
      - Multiply each team's lock_v1 implied probability by its tilt.
      - Multiply the operator-configured edge threshold by the pair
        uncertainty factor before checking edge >= threshold.

    >>> v = confederation_tilt_verdict("Brazil", "USA")
    >>> v.home_tilt > v.away_tilt
    True
    >>> v.is_cross_conf
    True
    """
    h_conf = confederation_of(home_team)
    a_conf = confederation_of(away_team)
    h_tilt = CONF_TILT.get(h_conf, 1.0)
    a_tilt = CONF_TILT.get(a_conf, 1.0)
    is_cross = h_conf != a_conf and h_conf != "UNK" and a_conf != "UNK"
    pair_key = frozenset({h_conf, a_conf})
    edge_mult = PAIR_UNCERTAINTY.get(pair_key, 1.05 if is_cross else 1.00)

    rationale_parts = []
    if h_tilt != 1.0:
        delta_pct = (h_tilt - 1.0) * 100
        rationale_parts.append(
            f"home {h_conf} {delta_pct:+.0f}% tilt (modern WC ppm = "
            f"{MODERN_WC_PPM.get(h_conf, '?')})"
        )
    if a_tilt != 1.0:
        delta_pct = (a_tilt - 1.0) * 100
        rationale_parts.append(
            f"away {a_conf} {delta_pct:+.0f}% tilt (modern WC ppm = "
            f"{MODERN_WC_PPM.get(a_conf, '?')})"
        )
    if edge_mult > 1.0:
        rationale_parts.append(
            f"pair {h_conf}x{a_conf} requires {edge_mult:.2f}x edge "
            f"(predictor Brier-based reliability)"
        )

    return ConfederationTiltVerdict(
        home_team=home_team,
        away_team=away_team,
        home_conf=h_conf,
        away_conf=a_conf,
        home_tilt=h_tilt,
        away_tilt=a_tilt,
        is_cross_conf=is_cross,
        edge_threshold_multiplier=edge_mult,
        rationale="; ".join(rationale_parts) if rationale_parts else "neutral",
    )


# ─── CONF_STYLE: capa MULTIDIMENSIONAL de estilo (paralela al tilt de RESULTADO) ────────────
# Investigación 2026-06-08, VERIFICADA contra cache (martj42 49k + StatsBomb 314). Cifras =
# cross_conf_pooled (inter-confederación INSESGADO). NO toca CONF_TILT/MODERN_WC_PPM/PAIR_UNCERTAINTY
# (validados, dimensión RESULTADO). Esto es EVIDENCIA de ESTILO por eje de mercado, sin multiplicador
# de probabilidad. DIRECCIÓN>magnitud, n chico. Detalle/honestidad: internal notes

@dataclass(frozen=True)
class ConfStyle:
    xgf: float            # xG-for cross-conf (insesgado)
    xga: float            # xG-against cross-conf (menor = mejor defensa de ocasión)
    corners_for: float
    corners_against: float
    ga_modern: float      # martj42 GA en WC-finals modernas (señal competitiva real)
    n_sb: int             # n StatsBomb cross-conf
    defense: str
    tempo: str
    finishing: str
    stability: str


CONF_STYLE: dict[str, ConfStyle] = {
    "CONMEBOL": ConfStyle(1.56, 0.88, 4.96, 3.72, 1.02, 57,
        "elite (xGA 0.88, 1º) — ventaja sobre UEFA es DEFENSA, no creación; GA empatadas",
        "directo/transición (d=1.43 vs UEFA, NO converge)", "clínico/eficiente", "HELD"),
    "UEFA": ConfStyle(1.46, 1.21, 5.13, 4.01, 1.01, 76,
        "elite (referencia) — solidez alta, ejecución-en-transición > dominio de balón",
        "control/build-up (POST-tiki-taka; España Euro24 fue vertical)", "+leve (over-finishing)", "HELD"),
    "CAF": ConfStyle(1.06, 1.39, 3.69, 5.51, 1.61, 35,
        "CONVERGIÓ a nivel resultado (GA 1.65→1.35, p=0.049 SIG) — NO cargar 'CAF encaja'; pero cede ocasión (xGA 1.39, córn-contra 5.51)",
        "atlético/transición + ABP (Marruecos-type)", "neutral", "SHIFTED↑ (defensa moderna no se fadea)"),
    "AFC": ConfStyle(0.96, 1.51, 3.68, 4.81, 1.73, 37,
        "bajo, chance-suppressed (mejora NS p=0.17) — perfil POR EQUIPO no por conf",
        "heterogéneo (Japón control / Golfo ultra-directo)", "neutral (upsets = varianza sobre xG)",
        "PARCIAL — AFC-vs-UEFA se ENSANCHÓ a favor UEFA"),
    "CONCACAF": ConfStyle(0.99, 1.64, 3.76, 4.98, 1.69, 41,
        "el PEOR (no convergió, NS) — peor en todo eje de ocasión",
        "físico/directo (ppm de quali inflado por localía)", "−0.20 (SUB-finaliza)",
        "no convergió; ×0.90 bien soportado; HOST override USA/MEX/CAN (no doble-contar)"),
    "OFC": ConfStyle(0.0, 0.0, 0.0, 0.0, 1.52, 0,
        "un-estimable (solo Nueva Zelanda, n bajo)", "un-estimable", "un-estimable",
        "placeholder baja confianza — NO inferir tendencia"),
}

_STRONG_CONF = {"CONMEBOL", "UEFA"}
_WEAK_CONF = {"CAF", "AFC", "CONCACAF"}


@dataclass(frozen=True)
class ConfStyleVerdict:
    home_conf: str
    away_conf: str
    lines: tuple[str, ...]
    """Líneas de evidencia de estilo para el brief (NO picks)."""
    cards_note: str
    possession_meta_flag: str


def confederation_style_verdict(home_team: str, away_team: str) -> ConfStyleVerdict | None:
    """Choque de confederaciones MULTIDIMENSIONAL (estilo) — evidencia, no auto-pick.
    None si alguna confederación es desconocida.

    >>> v = confederation_style_verdict("Brazil", "Mexico")
    >>> v.home_conf, v.away_conf
    ('CONMEBOL', 'CONCACAF')
    >>> any("Córner-share" in ln for ln in v.lines)
    True
    >>> nz = confederation_style_verdict("New Zealand", "Norway")  # OFC un-estimable → n/d, NO 0.0
    >>> any("n/d" in ln for ln in nz.lines) and not any("0.0/0.0" in ln for ln in nz.lines)
    True
    """
    h, a = confederation_of(home_team), confederation_of(away_team)
    sh, sa = CONF_STYLE.get(h), CONF_STYLE.get(a)
    if sh is None or sa is None:
        return None
    lines: list[str] = []
    for team, conf, s in ((home_team, h, sh), (away_team, a, sa)):
        # n_sb==0 (OFC/NZ) = un-estimable: NO mostrar 0.0 (se leería como "genera 0 xG"); marcar n/d.
        nums = (f"xGF/xGA {s.xgf}/{s.xga} · córn F/C {s.corners_for}/{s.corners_against} (n={s.n_sb})"
                if s.n_sb > 0 else "xGF/xGA n/d · córn F/C n/d (sin datos cross-conf)")
        lines.append(f"{team} ({conf}): def {s.defense} · tempo {s.tempo} · {nums}")
    # NOTA: P8 (córner al favorito-TM) usa VALOR DE PLANTILLA, no fuerza de confederación. Si el
    # favorito-TM NO es el lado fuerte por conf (raro), córner-share y P8 se CONTRADICEN → cruzar a mano.
    _p8 = "→ tilt córner al lado fuerte (mismo eje que P8 si el favorito-TM coincide; si difiere, conflicto → cruzar)"
    if h in _STRONG_CONF and a in _WEAK_CONF:
        lines.append(f"Córner-share: {home_team} ({h}) ~+1.3 córn/p vs {away_team} ({a}, suprime ocasión) {_p8}.")
    elif a in _STRONG_CONF and h in _WEAK_CONF:
        lines.append(f"Córner-share: {away_team} ({a}) ~+1.3 córn/p vs {home_team} ({h}, suprime ocasión) {_p8}.")
    elif {h, a} == _STRONG_CONF:
        lines.append("Córner-share: CONMEBOL≈UEFA (4.96 vs 5.13) → confederación NO los separa; córner por EQUIPO.")
    for team, conf in ((home_team, h), (away_team, a)):
        if conf == "CAF":
            lines.append(f"Temporal CAF: defensa convergió (p=0.049) → NO cargar 'CAF encaja goles' contra {team}.")
        elif conf == "CONCACAF":
            lines.append(f"Temporal CONCACAF: NO convergió (NS) → ×0.90 soportado; si {team} es host (USA/MEX/CAN) "
                         "el host-signal MANDA (no doble-contar con ×0.90).")
    return ConfStyleVerdict(
        home_conf=h, away_conf=a, lines=tuple(lines),
        cards_note="Tarjetas: NO hay prior de confederación (nivel país+DT+árbitro, NBER w13968) → "
                   "rutear a equipo + árbitro; 'CONMEBOL cínico / CONCACAF sucio / AFCON tarjetero' = estereotipo.",
        possession_meta_flag="Meta moderno: posesión ≠ control (decoupled desde WC2022) → NO inclinar "
                             "resultado/goles por posesión sola; el meta premia pragmático+transición.",
    )
