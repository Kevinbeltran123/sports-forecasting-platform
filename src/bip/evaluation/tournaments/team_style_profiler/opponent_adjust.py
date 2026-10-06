"""Ajuste por rival (per-market) + shrinkage de muestra chica para las stats de §6.6.

PROBLEMA (operator: "importantísimo"). El ajuste por rival hoy (SoS-Elo, script 42) toca SOLO
GF/GA. Las medias de córners/tarjetas/remates del brief son CRUDAS: un equipo que sacó 6.5
córners/partido vs rivales flojos se ve idéntico a uno que lo sacó vs fuertes — justo en los
mercados secundarios donde se supone que vive el edge.

MÉTODO (dos funciones puras, testeables):
1. ``opponent_adjust_rate`` — de-trend aditivo a una fuerza de rival de REFERENCIA. La sensibilidad
   (d(rate)/d(opp_Elo)) se estima fuera (pooled, en el spike 49) — típicamente NEGATIVA para los
   mercados "a favor" (rival más fuerte → menos córners/remates propios).
2. ``james_stein_shrink`` — shrinkage closed-form (Efron-Morris / EB) hacia la media de cohorte,
   para selecciones con n=10-20 bajo el DT actual. NO es un sampler PyMC (el Bayes jerárquico fue
   FALSIFICADO aquí); es aritmética cerrada. Devuelve también el peso de shrink para que el analista
   vea cuánto confiar.

Evidencia para el analista, NO un pick. La sensibilidad y el cohorte se validan en el spike
(walk-forward). Si el ajuste no baja el error out-of-sample, NO se cablea (goal-driven).
"""
from __future__ import annotations


def opponent_adjust_rate(
    raw_mean: float,
    team_mean_opp_elo: float,
    ref_elo: float,
    sensitivity: float,
) -> float:
    """De-trenda una tasa de mercado a una fuerza de rival de referencia.

    El ``raw_mean`` se midió contra rivales de Elo medio ``team_mean_opp_elo``; lo reexpresa como
    si hubiera sido contra rivales de ``ref_elo``.

        adjusted = raw_mean + sensitivity * (ref_elo - team_mean_opp_elo)

    ``sensitivity`` = d(rate)/d(opp_Elo) estimada pooled. Para un mercado "a favor" suele ser <0:
    si el equipo enfrentó rivales flojos (team_mean_opp_elo < ref_elo) → (ref - mean) > 0 → con
    sensitivity<0 la tasa se DEFLACTA (corrige el espejismo de calendario flojo). Y al revés.

    >>> round(opponent_adjust_rate(6.5, 1500.0, 1700.0, -0.004), 2)  # rivales flojos → deflacta
    5.7
    >>> round(opponent_adjust_rate(5.0, 1900.0, 1700.0, -0.004), 2)  # rivales duros → infla
    5.8
    """
    return raw_mean + sensitivity * (ref_elo - team_mean_opp_elo)


def james_stein_shrink(
    value: float,
    n: int,
    cohort_mean: float,
    within_var: float,
    between_var: float,
) -> tuple[float, float]:
    """Shrinkage closed-form de una media de muestra chica hacia ``cohort_mean``.

    Devuelve ``(shrunk, weight)`` donde ``weight`` ∈ [0,1] es cuánto se tira al cohorte
    (1 = todo prior, 0 = todo dato). ``within_var`` = varianza por-partido del mercado;
    ``between_var`` = varianza entre equipos de la media del mercado (señal real entre equipos).

        se2 = within_var / n          # varianza del estimador de la media del equipo
        weight = se2 / (between_var + se2)
        shrunk = (1 - weight) * value + weight * cohort_mean

    n grande o between_var grande → weight ↓ (confía en el dato). n chico o within_var grande →
    weight ↑ (tira al cohorte). Caso degenerado (n<=0 o between_var<=0) → todo cohorte.

    >>> v, w = james_stein_shrink(6.5, 10, 5.0, 4.0, 1.0)
    >>> round(w, 3), round(v, 3)
    (0.286, 6.071)
    >>> _, w_big = james_stein_shrink(6.5, 100, 5.0, 4.0, 1.0)  # más n → menos shrink
    >>> w_big < w
    True
    """
    if n <= 0 or between_var <= 0:
        return cohort_mean, 1.0
    se2 = within_var / n
    weight = se2 / (between_var + se2)
    weight = max(0.0, min(1.0, weight))
    return (1.0 - weight) * value + weight * cohort_mean, weight
