"""Confidence Modulator — section 3 (post-detection layer) of the design.

> Evidencia corroboratoria (no generadora). xg_vs_score_divergence,
> shot trend, possession trend. Multiplica Kelly fraction, no invierte
> dirección.

The modulator is **strictly post-detection**: it runs *after* the
thesis is generated, *after* the market is selected, *after* the gate
passes. Its only job is to scale the Kelly fraction up or down based
on corroborating evidence.

It must NOT:
- Generate new theses (that's the rule layer's job)
- Flip directions (that defeats the whole "evidence ≠ thesis" design)
- Veto picks below a hard floor (the gate already does that)

The multiplier is bounded in [0, 1.5]. Below 0.30 it returns 0, which
the caller can treat as a soft veto. The cap at 1.5 prevents over-confident
sizing on aligned signals (a Kelly of 1.5× the model's prior is already
aggressive; more invites variance).

Three evidence signals, each producing a centered multiplier:

- ``xg_aligned``: xg_vs_score_divergence agrees with thesis direction
- ``shot_trend_aligned``: shot acceleration aligns with thesis direction
- ``possession_aligned``: possession trend aligns

Signal magnitudes are tuned so the soft-veto floor is reachable when
all three signals oppose the thesis:

- xg_alignment:           ±0.35
- shot_trend_alignment:   ±0.20
- possession_alignment:   ±0.20

Max negative additive = -0.75 → multiplier 0.25 → triggers soft veto.
Max positive additive = +0.75 → multiplier 1.75 clamps to 1.5.
"""
from __future__ import annotations

from dataclasses import dataclass

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import MarketFamily, Thesis


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Audit-trail breakdown of the modulator's reasoning."""

    xg_alignment: float          # signed, [-0.35, +0.35]
    shot_trend_alignment: float  # signed, [-0.20, +0.20]
    possession_alignment: float  # signed, [-0.20, +0.20]
    multiplier: float            # clamped to [0.0, 1.5]


def _thesis_favours_home(thesis: Thesis) -> bool | None:
    """For the family + direction combination, is "home" the favoured side?

    Returns True/False/None:
    - True: thesis points to home (1X2 home, next_goal home, over for
      a home-dominant push, etc).
    - False: thesis points to away.
    - None: thesis is direction-neutral (over/under/yes/no on totals);
      alignment signals don't apply.
    """
    d = thesis.prediction.direction
    if d == "home":
        return True
    if d == "away":
        return False
    return None


def _xg_alignment(gsv: GameStateVector, thesis: Thesis) -> float:
    """xG-vs-score divergence agrees with the thesis direction.

    The divergence is positive when home is over-performing (out-shooting
    relative to the goal-state-conditional expectation). If the thesis
    favours home AND divergence is positive → corroborating → +0.20.
    If they disagree → −0.20.

    For totals-direction theses (over/under), we use the **absolute**
    divergence as a confidence proxy — high divergence in *any* direction
    raises the probability of regression (any goal coming). Bump +0.10.
    """
    div = gsv.xg.xg_vs_score_divergence
    favours_home = _thesis_favours_home(thesis)
    if favours_home is None:
        # Totals thesis: any large divergence boosts over, suppresses under.
        if thesis.prediction.direction in {"over", "yes"}:
            if abs(div) > 0.8:
                return 0.15
            return 0.0
        if thesis.prediction.direction in {"under", "no"}:
            if abs(div) > 0.8:
                return -0.15
            return 0.0
        return 0.0
    # Directional thesis
    if (favours_home and div > 0.3) or (not favours_home and div < -0.3):
        return 0.35
    if (favours_home and div < -0.3) or (not favours_home and div > 0.3):
        return -0.35
    return 0.0


def _shot_trend_alignment(gsv: GameStateVector, thesis: Thesis) -> float:
    """Compares xg_per_min_*_last_15 against pre-match λ_*.

    "Hot" side = the side whose recent rate ≥ 1.3× its prior rate.
    Aligned with thesis direction → +0.10. Opposite → −0.10.
    """
    home_hot = (
        gsv.xg.xg_per_min_home_last_15 * 90.0
        > gsv.priors.lambda_home_prematch * 1.15
    )
    away_hot = (
        gsv.xg.xg_per_min_away_last_15 * 90.0
        > gsv.priors.lambda_away_prematch * 1.15
    )
    favours_home = _thesis_favours_home(thesis)
    if favours_home is None:
        if home_hot or away_hot:
            if thesis.prediction.direction in {"over", "yes"}:
                return 0.10
            if thesis.prediction.direction in {"under", "no"}:
                return -0.10
        return 0.0
    if (favours_home and home_hot) or (not favours_home and away_hot):
        return 0.20
    if (favours_home and away_hot) or (not favours_home and home_hot):
        return -0.20
    return 0.0


def _possession_alignment(gsv: GameStateVector, thesis: Thesis) -> float:
    """Possession ≥ 60% for the favoured side → +0.10.

    Note: this is intentionally weaker than xG and shot trend signals.
    Possession alone is a poor proxy for chance creation (see the
    is_killing_clock detector in LiveMatchState)."""
    poss_home = gsv.flow.possession_home_5min
    favours_home = _thesis_favours_home(thesis)
    if favours_home is None:
        return 0.0
    if (favours_home and poss_home > 60.0) or (not favours_home and poss_home < 40.0):
        return 0.20
    if (favours_home and poss_home < 40.0) or (not favours_home and poss_home > 60.0):
        return -0.20
    return 0.0


def modulate(
    gsv: GameStateVector, thesis: Thesis,
) -> ConfidenceBreakdown:
    """Compute the confidence multiplier and its breakdown.

    Returns the breakdown so the audit log captures *which* evidence
    moved the multiplier (sec 7.2 contract).
    """
    xg = _xg_alignment(gsv, thesis)
    st = _shot_trend_alignment(gsv, thesis)
    poss = _possession_alignment(gsv, thesis)
    raw = 1.0 + xg + st + poss
    multiplier = max(0.0, min(1.5, raw))
    # Soft veto: very low confidence → return 0 so caller can skip.
    if multiplier < 0.30:
        multiplier = 0.0
    return ConfidenceBreakdown(
        xg_alignment=xg,
        shot_trend_alignment=st,
        possession_alignment=poss,
        multiplier=multiplier,
    )


__all__ = ["ConfidenceBreakdown", "modulate"]
