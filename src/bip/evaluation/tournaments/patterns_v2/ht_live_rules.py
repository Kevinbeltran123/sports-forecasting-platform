"""HT-state live rules — Lens 3 finding (internal research notes).

EVIDENCE. Combined-corpus favorite-stratified HT->FT matrix, n=255 fav-defined
from 314 StatsBomb matches. Empirical probabilities with 95% bootstrap CIs:

| HT state (fav-relative) | n   | P(fav_win) | P(draw) | P(fav_loss) | P(over2.5) | P(BTTS) |
|-------------------------|-----|------------|---------|-------------|------------|---------|
| tied                    | 124 | 0.468 [0.379, 0.556] | 0.363 [0.282, 0.452] | 0.169 [0.105, 0.242] | 0.315 | 0.411 |
| fav_up                  |  94 | 0.809 [0.734, 0.883] | 0.138 [0.074, 0.213] | 0.053 [0.011, 0.106] | 0.596 | 0.532 |
| fav_down                |  37 | 0.162 [0.054, 0.297] | 0.270 [0.135, 0.405] | 0.568 [0.405, 0.730] | 0.595 | 0.676 |

WHY. Lock_v1 is pre-match only — it does not condition on revealed HT
state. The empirical FT distribution given HT state is a strong gate for
live picks: e.g. when the favorite trails at HT, P(fav wins FT)=16% —
materially below typical Pinnacle live "favorite to win" pricing.

APPLICATION. For each live pick candidate, look up the empirical FT
probability conditional on the current HT state. Compare to Pinnacle's
live implied probability. Edge > configured threshold => alert.

LIMITATIONS. n=37 in fav_down has wide CIs. Recommend using the CI lower
bound for "fade favorite" decisions (conservative) and CI point for
"back fav consolidates" decisions (when fav_up CI is narrower).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HTLiveCell:
    n: int
    p_fav_win: float
    p_fav_win_ci_low: float
    p_fav_win_ci_high: float
    p_draw: float
    p_draw_ci_low: float
    p_draw_ci_high: float
    p_fav_loss: float
    p_over25: float
    p_btts: float


HT_LIVE_TABLE: dict[str, HTLiveCell] = {
    # HT state from favorite's perspective.
    "tied": HTLiveCell(
        n=124,
        p_fav_win=0.468,
        p_fav_win_ci_low=0.379,
        p_fav_win_ci_high=0.556,
        p_draw=0.363,
        p_draw_ci_low=0.282,
        p_draw_ci_high=0.452,
        p_fav_loss=0.169,
        p_over25=0.315,
        p_btts=0.411,
    ),
    "fav_up": HTLiveCell(
        n=94,
        p_fav_win=0.809,
        p_fav_win_ci_low=0.734,
        p_fav_win_ci_high=0.883,
        p_draw=0.138,
        p_draw_ci_low=0.074,
        p_draw_ci_high=0.213,
        p_fav_loss=0.053,
        p_over25=0.596,
        p_btts=0.532,
    ),
    "fav_down": HTLiveCell(
        n=37,
        p_fav_win=0.162,
        p_fav_win_ci_low=0.054,
        p_fav_win_ci_high=0.297,
        p_draw=0.270,
        p_draw_ci_low=0.135,
        p_draw_ci_high=0.405,
        p_fav_loss=0.568,
        p_over25=0.595,
        p_btts=0.676,
    ),
}


@dataclass(frozen=True)
class HTBaseRate:
    """Favorite-AGNOSTIC HT->FT base rate (absolute scoreline, not fav-relative).

    ``lo``/``hi`` are the reported interval (bootstrap CI or discovery/hold-out
    range — see ``note``). EVIDENCE for live secondary markets, not a pick.
    """

    label: str
    p: float
    lo: float
    hi: float
    n: int
    source: str
    note: str


# L3.2 — P(draw FT | HT 1-1). SHARPER than the lumped fav-relative "tied" cell
# (HT_LIVE_TABLE["tied"].p_draw = 36%, which mixes HT 0-0 and HT 1-1).
# Discovery 47% [24, 71] n=17 / hold-out 46% [18, 73] n=11 -> combined n=28 ~= 46%
# (borderline; needs more fixtures for STRONG). internal research notes L3.2.
P_DRAW_GIVEN_HT_1_1 = HTBaseRate(
    label="empate FT | HT 1-1",
    p=0.46,
    lo=0.18,
    hi=0.73,
    n=28,
    source="L3.2 (Papers V2)",
    note="vs implícita Pinnacle live-empate ~30-40% → EV+ si mercado <40%; n=28 borderline",
)

# P5b — comeback rate of the HT-trailing team (draws or wins by FT). Stable.
# Discovery 26.7% [16.7, 38.3] n=60 / hold-out 29.1% [20.4, 38.8] n=103 -> ~27-29%
# (lo/hi = discovery/hold-out point estimates, NOT a single bootstrap CI).
# internal research notes §2.6.
HT_TRAILING_COMEBACK = HTBaseRate(
    label="comeback del que pierde al HT (empata o gana)",
    p=0.28,
    lo=0.27,
    hi=0.29,
    n=163,
    source="P5b (Papers)",
    note="rango disc/hold-out 27-29%; sospechar si una casa ofrece comeback a implícita >30%",
)


def derive_ht_state(
    ht_home_goals: int,
    ht_away_goals: int,
    favorite_side: str,
) -> str | None:
    """Compute the favorite-relative HT state.

    Returns one of {"tied", "fav_up", "fav_down"} or None when favorite is
    undefined.

    >>> derive_ht_state(0, 0, "home")
    'tied'
    >>> derive_ht_state(1, 1, "away")
    'tied'
    >>> derive_ht_state(2, 1, "home")
    'fav_up'
    >>> derive_ht_state(0, 1, "home")
    'fav_down'
    >>> derive_ht_state(1, 0, "none") is None
    True
    """
    if favorite_side not in ("home", "away"):
        return None
    if ht_home_goals == ht_away_goals:
        return "tied"
    if favorite_side == "home":
        return "fav_up" if ht_home_goals > ht_away_goals else "fav_down"
    return "fav_up" if ht_away_goals > ht_home_goals else "fav_down"


@dataclass(frozen=True)
class HTLiveVerdict:
    """Lookup result for a live fixture in known HT state."""

    ht_state: str
    cell: HTLiveCell
    high_value_alerts: tuple[str, ...]
    """Markets with operationally high edge given this HT state.

    Heuristic: cells where the conditional probability is materially
    different from the pre-match marginal (>=15pp shift) and the n>=30.
    """


def ht_live_lookup(ht_state: str) -> HTLiveVerdict:
    """Returns empirical conditional distribution + flagged markets.

    Raises KeyError on unknown ``ht_state``.
    """
    cell = HT_LIVE_TABLE[ht_state]
    alerts: list[str] = []
    if ht_state == "fav_down":
        # P(fav wins) collapses to 16%. CI upper 30% still well below
        # typical pre-match favorite win prior. Fade live "fav to win".
        alerts.append("FADE_LIVE_FAV_WIN")
        # P(fav_loss) = 57% — back live underdog to win.
        alerts.append("BACK_LIVE_UNDERDOG_WIN")
    elif ht_state == "fav_up":
        # P(fav wins) = 81%, narrower CI. Back live fav-consolidates.
        alerts.append("BACK_LIVE_FAV_CONSOLIDATES")
    elif ht_state == "tied":
        # P(draw FT) = 36% — back live draw at odds > 1/0.36 = 2.78.
        alerts.append("BACK_LIVE_DRAW_IF_ODDS_GE_2_78")
    return HTLiveVerdict(
        ht_state=ht_state, cell=cell, high_value_alerts=tuple(alerts)
    )
