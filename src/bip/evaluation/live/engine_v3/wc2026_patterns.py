"""WC2026 historical-pattern rules — statistically validated subset.

Implements the four rules whose effect sizes survived FDR-BH q=0.10 on
discovery (n=115: WC2018 + Euro2020) **and** replicated on the held-out
validation tournaments (n=199: WC2022 + AFCON2023 + Copa2024 + Euro2024).
Full evidence: internal research notes.

Rules:
    A1  HT_00_LIVE_U25   live in-play gate; HT 0-0 → P(U2.5 FT)=80.7%
    A2  FAV_CORNER_TILT  pre-match; favorite +2.4 corners (group only)
    A3  FAV_TT_OVER15    pre-match; favorite +0.65 xG (group only)
    A4  AFCON_PARITY     filter that disables A2/A3 in AFCON regime

Each rule is a pure ``(MatchContext, OddsSnapshot) → PatternTrigger``
function. Rules are orthogonal to lock_v1 — they fire on markets the
predictor does not absorb (corners, team-totals, live U2.5).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from bip.evaluation.live.engine_v3.thesis import MarketFamily

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

# Empirical constants from internal research notes §1 + §2.
# P5a — combined n=114 HT 0-0 matches across all 6 tournaments.
P_U25_GIVEN_HT00 = 0.807
P_U25_GIVEN_HT00_CI_LOW = 0.728
P_U25_GIVEN_HT00_CI_HIGH = 0.877
BREAKEVEN_ODDS_HT00_U25 = 1 / P_U25_GIVEN_HT00  # ≈ 1.239

# A1 fires when odds are above this floor — i.e. there is positive EV
# with a safety margin over the point estimate (1.239). We use 1.30 to
# leave +4.9% EV minimum cushion vs the breakeven, accounting for the
# CI lower bound and bookmaker juice. Higher odds = more EV; no ceiling.
A1_MIN_ODDS = 1.30

# P4a — dog–fav xG gap −0.65 hold-out / −0.74 discovery (group stage).
FAV_XG_GAP_HOLDOUT = 0.65

# P8 — dog–fav corner gap −2.40 hold-out / −2.32 discovery (group stage).
FAV_CORNER_GAP_HOLDOUT = 2.40

# TM ratio threshold above which A2/A3 fire. Discovery used `>1.0`; v4 may
# refine. The dossier set showed P4a/P8 effect attenuates rapidly toward
# parity, so we use 1.5x as the conservative operational gate.
TM_RATIO_THRESHOLD = 1.5

# AFCON tournament identifiers. The parity gate disables A2/A3 when an
# AFCON match is being evaluated (intra-CAF), because squad market values
# are too compressed for the favorite-tilt patterns to apply.
AFCON_SLUGS: frozenset[str] = frozenset({"afcon_2023", "afcon_2025"})


class MatchContext(BaseModel):
    """Minimal pre-match identity for rule evaluation."""

    model_config = _MODEL_CONFIG
    match_id: str
    tournament_slug: str  # e.g. "world_cup_2026", "afcon_2025"
    home_team: str
    away_team: str
    phase: str = Field(..., description="'group' or 'knockout'")
    home_market_value_eur: float | None = None
    away_market_value_eur: float | None = None


class OddsSnapshot(BaseModel):
    """Single-market live odds snapshot. ``stage='pre_match' | 'live'``."""

    model_config = _MODEL_CONFIG
    stage: str
    market_family: MarketFamily
    selection: str  # e.g. "under_2_5", "ah_-2_corners_home"
    decimal_odds: float
    minute: int | None = None  # populated for live stage
    home_goals_ht: int | None = None
    away_goals_ht: int | None = None


class PatternTrigger(BaseModel):
    """Rule output. ``triggered=False`` means rule did not fire; the
    ``reason`` field still carries the audit string."""

    model_config = _MODEL_CONFIG
    rule_id: str
    triggered: bool
    market_family: MarketFamily | None = None
    direction: str | None = None
    max_kelly: float = 0.0
    reason: str
    breakeven_odds: float | None = None


# ── A1 — HT 0-0 → Under 2.5 live gate ────────────────────────────────────

def rule_a1_ht00_live_u25(ctx: MatchContext, odds: OddsSnapshot) -> PatternTrigger:
    """Trigger when HT score is 0-0 and live U2.5 odds are at or below the
    breakeven implied by P(U2.5 | HT 0-0) = 80.7%.

    Effective threshold: odds ≤ 1.30 → EV ≥ +4.9%. We cap recommended
    stake at 1/4 Kelly per internal research notes §2.1.
    """
    if odds.stage != "live":
        return PatternTrigger(
            rule_id="A1", triggered=False, reason="stage != live"
        )
    if odds.market_family != MarketFamily.GOALS or odds.selection != "under_2_5":
        return PatternTrigger(
            rule_id="A1", triggered=False, reason="market is not goals/under_2_5"
        )
    if odds.home_goals_ht is None or odds.away_goals_ht is None:
        return PatternTrigger(
            rule_id="A1", triggered=False, reason="HT score not in snapshot"
        )
    if odds.home_goals_ht != 0 or odds.away_goals_ht != 0:
        return PatternTrigger(
            rule_id="A1",
            triggered=False,
            reason=f"HT score {odds.home_goals_ht}-{odds.away_goals_ht} != 0-0",
        )
    if odds.decimal_odds < A1_MIN_ODDS:
        return PatternTrigger(
            rule_id="A1",
            triggered=False,
            market_family=MarketFamily.GOALS,
            direction="under",
            reason=f"live U2.5 odds {odds.decimal_odds:.3f} < {A1_MIN_ODDS} (insufficient EV cushion)",
            breakeven_odds=BREAKEVEN_ODDS_HT00_U25,
        )
    return PatternTrigger(
        rule_id="A1",
        triggered=True,
        market_family=MarketFamily.GOALS,
        direction="under",
        max_kelly=0.25,
        reason=f"HT 0-0 + U2.5 odds {odds.decimal_odds:.3f} ≥ {A1_MIN_ODDS}",
        breakeven_odds=BREAKEVEN_ODDS_HT00_U25,
    )


# ── A4 — AFCON parity filter (used internally by A2/A3) ─────────────────

def _is_afcon_regime(ctx: MatchContext) -> bool:
    """Returns True when the tournament is in the AFCON regime where
    squad-value-driven tilts have insufficient signal (§2.5)."""
    return ctx.tournament_slug in AFCON_SLUGS


def rule_a4_afcon_parity(ctx: MatchContext) -> PatternTrigger:
    """Reports whether A2/A3 should be suppressed for this match."""
    if _is_afcon_regime(ctx):
        return PatternTrigger(
            rule_id="A4",
            triggered=True,
            reason=f"tournament_slug={ctx.tournament_slug} is AFCON regime; "
            "disable A2/A3 (squad-value parity)",
        )
    return PatternTrigger(
        rule_id="A4",
        triggered=False,
        reason=f"tournament_slug={ctx.tournament_slug} not AFCON; A2/A3 enabled",
    )


# ── Favorite-side helpers (shared by A2, A3) ─────────────────────────────

def _favorite_side(ctx: MatchContext) -> tuple[str, float] | None:
    """Returns (favorite_team, ratio) or None if no clear favorite by MV.

    Ratio is max(home_mv, away_mv) / min(home_mv, away_mv).
    """
    if ctx.home_market_value_eur is None or ctx.away_market_value_eur is None:
        return None
    if ctx.home_market_value_eur <= 0 or ctx.away_market_value_eur <= 0:
        return None
    if ctx.home_market_value_eur == ctx.away_market_value_eur:
        return None
    if ctx.home_market_value_eur > ctx.away_market_value_eur:
        ratio = ctx.home_market_value_eur / ctx.away_market_value_eur
        return ctx.home_team, ratio
    ratio = ctx.away_market_value_eur / ctx.home_market_value_eur
    return ctx.away_team, ratio


def _gate_a2_a3(ctx: MatchContext, rule_id: str) -> PatternTrigger | None:
    """Common pre-conditions for A2 and A3. Returns a non-trigger reason
    if any gate fails, or None if all gates pass."""
    if ctx.phase != "group":
        return PatternTrigger(
            rule_id=rule_id,
            triggered=False,
            reason=f"phase={ctx.phase} != group",
        )
    if _is_afcon_regime(ctx):
        return PatternTrigger(
            rule_id=rule_id,
            triggered=False,
            reason="A4 AFCON-parity filter active",
        )
    fav = _favorite_side(ctx)
    if fav is None:
        return PatternTrigger(
            rule_id=rule_id,
            triggered=False,
            reason="no market-value data or values equal",
        )
    _, ratio = fav
    if ratio < TM_RATIO_THRESHOLD:
        return PatternTrigger(
            rule_id=rule_id,
            triggered=False,
            reason=f"TM ratio {ratio:.2f} < {TM_RATIO_THRESHOLD}",
        )
    return None


# ── A2 — Favorite Asian-corner handicap tilt ────────────────────────────

def rule_a2_favorite_corner_tilt(
    ctx: MatchContext, odds: OddsSnapshot
) -> PatternTrigger:
    """Tilt towards Asian-corner handicap on the favorite side in
    group-stage matches with TM ratio ≥ 1.5x.

    Expected favorite corner advantage: +2.40 (hold-out n=104). The
    operator should pick the AH line one notch below this gap to leave
    margin; the rule fires as a directional signal, not a line-specific
    bet.
    """
    gate = _gate_a2_a3(ctx, "A2")
    if gate is not None:
        return gate

    if odds.market_family not in (MarketFamily.CORNERS, MarketFamily.ASIAN_HANDICAP):
        return PatternTrigger(
            rule_id="A2",
            triggered=False,
            reason=f"market_family={odds.market_family} not corners/AH",
        )

    fav = _favorite_side(ctx)
    assert fav is not None  # gate would have returned
    favorite_team, ratio = fav

    direction = "home" if favorite_team == ctx.home_team else "away"
    return PatternTrigger(
        rule_id="A2",
        triggered=True,
        market_family=MarketFamily.CORNERS,
        direction=direction,
        max_kelly=0.25,
        reason=(
            f"group+non-AFCON+TM ratio {ratio:.2f}x; tilt corner-AH towards "
            f"{favorite_team} (expected +{FAV_CORNER_GAP_HOLDOUT:.2f} corner gap)"
        ),
    )


# ── A3 — Favorite Team-Total Over 1.5 ───────────────────────────────────

def rule_a3_favorite_team_total(
    ctx: MatchContext, odds: OddsSnapshot
) -> PatternTrigger:
    """Tilt towards favorite team-total over 1.5 in group-stage matches
    with TM ratio ≥ 1.5x.

    Expected favorite xG advantage: +0.65 (hold-out n=103). Do NOT apply
    this rule to 1X2 — lock_v1's Elo prior already absorbs that signal
    (see project_wc2026_v3_a_killed).
    """
    gate = _gate_a2_a3(ctx, "A3")
    if gate is not None:
        return gate

    if odds.market_family != MarketFamily.GOALS:
        return PatternTrigger(
            rule_id="A3",
            triggered=False,
            reason=f"market_family={odds.market_family} not goals",
        )

    fav = _favorite_side(ctx)
    assert fav is not None
    favorite_team, ratio = fav

    expected_selection_prefix = (
        "home_team_total_over_1_5" if favorite_team == ctx.home_team
        else "away_team_total_over_1_5"
    )
    if odds.selection != expected_selection_prefix:
        return PatternTrigger(
            rule_id="A3",
            triggered=False,
            reason=(
                f"selection={odds.selection} does not match favorite-team-total "
                f"expected={expected_selection_prefix}"
            ),
        )

    direction = "home" if favorite_team == ctx.home_team else "away"
    return PatternTrigger(
        rule_id="A3",
        triggered=True,
        market_family=MarketFamily.GOALS,
        direction=direction,
        max_kelly=0.25,
        reason=(
            f"group+non-AFCON+TM ratio {ratio:.2f}x; favorite={favorite_team}, "
            f"expected +{FAV_XG_GAP_HOLDOUT:.2f} xG advantage"
        ),
    )


# ── Market-value loader ──────────────────────────────────────────────────

_TM_PARQUET = (
    Path(__file__).resolve().parents[5]
    / "data" / "cache" / "transfermarkt" / "squad_values_current.parquet"
)


def load_market_values(path: Path = _TM_PARQUET) -> dict[str, float]:
    """Load current TM squad market values as {team_name: EUR}. Returns
    empty dict if the parquet is missing. Cache-friendly: callers should
    memoise."""
    if not path.exists():
        return {}
    df = pl.read_parquet(path)
    out: dict[str, float] = {}
    for row in df.iter_rows(named=True):
        team = row.get("team_name")
        mv = row.get("market_value_eur")
        if team and mv is not None and mv > 0:
            out[str(team)] = float(mv)
    return out


def enrich_context(
    ctx_partial: dict,
    market_values: dict[str, float],
) -> MatchContext:
    """Convenience builder: pulls home/away MV from a name→MV dict."""
    ctx_partial = dict(ctx_partial)  # don't mutate caller
    if "home_market_value_eur" not in ctx_partial:
        ctx_partial["home_market_value_eur"] = market_values.get(ctx_partial["home_team"])
    if "away_market_value_eur" not in ctx_partial:
        ctx_partial["away_market_value_eur"] = market_values.get(ctx_partial["away_team"])
    return MatchContext(**ctx_partial)


__all__ = [
    "AFCON_SLUGS",
    "BREAKEVEN_ODDS_HT00_U25",
    "FAV_CORNER_GAP_HOLDOUT",
    "FAV_XG_GAP_HOLDOUT",
    "MatchContext",
    "OddsSnapshot",
    "P_U25_GIVEN_HT00",
    "P_U25_GIVEN_HT00_CI_HIGH",
    "P_U25_GIVEN_HT00_CI_LOW",
    "PatternTrigger",
    "TM_RATIO_THRESHOLD",
    "enrich_context",
    "load_market_values",
    "rule_a1_ht00_live_u25",
    "rule_a2_favorite_corner_tilt",
    "rule_a3_favorite_team_total",
    "rule_a4_afcon_parity",
]
