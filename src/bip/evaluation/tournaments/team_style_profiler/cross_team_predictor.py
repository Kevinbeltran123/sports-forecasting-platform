"""Cross-team predictor — heart of the TSP system.

Cross two BettableProfiles (home + away) and emit market-specific
probabilities with confidence metadata.

OPERATOR DECISIONS (locked 2026-05-24):
  - HYBRID model: empirical for direct rates (BTTS, O/U, cards, corners),
    Bivariate Poisson for AH (needs full scoreline distribution).
  - BTTS: most realistic possible — use Bivariate Poisson scoreline grid
    (NOT naive independence). λ's derived from team_attack × opp_defense.
  - DO NOT emit 1X2 — lock_v1 owns that market. TSP stays in style space.
  - Sub-profile merging: inverse-CI-width weighting via weighted_combine.

OUTPUTS:
  Returns ``MarketPredictions`` — a dataclass with per-market probability
  + confidence flag (HIGH/MEDIUM/LOW based on combined CI widths). Caller
  (value_detector) uses confidence to gate alerts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bip.evaluation.tournaments.team_style_profiler.bettable_profile import (
    BettableProfile,
)
from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
    bivariate_poisson_grid,
    p_btts,
    p_home_covers_ah,
    p_home_pushes_ah,
    p_over_total,
    weighted_combine,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    DistributionStat,
)


# ─── Output structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketProb:
    """A single market probability + confidence metadata."""

    market: str
    """e.g. 'BTTS', 'O2.5', 'O3.5', 'corners_O9.5', 'cards_O4.5', 'AH_home_-1.5'."""

    probability: float
    """Empirical / model probability in [0, 1]."""

    confidence: str
    """HIGH | MEDIUM | LOW — derived from combined CI width."""

    rationale: str
    """Short audit string: which method (empirical/bivariate-poisson),
    what inputs were combined, sub-profile usage."""


@dataclass(frozen=True)
class MarketPredictions:
    """Bundle of all market predictions for one fixture."""

    home_team: str
    away_team: str
    home_source: str
    away_source: str

    # Goals markets
    over_2_5: MarketProb
    over_3_5: MarketProb
    under_2_5: MarketProb
    btts_yes: MarketProb
    btts_no: MarketProb

    # Corners markets
    corners_over_8_5: MarketProb | None
    corners_over_9_5: MarketProb | None
    corners_over_10_5: MarketProb | None

    # Cards markets
    cards_over_3_5: MarketProb | None
    cards_over_4_5: MarketProb | None
    cards_over_5_5: MarketProb | None

    # Asian handicap (home perspective)
    ah_home_minus_0_5: MarketProb | None
    ah_home_minus_1_5: MarketProb | None
    ah_away_minus_0_5: MarketProb | None
    ah_away_minus_1_5: MarketProb | None

    # Convenience metadata
    lambda_home: float
    """Derived home goal rate used for the bivariate Poisson grid."""
    lambda_away: float

    @property
    def all_markets(self) -> list[MarketProb]:
        """All emitted markets in display order, skipping unavailable ones."""
        out: list[MarketProb] = [
            self.over_2_5, self.over_3_5, self.under_2_5,
            self.btts_yes, self.btts_no,
        ]
        for m in [
            self.corners_over_8_5, self.corners_over_9_5, self.corners_over_10_5,
            self.cards_over_3_5, self.cards_over_4_5, self.cards_over_5_5,
            self.ah_home_minus_0_5, self.ah_home_minus_1_5,
            self.ah_away_minus_0_5, self.ah_away_minus_1_5,
        ]:
            if m is not None:
                out.append(m)
        return out


# ─── Internal helpers ───────────────────────────────────────────────────


def _confidence_from_width(width: float) -> str:
    """Map a CI half-width to a categorical confidence."""
    if width <= 0.10:
        return "HIGH"
    if width <= 0.20:
        return "MEDIUM"
    return "LOW"


def _is_usable(stat: DistributionStat, min_n: int = 5) -> bool:
    """A stat is usable when n >= min_n. Sub-profile threshold from schema."""
    return stat.n >= min_n


def _combined_with_sub(
    profile: BettableProfile,
    attr: str,
) -> DistributionStat:
    """Return the inverse-CI-weighted combination of profile.attr +
    profile.sub_profile_vs_opponent.attr (when sub-profile exists)."""
    general: DistributionStat = getattr(profile, attr)
    if profile.sub_profile_vs_opponent is None:
        return general
    sub_stat: DistributionStat | None = getattr(
        profile.sub_profile_vs_opponent, attr, None
    )
    return weighted_combine(general, sub_stat)


# ─── Lambda derivation ──────────────────────────────────────────────────


def derive_lambdas(
    home: BettableProfile, away: BettableProfile
) -> tuple[float, float]:
    """Derive (λ_home, λ_away) — the goal rates used in the bivariate grid.

    Standard attack × defense mixing per the operator's "more complete"
    request:

        λ_home = mean( home.attack, away.defense )
        λ_away = mean( away.attack, home.defense )

    where:
        home.attack       = home.goals_for_per_match  (merged w/ sub if present)
        away.defense      = away.goals_against_per_match (merged w/ sub if present)

    Sub-profile merge happens VIA the opposing-team's confederation lens
    on the home/away profile. I.e. when we ask for home's attack, we
    combine home's GENERAL attack with home's "vs <away.confederation>"
    sub-profile attack — because that's how home plays against teams from
    away's confederation specifically.
    """
    h_attack = _combined_with_sub(home, "goals_for_per_match")
    a_defense = _combined_with_sub(away, "goals_against_per_match")
    h_defense = _combined_with_sub(home, "goals_against_per_match")
    a_attack = _combined_with_sub(away, "goals_for_per_match")

    lam_h = (h_attack.mean + a_defense.mean) / 2.0
    lam_a = (a_attack.mean + h_defense.mean) / 2.0
    # Clamp to positive
    return max(lam_h, 0.05), max(lam_a, 0.05)


# ─── Per-market predictions ─────────────────────────────────────────────


def predict_btts(
    home: BettableProfile, away: BettableProfile, grid: np.ndarray
) -> tuple[MarketProb, MarketProb]:
    """BTTS yes / no using the bivariate Poisson scoreline grid.

    Per operator: "most complete and realistic" — we use the FULL
    scoreline distribution (NOT naive independence). When the grid has
    rho > 0, BTTS is automatically adjusted for goal-correlation.
    """
    p = p_btts(grid)
    p_no = 1.0 - p
    # Confidence from combined CI width of both teams' goal rates
    width = (
        home.goals_for_per_match.ci_width + away.goals_for_per_match.ci_width
    ) / 2
    conf = _confidence_from_width(width)
    rationale = (
        f"bivariate Poisson grid, λ derived from "
        f"home_attack×away_defense (n={home.goals_for_per_match.n}, "
        f"{away.goals_against_per_match.n}). "
        f"{'sub-profile merged' if home.sub_profile_vs_opponent else 'general only'}."
    )
    return (
        MarketProb("BTTS_yes", p, conf, rationale),
        MarketProb("BTTS_no", p_no, conf, rationale),
    )


def predict_over_under_goals(
    home: BettableProfile,
    away: BettableProfile,
    grid: np.ndarray,
    line: float,
) -> tuple[MarketProb, MarketProb]:
    """O/U for total goals from bivariate Poisson grid."""
    p_over = p_over_total(grid, line)
    p_under = 1.0 - p_over
    # Sanity: also pull empirical O/U rate and weight 50/50 if line == 2.5
    if line == 2.5 and _is_usable(home.over_25_rate) and _is_usable(away.over_25_rate):
        emp = (home.over_25_rate.mean + away.over_25_rate.mean) / 2
        # Hybrid: 60% Poisson model + 40% empirical anchor
        p_over_hybrid = 0.6 * p_over + 0.4 * emp
        p_over = p_over_hybrid
        p_under = 1.0 - p_over
        method = "hybrid (Poisson 60% + empirical 40%)"
    elif line == 3.5 and _is_usable(home.over_35_rate) and _is_usable(away.over_35_rate):
        emp = (home.over_35_rate.mean + away.over_35_rate.mean) / 2
        p_over_hybrid = 0.6 * p_over + 0.4 * emp
        p_over = p_over_hybrid
        p_under = 1.0 - p_over
        method = "hybrid (Poisson 60% + empirical 40%)"
    else:
        method = "bivariate Poisson grid"

    width = (
        home.goals_for_per_match.ci_width + away.goals_for_per_match.ci_width
    ) / 2
    conf = _confidence_from_width(width)
    rationale = f"O/U {line} via {method}. λ_h+λ_a={grid.shape[0]}-point grid."
    return (
        MarketProb(f"O{line}", p_over, conf, rationale),
        MarketProb(f"U{line}", p_under, conf, rationale),
    )


def predict_over_corners(
    home: BettableProfile, away: BettableProfile, line: float
) -> MarketProb | None:
    """O/U corners — pure empirical Poisson approx.

    Uses team-level corners_for + opponent's corners_against (proxy for
    'this match generates X corners total'). Poisson with λ = total mean.
    """
    h_for = _combined_with_sub(home, "corners_for_per_match")
    a_for = _combined_with_sub(away, "corners_for_per_match")
    h_against = home.corners_against_per_match
    a_against = away.corners_against_per_match

    if not (
        _is_usable(h_for) and _is_usable(a_for)
        and _is_usable(h_against) and _is_usable(a_against)
    ):
        return None

    expected_home_corners = (h_for.mean + a_against.mean) / 2
    expected_away_corners = (a_for.mean + h_against.mean) / 2
    lam_total = expected_home_corners + expected_away_corners

    # P(total > line) for X ~ Poisson(λ_total)
    from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
        poisson_sf,
    )
    p_over = poisson_sf(int(line), lam_total)

    width = (h_for.ci_width + a_for.ci_width) / 2
    conf = _confidence_from_width(width)
    rationale = (
        f"corners O/U {line}: Poisson(λ={lam_total:.2f}) from "
        f"home(for+against)={expected_home_corners:.2f}, "
        f"away(for+against)={expected_away_corners:.2f}"
    )
    return MarketProb(f"corners_O{line}", p_over, conf, rationale)


def predict_over_cards(
    home: BettableProfile, away: BettableProfile, line: float
) -> MarketProb | None:
    """O/U yellow cards (total in match) — Poisson approx."""
    h_yc = _combined_with_sub(home, "yellow_cards_per_match")
    a_yc = _combined_with_sub(away, "yellow_cards_per_match")
    if not (_is_usable(h_yc) and _is_usable(a_yc)):
        return None
    lam_total = h_yc.mean + a_yc.mean
    from bip.evaluation.tournaments.team_style_profiler.predictor_helpers import (
        poisson_sf,
    )
    p_over = poisson_sf(int(line), lam_total)

    width = (h_yc.ci_width + a_yc.ci_width) / 2
    conf = _confidence_from_width(width)
    rationale = (
        f"cards O/U {line}: Poisson(λ={lam_total:.2f}) summing both teams' "
        f"yellow rates"
    )
    return MarketProb(f"cards_O{line}", p_over, conf, rationale)


def predict_asian_handicap(
    home: BettableProfile,
    away: BettableProfile,
    grid: np.ndarray,
    line: float,
    side: str = "home",
) -> MarketProb:
    """Asian handicap on home or away with given line.

    For side='home', line is the home team's spread:
      -0.5 => home must win outright
      -1.5 => home must win by 2+
      +0.5 => home wins or draws

    For side='away', the perspective flips (away covers when home loses
    by enough or doesn't win).
    """
    if side == "home":
        # P(home_goals - away_goals > -line) = home covers
        # Note: AH "home -1.5" means we want home_goals - away_goals > 1.5
        p_cover = p_home_covers_ah(grid, -line)
        push = p_home_pushes_ah(grid, -line)
    else:
        # Away side: home_goals - away_goals < line for away to cover
        n = grid.shape[0]
        p_cover = 0.0
        for h in range(n):
            for a in range(n):
                if (h - a) < line:
                    p_cover += grid[h, a]
        p_cover = float(p_cover)
        push = p_home_pushes_ah(grid, line)

    width = (
        home.goals_for_per_match.ci_width + away.goals_for_per_match.ci_width
    ) / 2
    conf = _confidence_from_width(width)
    rationale = (
        f"AH {side} line={line:+}: bivariate Poisson grid. "
        f"Push prob (whole-line): {push:.4f}."
    )
    return MarketProb(f"AH_{side}_{line:+}", p_cover, conf, rationale)


# ─── Main entry ─────────────────────────────────────────────────────────


def predict_markets(
    home: BettableProfile,
    away: BettableProfile,
    rho: float = 0.0,
) -> MarketPredictions:
    """Compute all market predictions for a fixture.

    Args:
        home: BettableProfile for the home team (own_tsv or cohort).
        away: BettableProfile for the away team.
        rho: correlation parameter for bivariate Poisson. Default 0.0
             matches lock_v1. >0 introduces positive goal-correlation.

    Returns:
        MarketPredictions with all supported markets.
    """
    lam_h, lam_a = derive_lambdas(home, away)
    grid = bivariate_poisson_grid(lam_h, lam_a, rho=rho)

    btts_yes, btts_no = predict_btts(home, away, grid)
    o25, u25 = predict_over_under_goals(home, away, grid, 2.5)
    o35, _ = predict_over_under_goals(home, away, grid, 3.5)

    return MarketPredictions(
        home_team=home.team_name,
        away_team=away.team_name,
        home_source=home.source,
        away_source=away.source,
        over_2_5=o25,
        over_3_5=o35,
        under_2_5=u25,
        btts_yes=btts_yes,
        btts_no=btts_no,
        corners_over_8_5=predict_over_corners(home, away, 8.5),
        corners_over_9_5=predict_over_corners(home, away, 9.5),
        corners_over_10_5=predict_over_corners(home, away, 10.5),
        cards_over_3_5=predict_over_cards(home, away, 3.5),
        cards_over_4_5=predict_over_cards(home, away, 4.5),
        cards_over_5_5=predict_over_cards(home, away, 5.5),
        ah_home_minus_0_5=predict_asian_handicap(home, away, grid, -0.5, "home"),
        ah_home_minus_1_5=predict_asian_handicap(home, away, grid, -1.5, "home"),
        ah_away_minus_0_5=predict_asian_handicap(home, away, grid, -0.5, "away"),
        ah_away_minus_1_5=predict_asian_handicap(home, away, grid, -1.5, "away"),
        lambda_home=lam_h,
        lambda_away=lam_a,
    )
