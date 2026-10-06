"""Match Dossier Generator — combines TSV knowledge + patterns_v2 tendencies.

PHILOSOPHY (operator directive 2026-05-25):
  "El objetivo NO es predecir a la perfección. Es CONOCER cada selección a la
  perfección y combinarlo con las tendencias para identificar picks probables."

This module turns the team-level knowledge (TSV) + the structural patterns
(patterns_v2) into a structured per-fixture dossier with:
  1. Identity + context (DT, n_matches, flags, host status, etc.)
  2. Team fundamentals (full TSV in readable form)
  3. Sub-profiles vs opponent confederation + patterns v2 applicable
  4. Combined market signals (rules where knowledge + tendencies align)
  5. Ranked picks (categorized + scored) + risk flags

NOT a probabilistic predictor — a knowledge synthesizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from bip.evaluation.tournaments.patterns_v2 import (
    confederation_tilt_verdict,
    host_nation_verdict,
    regime_warning_for_fixture,
    team_transfer_verdict,
)
from bip.evaluation.tournaments.team_style_profiler.tactical_identity import (
    MatchupRead,
    TacticalIdentity,
    read_matchup,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    TeamStyleVector,
)


PickCategory = Literal["STRONG", "MODERATE", "EXPLORATORY", "SKIP"]


@dataclass(frozen=True)
class Pick:
    """A single market pick with category + score + rationale."""

    market: str
    """e.g. 'Under 2.5', 'BTTS Yes', 'Home AH -0.5', 'Fade USA'."""

    direction: str
    """Operator-facing direction: 'BACK' or 'LAY' or 'FADE'."""

    category: PickCategory
    score: float
    """Composite score in [0, 1] — higher = more evidence."""

    rationale: list[str]
    """Bullet list of why this pick: rule names + numbers."""

    risk_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DossierContext:
    """Match-level context that feeds the dossier rules."""

    home_team: str
    away_team: str
    home_tsv: TeamStyleVector | None
    away_tsv: TeamStyleVector | None
    tournament_slug: str
    fixture_date: date
    home_tid: TacticalIdentity | None = None
    away_tid: TacticalIdentity | None = None
    """Tactical identities (planteamiento). When both present, the dossier
    carries a game-plan read of the matchup."""


@dataclass(frozen=True)
class MatchDossier:
    """Full structured output of the dossier generator."""

    context: DossierContext

    # Section 1: identity + context already in context

    # Section 2: per-team fundamentals (the TSV summaries)
    # Just exposed via context.home_tsv / context.away_tsv

    # Section 3: applicable patterns
    host_verdict: dict
    """From host_nation_verdict — host_in_fixture, downgrade_factor, rationale."""

    conf_tilt: dict
    """From confederation_tilt_verdict — tilts + edge_threshold_multiplier."""

    home_team_transfer: dict
    away_team_transfer: dict

    regime_warning: dict | None
    """Only for modern era + HT 0-0 (pre-match: None unless future flag)."""

    # Section 4: ranked picks
    picks: list[Pick]

    # Generation metadata
    generated_at: datetime

    # Section 3b: game-plan read (planteamiento) — present when both tactical
    # identities are supplied in the context.
    matchup_read: MatchupRead | None = None


# ─── Rule helpers ───────────────────────────────────────────────────────


def _is_defensive_team(tsv: TeamStyleVector) -> bool:
    """High clean sheet rate + low GA."""
    return (
        tsv.clean_sheet_rate.mean > 0.45
        and tsv.goals_against_per_match.mean < 1.0
    )


def _is_high_scoring(tsv: TeamStyleVector) -> bool:
    return tsv.goals_for_per_match.mean > 2.0


def _is_card_heavy(tsv: TeamStyleVector) -> bool:
    return tsv.yellow_cards_per_match.mean > 2.5 or tsv.fouls_per_match.mean > 14.0


def _is_corner_heavy(tsv: TeamStyleVector) -> bool:
    return tsv.corners_for_per_match.mean > 6.0


def _profile_confidence_score(tsv: TeamStyleVector | None) -> float:
    """[0, 1] confidence score from sample size + flag."""
    if tsv is None:
        return 0.0
    if tsv.flag == "red":
        return 0.2
    if tsv.flag == "yellow":
        return 0.5
    # green
    n = tsv.n_matches
    if n >= 30:
        return 1.0
    if n >= 20:
        return 0.85
    return 0.7


def _aligned_score(home_signal: bool, away_signal: bool) -> float:
    """Both teams point in same direction → high signal."""
    if home_signal and away_signal:
        return 1.0
    if home_signal or away_signal:
        return 0.5
    return 0.0


# ─── Rule evaluations producing Picks ───────────────────────────────────


def _rule_under_25(ctx: DossierContext) -> Pick | None:
    """Both teams defensive + low total goals tendency."""
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale: list[str] = []
    score = 0.0

    if _is_defensive_team(h):
        rationale.append(
            f"{ctx.home_team}: CS rate {h.clean_sheet_rate.mean:.0%}, "
            f"GA {h.goals_against_per_match.mean:.2f}/match (defensive)"
        )
        score += 0.3
    if _is_defensive_team(a):
        rationale.append(
            f"{ctx.away_team}: CS rate {a.clean_sheet_rate.mean:.0%}, "
            f"GA {a.goals_against_per_match.mean:.2f}/match (defensive)"
        )
        score += 0.3

    combined_o25 = (h.over_25_rate.mean + a.over_25_rate.mean) / 2
    if combined_o25 < 0.40:
        rationale.append(
            f"Combined O2.5 rates: {h.over_25_rate.mean:.0%} + "
            f"{a.over_25_rate.mean:.0%} → both teams play low-scoring"
        )
        score += 0.3

    combined_total = (h.mean_total_goals.mean + a.mean_total_goals.mean) / 2
    if combined_total < 2.3:
        rationale.append(
            f"Combined avg total goals: {combined_total:.2f} (low-scoring matchup)"
        )
        score += 0.2

    # CONMEBOL CONMEBOL or vs UEFA = defensive
    if h.confederation == "CONMEBOL" and a.confederation in ("CONMEBOL", "UEFA"):
        rationale.append(
            "CONMEBOL match: CONMEBOL defends well in WC (mean GA 0.67, "
            "Under-2.5 rate ~60% per patterns_v2 iter 4)"
        )
        score += 0.15

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0)
    confidence_mult = (_profile_confidence_score(h) + _profile_confidence_score(a)) / 2
    score *= confidence_mult
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="Under 2.5 goals",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_over_25(ctx: DossierContext) -> Pick | None:
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale = []
    score = 0.0

    if _is_high_scoring(h):
        rationale.append(
            f"{ctx.home_team}: high-scoring ({h.goals_for_per_match.mean:.2f} GF/match)"
        )
        score += 0.25
    if _is_high_scoring(a):
        rationale.append(
            f"{ctx.away_team}: high-scoring ({a.goals_for_per_match.mean:.2f} GF/match)"
        )
        score += 0.25

    combined_o25 = (h.over_25_rate.mean + a.over_25_rate.mean) / 2
    if combined_o25 > 0.60:
        rationale.append(
            f"Combined O2.5 rates: {h.over_25_rate.mean:.0%} + "
            f"{a.over_25_rate.mean:.0%}"
        )
        score += 0.25

    # UEFA tendency
    if h.confederation == "UEFA" and a.confederation == "UEFA":
        rationale.append("UEFA vs UEFA: O2.5% baseline 60% per cohort stats")
        score += 0.10

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0) * ((_profile_confidence_score(h) + _profile_confidence_score(a)) / 2)
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="Over 2.5 goals",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_btts(ctx: DossierContext) -> Pick | None:
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale = []
    score = 0.0

    if h.btts_rate.mean > 0.55:
        rationale.append(f"{ctx.home_team} BTTS rate {h.btts_rate.mean:.0%}")
        score += 0.25
    if a.btts_rate.mean > 0.55:
        rationale.append(f"{ctx.away_team} BTTS rate {a.btts_rate.mean:.0%}")
        score += 0.25

    # Both concede some goals
    if h.goals_against_per_match.mean > 1.0 and a.goals_against_per_match.mean > 1.0:
        rationale.append("Both teams concede regularly")
        score += 0.20

    # Both score some goals
    if h.goals_for_per_match.mean > 1.2 and a.goals_for_per_match.mean > 1.2:
        rationale.append("Both teams score consistently")
        score += 0.20

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0) * ((_profile_confidence_score(h) + _profile_confidence_score(a)) / 2)
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="BTTS Yes",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_btts_no(ctx: DossierContext) -> Pick | None:
    """At least one team likely to keep clean sheet OR not score."""
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale = []
    score = 0.0

    if h.clean_sheet_rate.mean > 0.50:
        rationale.append(
            f"{ctx.home_team}: clean sheets {h.clean_sheet_rate.mean:.0%} of matches"
        )
        score += 0.25
    if a.clean_sheet_rate.mean > 0.50:
        rationale.append(
            f"{ctx.away_team}: clean sheets {a.clean_sheet_rate.mean:.0%}"
        )
        score += 0.25

    # Low BTTS rates
    avg_btts = (h.btts_rate.mean + a.btts_rate.mean) / 2
    if avg_btts < 0.40:
        rationale.append(f"Combined BTTS rates avg {avg_btts:.0%} (low)")
        score += 0.20

    # One team weak attack
    if h.goals_for_per_match.mean < 1.0 or a.goals_for_per_match.mean < 1.0:
        weak = ctx.home_team if h.goals_for_per_match.mean < 1.0 else ctx.away_team
        weak_gf = min(h.goals_for_per_match.mean, a.goals_for_per_match.mean)
        rationale.append(f"{weak} weak attack ({weak_gf:.2f} GF/match)")
        score += 0.20

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0) * ((_profile_confidence_score(h) + _profile_confidence_score(a)) / 2)
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="BTTS No",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_corners_over(ctx: DossierContext) -> Pick | None:
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale = []
    score = 0.0

    if _is_corner_heavy(h):
        rationale.append(
            f"{ctx.home_team}: {h.corners_for_per_match.mean:.1f} corners/match"
        )
        score += 0.30
    if _is_corner_heavy(a):
        rationale.append(
            f"{ctx.away_team}: {a.corners_for_per_match.mean:.1f} corners/match"
        )
        score += 0.30

    # Mismatch favoring corners (P8 from v1)
    expected_total = (
        h.corners_for_per_match.mean
        + a.corners_for_per_match.mean
        + h.corners_against_per_match.mean
        + a.corners_against_per_match.mean
    ) / 2
    if expected_total > 10.5:
        rationale.append(
            f"Expected total ~{expected_total:.1f} corners"
        )
        score += 0.25

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0) * ((_profile_confidence_score(h) + _profile_confidence_score(a)) / 2)
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="Corners Over 9.5",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_cards_over(ctx: DossierContext) -> Pick | None:
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    rationale = []
    score = 0.0

    if _is_card_heavy(h):
        rationale.append(
            f"{ctx.home_team}: {h.yellow_cards_per_match.mean:.1f} YC/match, "
            f"{h.fouls_per_match.mean:.1f} fouls"
        )
        score += 0.30
    if _is_card_heavy(a):
        rationale.append(
            f"{ctx.away_team}: {a.yellow_cards_per_match.mean:.1f} YC/match, "
            f"{a.fouls_per_match.mean:.1f} fouls"
        )
        score += 0.30

    expected_cards = h.yellow_cards_per_match.mean + a.yellow_cards_per_match.mean
    if expected_cards > 5.0:
        rationale.append(f"Combined expected yellows: {expected_cards:.1f}")
        score += 0.20

    if score < 0.4 or not rationale:
        return None
    score = min(score, 1.0) * ((_profile_confidence_score(h) + _profile_confidence_score(a)) / 2)
    category: PickCategory = (
        "STRONG" if score >= 0.65 else "MODERATE" if score >= 0.40 else "EXPLORATORY"
    )
    return Pick(
        market="Cards Over 4.5",
        direction="BACK",
        category=category,
        score=score,
        rationale=rationale,
    )


def _rule_fade_host(ctx: DossierContext) -> Pick | None:
    """L5.6: host nation under-performs vs predictor expectation."""
    host_v = host_nation_verdict(
        ctx.home_team, ctx.away_team, ctx.tournament_slug
    )
    if not host_v.host_in_fixture:
        return None
    # Which side is the host?
    from bip.evaluation.tournaments.patterns_v2 import HOST_NATIONS
    hosts = HOST_NATIONS.get(ctx.tournament_slug, frozenset())
    host_side = "home" if ctx.home_team in hosts else "away"
    rationale = [
        f"Host nation ({host_side}) in fixture: {ctx.home_team if host_side=='home' else ctx.away_team}",
        "L5.6 evidence: hosts 3.32× over-represented in worst-Brier decile (p=0.0065)",
        "Lock_v1 lacks host-advantage feature → typically over-rates hosts",
    ]
    return Pick(
        market=f"FADE {ctx.home_team if host_side=='home' else ctx.away_team} (host)",
        direction="LAY",
        category="MODERATE",
        score=0.55,
        rationale=rationale,
        risk_flags=["L5.6 host effect — moderate sample n=15"],
    )


def _rule_conmebol_uefa(ctx: DossierContext) -> Pick | None:
    """Iter 4: CONMEBOL beats UEFA 43% in modern WC."""
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None
    if not (
        (h.confederation == "CONMEBOL" and a.confederation == "UEFA")
        or (h.confederation == "UEFA" and a.confederation == "CONMEBOL")
    ):
        return None
    conmebol_team = ctx.home_team if h.confederation == "CONMEBOL" else ctx.away_team
    rationale = [
        f"CONMEBOL ({conmebol_team}) vs UEFA matchup",
        "Iter 4 evidence: modern WC (>=2012) CONMEBOL wins UEFA 43% (n=37, "
        "martj42)",
        "Lock_v1's strength prior under-weights CONMEBOL WC dominance",
    ]
    return Pick(
        market=f"Tilt {conmebol_team} (CONMEBOL vs UEFA)",
        direction="BACK",
        category="MODERATE",
        score=0.50,
        rationale=rationale,
    )


def _rule_favorite_dominance(ctx: DossierContext) -> Pick | None:
    """One team clearly superior in both attack AND defense → back to win."""
    h, a = ctx.home_tsv, ctx.away_tsv
    if h is None or a is None:
        return None

    # Compute relative dominance per dimension
    h_attack_edge = h.goals_for_per_match.mean - a.goals_against_per_match.mean
    a_attack_edge = a.goals_for_per_match.mean - h.goals_against_per_match.mean
    h_def_edge = a.goals_for_per_match.mean - h.goals_against_per_match.mean
    a_def_edge = h.goals_for_per_match.mean - a.goals_against_per_match.mean

    # Home dominant
    if h.goals_for_per_match.mean - a.goals_for_per_match.mean > 0.4 and \
       h.goals_against_per_match.mean < a.goals_against_per_match.mean - 0.4 and \
       h.clean_sheet_rate.mean > a.clean_sheet_rate.mean + 0.15:
        rationale = [
            f"{ctx.home_team} dominates: GF {h.goals_for_per_match.mean:.2f} vs "
            f"{a.goals_for_per_match.mean:.2f}; GA {h.goals_against_per_match.mean:.2f} "
            f"vs {a.goals_against_per_match.mean:.2f}",
            f"Clean sheet gap: {ctx.home_team} {h.clean_sheet_rate.mean:.0%} vs "
            f"{ctx.away_team} {a.clean_sheet_rate.mean:.0%}",
        ]
        score = 0.55 + min(0.25, abs(h.goals_for_per_match.mean - a.goals_for_per_match.mean) * 0.2)
        score *= (_profile_confidence_score(h) + _profile_confidence_score(a)) / 2
        category: PickCategory = "STRONG" if score >= 0.65 else "MODERATE"
        return Pick(
            market=f"Back {ctx.home_team} to win",
            direction="BACK",
            category=category,
            score=score,
            rationale=rationale,
        )
    # Away dominant (rare in WC home advantage)
    if a.goals_for_per_match.mean - h.goals_for_per_match.mean > 0.4 and \
       a.goals_against_per_match.mean < h.goals_against_per_match.mean - 0.4 and \
       a.clean_sheet_rate.mean > h.clean_sheet_rate.mean + 0.15:
        rationale = [
            f"{ctx.away_team} dominates: GF {a.goals_for_per_match.mean:.2f} vs "
            f"{h.goals_for_per_match.mean:.2f}; GA {a.goals_against_per_match.mean:.2f} "
            f"vs {h.goals_against_per_match.mean:.2f}",
            f"Clean sheet gap: {ctx.away_team} {a.clean_sheet_rate.mean:.0%} vs "
            f"{ctx.home_team} {h.clean_sheet_rate.mean:.0%}",
        ]
        score = 0.55 + min(0.25, abs(a.goals_for_per_match.mean - h.goals_for_per_match.mean) * 0.2)
        score *= (_profile_confidence_score(h) + _profile_confidence_score(a)) / 2
        category: PickCategory = "STRONG" if score >= 0.65 else "MODERATE"
        return Pick(
            market=f"Back {ctx.away_team} to win (or draw-no-bet)",
            direction="BACK",
            category=category,
            score=score,
            rationale=rationale,
        )
    return None


def _rule_team_transfer(ctx: DossierContext) -> list[Pick]:
    """Apply iter 2 team-level transfer tilts."""
    out = []
    for side, team in [("home", ctx.home_team), ("away", ctx.away_team)]:
        v = team_transfer_verdict(team)
        if not v.has_transfer_signal:
            continue
        if v.tilt > 1.0:
            direction = "BACK"
            category: PickCategory = "MODERATE"
            score = 0.5
        else:
            direction = "FADE"
            category = "MODERATE"
            score = 0.5
        out.append(Pick(
            market=f"{direction} {team} (cross-tournament signal)",
            direction=direction,
            category=category,
            score=score,
            rationale=[
                f"Team-transfer signal for {team} (iter 2)",
                v.evidence,
                f"Tilt factor: ×{v.tilt:.2f}",
            ],
        ))
    return out


# ─── Tactical cross-confirmation (game-plan ↔ picks) ─────────────────────


def _axis_side(market: str) -> tuple[str | None, str | None]:
    """Map a pick/lean market string to a (betting axis, side) for alignment."""
    m = market.lower()
    if "under 2.5" in m:
        return ("goals", "low")
    if "over 2.5" in m:
        return ("goals", "high")
    if "btts no" in m:
        return ("btts", "no")
    if "btts" in m and ("yes" in m or "sí" in m or " si" in m):
        return ("btts", "yes")
    if "corner" in m or "córner" in m:
        if "under" in m:
            return ("corners", "under")
        if "over" in m:
            return ("corners", "over")
    return (None, None)


def _apply_tactical_confirmation(
    picks: list[Pick], read: "MatchupRead | None"
) -> list[Pick]:
    """Cross-confirm TSV picks against the game-plan leans.

    A BACK lean on the same axis+side boosts the pick (+rationale, +score); an
    opposite-side BACK lean or a same-side CAUTION lean adds a risk flag. Does
    NOT create new picks — the tactical-only leans stay in the §1b read.
    """
    if read is None or not picks:
        return picks
    out: list[Pick] = []
    for p in picks:
        axis_p, side_p = _axis_side(p.market)
        if axis_p is None or p.direction not in ("BACK",):
            out.append(p)
            continue
        new_rationale = list(p.rationale)
        new_flags = list(p.risk_flags)
        bump = 0.0
        for lean in read.market_leans:
            axis_l, side_l = _axis_side(lean.market)
            if axis_l != axis_p:
                continue
            if lean.direction == "CAUTION" and side_l == side_p:
                new_flags.append(f"Planteamiento — {lean.rationale}")
            elif lean.direction == "BACK" and side_l == side_p:
                new_rationale.append(f"✓ Planteamiento confirma: {lean.rationale}")
                bump += 0.10
            elif lean.direction == "BACK" and side_l != side_p:
                new_flags.append(
                    f"Planteamiento sugiere lo contrario ({lean.market}): {lean.rationale}"
                )
        if bump == 0.0 and new_flags == list(p.risk_flags):
            out.append(p)
            continue
        new_score = min(1.0, p.score + bump)
        category: PickCategory = (
            "STRONG" if new_score >= 0.65 else "MODERATE" if new_score >= 0.40 else "EXPLORATORY"
        )
        out.append(Pick(
            market=p.market, direction=p.direction, category=category,
            score=new_score, rationale=new_rationale, risk_flags=new_flags,
        ))
    return out


# ─── Main entry point ──────────────────────────────────────────────────


def _collect_risk_flags(ctx: DossierContext) -> list[str]:
    flags = []
    for side, tsv, name in [
        ("home", ctx.home_tsv, ctx.home_team),
        ("away", ctx.away_tsv, ctx.away_team),
    ]:
        if tsv is None:
            flags.append(f"{name}: NO profile available — using cohort fallback")
            continue
        if tsv.flag == "red":
            flags.append(
                f"{name}: RED flag ({tsv.n_matches} matches with "
                f"{tsv.coach.coach_name}) — perfil no confiable"
            )
        elif tsv.flag == "yellow":
            flags.append(
                f"{name}: YELLOW flag ({tsv.n_matches} matches with "
                f"{tsv.coach.coach_name}) — confianza media"
            )
        # New coach within 18 months
        days_in_role = (ctx.fixture_date - tsv.coach.start_date).days
        if days_in_role < 365:
            flags.append(
                f"{name}: DT {tsv.coach.coach_name} contratado hace solo "
                f"{days_in_role} días"
            )
    return flags


def generate_dossier(ctx: DossierContext) -> MatchDossier:
    """Build the full match dossier."""
    # Pattern verdicts
    host_v = host_nation_verdict(ctx.home_team, ctx.away_team, ctx.tournament_slug)
    conf_v = confederation_tilt_verdict(ctx.home_team, ctx.away_team)
    home_tt = team_transfer_verdict(ctx.home_team)
    away_tt = team_transfer_verdict(ctx.away_team)
    # Regime: pre-match without HT score → not active but reportable
    regime_v = regime_warning_for_fixture(ctx.fixture_date, ht_score=None)

    # Game-plan read (planteamiento) when both tactical identities are present
    matchup_read = (
        read_matchup(ctx.home_tid, ctx.away_tid)
        if ctx.home_tid is not None and ctx.away_tid is not None
        else None
    )

    # Evaluate rules
    picks: list[Pick] = []
    for rule in [
        _rule_under_25, _rule_over_25, _rule_btts, _rule_btts_no,
        _rule_corners_over, _rule_cards_over, _rule_fade_host,
        _rule_conmebol_uefa, _rule_favorite_dominance,
    ]:
        pick = rule(ctx)
        if pick:
            picks.append(pick)
    picks.extend(_rule_team_transfer(ctx))

    # Cross-confirm picks against the game-plan read (boost aligned, flag conflicts)
    picks = _apply_tactical_confirmation(picks, matchup_read)

    # Sort: STRONG → MODERATE → EXPLORATORY, score desc within category
    cat_order = {"STRONG": 0, "MODERATE": 1, "EXPLORATORY": 2, "SKIP": 3}
    picks.sort(key=lambda p: (cat_order.get(p.category, 9), -p.score))

    # Attach risk flags to top picks where relevant
    risk_flags = _collect_risk_flags(ctx)
    if risk_flags:
        picks = [
            Pick(
                market=p.market, direction=p.direction, category=p.category,
                score=p.score, rationale=p.rationale,
                risk_flags=list(p.risk_flags) + risk_flags,
            )
            for p in picks
        ]

    return MatchDossier(
        context=ctx,
        host_verdict={
            "host_in_fixture": host_v.host_in_fixture,
            "downgrade_factor": host_v.downgrade_factor,
            "rationale": host_v.rationale,
        },
        conf_tilt={
            "home_conf": conf_v.home_conf,
            "away_conf": conf_v.away_conf,
            "home_tilt": conf_v.home_tilt,
            "away_tilt": conf_v.away_tilt,
            "is_cross_conf": conf_v.is_cross_conf,
            "edge_threshold_multiplier": conf_v.edge_threshold_multiplier,
            "rationale": conf_v.rationale,
        },
        home_team_transfer={
            "has_signal": home_tt.has_transfer_signal,
            "tilt": home_tt.tilt,
            "evidence": home_tt.evidence,
        },
        away_team_transfer={
            "has_signal": away_tt.has_transfer_signal,
            "tilt": away_tt.tilt,
            "evidence": away_tt.evidence,
        },
        regime_warning={
            "is_modern_era": regime_v.is_modern_era,
            "warning_active": regime_v.warning_active,
            "rationale": regime_v.rationale,
        } if regime_v.is_modern_era else None,
        picks=picks,
        generated_at=datetime.now(),
        matchup_read=matchup_read,
    )
