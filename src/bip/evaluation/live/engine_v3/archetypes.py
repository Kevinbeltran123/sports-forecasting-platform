"""Rule layer of the Causal Hypothesis Generator.

12 archetype detectors from sec 4-bis of the v3 engine design notes, each
implemented as a pure ``(GSV) -> Thesis | None`` function. The generator
calls all detectors and collects the non-None ones — a single GSV
frame can activate multiple archetypes (e.g., red card AND late game
state → both #1 and #11 fire).

**Why pure functions, not classes**: deterministic, easy to unit-test
in isolation, and the dispatch loop is trivially parallelizable if it
ever becomes a hotspot. The current 12 detectors run in <2ms total
on a typical GSV.

**Why we cap at 12 archetypes**: per risk #1 in sec 8, rule explosion
is the failure mode. The hard ceiling is 30 — anything beyond gets
absorbed by the pattern layer (kNN over state embeddings).

Confidence priors are anchors, not learned. They come from operator
domain knowledge + the SYNTHESIS.md historical hit-rate evidence. The
real calibration happens in Conditional Predictor / per-archetype
isotonic per sec 5 of the design.
"""
from __future__ import annotations

from collections.abc import Callable

from bip.evaluation.live.engine_v3.gsv import GameStateVector
from bip.evaluation.live.engine_v3.thesis import (
    CausalChain,
    CausalStep,
    ConditionalShift,
    GSVPredicate,
    InvalidationTrigger,
    MarketFamily,
    Thesis,
    ThesisArchetype,
    ThesisSource,
    build_horizon,
)

# ──────────────────────────────────────────────────────────────────────
# Helper builders
# ──────────────────────────────────────────────────────────────────────


def _src(rule_id: str) -> ThesisSource:
    return ThesisSource(layer="rule", identifier=rule_id)


def _mk_thesis(
    *,
    arch: ThesisArchetype,
    rule_id: str,
    minute: int,
    premise: list[GSVPredicate],
    chain: list[CausalStep],
    family: MarketFamily,
    direction: str,
    magnitude_pp: float,
    horizon_label: str,
    invalidations: list[InvalidationTrigger],
    confidence_prior: float,
) -> Thesis:
    return Thesis(
        id=f"{rule_id}@m{minute}",
        archetype=arch,
        premise=premise,
        mechanism=CausalChain(steps=chain),
        prediction=ConditionalShift(
            family=family,
            direction=direction,  # type: ignore[arg-type]
            magnitude_pp=magnitude_pp,
            horizon=build_horizon(horizon_label, minute),  # type: ignore[arg-type]
        ),
        invalidation_triggers=invalidations,
        confidence_prior=confidence_prior,
        source=_src(rule_id),
        activated_at_minute=minute,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 1 — Red card to AWAY before minute 30
# ──────────────────────────────────────────────────────────────────────


def detect_red_card_away_early(gsv: GameStateVector) -> Thesis | None:
    if gsv.numerical.red_cards_away < 1:
        return None
    if gsv.time.minute >= 30:
        return None
    # Home must be at least neutrally positioned in xG before this fires —
    # if home was already losing the xG battle, a red card doesn't auto-flip.
    if gsv.xg.xg_diff < -0.4:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.RED_CARD_AWAY_EARLY,
        rule_id="A1",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="numerical.red_cards_away", op="ge", value=1),
            GSVPredicate(path="time.minute", op="lt", value=30),
            GSVPredicate(path="xg.xg_diff", op="ge", value=-0.4),
        ],
        chain=[
            CausalStep(
                cause="away_red_card_early",
                effect="away_drops_block_deep",
                mechanism="numerical inferiority forces defensive shape",
            ),
            CausalStep(
                cause="home_sustained_pressure",
                effect="corners_2H_rate_up",
                mechanism="entries to final third without conversion",
            ),
        ],
        family=MarketFamily.CORNERS,
        direction="over",
        magnitude_pp=0.08,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="away scores → strategy resets, line repriced",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="home red equalises numerics",
            ),
        ],
        confidence_prior=0.62,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 2 — Dominant team trailing at 25-45' (the Napoli case)
# ──────────────────────────────────────────────────────────────────────


def detect_dominant_losing_napoli(gsv: GameStateVector) -> Thesis | None:
    if not gsv.score.dominant_losing:
        return None
    if not (25 <= gsv.time.minute <= 45):
        return None
    # Require xG divergence that *supports* the regression argument —
    # dominant team must actually be outshooting the leader by enough.
    if abs(gsv.xg.xg_vs_score_divergence) < 0.3:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.DOMINANT_LOSING_NAPOLI,
        rule_id="A2",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.dominant_losing", op="eq", value=True),
            GSVPredicate(path="time.minute", op="between", value=[25, 45]),
            GSVPredicate(path="xg.xg_vs_score_divergence", op="ge", value=0.3),
        ],
        chain=[
            CausalStep(
                cause="dominant_team_trails_and_outshoots",
                effect="urgency_increases",
                mechanism="favourite must now create more, defends less",
            ),
            CausalStep(
                cause="leader_concedes_territory",
                effect="next_goal_likely_for_dominant",
                mechanism="regression to pre-match expectation + push",
            ),
        ],
        family=MarketFamily.NEXT_GOAL,
        direction="home" if gsv.score.dominant_team_id == gsv.home_team_id else "away",
        magnitude_pp=0.10,
        horizon_label="rest_of_half",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="trailing team double-down → thesis collapses",
            ),
            InvalidationTrigger(
                kind="halftime_reached",
                description="HT reset disrupts momentum thesis",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="dominant team red erases the push thesis",
            ),
        ],
        confidence_prior=0.58,
    )


def detect_napoli_btts_companion(gsv: GameStateVector) -> Thesis | None:
    """Phase-2 companion to A2 — BTTS Yes for the Napoli regime.

    When the Napoli predicate holds (dominant trailing, 25-45', xG
    divergence ≥ 0.3) AND the *underdog* has scored but the *dominant*
    side has not, "next goal to dominant" implies both teams will have
    scored by end of game. That's a BTTS Yes thesis with high prior
    confidence — and unlocks the BTTSPredictor that's otherwise dormant.

    Why not collapse it into A2 itself: market families differ. A2's
    market routing is NEXT_GOAL/GOALS; this companion's is BTTS. The
    market_selector decides which (or both) markets to score, and the
    no-bet gate evaluates them independently.

    Skip rules (informed by internal research notes empirical analysis
    of v2 outcomes: btts minute 15-30 had ROI −40%):
        - require minute ≥ 30 (Day-1+2 v2 data: BTTS pre-30 is a
          systematic ROI loser)
        - require dominant_team has NOT yet scored (otherwise BTTS Yes
          may already be settled or trivial)
        - require underdog has scored EXACTLY one (so the BTTS Yes
          conclusion follows naturally from the napoli prediction)
    """
    if not gsv.score.dominant_losing:
        return None
    if not (30 <= gsv.time.minute <= 45):
        return None
    if abs(gsv.xg.xg_vs_score_divergence) < 0.3:
        return None
    dom_is_home = gsv.score.dominant_team_id == gsv.home_team_id
    dom_goals = gsv.score.home_goals if dom_is_home else gsv.score.away_goals
    underdog_goals = gsv.score.away_goals if dom_is_home else gsv.score.home_goals
    if dom_goals != 0:
        return None
    if underdog_goals < 1:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.DOMINANT_LOSING_NAPOLI,  # same archetype id, different market family
        rule_id="A2b",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.dominant_losing", op="eq", value=True),
            GSVPredicate(path="time.minute", op="between", value=[30, 45]),
            GSVPredicate(path="xg.xg_vs_score_divergence", op="ge", value=0.3),
        ],
        chain=[
            CausalStep(
                cause="dominant_trails_with_xg_pressure",
                effect="dominant_likely_scores_in_remaining_match",
                mechanism="urgency + outshooting → P(dominant ≥1) is high",
            ),
            CausalStep(
                cause="underdog_has_already_scored",
                effect="btts_yes_resolves_when_dominant_scores",
                mechanism="both teams scored ≥1 → BTTS Yes settled",
            ),
        ],
        family=MarketFamily.BTTS,
        direction="yes",
        magnitude_pp=0.08,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="underdog 2-up → state changes, regenerate",
            ),
            InvalidationTrigger(
                kind="goal_for_dominant",
                description="BTTS Yes trivially resolves — pick stops being a bet",
            ),
        ],
        confidence_prior=0.60,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 3 — 0-0 at 75', low xG sum, cagey closed
# ──────────────────────────────────────────────────────────────────────


def detect_late_cagey_zero_zero(gsv: GameStateVector) -> Thesis | None:
    if not gsv.is_late_cagey_zero_zero:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.LATE_CAGEY_ZERO_ZERO,
        rule_id="A3",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.home_goals", op="eq", value=0),
            GSVPredicate(path="score.away_goals", op="eq", value=0),
            GSVPredicate(path="time.minute", op="ge", value=75),
            GSVPredicate(path="tactical.game_phase", op="eq", value="cagey_closed"),
        ],
        chain=[
            CausalStep(
                cause="both_sides_play_for_point",
                effect="match_likely_ends_no_goal",
                mechanism="commentary + xG confirm low-intent territory game",
            ),
        ],
        family=MarketFamily.GOALS,
        direction="under",
        magnitude_pp=0.07,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="any_goal",
                description="0-0 broken → thesis dies trivially",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="red card cracks the equilibrium",
            ),
            InvalidationTrigger(
                kind="red_card_underdog",
                description="red card cracks the equilibrium",
            ),
        ],
        confidence_prior=0.64,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 4 — Lead 2-0 by 60', defensive sub from the leader
# ──────────────────────────────────────────────────────────────────────


def detect_lead_two_defensive_sub(gsv: GameStateVector) -> Thesis | None:
    leader_diff = abs(gsv.score.goal_diff)
    if leader_diff < 2 or not (55 <= gsv.time.minute <= 75):
        return None
    leader_is_home = gsv.score.goal_diff >= 2
    defensive_subs = [
        s for s in gsv.roster.recent_subs_5min
        if s.role_signal == "defensive"
        and ((leader_is_home and s.team_id == gsv.home_team_id)
             or (not leader_is_home and s.team_id == gsv.away_team_id))
    ]
    if not defensive_subs:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.LEAD_TWO_DEFENSIVE_SUB,
        rule_id="A4",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.goal_diff", op="in", value=[2, -2, 3, -3]),
            GSVPredicate(path="time.minute", op="between", value=[55, 75]),
            GSVPredicate(path="roster.recent_subs_5min", op="contains", value="defensive"),
        ],
        chain=[
            CausalStep(
                cause="leader_defensive_sub",
                effect="leader_drops_block",
                mechanism="manager-intent signal: lock the result",
            ),
            CausalStep(
                cause="trailing_team_has_territory_without_quality",
                effect="corners_for_trailing_up_goals_flat",
                mechanism="possession in deep zones, low xG per shot",
            ),
        ],
        family=MarketFamily.CORNERS,
        direction="over",
        magnitude_pp=0.06,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="trailing scores → tactical reset",
            ),
            InvalidationTrigger(
                kind="key_sub_dominant_offensive",
                description="leader reverses intent with an offensive sub",
            ),
        ],
        confidence_prior=0.57,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 5 — Tied at 60', xG_diff > 1.5 (regression to xG)
# ──────────────────────────────────────────────────────────────────────


def detect_regression_to_xg(gsv: GameStateVector) -> Thesis | None:
    # xG-diff threshold lowered 1.5 → 0.7 (= empirical p90 of tied-60'
    # frames in internal research notes). 1.5 captured only 2/151
    # frames in Day-3 (>p90 outliers); 0.7 captures the right tail.
    if gsv.score.goal_diff != 0 or gsv.time.minute < 60:
        return None
    if abs(gsv.xg.xg_diff) < 0.7:
        return None
    high_xg_home = gsv.xg.xg_diff > 0
    return _mk_thesis(
        arch=ThesisArchetype.REGRESSION_TO_XG,
        rule_id="A5",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.goal_diff", op="eq", value=0),
            GSVPredicate(path="time.minute", op="ge", value=60),
            GSVPredicate(path="xg.xg_diff", op="gt" if high_xg_home else "lt",
                         value=0.7 if high_xg_home else -0.7),
        ],
        chain=[
            CausalStep(
                cause="high_xg_team_underconverted",
                effect="regression_to_mean",
                mechanism="finishing variance reverts over remaining minutes",
            ),
        ],
        family=MarketFamily.NEXT_GOAL,
        direction="home" if high_xg_home else "away",
        magnitude_pp=0.07,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="opposite team scores → thesis dies",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="numerical loss erodes the push",
            ),
        ],
        confidence_prior=0.55,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 6 — Cards momentum + strict ref
# ──────────────────────────────────────────────────────────────────────


def detect_cards_momentum_strict_ref(gsv: GameStateVector) -> Thesis | None:
    total_yellows = gsv.cards.yellows[0] + gsv.cards.yellows[1]
    if total_yellows < 5 or gsv.time.minute < 70:
        return None
    if gsv.cards.ref_card_rate_prior < 5.0:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.CARDS_MOMENTUM_STRICT_REF,
        rule_id="A6",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="cards.yellows", op="ge", value=5),
            GSVPredicate(path="time.minute", op="ge", value=70),
            GSVPredicate(path="cards.ref_card_rate_prior", op="ge", value=5.0),
        ],
        chain=[
            CausalStep(
                cause="card_cluster_under_strict_ref",
                effect="more_cards_in_remaining_minutes",
                mechanism="referee tolerance threshold + late-game tactical fouls",
            ),
        ],
        family=MarketFamily.CARDS,
        direction="over",
        magnitude_pp=0.08,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="any_goal",
                description="goal lowers urgency / cards drop",
            ),
            InvalidationTrigger(
                kind="ref_card_strict_regime",
                description="prior was miscalibrated for this ref",
            ),
        ],
        confidence_prior=0.60,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 7 — Underdog leads + sustained siege
# ──────────────────────────────────────────────────────────────────────


def detect_underdog_leads_siege(gsv: GameStateVector) -> Thesis | None:
    if gsv.score.dominant_team_id is None:
        return None
    if abs(gsv.score.goal_diff) != 1:
        return None
    leader_is_home = gsv.score.goal_diff == 1
    leader_id = gsv.home_team_id if leader_is_home else gsv.away_team_id
    if leader_id == gsv.score.dominant_team_id:
        return None  # not an underdog leading
    if gsv.time.minute < 70:
        return None
    underdog_phase = gsv.tactical.home_phase if leader_is_home else gsv.tactical.away_phase
    if underdog_phase not in ("parking_bus", "collapsing"):
        return None
    return _mk_thesis(
        arch=ThesisArchetype.UNDERDOG_LEADS_SIEGE,
        rule_id="A7",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.dominant_losing", op="eq", value=True),
            GSVPredicate(path="time.minute", op="ge", value=70),
            GSVPredicate(path="tactical.home_phase" if leader_is_home else "tactical.away_phase",
                         op="in", value=["parking_bus", "collapsing"]),
        ],
        chain=[
            CausalStep(
                cause="underdog_parks_bus",
                effect="dominant_accumulates_corners_late",
                mechanism="territorial pressure converted to set pieces",
            ),
            CausalStep(
                cause="dominant_attacks_force_underdog_fouls",
                effect="underdog_cards_up",
                mechanism="tactical fouling near box",
            ),
        ],
        family=MarketFamily.CORNERS,
        direction="over",
        magnitude_pp=0.07,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_dominant",
                description="equaliser → siege thesis ends",
            ),
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="underdog 2-up → game state flips",
            ),
        ],
        confidence_prior=0.61,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 8 — Open game (3+ goals before 60', aggressive formations)
# ──────────────────────────────────────────────────────────────────────


def detect_open_game(gsv: GameStateVector) -> Thesis | None:
    """A8 — open game (3+ goals before 60'). T3 relaxation (2026-05-14):
    formation data is always ``unknown`` in production (Sportmonks does
    not populate per-fixture formations on the live feed; Day-4 sampled
    31/31 fixtures with ``formation_home='unknown'`` and
    ``formation_away='unknown'``). The aggressive-formation gate was
    a proxy for "match is genuinely open" — but ``is_open_game`` (= 3+
    goals before 60') is itself a stronger empirical signal than the
    formation proxy. We fire on the score+minute predicate and let the
    no-bet gate downstream handle calibration / OOD filtering.

    If the formation feed ever lights up we can re-tighten via
    invalidation triggers; the proxy is no longer required for emission.
    """
    if not gsv.is_open_game:
        return None
    aggressive_formations = {"4-3-3", "3-4-3", "4-2-4", "3-3-4"}
    has_aggressive_formation = (
        gsv.roster.formation_home in aggressive_formations
        or gsv.roster.formation_away in aggressive_formations
    )
    # Higher prior when formation confirms — both signals point the same
    # way. Otherwise fall back to score-based prior alone.
    confidence_prior = 0.65 if has_aggressive_formation else 0.55
    return _mk_thesis(
        arch=ThesisArchetype.OPEN_GAME_FORMATIONS,
        rule_id="A8",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.home_goals", op="ge", value=0),
            GSVPredicate(path="time.minute", op="le", value=60),
        ],
        chain=[
            CausalStep(
                cause="3+_goals_early",
                effect="prematch_priors_obsolete",
                mechanism="structure demonstrably open, rate forward-extrapolates",
            ),
        ],
        family=MarketFamily.GOALS,
        direction="over",
        magnitude_pp=0.09,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="red_card_dominant",
                description="red shuts the game down",
            ),
            InvalidationTrigger(
                kind="red_card_underdog",
                description="red shuts the game down",
            ),
            InvalidationTrigger(
                kind="formation_change",
                description="manager switches to a defensive shape",
            ),
        ],
        confidence_prior=confidence_prior,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 9 — Key playmaker subbed off (injury)
# ──────────────────────────────────────────────────────────────────────


def detect_key_playmaker_off(gsv: GameStateVector) -> Thesis | None:
    if not any(gsv.roster.key_player_off):
        return None
    home_off, away_off = gsv.roster.key_player_off
    # We only fire when the team that lost the playmaker was the
    # *dominant* (creator) side — otherwise the effect is weak.
    if home_off and gsv.score.dominant_team_id != gsv.home_team_id:
        return None
    if away_off and gsv.score.dominant_team_id != gsv.away_team_id:
        return None
    direction_team = "home" if home_off else "away"
    return _mk_thesis(
        arch=ThesisArchetype.KEY_PLAYMAKER_OFF,
        rule_id="A9",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="roster.key_player_off", op="contains", value=True),
        ],
        chain=[
            CausalStep(
                cause="creator_subbed_off",
                effect="team_shot_rate_drops",
                mechanism="key_passes loss not absorbed by replacement",
            ),
        ],
        family=MarketFamily.GOALS,
        direction="under",
        magnitude_pp=0.05,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="any_goal",
                description="any goal disrupts the under thesis",
            ),
        ],
        confidence_prior=0.52,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 10 — Second-half reset (HT formation change by trailing side)
# ──────────────────────────────────────────────────────────────────────


def detect_second_half_reset(gsv: GameStateVector) -> Thesis | None:
    """A10 — second-half reset (HT formation change by trailing favourite).
    T3 relaxation (2026-05-14): ``formation_changes`` is empty in 31/31
    production fixtures (Sportmonks does not emit formation-change
    events on the live feed). The causal mechanism — favourite trails
    at HT and pushes with new shape after the break — is captured by
    ``dominant_losing=True`` at min 46-50 with positive xG divergence;
    the formation-change event was a proxy for "manager intent to
    change" and is not load-bearing.

    Substitute requirement: ``xg_vs_score_divergence`` must be ≥ 0.5 —
    a stronger version of A2's 0.3 threshold so we don't double-fire
    A10 on every dominant-trailing 2H frame. The 0.5 floor isolates the
    "favourite genuinely should be ahead" subset of the dominant-losing
    population.
    """
    if gsv.time.period != "2H" or not (46 <= gsv.time.minute <= 50):
        return None
    if not gsv.score.dominant_losing:
        return None
    if abs(gsv.xg.xg_vs_score_divergence) < 0.5:
        return None
    # Formation change by the dominant (losing) team, if available, is
    # additive confidence but not required for emission.
    dom_id = gsv.score.dominant_team_id
    has_formation_change = any(
        c.team_id == dom_id for c in gsv.roster.formation_changes
    )
    confidence_prior = 0.68 if has_formation_change else 0.60
    return _mk_thesis(
        arch=ThesisArchetype.SECOND_HALF_RESET,
        rule_id="A10",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="time.period", op="eq", value="2H"),
            GSVPredicate(path="time.minute", op="between", value=[46, 50]),
            GSVPredicate(path="score.dominant_losing", op="eq", value=True),
            GSVPredicate(path="xg.xg_vs_score_divergence", op="ge", value=0.5),
        ],
        chain=[
            CausalStep(
                cause="favourite_trails_at_ht_with_xg_pressure",
                effect="books_repriced_with_lag",
                mechanism="manual line review takes 5-10 min after restart",
            ),
            CausalStep(
                cause="favourite_pushes_in_2H",
                effect="next_goal_or_2H_goals_up",
                mechanism="information asymmetry vs book closing-line",
            ),
        ],
        family=MarketFamily.GOALS,
        direction="over",
        magnitude_pp=0.10,
        horizon_label="next_15",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="trailing scores again → reset killed",
            ),
            InvalidationTrigger(
                kind="minute_threshold_passed",
                description="window closes after minute 55",
                payload={"minute": 55},
            ),
        ],
        confidence_prior=confidence_prior,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 11 — Sustained 11v10 for the dominant team (≥30 min)
# ──────────────────────────────────────────────────────────────────────


def detect_numerical_sustained(gsv: GameStateVector) -> Thesis | None:
    if gsv.numerical.numerical_advantage <= 0:
        return None
    # Sportmonks emits red card minute; we approximate "sustained" as
    # "red card was at least 30 minutes ago". Without the actual red
    # minute on the GSV directly we use the numerical_advantage + minute
    # heuristic: minute - first_red_minute >= 30. The first-red-minute
    # is on LiveMatchState; we extract via a derived path if present.
    if gsv.time.minute < 30:
        return None
    if gsv.score.dominant_losing:
        return None
    leader_is_home = gsv.numerical.numerical_advantage > 0
    return _mk_thesis(
        arch=ThesisArchetype.NUMERICAL_SUSTAINED,
        rule_id="A11",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="numerical.numerical_advantage", op="gt", value=0),
            GSVPredicate(path="time.minute", op="ge", value=30),
            GSVPredicate(path="score.dominant_losing", op="eq", value=False),
        ],
        chain=[
            CausalStep(
                cause="sustained_numerical_advantage_for_leader",
                effect="corners_pile_up_nonlinear",
                mechanism="opponent tires + presses cheaper",
            ),
        ],
        family=MarketFamily.CORNERS,
        direction="over",
        magnitude_pp=0.07,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="goal_for_underdog",
                description="underdog scores → resets territorial dynamics",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="leader gets a red → numerics equalise",
            ),
        ],
        confidence_prior=0.65,
    )


# ──────────────────────────────────────────────────────────────────────
# Archetype 12 — Cruise mode (leader 1-0 at 80', controlling, low shots)
# ──────────────────────────────────────────────────────────────────────


def detect_cruise_mode(gsv: GameStateVector) -> Thesis | None:
    if abs(gsv.score.goal_diff) != 1 or gsv.time.minute < 80:
        return None
    leader_is_home = gsv.score.goal_diff == 1
    leader_phase = gsv.tactical.home_phase if leader_is_home else gsv.tactical.away_phase
    if leader_phase not in ("controlling", "parking_bus"):
        return None
    # Shot rate in last 10 min must be low (cruise = teams not pushing).
    recent_total = gsv.xg.xg_per_min_home_last_15 + gsv.xg.xg_per_min_away_last_15
    if recent_total > 0.05:
        return None
    return _mk_thesis(
        arch=ThesisArchetype.CRUISE_MODE,
        rule_id="A12",
        minute=gsv.time.minute,
        premise=[
            GSVPredicate(path="score.goal_diff", op="in", value=[1, -1]),
            GSVPredicate(path="time.minute", op="ge", value=80),
            GSVPredicate(path="tactical.home_phase" if leader_is_home else "tactical.away_phase",
                         op="in", value=["controlling", "parking_bus"]),
        ],
        chain=[
            CausalStep(
                cause="leader_controls_tempo_late",
                effect="match_likely_finishes_at_current_score",
                mechanism="trailing side fatigue + leader_clock_management",
            ),
        ],
        family=MarketFamily.GOALS,
        direction="under",
        magnitude_pp=0.06,
        horizon_label="rest_of_match",
        invalidations=[
            InvalidationTrigger(
                kind="any_goal",
                description="any goal kills the thesis",
            ),
            InvalidationTrigger(
                kind="red_card_dominant",
                description="leader red opens the game",
            ),
        ],
        confidence_prior=0.59,
    )


# ──────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────


_ARCHETYPE_DETECTORS: tuple[Callable[[GameStateVector], Thesis | None], ...] = (
    detect_red_card_away_early,
    detect_dominant_losing_napoli,
    detect_napoli_btts_companion,
    detect_late_cagey_zero_zero,
    detect_lead_two_defensive_sub,
    detect_regression_to_xg,
    detect_cards_momentum_strict_ref,
    detect_underdog_leads_siege,
    detect_open_game,
    detect_key_playmaker_off,
    detect_second_half_reset,
    # detect_numerical_sustained removed: net-negative both sample days
    # (D3: -5.79u wr=0.50 n=42; D4: -1.00u; MES anti-predictive on corners).
    # Function + __all__ export retained for import stability.
    detect_cruise_mode,
)


def generate_theses(gsv: GameStateVector) -> list[Thesis]:
    """Rule-layer generator. Calls every detector and returns the
    non-None set.

    Multiple archetypes may fire on the same GSV: e.g., an early red AND
    later numerical-sustained both have premises that can both be true
    at minute 65 of an 11v10 game. The Market Selector consumes the full
    list and picks the best ``thesis × market`` pair.
    """
    out: list[Thesis] = []
    for det in _ARCHETYPE_DETECTORS:
        thesis = det(gsv)
        if thesis is not None:
            out.append(thesis)
    return out


__all__ = [
    "detect_cards_momentum_strict_ref",
    "detect_cruise_mode",
    "detect_dominant_losing_napoli",
    "detect_key_playmaker_off",
    "detect_late_cagey_zero_zero",
    "detect_lead_two_defensive_sub",
    "detect_napoli_btts_companion",
    "detect_numerical_sustained",
    "detect_open_game",
    "detect_red_card_away_early",
    "detect_regression_to_xg",
    "detect_second_half_reset",
    "detect_underdog_leads_siege",
    "generate_theses",
]
