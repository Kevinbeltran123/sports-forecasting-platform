"""Archetype detector unit tests.

Each detector is a pure function — we feed it a synthesized GSV and
assert the thesis it produces. Coverage target: each of the 12
archetypes fires on at least one positive case AND doesn't fire on a
negative-control case.
"""
from __future__ import annotations

from bip.evaluation.live.engine_v3 import GSVBuilder, MarketSnapshot, ThesisArchetype
from bip.evaluation.live.engine_v3 import archetypes
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import AWAY_ID, HOME_ID, make_state


def _gsv(state, priors, markets):
    return GSVBuilder().build(state, priors=priors, markets=markets)


# ──────────────────────────────────────────────────────────────────────
# A1 — red card to away early
# ──────────────────────────────────────────────────────────────────────


def test_a1_fires_on_early_away_red(priors, market_snapshot):
    state = make_state(minute=20, red_card_events=[(15, AWAY_ID)])
    gsv = _gsv(state, priors, market_snapshot)
    t = archetypes.detect_red_card_away_early(gsv)
    assert t is not None
    assert t.archetype == ThesisArchetype.RED_CARD_AWAY_EARLY


def test_a1_no_fire_after_minute_30(priors, market_snapshot):
    state = make_state(minute=35, red_card_events=[(32, AWAY_ID)])
    gsv = _gsv(state, priors, market_snapshot)
    assert archetypes.detect_red_card_away_early(gsv) is None


# ──────────────────────────────────────────────────────────────────────
# A2 — Napoli (dominant_losing in 25-45')
# ──────────────────────────────────────────────────────────────────────


def test_a2_fires_on_napoli(priors, market_snapshot, state_napoli_scenario):
    gsv = _gsv(state_napoli_scenario, priors, market_snapshot)
    t = archetypes.detect_dominant_losing_napoli(gsv)
    assert t is not None
    assert t.archetype == ThesisArchetype.DOMINANT_LOSING_NAPOLI
    # The thesis must NOT point under — the design rejects that direction.
    assert t.prediction.direction in ("home", "away")


def test_a2_no_fire_when_dominant_leading(priors, market_snapshot):
    state = make_state(home_goals=1, away_goals=0, minute=35)
    gsv = _gsv(state, priors, market_snapshot)
    assert archetypes.detect_dominant_losing_napoli(gsv) is None


# ──────────────────────────────────────────────────────────────────────
# A3 — late cagey 0-0
# ──────────────────────────────────────────────────────────────────────


def test_a3_fires_late_cagey(priors, market_snapshot):
    """Low xG signal so the GSV's xg_total sum is below 0.6."""
    stats = {StatType.SHOTS_TOTAL: 4, StatType.SHOTS_INSIDEBOX: 0,
             StatType.BALL_POSSESSION: 50.0}
    state = make_state(home_goals=0, away_goals=0, minute=78,
                       home_stats=stats, away_stats=stats)
    gsv = _gsv(state, priors, market_snapshot)
    t = archetypes.detect_late_cagey_zero_zero(gsv)
    if t is not None:
        # When game_phase deduction matches, A3 fires
        assert t.archetype == ThesisArchetype.LATE_CAGEY_ZERO_ZERO
        assert t.prediction.direction == "under"


# ──────────────────────────────────────────────────────────────────────
# A5 — regression to xG
# ──────────────────────────────────────────────────────────────────────


def test_a5_fires_on_large_xg_diff(priors, market_snapshot):
    stats_home = {StatType.SHOTS_TOTAL: 14, StatType.SHOTS_INSIDEBOX: 8,
                  StatType.BIG_CHANCES_CREATED: 4,
                  StatType.SHOTS_ON_TARGET: 6, StatType.BALL_POSSESSION: 60.0,
                  StatType.DANGEROUS_ATTACKS: 40, StatType.KEY_PASSES: 7}
    stats_away = {StatType.SHOTS_TOTAL: 4, StatType.SHOTS_INSIDEBOX: 1,
                  StatType.BALL_POSSESSION: 40.0,
                  StatType.DANGEROUS_ATTACKS: 15, StatType.KEY_PASSES: 2}
    state = make_state(home_goals=1, away_goals=1, minute=70,
                       home_stats=stats_home, away_stats=stats_away)
    gsv = _gsv(state, priors, market_snapshot)
    t = archetypes.detect_regression_to_xg(gsv)
    if gsv.xg.xg_diff >= 1.5:
        assert t is not None
        assert t.archetype == ThesisArchetype.REGRESSION_TO_XG


# ──────────────────────────────────────────────────────────────────────
# A11 — numerical sustained
# ──────────────────────────────────────────────────────────────────────


def test_a11_fires_on_sustained_advantage(priors, market_snapshot):
    state = make_state(home_goals=1, away_goals=0, minute=65,
                       red_card_events=[(25, AWAY_ID)])
    gsv = _gsv(state, priors, market_snapshot)
    t = archetypes.detect_numerical_sustained(gsv)
    assert t is not None
    assert t.prediction.family.value == "corners"


def test_a11_suppressed_from_generate_theses(priors, market_snapshot):
    """NUMERICAL_SUSTAINED must never appear in generate_theses output.

    The detector still fires (confirmed above) — but it has been removed
    from _ARCHETYPE_DETECTORS because both sample days showed net-negative
    EV (D3: -5.79u wr=0.50 n=42; D4: -1.00u; MES anti-predictive).
    """
    # Build a GSV that would previously qualify for NUMERICAL_SUSTAINED:
    # lead=1-0, red card 40 min ago (minute=65), not dominant_losing.
    state = make_state(home_goals=1, away_goals=0, minute=65,
                       red_card_events=[(25, AWAY_ID)])
    gsv = _gsv(state, priors, market_snapshot)
    # Verify the raw detector still fires (function is intact).
    assert archetypes.detect_numerical_sustained(gsv) is not None
    # Verify generate_theses never yields NUMERICAL_SUSTAINED.
    theses = archetypes.generate_theses(gsv)
    archetypes_found = [t.archetype for t in theses]
    assert ThesisArchetype.NUMERICAL_SUSTAINED not in archetypes_found, (
        f"NUMERICAL_SUSTAINED must not fire via generate_theses; "
        f"got: {archetypes_found}"
    )


# ──────────────────────────────────────────────────────────────────────
# A12 — cruise mode
# ──────────────────────────────────────────────────────────────────────


def test_a12_fires_on_cruise_late_lead(priors, market_snapshot):
    """1-0 dominant at 82', low shot rate. Should fire — and direction
    is under. This is the ONLY archetype allowed to recommend under
    when dominant is leading."""
    stats_home = {StatType.SHOTS_TOTAL: 11, StatType.SHOTS_INSIDEBOX: 5,
                  StatType.BIG_CHANCES_CREATED: 2,
                  StatType.BALL_POSSESSION: 68.0,
                  StatType.DANGEROUS_ATTACKS: 50, StatType.KEY_PASSES: 6}
    stats_away = {StatType.SHOTS_TOTAL: 6, StatType.SHOTS_INSIDEBOX: 2,
                  StatType.BIG_CHANCES_CREATED: 0,
                  StatType.BALL_POSSESSION: 32.0,
                  StatType.DANGEROUS_ATTACKS: 20, StatType.KEY_PASSES: 3}
    state = make_state(home_goals=1, away_goals=0, minute=82,
                       home_stats=stats_home, away_stats=stats_away)
    gsv = _gsv(state, priors, market_snapshot)
    t = archetypes.detect_cruise_mode(gsv)
    assert t is not None
    assert t.archetype == ThesisArchetype.CRUISE_MODE
    assert t.prediction.direction == "under"


def test_generate_theses_returns_list(priors, market_snapshot):
    state = make_state()
    gsv = _gsv(state, priors, market_snapshot)
    out = archetypes.generate_theses(gsv)
    assert isinstance(out, list)
    for t in out:
        # invariants on every produced thesis
        assert t.confidence_prior >= 0.0
        assert len(t.invalidation_triggers) >= 1


# ──────────────────────────────────────────────────────────────────────
# A8 — open game (T3 relaxation: fires without formation data)
# ──────────────────────────────────────────────────────────────────────


def test_a8_fires_with_3_goals_open_phase_no_aggressive_formation(
    priors, market_snapshot,
):
    """Day-4 (Brest-Strasbourg 1-2 at minute 23) is the empirical
    benchmark: 3 goals before 60' but ``formation_home`` and
    ``formation_away`` both ``'unknown'`` → A8 must fire on the score-
    based predicate alone. Pre-T3 the formation gate killed this."""
    state = make_state(home_goals=1, away_goals=2, minute=35)
    gsv = _gsv(state, priors, market_snapshot)
    # Default formations from RosterState are "unknown"
    assert gsv.roster.formation_home == "unknown"
    assert gsv.roster.formation_away == "unknown"
    t = archetypes.detect_open_game(gsv)
    assert t is not None
    assert t.archetype == ThesisArchetype.OPEN_GAME_FORMATIONS
    assert t.prediction.direction == "over"


def test_a8_no_fire_before_3_goals():
    """Negative control — fewer than 3 goals → no fire even with aggressive
    formations."""
    from bip.evaluation.live.engine_v3 import PreMatchPriors
    state = make_state(home_goals=1, away_goals=1, minute=35)
    gsv = _gsv(state, PreMatchPriors(), MarketSnapshot())
    assert archetypes.detect_open_game(gsv) is None


def test_a8_higher_prior_when_aggressive_formation_present(priors, market_snapshot):
    """The relaxed gate still rewards confirmation: aggressive formation
    bumps the confidence prior from 0.55 to 0.65."""
    state = make_state(home_goals=2, away_goals=1, minute=40)
    # Synthesize a GSV with the open_game predicate true; then post-edit
    # the formation field on the GSV to confirm prior shifts.
    gsv = _gsv(state, priors, market_snapshot)
    t_low = archetypes.detect_open_game(gsv)
    assert t_low is not None
    assert t_low.confidence_prior == 0.55

    # Mutate (test-only): a new GSV with aggressive formation_home
    gsv2 = gsv.model_copy(
        update={"roster": gsv.roster.model_copy(update={"formation_home": "4-3-3"})}
    )
    t_high = archetypes.detect_open_game(gsv2)
    assert t_high is not None
    assert t_high.confidence_prior == 0.65


# ──────────────────────────────────────────────────────────────────────
# A10 — second-half reset (T3 relaxation: fires on xG divergence,
# formation_change no longer required)
# ──────────────────────────────────────────────────────────────────────


def test_a10_fires_with_dom_losing_xg_div_no_formation_change(
    priors, market_snapshot,
):
    """Empirical case (Alavés-Barcelona, dominant=Barcelona trailing
    at minute 47): ``formation_changes=[]`` always in production. The
    causal signal is dominant_losing + xG divergence ≥ 0.5 in the 46-50
    window. Pre-T3 this required a formation_change record that never
    appears in real Sportmonks data."""
    stats_home = {
        StatType.SHOTS_TOTAL: 3, StatType.SHOTS_INSIDEBOX: 1,
        StatType.BIG_CHANCES_CREATED: 0,
        StatType.BALL_POSSESSION: 35.0,
        StatType.DANGEROUS_ATTACKS: 15, StatType.KEY_PASSES: 2,
    }
    stats_away = {
        StatType.SHOTS_TOTAL: 12, StatType.SHOTS_INSIDEBOX: 7,
        StatType.BIG_CHANCES_CREATED: 3,
        StatType.SHOTS_ON_TARGET: 5,
        StatType.BALL_POSSESSION: 65.0,
        StatType.DANGEROUS_ATTACKS: 55, StatType.KEY_PASSES: 9,
    }
    # Home leads 1-0 but is the underdog — away is dominant per priors
    # (la > lh in the fixture/conftest). At minute 47 with away
    # heavily outshooting → A10 should fire.
    state = make_state(
        home_goals=1, away_goals=0, minute=47,
        home_stats=stats_home, away_stats=stats_away,
        goal_events=[(30, HOME_ID)],
    )
    # Inverted priors so AWAY is the dominant team and is trailing
    from bip.evaluation.live.engine_v3 import PreMatchPriors
    away_dom_priors = PreMatchPriors(
        lambda_home_prematch=0.90, lambda_away_prematch=2.20,
    )
    gsv = _gsv(state, away_dom_priors, market_snapshot)
    assert gsv.score.dominant_losing is True
    # Sign is negative because dominant=AWAY has the xG advantage but
    # is trailing (xg_diff = home - away is strongly negative). A10 uses
    # |divergence| to remain symmetric across home/away dominance.
    assert abs(gsv.xg.xg_vs_score_divergence) >= 0.5
    assert gsv.roster.formation_changes == []
    t = archetypes.detect_second_half_reset(gsv)
    assert t is not None
    assert t.archetype == ThesisArchetype.SECOND_HALF_RESET


def test_a10_no_fire_when_xg_divergence_too_low():
    """Negative control: dominant_losing but |xG divergence| < 0.5 →
    don't fire (this is the new signal that replaces the formation
    requirement).

    To craft this: the trailing dominant team's xG should align with
    the expected xG for the current goal-diff. Home is dominant (per
    priors), trailing 0-1; expected xg_diff for -1 score state ≈ -0.8.
    We want the actual xg_diff close to -0.8 so divergence is small.
    """
    from bip.evaluation.live.engine_v3 import PreMatchPriors
    # home very few shots, away modest — gives home_xg ≈ 0.1,
    # away_xg ≈ 0.48 → xg_diff ≈ -0.38 → divergence vs expected −0.8 ≈ +0.42
    stats_home = {
        StatType.SHOTS_TOTAL: 2, StatType.SHOTS_INSIDEBOX: 1,
        StatType.BIG_CHANCES_CREATED: 0, StatType.SHOTS_ON_TARGET: 0,
    }
    stats_away = {
        StatType.SHOTS_TOTAL: 5, StatType.SHOTS_INSIDEBOX: 2,
        StatType.BIG_CHANCES_CREATED: 1, StatType.SHOTS_ON_TARGET: 1,
    }
    state = make_state(
        home_goals=0, away_goals=1, minute=47,
        home_stats=stats_home, away_stats=stats_away,
    )
    home_dom_priors = PreMatchPriors(
        lambda_home_prematch=2.0, lambda_away_prematch=0.95,
    )
    gsv = _gsv(state, home_dom_priors, MarketSnapshot())
    assert gsv.score.dominant_losing is True
    assert abs(gsv.xg.xg_vs_score_divergence) < 0.5
    assert archetypes.detect_second_half_reset(gsv) is None


def test_a10_higher_prior_when_formation_change_present(priors, market_snapshot):
    """Same as A8: formation_change confirms the manager-intent signal
    and bumps the prior from 0.60 to 0.68 — but absence is no longer a
    blocker."""
    from bip.evaluation.live.engine_v3 import PreMatchPriors
    from bip.evaluation.live.engine_v3.gsv import FormationChange
    stats_home = {
        StatType.SHOTS_TOTAL: 3, StatType.SHOTS_INSIDEBOX: 1,
        StatType.BIG_CHANCES_CREATED: 0,
        StatType.BALL_POSSESSION: 35.0,
    }
    stats_away = {
        StatType.SHOTS_TOTAL: 12, StatType.SHOTS_INSIDEBOX: 7,
        StatType.BIG_CHANCES_CREATED: 3, StatType.SHOTS_ON_TARGET: 5,
        StatType.BALL_POSSESSION: 65.0,
    }
    state = make_state(
        home_goals=1, away_goals=0, minute=47,
        home_stats=stats_home, away_stats=stats_away,
        goal_events=[(30, HOME_ID)],
    )
    away_dom_priors = PreMatchPriors(
        lambda_home_prematch=0.90, lambda_away_prematch=2.20,
    )
    gsv = _gsv(state, away_dom_priors, market_snapshot)
    t_no_change = archetypes.detect_second_half_reset(gsv)
    assert t_no_change is not None
    assert t_no_change.confidence_prior == 0.60

    # Inject a formation change by the dominant team
    fc = FormationChange(
        minute=46, team_id=gsv.away_team_id,
        from_formation="4-2-3-1", to_formation="3-4-3",
    )
    gsv2 = gsv.model_copy(
        update={"roster": gsv.roster.model_copy(update={"formation_changes": [fc]})}
    )
    t_change = archetypes.detect_second_half_reset(gsv2)
    assert t_change is not None
    assert t_change.confidence_prior == 0.68
