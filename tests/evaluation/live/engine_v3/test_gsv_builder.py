"""GSV builder smoke + derived-signal correctness tests.

The critical contract: the load-bearing derived fields
(``dominant_losing``, ``xg_vs_score_divergence``, ``tactical.home_phase``)
must be set correctly. If the builder gets these wrong, every
downstream archetype detector misfires.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bip.evaluation.live.engine_v3 import (
    GSVBuilder,
    MarketLine,
    MarketSnapshot,
    PreMatchPriors,
)
from bip.sports.football.sportmonks.types import StatType
from tests.evaluation.live.engine_v3.conftest import AWAY_ID, HOME_ID, make_state


def test_gsv_builder_smoke(priors, market_snapshot):
    state = make_state()
    out = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    assert out.fixture_id == state.fixture_id
    assert out.state_version == 1
    assert out.home_team_id == state.home_team_id
    # state_version monotonically increases per-fixture
    out2 = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    assert out2.state_version == 1  # different builder


def test_gsv_builder_versions_increment_per_call():
    builder = GSVBuilder()
    state = make_state()
    priors_arg = builder.build(state, priors=__import__(
        "bip.evaluation.live.engine_v3", fromlist=["PreMatchPriors"]
    ).PreMatchPriors(), markets=MarketSnapshot())
    second = builder.build(state, priors=priors_arg.priors, markets=MarketSnapshot())
    assert second.state_version == priors_arg.state_version + 1


def test_dominant_losing_true_when_favourite_trails(priors, market_snapshot, state_napoli_scenario):
    """Home is dominant per priors (λ_home > λ_away). Home is trailing 0-1.
    Therefore dominant_losing must be True — the load-bearing predicate
    for the no-bet rule #2."""
    out = GSVBuilder().build(
        state_napoli_scenario,
        priors=priors,
        markets=market_snapshot,
    )
    assert out.score.dominant_team_id == HOME_ID
    assert out.score.dominant_losing is True
    assert out.is_dominant_losing is True


def test_xg_vs_score_divergence_positive_for_dominant_underdog(priors, market_snapshot, state_napoli_scenario):
    """At Napoli scenario: home outshoots heavily but trails. Expected
    xg_diff_for_score(-1) ≈ -0.8 → actual xg_diff is strongly positive
    → divergence very positive."""
    out = GSVBuilder().build(
        state_napoli_scenario,
        priors=priors,
        markets=market_snapshot,
    )
    # Home has 3 big_chances + 6 shots_in_box → live xG well over 1.0;
    # away has 0 big_chances + 1 sib → xG near 0.1
    assert out.xg.home_xg_total > out.xg.away_xg_total
    assert out.xg.xg_vs_score_divergence > 0.5


def test_tactical_phase_chasing_when_losing_with_pressure(priors, market_snapshot, state_napoli_scenario):
    """A dominant-losing team that's pressing hard (lots of dangerous
    attacks + key passes) should be tagged ``chasing`` or ``pressing``.
    Either qualifies — the v3 doesn't distinguish them strictly here."""
    out = GSVBuilder().build(
        state_napoli_scenario,
        priors=priors,
        markets=market_snapshot,
    )
    assert out.tactical.home_phase in ("chasing", "pressing")


def test_numerical_advantage_after_red_card(priors, market_snapshot):
    state = make_state(red_card_events=[(15, 200)])  # AWAY_ID red
    out = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    assert out.numerical.numerical_advantage == 1
    assert out.numerical.red_cards_away == 1


def test_cruise_mode_phase_late_lead(priors, market_snapshot):
    """Leader at minute 82 with 1-0 and modest shot rate should be
    tagged ``parking_bus`` or ``controlling``, enabling cruise_mode."""
    stats_home = {StatType.SHOTS_TOTAL: 10, StatType.BALL_POSSESSION: 68.0,
                  StatType.SHOTS_INSIDEBOX: 5}
    stats_away = {StatType.SHOTS_TOTAL: 8, StatType.BALL_POSSESSION: 32.0,
                  StatType.SHOTS_INSIDEBOX: 2}
    state = make_state(home_goals=1, away_goals=0, minute=82,
                       home_stats=stats_home, away_stats=stats_away)
    out = GSVBuilder().build(state, priors=priors, markets=market_snapshot)
    assert out.tactical.home_phase in ("parking_bus", "controlling")
    assert out.score.goal_diff == 1


# ── Dominant-team-id resolution (T2 fix v2: team-named markets) ─────────


def _markets_with_team_named_bttsxresult(
    *,
    home_team: str,
    away_team: str,
    home_yes_decimal: float,
    home_no_decimal: float,
    away_yes_decimal: float,
    away_no_decimal: float,
) -> MarketSnapshot:
    """Synthesize the Sportmonks ``result___both_teams_to_score_<team>_/_<yes|no>``
    lines that the dominant-team resolver consults. Six outcomes per
    fixture but we only need the four team-specific ones."""
    now = datetime.now(timezone.utc)
    norm_home = home_team.lower().replace(" ", "_")
    norm_away = away_team.lower().replace(" ", "_")
    return MarketSnapshot(
        lines={
            f"result___both_teams_to_score_{norm_home}_/_yes": MarketLine(
                market_id=f"result___both_teams_to_score_{norm_home}_/_yes",
                side_a_decimal=home_yes_decimal, max_stake_cap=200.0,
                last_update_utc=now,
            ),
            f"result___both_teams_to_score_{norm_home}_/_no": MarketLine(
                market_id=f"result___both_teams_to_score_{norm_home}_/_no",
                side_a_decimal=home_no_decimal, max_stake_cap=200.0,
                last_update_utc=now,
            ),
            f"result___both_teams_to_score_{norm_away}_/_yes": MarketLine(
                market_id=f"result___both_teams_to_score_{norm_away}_/_yes",
                side_a_decimal=away_yes_decimal, max_stake_cap=200.0,
                last_update_utc=now,
            ),
            f"result___both_teams_to_score_{norm_away}_/_no": MarketLine(
                market_id=f"result___both_teams_to_score_{norm_away}_/_no",
                side_a_decimal=away_no_decimal, max_stake_cap=200.0,
                last_update_utc=now,
            ),
        }
    )


def _state_with_names(home_name: str, away_name: str):
    """Override the conftest default Home FC / Away FC names so the
    team-named market matcher has something to bind to."""
    from dataclasses import replace
    return replace(make_state(), home_team_name=home_name, away_team_name=away_name)


def test_dominant_palace_city_market_says_city_overrides_coin_flip_priors():
    """Day-4 Palace vs Man City empirical case — updated for Wave-3 ML-λ tier.

    Wave-3 adds an ML-λ tier (Dixon-Coles) ABOVE the bookmaker market.
    When the λ-gap is DECISIVE (|lh - la| >= 0.15), ML-λ overrides market
    as the more reliable signal. The original Sportmonks λ_h=1.64/λ_a=0.86
    gap (0.78) would now invoke the ML-λ tier.

    This test is updated to use a COIN-FLIP λ (gap=0.08 < 0.15) to verify
    that the bookmaker market (City strongly favoured) still wins when the
    ML-λ signal is inconclusive. The empirical insight that the MARKET is
    sharper than SPORTMONKS λ in ambiguous cases is preserved — it is now
    the fallback when ML-λ defers.

    Historical note: the original test used λ_h=1.64/λ_a=0.86 to represent
    Sportmonks' prediction (which was wrong — City is clearly stronger).
    A properly fitted Dixon-Coles ML-λ would show City as stronger and
    would correctly invoke the ML-λ tier without relying on market. The
    coin-flip scenario here tests the market-fallback tier specifically.
    """
    priors = PreMatchPriors(
        lambda_home_prematch=1.25, lambda_away_prematch=1.17,  # gap=0.08 < 0.15 → coin-flip
    )
    state = _state_with_names("Crystal Palace", "Man City")
    # Real Day-4 odds: Palace_yes=13.0 Palace_no=17.0 → P=13.6%
    #                  City_yes=2.62  City_no=2.37  → P=80.4%
    markets = _markets_with_team_named_bttsxresult(
        home_team="Crystal Palace", away_team="Man City",
        home_yes_decimal=13.0, home_no_decimal=17.0,
        away_yes_decimal=2.62, away_no_decimal=2.37,
    )
    out = GSVBuilder().build(state, priors=priors, markets=markets)
    assert out.score.dominant_team_id == AWAY_ID  # City wins via market tier


def test_dominant_priors_used_when_no_team_named_markets():
    """Bookmaker signal absent → fall back to λ. This covers fixtures
    where the BTTS-x-result odds set hasn't been emitted yet."""
    priors = PreMatchPriors(
        lambda_home_prematch=1.80, lambda_away_prematch=0.90,
    )
    state = make_state()
    out = GSVBuilder().build(state, priors=priors, markets=MarketSnapshot())
    assert out.score.dominant_team_id == HOME_ID


def test_dominant_priors_used_when_market_is_coin_flip():
    """Gap in implied P(win) < 4 pp → market gives no signal."""
    priors = PreMatchPriors(
        lambda_home_prematch=1.35, lambda_away_prematch=1.10,
    )
    state = _state_with_names("Charlotte", "New York City")
    # Synthesize even probabilities (~42% each)
    markets = _markets_with_team_named_bttsxresult(
        home_team="Charlotte", away_team="New York City",
        home_yes_decimal=4.0, home_no_decimal=4.5,
        away_yes_decimal=4.0, away_no_decimal=4.5,
    )
    out = GSVBuilder().build(state, priors=priors, markets=markets)
    # Market coin-flip → λ wins → home dominant (lh > la)
    assert out.score.dominant_team_id == HOME_ID


def test_dominant_market_picks_home_when_lambda_coin_flip():
    """Market picks home when ML-λ is inconclusive (coin-flip gap).

    Wave-3 ML-λ tier: when |lh - la| < 0.15, λ is a coin-flip and the
    bookmaker market is consulted. Here Olympiacos is heavily favoured
    by the market (P≈74%), and the λ-gap is small (0.10 < 0.15) → home
    wins via market tier (tier 3).

    Updated from the pre-Wave-3 test that used λ_h=1.00/λ_a=1.50 (gap=0.50
    which now triggers the ML-λ tier picking away). The new scenario uses
    near-equal λ to isolate the market-fallback path.
    """
    priors = PreMatchPriors(
        lambda_home_prematch=1.25, lambda_away_prematch=1.15,  # gap=0.10 < 0.15 → coin-flip
    )
    state = _state_with_names("Olympiacos", "Panathinaikos")
    # Olympiacos heavy fav: P ≈ 74%; Pana P ≈ 14%
    markets = _markets_with_team_named_bttsxresult(
        home_team="Olympiacos", away_team="Panathinaikos",
        home_yes_decimal=2.7, home_no_decimal=2.7,
        away_yes_decimal=14.0, away_no_decimal=14.0,
    )
    out = GSVBuilder().build(state, priors=priors, markets=markets)
    assert out.score.dominant_team_id == HOME_ID


def test_dominant_uses_strong_elo_diff_when_available():
    """elo_diff ≥ 25 trumps both market and priors.

    Production observed value: always 0.0 today. Kept as a forward-
    compatibility hook for when the feed is wired up.
    """
    priors = PreMatchPriors(
        lambda_home_prematch=1.10, lambda_away_prematch=1.50,
        elo_diff=120.0,  # home strongly favoured by elo
    )
    state = _state_with_names("Strong Home", "Weaker Away")
    # Market also says away; elo overrides both
    markets = _markets_with_team_named_bttsxresult(
        home_team="Strong Home", away_team="Weaker Away",
        home_yes_decimal=10.0, home_no_decimal=10.0,
        away_yes_decimal=2.5, away_no_decimal=2.5,
    )
    out = GSVBuilder().build(state, priors=priors, markets=markets)
    assert out.score.dominant_team_id == HOME_ID


def test_dominant_explicit_arg_still_overrides_everything():
    """Caller-provided ``dominant_team_id`` short-circuits the heuristic
    entirely — the anti-Napoli regression suite relies on this."""
    priors = PreMatchPriors(
        lambda_home_prematch=1.80, lambda_away_prematch=0.90,
    )
    state = make_state()
    out = GSVBuilder().build(
        state, priors=priors, markets=MarketSnapshot(),
        dominant_team_id=AWAY_ID,
    )
    assert out.score.dominant_team_id == AWAY_ID


def test_numeric_fulltime_result_lines_are_ignored():
    """Day-4 evidence: ``fulltime_result_1/2`` is unreliable (8/31
    fixtures disagree with team-named markets). The resolver MUST NOT
    consult those lines. We verify by giving inverted numeric labels
    and confirming the resolver still picks the team-named favourite.

    Wave-3 update: λ-gap must be coin-flip (< 0.15) so the ML-λ tier
    defers to market, allowing us to test the numeric-line-ignore behavior
    of the market tier specifically. Original priors λ_h=1.64/λ_a=0.86
    now trigger ML-λ tier — updated to near-equal λ.
    """
    priors = PreMatchPriors(
        lambda_home_prematch=1.25, lambda_away_prematch=1.17,  # gap=0.08 < 0.15 → coin-flip
    )
    state = _state_with_names("Crystal Palace", "Man City")
    markets = _markets_with_team_named_bttsxresult(
        home_team="Crystal Palace", away_team="Man City",
        home_yes_decimal=13.0, home_no_decimal=17.0,
        away_yes_decimal=2.62, away_no_decimal=2.37,
    )
    # Inject a misleading numeric line that says home is heavy fav
    now = datetime.now(timezone.utc)
    markets.lines["fulltime_result_1"] = MarketLine(
        market_id="fulltime_result_1", side_a_decimal=1.30,
        max_stake_cap=100.0, last_update_utc=now,
    )
    markets.lines["fulltime_result_2"] = MarketLine(
        market_id="fulltime_result_2", side_a_decimal=8.50,
        max_stake_cap=100.0, last_update_utc=now,
    )
    out = GSVBuilder().build(state, priors=priors, markets=markets)
    assert out.score.dominant_team_id == AWAY_ID  # team-named wins
