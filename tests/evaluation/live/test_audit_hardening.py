"""Regression tests for the 2026-05-09 audit-hardening pass.

These tests lock in the kills for §3 clusters A-E and validate the new
signals from §B. Each test names the cluster or signal it covers in its
docstring so future refactors can trace what they're protecting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import (
    MARKET_AWAY_CLEAN_SHEET,
    MARKET_BTTS,
    MARKET_BTTS_SECOND_HALF,
    MARKET_CORNERS_TOTAL_8_5,
    MARKET_CORNERS_TOTAL_9_5,
    MARKET_CORNERS_TOTAL_10_5,
    MARKET_DOUBLE_CHANCE,
    MARKET_DRAW_NO_BET,
    MARKET_FULLTIME_RESULT,
    MARKET_HOME_CLEAN_SHEET,
    MARKET_OU_05,
    MARKET_OU_15,
    MARKET_OU_25,
    MARKET_OU_35,
    MARKET_TEAM_TO_SCORE_FIRST,
    LiveMatchPredictor,
    MarketProbabilities,
)
from bip.evaluation.live.value_detector import (
    DEFAULT_INFO_DENSITY_FLOOR,
    LivePick,
    ValueDetector,
)
from bip.sports.football.sportmonks.schemas import Odd
from bip.sports.football.sportmonks.types import (
    MarketID,
    PredictionType,
    StatType,
)


# ── Shared builders ─────────────────────────────────────────────────────────


_BALANCED_PRE_MATCH = {
    PredictionType.FULLTIME_RESULT_PROBABILITY: {"home": 40, "draw": 30, "away": 30},
    PredictionType.OVER_UNDER_2_5_PROBABILITY: {"yes": 55, "no": 45},
    PredictionType.OVER_UNDER_1_5_PROBABILITY: {"yes": 78, "no": 22},
    PredictionType.OVER_UNDER_3_5_PROBABILITY: {"yes": 25, "no": 75},
    PredictionType.BTTS_PROBABILITY: {"yes": 60, "no": 40},
    PredictionType.HOME_OVER_UNDER_0_5_PROBABILITY: {"yes": 75, "no": 25},
    PredictionType.AWAY_OVER_UNDER_0_5_PROBABILITY: {"yes": 70, "no": 30},
    PredictionType.HOME_OVER_UNDER_1_5_PROBABILITY: {"yes": 40, "no": 60},
    PredictionType.AWAY_OVER_UNDER_1_5_PROBABILITY: {"yes": 35, "no": 65},
    PredictionType.TEAM_TO_SCORE_FIRST_PROBABILITY: {
        "home": 45, "draw": 15, "away": 40,
    },
    PredictionType.DOUBLE_CHANCE_PROBABILITY: {
        "draw_home": 70, "draw_away": 60, "home_away": 70,
    },
}


def _make_state(
    *,
    minute: int = 30,
    home_goals: int = 0,
    away_goals: int = 0,
    is_live: bool = True,
    home_pressure: list[float] | None = None,
    away_pressure: list[float] | None = None,
    home_stats: dict[int, float] | None = None,
    away_stats: dict[int, float] | None = None,
    sportmonks: dict[int, dict] | None = None,
    goal_events: list[tuple[int, int]] | None = None,
    red_card_events: list[tuple[int, int]] | None = None,
    yellow_card_events: list[tuple[int, int, int]] | None = None,
    substitution_events: list[tuple[int, int, int | None]] | None = None,
    snapshot_taken_at: datetime | None = None,
) -> LiveMatchState:
    return LiveMatchState(
        fixture_id=1,
        home_team_id=10,
        away_team_id=20,
        home_team_name="Home",
        away_team_name="Away",
        home_goals=home_goals,
        away_goals=away_goals,
        minute=minute,
        period_id=2 if minute >= 45 else 1,
        is_live=is_live,
        is_half_time=False,
        is_finished=False,
        home_stats=home_stats or {},
        away_stats=away_stats or {},
        home_pressure_recent=home_pressure or [],
        away_pressure_recent=away_pressure or [],
        sportmonks_predictions=sportmonks or _BALANCED_PRE_MATCH,
        goal_events=goal_events or [],
        red_card_events=red_card_events or [],
        yellow_card_events=yellow_card_events or [],
        substitution_events=substitution_events or [],
        snapshot_taken_at=snapshot_taken_at,
    )


def _odd(
    *,
    market_id: int, label: str, value: str,
    suspended: bool = False, stopped: bool = False,
    total: str | None = None,
    bookmaker_id: int = 2, fixture_id: int = 1,
    latest_bookmaker_update: datetime | None = None,
) -> Odd:
    body = {
        "id": id((market_id, label, value, total)),
        "fixture_id": fixture_id,
        "market_id": market_id,
        "bookmaker_id": bookmaker_id,
        "label": label,
        "value": value,
        "suspended": suspended,
        "stopped": stopped,
        "total": total,
    }
    if latest_bookmaker_update is not None:
        body["latest_bookmaker_update"] = latest_bookmaker_update.isoformat()
    return Odd.model_validate(body)


def _probs_with_conf(
    *,
    market_probs: dict[str, dict[str, float]],
    confidences: dict[str, float] | None = None,
    fixture_id: int = 1,
    minute: int = 30,
    snapshot_kind: str = "live",
) -> MarketProbabilities:
    return MarketProbabilities(
        fixture_id=fixture_id,
        minute=minute,
        snapshot_kind=snapshot_kind,
        by_market=market_probs,
        sources={k: "test" for k in market_probs},
        confidences=confidences or {},
    )


# ── §3 Cluster C — informational-density gate ──────────────────────────────


class TestClusterCInfoDensity:
    """Cluster C: at min 8 with no live state, model must NOT diverge from
    Sportmonks prior. Informational-density gate kills the burst."""

    def test_min_8_no_data_returns_sportmonks_prior(self):
        state = _make_state(minute=8, is_live=True)
        # No pressure, no stats → density well below 0.40
        assert state.informational_density < 0.40
        ft = LiveMatchPredictor().predict(state).by_market[MARKET_FULLTIME_RESULT]
        # Should match Sportmonks prior verbatim (40/30/30)
        assert ft["home"] == pytest.approx(0.40, abs=1e-3)
        assert ft["draw"] == pytest.approx(0.30, abs=1e-3)
        assert ft["away"] == pytest.approx(0.30, abs=1e-3)

    def test_density_lifted_after_first_goal_but_minute_floored(self):
        """Bug #7 fix: previously any goal forced density=1.0 regardless of
        minute. Now event_term only saturates the data axis; the minute
        floor still gates. At minute 15 with one goal: minute_term=0.6,
        event_term=1.0 → density = 0.6 × (0.4 + 0.6 × 1.0) = 0.6.
        """
        state = _make_state(minute=15, home_goals=1)
        density = state.informational_density
        assert density == pytest.approx(0.60, abs=0.01)
        # Above the 0.20 hard floor → live-adjustments engage. Above the
        # 0.40 predictor threshold → live mode active.
        assert density > DEFAULT_INFO_DENSITY_FLOOR

    def test_density_lifted_with_red_card_but_minute_floored(self):
        """Bug #7 fix: same for red cards. At minute 15: density = 0.6,
        not 1.0. Means a red at min 5 (minute_term=0.20) caps density at
        0.20 — stays at the floor until minute accrues."""
        state = _make_state(
            minute=15, home_goals=0, away_goals=0,
            red_card_events=[(14, 10)],
        )
        density = state.informational_density
        assert density == pytest.approx(0.60, abs=0.01)

    def test_density_minute_5_red_card_still_below_floor(self):
        """Critical regression: a red card at min 5 must NOT bypass the
        info-density floor. Old code returned 1.0; new code returns 0.20
        which is AT the floor → live emission still gated.
        """
        state = _make_state(
            minute=5, home_goals=0, away_goals=0,
            red_card_events=[(4, 10)],
        )
        density = state.informational_density
        # minute_term=0.20, event_term=1.0 → density=0.20 (== floor)
        assert density == pytest.approx(0.20, abs=0.01)

    def test_value_detector_drops_picks_below_density_floor(self):
        # state with density very low (min=5, no data, no goals)
        state = _make_state(minute=5, is_live=True)
        assert state.informational_density < DEFAULT_INFO_DENSITY_FLOOR
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds,
            home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []


# ── §3 Cluster A — bookmaker coherence ─────────────────────────────────────


class TestClusterABookmakerCoherence:
    """Cluster A: DC X2 == draw price → bookmaker is internally inconsistent.
    Detector must skip those markets entirely."""

    def test_dc_x2_equals_draw_skips_both_markets(self):
        state = _make_state(minute=56, home_goals=1, away_goals=2)  # density=1 (goals)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.10, "draw": 0.53, "away": 0.37},
            MARKET_DOUBLE_CHANCE: {"1x": 0.63, "x2": 0.90, "12": 0.47},
        })
        # Bookmaker offers draw @ 4.50 (implied 0.222) and X2 @ 4.50 (0.222)
        # → X2 implied < draw implied is impossible (X2 mass ⊇ draw mass).
        # Even putting them equal trips the cluster-A near-equality flag.
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="9.00"),
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Draw", value="4.50"),
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Away", value="2.50"),
            _odd(market_id=MarketID.DOUBLE_CHANCE, label="1X", value="3.00"),
            _odd(market_id=MarketID.DOUBLE_CHANCE, label="X2", value="4.50"),
            _odd(market_id=MarketID.DOUBLE_CHANCE, label="12", value="2.00"),
        ]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds,
            home_team_name="Lanús", away_team_name="Argentinos", state=state,
        )
        # All FTR + DC picks must be suppressed by the coherence gate
        emitted_markets = {p.market for p in picks}
        assert MARKET_FULLTIME_RESULT not in emitted_markets
        assert MARKET_DOUBLE_CHANCE not in emitted_markets

    def test_ou_monotonicity_violation_skips_all_overs(self):
        # P(over 0.5) < P(over 1.5) is mathematically impossible
        state = _make_state(minute=40, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_OU_15: {"over": 0.85, "under": 0.15},
            MARKET_OU_25: {"over": 0.55, "under": 0.45},
        })
        odds = [
            _odd(market_id=MarketID.MATCH_GOALS, label="Over", value="3.00", total="0.5"),
            _odd(market_id=MarketID.MATCH_GOALS, label="Under", value="1.40", total="0.5"),
            _odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.30", total="1.5"),
            _odd(market_id=MarketID.MATCH_GOALS, label="Under", value="3.50", total="1.5"),
        ]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert all(p.market not in {"ou_0_5", "ou_1_5", "ou_2_5", "ou_3_5"}
                   for p in picks)


# ── §3 Cluster B — extreme-edge hard drop + valuebet gate ──────────────────


class TestClusterBExtremeEdge:
    """Cluster B: P=0.976 @ 7.00 → +582% EV. Default drop_extreme=True
    must HARD-DROP (not just flag) such picks."""

    def test_extreme_edge_dropped_by_default(self):
        state = _make_state(minute=61, home_goals=2, away_goals=0)  # density=1
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.976, "draw": 0.020, "away": 0.004},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="7.00")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="Nashville", away_team_name="DC United",
            state=state,
        )
        assert picks == []

    def test_extreme_edge_kept_when_drop_extreme_disabled(self):
        state = _make_state(minute=61, home_goals=2, away_goals=0)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.976, "draw": 0.020, "away": 0.004},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="7.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False,
            min_logical_score_flag=0.0,  # no logical-score floor
            min_logical_score_emit=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Should appear (flagged) when explicit override
        assert len(picks) == 1
        assert picks[0].flagged_reason and (
            picks[0].flagged_reason.startswith("edge_above_sanity_cap")
            or picks[0].flagged_reason.startswith("extreme_edge_no_sm_confirmation")
        )

    def test_valuebet_disagreement_drops_pick(self):
        sm = dict(_BALANCED_PRE_MATCH)
        sm[PredictionType.VALUEBET] = {
            "bet": "draw", "is_value": True, "fair_odd": 3.0, "odd": 3.5,
        }
        state = _make_state(minute=40, home_goals=1, sportmonks=sm)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Sportmonks valuebet flagged "draw" — picking "home" conflicts → drop.
        assert picks == []


# ── §3 Cluster D — credibility-interval gate ───────────────────────────────


class TestClusterDConfidenceInterval:
    """Cluster D: +6.14% EV on a 1.20 odd is within model noise. The CI
    gate demands edge ≥ 2 × CI half-width × odd."""

    def test_low_edge_below_ci_threshold_dropped(self):
        # P=0.884 @ 1.20 → EV=6.08%. CI=0.04 → required = 200×0.04×1.20 = 9.6%.
        state = _make_state(minute=8, home_goals=1)  # density=1 (goal)
        probs = _probs_with_conf(
            market_probs={
                MARKET_FULLTIME_RESULT: {"home": 0.884, "draw": 0.080, "away": 0.036},
            },
            confidences={MARKET_FULLTIME_RESULT: 0.04},
        )
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.20")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="Portland", away_team_name="SKC",
            state=state,
        )
        assert picks == []

    def test_high_edge_above_ci_threshold_passes(self):
        state = _make_state(minute=30, home_goals=1)
        probs = _probs_with_conf(
            market_probs={
                MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
            },
            confidences={MARKET_FULLTIME_RESULT: 0.04},
        )
        # EV at odd=2.00 = 0.55 × 2 - 1 = 10%. Required = 200×0.04×2 = 16%.
        # So 10% should DROP. Use 2.50 instead → EV = 37.5% > 20% threshold.
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.50")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        assert picks[0].edge_pct == pytest.approx(37.5, abs=0.5)


# ── §3 Cluster E — bundle deduplication ────────────────────────────────────


class TestClusterEBundleDedup:
    """Cluster E: btts/no + home_ou_1_5/over fire on the same fixture as
    correlated marginals of one underlying score state. Bundle dedup must
    pick the single highest-information one per cluster."""

    def test_correlated_picks_collapse_within_cluster(self):
        state = _make_state(minute=35, home_goals=1)  # density=1
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
            MARKET_DOUBLE_CHANCE: {"1x": 0.80, "x2": 0.45, "12": 0.75},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.20"),
            _odd(market_id=MarketID.DOUBLE_CHANCE, label="1X", value="1.40"),
        ]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Both in 1X2 cluster → exactly one survives
        cluster_picks = [p for p in picks if p.market in
                         {MARKET_FULLTIME_RESULT, MARKET_DOUBLE_CHANCE}]
        assert len(cluster_picks) == 1

    def test_picks_in_different_clusters_both_survive(self):
        state = _make_state(minute=35, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
            MARKET_BTTS: {"yes": 0.65, "no": 0.35},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.20"),
            _odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="Yes", value="1.80"),
        ]
        # Opt out of Day-1 audit gates: this test exercises cluster-based
        # bundle dedup, not the post-audit cascade rules.
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
            market_blacklist=frozenset(),
            ban_positive_side_binaries=False,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # 1X2 cluster + BTTS cluster → both survive
        assert len(picks) == 2


# ── B-1 corners market ────────────────────────────────────────────────────


class TestCornersMarket:
    def test_corners_under_when_low_pace(self):
        state = _make_state(
            minute=70,
            home_stats={StatType.CORNERS: 2}, away_stats={StatType.CORNERS: 1},
        )
        probs = LiveMatchPredictor().predict(state).by_market.get(
            MARKET_CORNERS_TOTAL_8_5
        )
        assert probs is not None
        # 3 corners by min 70 → very unlikely to reach 8.5
        assert probs["under"] > 0.80

    def test_corners_over_when_already_above_line(self):
        state = _make_state(
            minute=70,
            home_stats={StatType.CORNERS: 7}, away_stats={StatType.CORNERS: 5},
        )
        probs = LiveMatchPredictor().predict(state).by_market.get(
            MARKET_CORNERS_TOTAL_8_5
        )
        assert probs is not None
        assert probs["over"] == 1.0

    def test_no_corners_emission_below_min10(self):
        state = _make_state(minute=8, home_stats={StatType.CORNERS: 1})
        probs = LiveMatchPredictor().predict(state)
        assert MARKET_CORNERS_TOTAL_8_5 not in probs.by_market


# ── B-15 / B-16 cross-checks (already exercised in cluster B/E above,
# add an isolated assertion for the score-grid disagreement gate) ─────────


class TestCorrectScoreGridDisagreement:
    def test_grid_disagreement_flags_pick(self):
        sm = dict(_BALANCED_PRE_MATCH)
        # Sportmonks score grid says draw is improbable (~10%); we say 60%
        scores = {f"{h}-{a}": (15.0 if h != a else 1.0)
                  for h in range(4) for a in range(4)}
        scores["other"] = 5.0
        sm[PredictionType.CORRECT_SCORE_PROBABILITY] = {"scores": scores}
        state = _make_state(minute=40, home_goals=0, sportmonks=sm)
        # Force info-density override via 1+ goal? state has 0 goals — use red card.
        state = _make_state(
            minute=40, sportmonks=sm,
            red_card_events=[(35, 10)],
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.20, "draw": 0.60, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Draw", value="2.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False,
            min_logical_score_flag=0.0, min_logical_score_emit=0.0,
            enforce_ci_gate=False,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # The pick should be present BUT flagged by the score-grid mismatch
        assert len(picks) == 1
        assert picks[0].flagged_reason is not None
        assert picks[0].flagged_reason.startswith(("correct_score_disagrees",
                                                    "red_card_inflates_draw"))


# ── New markets sanity ────────────────────────────────────────────────────


class TestNewMarketsEmitted:
    def test_draw_no_bet_emitted_with_balanced_state(self):
        state = _make_state(minute=30, home_goals=1)
        probs = LiveMatchPredictor().predict(state)
        dnb = probs.by_market.get(MARKET_DRAW_NO_BET)
        assert dnb is not None
        assert dnb["home"] + dnb["away"] == pytest.approx(1.0, abs=1e-6)

    def test_btts_2h_emitted(self):
        state = _make_state(minute=30, home_goals=1)
        probs = LiveMatchPredictor().predict(state)
        b2h = probs.by_market.get(MARKET_BTTS_SECOND_HALF)
        assert b2h is not None
        assert b2h["yes"] + b2h["no"] == pytest.approx(1.0, abs=1e-6)

    def test_team_clean_sheet_emitted(self):
        state = _make_state(minute=30, home_goals=1)
        probs = LiveMatchPredictor().predict(state)
        assert MARKET_HOME_CLEAN_SHEET in probs.by_market
        assert MARKET_AWAY_CLEAN_SHEET in probs.by_market

    def test_team_to_score_first_only_when_no_goals(self):
        scoreless = LiveMatchPredictor().predict(_make_state(minute=20))
        scored = LiveMatchPredictor().predict(_make_state(minute=20, home_goals=1))
        assert MARKET_TEAM_TO_SCORE_FIRST in scoreless.by_market
        assert MARKET_TEAM_TO_SCORE_FIRST not in scored.by_market

    def test_confidences_emitted_for_all_markets(self):
        state = _make_state(minute=30, home_goals=1)
        probs = LiveMatchPredictor().predict(state)
        # Every emitted market should have a confidence entry
        missing = set(probs.by_market) - set(probs.confidences)
        assert missing == set()


# ── B-3 killing-the-clock ─────────────────────────────────────────────────


class TestKillingTheClock:
    def test_high_possession_no_penetration_detected(self):
        # 70% possession but 0 shots, 0 key passes at min 60+
        state = _make_state(
            minute=75, home_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 70.0,
                StatType.SHOTS_TOTAL: 0,
                StatType.KEY_PASSES: 0,
            },
        )
        assert state.is_killing_clock("home") is True
        assert state.is_killing_clock("away") is False

    def test_high_possession_with_penetration_not_killing(self):
        state = _make_state(
            minute=75, home_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 70.0,
                StatType.SHOTS_TOTAL: 12,
                StatType.KEY_PASSES: 8,
            },
        )
        assert state.is_killing_clock("home") is False

    def test_killing_clock_inactive_before_min60(self):
        state = _make_state(
            minute=55,
            home_stats={
                StatType.BALL_POSSESSION: 75.0,
                StatType.SHOTS_TOTAL: 0,
                StatType.KEY_PASSES: 0,
            },
        )
        assert state.is_killing_clock("home") is False


# ── Yellow-card + substitution events ─────────────────────────────────────


class TestEventExtensions:
    def test_yellow_card_count(self):
        state = _make_state(
            yellow_card_events=[(20, 10, 999), (30, 10, 998), (40, 20, 997)],
        )
        assert state.yellow_card_count_home == 2
        assert state.yellow_card_count_away == 1
        assert state.players_booked_home == {999, 998}
        assert state.players_booked_away == {997}

    def test_substitution_count(self):
        state = _make_state(
            substitution_events=[(60, 10, 100), (65, 20, 200), (70, 10, 101)],
        )
        assert state.substitutions_home == 2
        assert state.substitutions_away == 1


# ── C-3 blackout windows ──────────────────────────────────────────────────


class TestBlackouts:
    def test_late_minute_blackout(self):
        state = _make_state(minute=89, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.85, "draw": 0.10, "away": 0.05},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.50")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []

    def test_post_event_blackout(self):
        # Goal at min 60, current minute 60 → blackout triggers
        state = _make_state(minute=60, home_goals=1, goal_events=[(60, 10)])
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []


# ── C-1 stale odd ─────────────────────────────────────────────────────────


class TestStaleOdd:
    def test_stale_odd_dropped(self):
        # Goal at minute 50; snapshot taken at 13:30 UTC. Wall-clock minute-50
        # was approximately 13:20 (10 minutes earlier). An odd last updated
        # at 13:00 is older than that goal by 20 minutes → stale.
        snapshot_taken = datetime(2026, 5, 9, 13, 30, tzinfo=timezone.utc)
        odd_update = datetime(2026, 5, 9, 13, 0, tzinfo=timezone.utc)
        state = _make_state(
            minute=60, home_goals=1, goal_events=[(50, 10)],
            snapshot_taken_at=snapshot_taken,
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(
            market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
            latest_bookmaker_update=odd_update,
        )]
        picks = ValueDetector(min_edge_pct=3.0).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []

    def test_fresh_odd_accepted(self):
        snapshot_taken = datetime(2026, 5, 9, 13, 30, tzinfo=timezone.utc)
        # Odd updated 5 seconds before snapshot — fresh
        odd_update = snapshot_taken - timedelta(seconds=5)
        state = _make_state(
            minute=30, home_goals=1, goal_events=[(20, 10)],
            snapshot_taken_at=snapshot_taken,
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(
            market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
            latest_bookmaker_update=odd_update,
        )]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1


# ── §D logical-score cascade ──────────────────────────────────────────────


class TestLogicalScore:
    def test_size_consistency_scales_down_extreme_edges(self):
        state = _make_state(minute=30, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.70, "draw": 0.20, "away": 0.10},
        })
        # +75% EV at 2.50 → edge_dec=0.75 → c_size = 0.20
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.50")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_flag=0.0, min_logical_score_emit=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        # The stake should be much smaller than the cap due to size_scale=0.20
        assert picks[0].logical_components["size_consistency"] == pytest.approx(0.20)

    def test_logical_score_min_aggregation(self):
        # Manufactured pick with all components ≥ 0.85 except c_size = 0.20
        state = _make_state(minute=30, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.70, "draw": 0.20, "away": 0.10},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="3.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_flag=0.0, min_logical_score_emit=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # filter_score = min over filter components (excludes size_consistency)
        score = picks[0].logical_score
        components = picks[0].logical_components
        filter_components = {
            k: v for k, v in components.items() if k != "size_consistency"
        }
        assert score == pytest.approx(min(filter_components.values()))


# ── Phase 1a regression tests (post-audit-fix, 2026-05-09 night #2) ──────────


class TestPhase1aBackoutBugs:
    """Regression suite for the bugs identified in the post-implementation audit:
    #1 BTTS-2H math, #4 c_size cascade, #6 killing-clock units, #11 VALUEBET
    coverage, #15 lambda silent default. Each test FAILED prior to the fix.
    """

    # ── #1 BTTS-2H — both teams already scored ──────────────────────────
    def test_btts_2h_does_not_return_one_when_both_already_scored(self):
        """Bug #1: previously P(yes)=1.0 because _team_to_score_remaining
        short-circuits to 1.0 for already-scored teams. Bookmaker BTTS-2H
        prices "both score in the SECOND HALF period" — independent of 1H
        goals. Fix uses _team_score_in_remaining_uncond.
        """
        # Min 60 (2H), both teams scored in 1H — but BTTS-2H must reflect
        # only what happens in remaining 30 min.
        state = _make_state(
            minute=60, home_goals=1, away_goals=1,
            goal_events=[(20, 10), (35, 20)],
        )
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        btts2h = probs.by_market.get(MARKET_BTTS_SECOND_HALF)
        assert btts2h is not None
        # With balanced ~75% per-team OU 0.5 priors and only 30 min left,
        # P(both score in remaining) should be << 1.0. Concretely,
        # λ_remaining ≈ 0.46 per side → P(score)=0.37 → product ≈ 0.14.
        assert btts2h["yes"] < 0.50, (
            f"BTTS-2H yes={btts2h['yes']:.3f} should be < 0.50; bug #1 "
            f"would emit 1.0 because both teams already scored."
        )
        assert btts2h["no"] > 0.50

    def test_btts_2h_does_not_return_one_when_only_one_scored(self):
        """Same bug #1, second branch: when only home has scored, the
        previous code also short-circuits home to 1.0, leaving yes = away_p
        rather than the true product."""
        state = _make_state(
            minute=60, home_goals=1, away_goals=0,
            goal_events=[(20, 10)],
        )
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        btts2h = probs.by_market.get(MARKET_BTTS_SECOND_HALF)
        assert btts2h is not None
        # Both teams need to score in 2H — neither short-circuits to 1.0.
        # Home λ_remaining ≈ 0.46 → P(score)=0.37; same for away → 0.37.
        # Product ≈ 0.14, not the buggy 0.37.
        assert btts2h["yes"] < 0.30

    # ── #4 c_size cascade — moderate-edge picks size-down via c_size ──
    def test_moderate_edge_emits_with_size_halved(self):
        """Bug #4: c_size component (0.5 for edge_dec ∈ (0.20, 0.40]) used
        to enter the cascade min() — at logical_score=0.5 it would still
        emit-with-flag, but if any other dimension was below 0.5 the pick
        dropped without the stake reduction at line 567 ever firing.

        With the fix, c_size is OUT of the cascade and only modulates
        stake. For edge_dec ∈ (0.20, 0.40], c_size=0.5 → stake halved.
        """
        state = _make_state(minute=40, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.65, "draw": 0.20, "away": 0.15},
        })
        # Edge ≈ 30% (P=0.65 × odd=2.00 - 1) → edge_dec=0.30 → c_size=0.5
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_flag=0.40, min_logical_score_emit=0.70,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        # c_size must be 0.5 (stake-modulator in the 20-40% range)
        assert picks[0].logical_components["size_consistency"] == pytest.approx(0.5)
        # The size_consistency dimension must NOT have entered the cascade
        # min() — logical_score equals min over filter components only.
        comps = picks[0].logical_components
        filter_min = min(v for k, v in comps.items() if k != "size_consistency")
        assert picks[0].logical_score == pytest.approx(filter_min)

    def test_extreme_edge_dropped_by_c_odds_not_c_size(self):
        """Sister test: at edge_dec > 0.40, c_odds=0 hard-drops the pick
        in cascade. This is intended (operator: 'EV > 50% should be
        DROPPED'). The c_size dimension just stops being relevant here.
        """
        state = _make_state(minute=40, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.70, "draw": 0.20, "away": 0.10},
        })
        # Edge = 75% → c_odds=0 → cascade kills via odds_credibility.
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.50")]
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_flag=0.40,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Dropped — but by c_odds, not c_size. (Operator intent honoured.)
        assert picks == []

    # ── #6 killing-the-clock unit fix ───────────────────────────────────
    def test_killing_clock_threshold_uses_possession_minutes(self):
        """Bug #6: previously divided shots by raw possession PERCENTAGE
        (0-100), which is dimensionally wrong. Fix divides by minutes-of-
        possession (minute × poss/100).
        """
        # Min 70, 70% possession, only 4 shots + 1 key pass.
        # Old formula: 5 / 70 = 0.071 < 0.10 → True.
        # New formula: poss_min = 70 × 0.7 = 49; 5/49 = 0.102; < 0.20 → True.
        state_killing = _make_state(
            minute=70,
            home_stats={
                StatType.BALL_POSSESSION: 70,
                StatType.SHOTS_TOTAL: 4,
                StatType.KEY_PASSES: 1,
            },
        )
        assert state_killing.is_killing_clock("home") is True

        # Same possession + minute, but 14 shots + 6 key passes (active).
        # New formula: 20/49 = 0.408; > 0.20 → False.
        state_active = _make_state(
            minute=70,
            home_stats={
                StatType.BALL_POSSESSION: 70,
                StatType.SHOTS_TOTAL: 14,
                StatType.KEY_PASSES: 6,
            },
        )
        assert state_active.is_killing_clock("home") is False

    def test_killing_clock_requires_min_possession_minutes(self):
        """Edge case: very early in 2H with high possession but tiny
        absolute time — must NOT fire (rate too noisy)."""
        # Minute 60, but possession just measured ~7% × 60 = 4.2 poss-min.
        # Below the 5.0 floor — should not fire even with zero shots.
        state = _make_state(
            minute=60,
            home_stats={
                StatType.BALL_POSSESSION: 7,  # absurd, but tests the floor
                StatType.SHOTS_TOTAL: 0,
                StatType.KEY_PASSES: 0,
            },
        )
        # Also fails the 65% gate, but the poss-min floor is the inner safety
        assert state.is_killing_clock("home") is False

    # ── #11 VALUEBET coverage — new markets no longer falsely flagged ───
    def test_extreme_edge_on_new_market_uses_sanity_cap_not_vb_check(self):
        """Bug #11: previously >100% EV on markets without VALUEBET coverage
        (DNB, BTTS-2H, corners, team-clean-sheet, team-to-score-first,
        OUs) was ALWAYS flagged `extreme_edge_no_sm_confirmation` because
        _selections_align returned False (market not in _VB_ALIGN). Fix
        gates the VB check on `market in _VB_ALIGN`. The 50% sanity cap
        still hard-drops these via `edge_above_sanity_cap`.
        """
        state = _make_state(minute=30, home_goals=1)
        # +52% EV on team_to_score_first (not in _VB_ALIGN)
        probs = _probs_with_conf(market_probs={
            MARKET_TEAM_TO_SCORE_FIRST: {"home": 0.55, "away": 0.35, "none": 0.10},
        })
        odds = [_odd(
            market_id=MarketID.FIRST_GOAL, label="Home", value="2.80",
        )]
        # With drop_extreme=True (default), edge_above_sanity_cap fires (>50%).
        picks = ValueDetector(
            min_edge_pct=3.0, drop_extreme=True, enforce_ci_gate=False,
            sanity_edge_cap=50.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Drop reason should be sanity_cap, NOT extreme_edge_no_sm_confirmation.
        # We can't inspect the drop reason directly here, but we confirm the
        # behaviour by checking that with sanity_edge_cap=200 (so edge_cap
        # doesn't fire), the pick now passes — proving #11 unblocks it.
        picks_relaxed = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            sanity_edge_cap=200.0,  # bypass cap so VB-coverage logic is the
                                    # sole gate
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks_relaxed) == 1, (
            "team_to_score_first pick blocked by extreme_edge_no_sm_confirmation "
            "even though that market has no VALUEBET coverage (bug #11)."
        )
        # And critically: NOT flagged with the false reason.
        assert (picks_relaxed[0].flagged_reason or "").startswith(
            "extreme_edge_no_sm_confirmation"
        ) is False

    # ── #2 drop logging — on_decision callback records every gate ──────
    def test_on_decision_callback_fires_for_below_min_edge(self):
        """Bug #2: previously gate-drops were silent — pick_decisions only
        recorded emits/flags. Fix: ValueDetector.evaluate() accepts
        on_decision callback that fires for every drop with full context.
        """
        state = _make_state(minute=30, home_goals=1)
        # Probability barely positive (edge = 1%, below default 3%)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.51, "draw": 0.30, "away": 0.19},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        decisions: list[dict] = []
        ValueDetector(min_edge_pct=3.0, enforce_ci_gate=False).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
            on_decision=lambda **kw: decisions.append(kw),
        )
        # Pick was dropped via below_min_edge — must be recorded.
        assert any(d["drop_reason"] == "below_min_edge" for d in decisions)
        rec = next(d for d in decisions if d["drop_reason"] == "below_min_edge")
        assert rec["decision"] == "drop"
        assert rec["market"] == MARKET_FULLTIME_RESULT
        assert rec["selection"] == "home"
        assert rec["edge_pct"] is not None
        # CI half-width and logical_score not yet computed at this gate.
        assert rec["logical_score"] is None

    def test_on_decision_callback_fires_for_blackout(self):
        """Global blackouts must log drop records per-market×selection."""
        state = _make_state(minute=89, home_goals=1)  # late_minute_blackout
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        decisions: list[dict] = []
        ValueDetector().evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
            on_decision=lambda **kw: decisions.append(kw),
        )
        assert any(d["drop_reason"] == "late_minute_blackout" for d in decisions)
        # One per (market, selection) — 3 selections in 1X2.
        blackout_drops = [d for d in decisions if d["drop_reason"] == "late_minute_blackout"]
        assert len(blackout_drops) == 3
        assert {d["selection"] for d in blackout_drops} == {"home", "draw", "away"}

    def test_on_decision_callback_fires_for_low_info_density(self):
        """Cluster-C info-density floor must log drop records."""
        # Minute 8, no events, no stats → density well below 0.20 floor.
        state = _make_state(minute=8)
        probs = _probs_with_conf(market_probs={
            MARKET_BTTS: {"yes": 0.65, "no": 0.35},
        })
        odds = [_odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="Yes", value="2.00")]
        decisions: list[dict] = []
        ValueDetector().evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
            on_decision=lambda **kw: decisions.append(kw),
        )
        assert any(d["drop_reason"] == "low_info_density" for d in decisions)

    def test_on_decision_callback_fires_for_critical_flag_drop(self):
        """Critical flags (extreme edge, etc.) must log canonical reasons,
        not the verbose flagged_reason string."""
        state = _make_state(minute=30, home_goals=1)
        # +200% EV — sanity_edge_cap (50%) fires; drop_extreme=True drops it
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.95, "draw": 0.03, "away": 0.02},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="3.00")]
        decisions: list[dict] = []
        ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False, drop_extreme=True,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
            on_decision=lambda **kw: decisions.append(kw),
        )
        # Canonical drop_reason matches the _CRITICAL_FLAGS prefix —
        # not the full "edge_above_sanity_cap (>50%)" string.
        critical = [d for d in decisions if d.get("decision") == "drop"]
        assert len(critical) >= 1
        assert any(
            d["drop_reason"] == "edge_above_sanity_cap" for d in critical
        )

    def test_on_decision_callback_optional_no_op_when_none(self):
        """Backward compat: existing callers that don't pass on_decision
        continue to work; drops happen silently as before."""
        state = _make_state(minute=8)  # forces low_info_density drop
        probs = _probs_with_conf(market_probs={
            MARKET_BTTS: {"yes": 0.65, "no": 0.35},
        })
        odds = [_odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="Yes", value="2.00")]
        # No on_decision → must not raise
        result = ValueDetector().evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert result == []

    # ── #5 stale-odd math — wall-clock independent of match-minute ──────
    def test_stale_odd_robust_against_match_minute_distortion(self):
        """Bug #5: previous code computed ``last_event_time = snapshot_at -
        timedelta(minutes=match_minute_delta)`` which assumed 1 match-min ≈
        1 wall-clock-min. False during halftime (+15 min), stoppage time,
        VAR pauses, ET, etc. A pick captured at minute 60 with goal at
        minute 50 in real time was 10-25 wall-clock minutes after the
        goal — but the old code dated the goal to (snapshot - 10 min).

        Fix: stale-odd uses snapshot-relative odd age (no match-minute
        conversion). An odd is stale if (a) it hasn't refreshed in 5 min
        before the snapshot, OR (b) a material event landed within last
        3 match-minutes AND the odd is older than 30s.
        """
        # Scenario A: 2nd half snapshot, match-minute 60, goal at min 50,
        # but real wall-clock time means goal happened ~25 min ago because
        # of a long halftime + stoppage. Odd was updated 10s before the
        # snapshot — clearly fresh.
        snapshot_at = datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc)
        # Match has been going 25+ wall-clock minutes since the goal —
        # but the odd is only 10s old.
        odd_update = snapshot_at - timedelta(seconds=10)
        state = _make_state(
            minute=60, home_goals=1, goal_events=[(50, 10)],
            snapshot_taken_at=snapshot_at,
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(
            market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
            latest_bookmaker_update=odd_update,
        )]
        picks = ValueDetector(min_edge_pct=3.0, enforce_ci_gate=False).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Odd is fresh (10s old) — must NOT be stale regardless of where
        # match-minute math says the goal happened.
        assert len(picks) == 1, (
            "Fresh odd dropped as stale because the old code computed "
            "last_event_time using match-minute arithmetic that doesn't "
            "account for halftime / stoppage time."
        )

    def test_stale_odd_post_event_within_3_match_minutes(self):
        """Tight gate: when a goal landed within last 3 match-minutes, the
        odd must be very fresh (≤ 30s)."""
        snapshot_at = datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc)
        # Goal at minute 60, current minute 62 — 2 match-minutes after goal.
        # Odd is 60s old: passes hard freshness (300s), but fails post-event.
        odd_update = snapshot_at - timedelta(seconds=60)
        state = _make_state(
            minute=62, home_goals=1, goal_events=[(60, 10)],
            snapshot_taken_at=snapshot_at,
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(
            market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
            latest_bookmaker_update=odd_update,
        )]
        picks = ValueDetector(min_edge_pct=3.0, enforce_ci_gate=False).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == [], (
            "Post-event tight gate should drop odds older than 30s when a "
            "material event landed within the last 3 match-minutes."
        )

    def test_stale_odd_hard_freshness_floor(self):
        """Hard floor: 5+ minutes since odd update = abandoned, drop."""
        snapshot_at = datetime(2026, 5, 9, 14, 30, tzinfo=timezone.utc)
        # Odd 6 minutes old — beyond hard freshness floor (300s).
        odd_update = snapshot_at - timedelta(seconds=360)
        state = _make_state(
            minute=70, home_goals=1, goal_events=[(20, 10)],  # event long ago
            snapshot_taken_at=snapshot_at,
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(
            market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
            latest_bookmaker_update=odd_update,
        )]
        picks = ValueDetector(min_edge_pct=3.0, enforce_ci_gate=False).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == [], "5+ min old odd must be dropped (hard freshness)"

    # ── #8 pressure_avg empty window — no false PRESSURE_DIFF trigger ──
    def test_pressure_nudge_skipped_when_one_side_has_no_samples(self):
        """Bug #8: previously when only one side had pressure samples, the
        other side's pressure_avg defaulted to 0, producing a fake huge
        diff that triggered PRESSURE_DIFF_TRIGGER (25). Fix gates the
        nudge on BOTH sides having data.
        """
        # Home has high pressure samples, away has none. Old code:
        # diff = 65 - 0 = 65 → fires shift. New code: skips entirely.
        state_a = _make_state(
            minute=30, home_goals=1,
            home_pressure=[60, 65, 70, 65, 60],
            away_pressure=[],  # asymmetric data
        )
        # Same scenario but BOTH sides have data — nudge SHOULD fire.
        state_b = _make_state(
            minute=30, home_goals=1,
            home_pressure=[60, 65, 70, 65, 60],
            away_pressure=[20, 15, 10, 15, 20],  # diff = 65-16 = ~49
        )
        predictor = LiveMatchPredictor()
        probs_a = predictor.predict(state_a).by_market[MARKET_FULLTIME_RESULT]
        probs_b = predictor.predict(state_b).by_market[MARKET_FULLTIME_RESULT]
        # In state_a, home prob should be close to the Dixon-Robinson
        # base (no pressure nudge); in state_b, home prob should be
        # measurably HIGHER (pressure nudge fires for home).
        assert probs_b["home"] > probs_a["home"] + 0.01, (
            f"Pressure nudge failed to fire for state_b "
            f"(home={probs_b['home']:.3f} vs state_a={probs_a['home']:.3f})"
        )

    # ── #9 c_consilience must NOT double-count density ─────────────────
    def test_consilience_does_not_drop_with_low_density(self):
        """Bug #9: previously density was added to consilience_terms,
        meaning low density dragged consilience down even when both
        Sportmonks cross-check sources (grid + valuebet) agreed.
        """
        # Construct a state with:
        # - Low density (early minute, no events) — but density is
        #   gated by the predictor anyway, so we test consilience by
        #   manually emitting probs+state with VALUEBET aligned.
        # - VALUEBET aligned with our pick → consilience contributor = 1.0
        # - Density component is in c_info, NOT consilience.
        sm = dict(_BALANCED_PRE_MATCH)
        sm[PredictionType.VALUEBET] = {
            "bet": "home", "is_value": True,
        }
        state = _make_state(
            minute=30, home_goals=1, sportmonks=sm,
            # Lots of stats so density is high
            home_stats={
                StatType.SHOTS_TOTAL: 8, StatType.KEY_PASSES: 5,
                StatType.SHOTS_ON_TARGET: 3, StatType.CORNERS: 4,
            },
        )
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        # consilience should be 1.0 (VALUEBET aligned, no grid penalty)
        assert picks[0].logical_components["consilience"] == pytest.approx(1.0)

    # ── #10 c_consilience separate defaults: state-None vs no-sources ──
    def test_consilience_no_sources_gives_flag_default(self):
        """Bug #10: previously when state existed but Sportmonks had no
        VALUEBET / CORRECT_SCORE for this market, c_consilience used the
        same 0.85 default as the test-fixture path (state=None) — too
        high to trigger flagging. Fix splits: 0.85 only when state=None,
        0.65 (below emit threshold) when state exists but no sources.
        """
        # _BALANCED_PRE_MATCH has no VALUEBET, no CORRECT_SCORE_PROBABILITY.
        state = _make_state(minute=30, home_goals=1)
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        # State exists, no SM sources for this market → 0.65 default
        assert picks[0].logical_components["consilience"] == pytest.approx(0.65)

    def test_consilience_state_none_uses_test_default(self):
        """Sister test: when state IS None (unit-test fixture path),
        consilience defaults to 0.85 — high enough to not penalize tests
        that don't construct realistic Sportmonks state."""
        probs = _probs_with_conf(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=None,
        )
        assert len(picks) == 1
        assert picks[0].logical_components["consilience"] == pytest.approx(0.85)

    # ── #16 killing-clock proportional damp ────────────────────────────
    def test_killing_clock_one_side_dampens_half(self):
        """Bug #16: previously OU markets applied full 15% λ damp when
        EITHER team was killing clock. Asymmetric (only one team killing)
        should only apply ~7.5% damp because the other team still
        contributes their full attacking λ.

        Compare two states with IDENTICAL score / minute / SM priors —
        the only difference is whether one or both teams are killing clock.
        """
        # SAME score (1-1) and minute (70). Only difference: how many
        # teams meet the killing-clock pattern (high poss + low productivity).
        state_one_kills = _make_state(
            minute=70, home_goals=1, away_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 70,    # home kills clock
                StatType.SHOTS_TOTAL: 4, StatType.KEY_PASSES: 1,
            },
            away_stats={
                StatType.BALL_POSSESSION: 30,    # away active (low poss)
                StatType.SHOTS_TOTAL: 8, StatType.KEY_PASSES: 4,
            },
        )
        state_both_kill = _make_state(
            minute=70, home_goals=1, away_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 65, StatType.SHOTS_TOTAL: 4,
                StatType.KEY_PASSES: 1,          # home kills
            },
            away_stats={
                StatType.BALL_POSSESSION: 65, StatType.SHOTS_TOTAL: 3,
                StatType.KEY_PASSES: 1,          # away also kills
            },
        )
        assert state_one_kills.is_killing_clock("home") is True
        assert state_one_kills.is_killing_clock("away") is False
        assert state_both_kill.is_killing_clock("home") is True
        assert state_both_kill.is_killing_clock("away") is True

        predictor = LiveMatchPredictor()
        probs_one = predictor.predict(state_one_kills).by_market.get(MARKET_OU_25)
        probs_both = predictor.predict(state_both_kill).by_market.get(MARKET_OU_25)
        assert probs_one is not None and probs_both is not None
        # Same score → same goals_needed, same base λ. Only damp factor
        # differs: 0.925× (one kills) vs 0.85× (both kill).
        # Lower λ → lower P(over). So both-kill < one-kill.
        assert probs_both["over"] < probs_one["over"], (
            f"both-kill should suppress overs more than one-kill: "
            f"both={probs_both['over']:.3f} vs one={probs_one['over']:.3f}"
        )

    # ── #12 CORRECT_SCORE relative tolerance ────────────────────────────
    def test_correct_score_high_prob_market_tolerates_pp_noise(self):
        """Bug #12: the previous absolute 5pp tolerance flagged perfectly
        normal noise on high-probability markets (e.g., OU 0.5 with sm=0.92
        vs ours=0.85 — 7pp absolute, 8% relative). Fix uses mixed tolerance:
        max(5pp, 30% × min(sm, ours)) — at high probs the relative term
        gives generous slack.
        """
        # OU 0.5 with our_p=0.85 vs SM grid implied = 0.92.
        # Old: 7pp diff > 5pp → FLAG (false alarm).
        # New: tolerance = max(0.05, 0.30 × 0.85) = 0.255 → diff 0.07 < 0.255 → OK.
        sm = dict(_BALANCED_PRE_MATCH)
        # Build a CORRECT_SCORE grid where P(over 0.5) = 0.92
        # (everything except 0-0): grid[0,0]=8, others sum to 92.
        sm[PredictionType.CORRECT_SCORE_PROBABILITY] = {
            "scores": {
                "0-0": 8.0,
                "1-0": 25.0, "0-1": 20.0, "1-1": 22.0,
                "2-0": 10.0, "0-2": 8.0, "2-1": 4.0, "1-2": 3.0,
            },
        }
        state = _make_state(minute=30, home_goals=0, sportmonks=sm)
        # Our P(over 0.5) = 0.85 (slightly lower than SM 0.92)
        probs = _probs_with_conf(market_probs={
            MARKET_OU_05: {"over": 0.85, "under": 0.15},
        })
        odds = [_odd(
            market_id=MarketID.MATCH_GOALS, label="Over", value="1.20",
            total="0.5",
        )]
        picks = ValueDetector(
            min_edge_pct=0.0,  # accept any edge so we test only the gate
            enforce_ci_gate=False, drop_extreme=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
            # Opt out of post-audit gates: this test exercises the
            # correct-score-disagrees tolerance gate only.
            ban_positive_side_binaries=False,
            drop_over_zero_zero=False,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Should NOT be flagged for correct_score_disagrees on 7pp noise
        assert len(picks) == 1
        assert (picks[0].flagged_reason or "").startswith(
            "correct_score_disagrees"
        ) is False, f"unexpected flag: {picks[0].flagged_reason}"

    def test_correct_score_low_prob_market_strict_tolerance(self):
        """Sister test: at LOW probs (e.g., OU 3.5 with sm=0.20 vs ours=0.40),
        a 20pp absolute disagreement = 100% relative — must flag."""
        sm = dict(_BALANCED_PRE_MATCH)
        # Build grid where P(over 3.5) = 0.20
        sm[PredictionType.CORRECT_SCORE_PROBABILITY] = {
            "scores": {
                "0-0": 15.0, "1-0": 18.0, "0-1": 17.0, "1-1": 16.0,
                "2-0": 7.0, "0-2": 7.0, "2-1": 5.0, "1-2": 5.0,
                "3-1": 4.0, "1-3": 3.0, "2-2": 3.0,
            },
        }
        state = _make_state(minute=30, home_goals=0, sportmonks=sm)
        probs = _probs_with_conf(market_probs={
            MARKET_OU_35: {"over": 0.40, "under": 0.60},  # twice SM's
        })
        odds = [_odd(
            market_id=MarketID.MATCH_GOALS, label="Over", value="3.00",
            total="3.5",
        )]
        picks = ValueDetector(
            min_edge_pct=0.0, enforce_ci_gate=False,
            drop_extreme=True,  # critical flags hard-drop
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # 20pp diff > tolerance max(0.05, 0.30 × 0.20)=0.06 → flagged → dropped
        assert picks == []

    # ── #13 + #14 bundle dedup — confidence-weighted, flagged-aware ─────
    def test_bundle_dedup_prefers_high_confidence_over_coin_flip(self):
        """Bug #13: previously the rank key was edge × p × (1−p), which
        MAXIMIZED at p=0.5. A coin-flip pick of equal edge would beat a
        high-confidence pick. Fix: edge × (1 − p × (1−p)), which rewards
        confidence (peaks at p=0 or p=1).
        """
        # Two picks in the same 1X2 cluster. Both have ~similar edges,
        # but one is coin-flip (p=0.55) and the other is confident (p=0.85).
        state = _make_state(minute=40, home_goals=1)
        probs = _probs_with_conf(market_probs={
            # FT result: home win at p=0.85 (confident pick on home)
            MARKET_FULLTIME_RESULT: {"home": 0.85, "draw": 0.10, "away": 0.05},
            # Double chance 1X at p=0.95 (HIGHLY confident — same direction)
            # but at lower bookie odd → smaller absolute edge.
            MARKET_DOUBLE_CHANCE: {"1x": 0.95, "x2": 0.15, "12": 0.90},
        })
        # Edges:
        #   FTR home @ 1.30: 0.85 × 1.30 - 1 = 10.5%
        #   DC 1x @ 1.15: 0.95 × 1.15 - 1 = 9.25%
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.30"),
            _odd(market_id=MarketID.DOUBLE_CHANCE, label="1X", value="1.15"),
        ]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False, drop_extreme=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Bundle dedup collapses to ONE pick from the 1X2 cluster.
        ones = [p for p in picks if p.market in (
            MARKET_FULLTIME_RESULT, MARKET_DOUBLE_CHANCE,
        )]
        assert len(ones) == 1
        # Old key: edge × p×(1-p)
        #   FTR: 10.5 × 0.85×0.15 = 1.34
        #   DC:   9.25 × 0.95×0.05 = 0.44 → FTR wins (LUCKILY)
        # New key: edge × (1 - p×(1-p))
        #   FTR: 10.5 × (1 - 0.1275) = 9.16
        #   DC:   9.25 × (1 - 0.0475) = 8.81  → FTR still wins
        # Either way, FTR home wins because edge AND confidence are higher.
        # BUT: if we lowered FTR confidence to coin-flip (p=0.55), old code
        # would have flipped. Let's verify the rank function directly.
        from bip.evaluation.live.value_detector import _bundle_dedup_picks
        # Build two picks with EQUAL edge, different confidence.
        coin_flip = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_FULLTIME_RESULT, selection="home",
            bookmaker_id=2, bookmaker_odd=2.00, our_probability=0.55,
            fair_odd=1.82, edge_pct=10.0,
            kelly_fraction_full=0.05, suggested_stake_pct=1.0,
        )
        confident = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_DOUBLE_CHANCE, selection="1x",
            bookmaker_id=2, bookmaker_odd=1.15, our_probability=0.96,
            fair_odd=1.04, edge_pct=10.0,
            kelly_fraction_full=0.05, suggested_stake_pct=1.0,
        )
        result = _bundle_dedup_picks([coin_flip, confident])
        assert len(result) == 1
        assert result[0].our_probability == 0.96, (
            "Confident pick should win at equal edge; old code picked coin-flip."
        )

    def test_bundle_dedup_prefers_clean_over_flagged(self):
        """Bug #14: previously a flagged pick with higher edge could
        displace a clean (unflagged) pick of lower edge. Fix: clean
        picks always win regardless of edge.
        """
        from bip.evaluation.live.value_detector import _bundle_dedup_picks
        clean = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_FULLTIME_RESULT, selection="home",
            bookmaker_id=2, bookmaker_odd=2.00, our_probability=0.55,
            fair_odd=1.82, edge_pct=5.0,  # lower edge
            kelly_fraction_full=0.03, suggested_stake_pct=0.6,
            flagged_reason=None,  # CLEAN
        )
        flagged = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_DOUBLE_CHANCE, selection="1x",
            bookmaker_id=2, bookmaker_odd=1.40, our_probability=0.78,
            fair_odd=1.28, edge_pct=12.0,  # HIGHER edge
            kelly_fraction_full=0.10, suggested_stake_pct=1.5,
            flagged_reason="logical_score_below_emit",
        )
        result = _bundle_dedup_picks([clean, flagged])
        assert len(result) == 1
        assert result[0].flagged_reason is None, (
            "Clean pick must win over flagged regardless of edge."
        )

    # ── #18 confidence_1x2 multi-axis (max spread, not draw-only) ──────
    def test_confidence_1x2_uses_max_spread_across_all_three_outcomes(self):
        """Bug #18: previously CI was derived from disagreement on P(draw)
        only — meaning a pick on `home` got the same CI as one on `draw`,
        even if the home-axis disagreement was much wider. Fix: max
        spread across home/draw/away.
        """
        # Construct sportmonks state where:
        # - 1X2 prediction says home=0.30, draw=0.30, away=0.40
        # - DC prediction implies home=0.50 (via 1 - X2=0.50), draw=0.30,
        #   away=0.20 (very different home/away marginals from 1X2)
        # - No CORRECT_SCORE
        # Old code: draw spread = max(0.30, 0.30) - min = 0 → CI floor.
        # New code: home spread = 0.50-0.30 = 0.20 → CI = 0.20/2 + 0.015 = 0.115.
        sm = dict(_BALANCED_PRE_MATCH)
        sm[PredictionType.FULLTIME_RESULT_PROBABILITY] = {
            "home": 30, "draw": 30, "away": 40,
        }
        sm[PredictionType.DOUBLE_CHANCE_PROBABILITY] = {
            "draw_home": 60,    # 1X = home + draw → = 0.60
            "draw_away": 50,    # X2 = draw + away → = 0.50
            "home_away": 70,    # 12 = home + away → = 0.70
        }
        # Imply: P(home) = 1 - P(X2) = 0.50
        #        P(draw) = 1 - P(12) = 0.30
        #        P(away) = 1 - P(1X) = 0.40
        state = _make_state(minute=30, home_goals=1, sportmonks=sm)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        ci = probs.confidences.get(MARKET_FULLTIME_RESULT)
        assert ci is not None
        # Home axis: SM_FT=0.30 vs SM_DC=0.50 → spread=0.20
        # Draw axis: SM_FT=0.30 vs SM_DC=0.30 → spread=0
        # Away axis: SM_FT=0.40 vs SM_DC=0.40 → spread=0
        # Max spread = 0.20 → CI = 0.20/2 + 0.015 = 0.115
        assert ci == pytest.approx(0.115, abs=0.005), (
            f"Multi-axis CI = {ci:.3f}, expected ~0.115 (home-axis spread). "
            f"Old draw-only code would yield 0.02 floor."
        )

    # ── #19 coherence margin normalization ─────────────────────────────
    def test_coherence_dc_monotonicity_normalises_for_margin(self):
        """Bug #19: 1X2 (6% margin) vs DC (3% margin) compared raw could
        falsely trip 'X2 < draw' from margin asymmetry alone. Fix: fair
        normalisation. Cluster-A still uses raw (it's a pricing pathology).
        """
        # 1X2 with 6% margin: home@2.10 (0.476), draw@3.40 (0.294), away@3.40 (0.294)
        # Sum raw = 1.064 → fair: home=0.448, draw=0.276, away=0.276
        # DC with 3% margin: 1X@1.45 (0.690), 12@1.45 (0.690), X2@2.85 (0.351)
        # Sum raw = 1.731. Fair (×2/sum): 1X=0.797, 12=0.797, X2=0.405
        # Raw X2 (0.351) > raw draw (0.294) → no monotonicity violation.
        # Fair X2 (0.405) > fair draw (0.276) → still no violation. ✓ coherent.
        odds_by_market = {
            MarketID.FULLTIME_RESULT: [
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.10"),
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Draw", value="3.40"),
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Away", value="3.40"),
            ],
            MarketID.DOUBLE_CHANCE: [
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="1X", value="1.45"),
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="12", value="1.45"),
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="X2", value="2.85"),
            ],
        }
        # Use the internal coherence function via the detector
        detector = ValueDetector()
        skip = detector._bookmaker_coherence_flags(odds_by_market)
        # Healthy markets: no skip
        assert MARKET_FULLTIME_RESULT not in skip
        assert MARKET_DOUBLE_CHANCE not in skip

    def test_coherence_cluster_a_pathology_still_caught_after_norm(self):
        """Cluster-A check stays on RAW (X2 odd ≈ draw odd), unaffected
        by normalisation."""
        odds_by_market = {
            MarketID.FULLTIME_RESULT: [
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.10"),
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Draw", value="3.40"),
                _odd(market_id=MarketID.FULLTIME_RESULT, label="Away", value="3.40"),
            ],
            MarketID.DOUBLE_CHANCE: [
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="1X", value="1.40"),
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="12", value="1.45"),
                _odd(market_id=MarketID.DOUBLE_CHANCE, label="X2", value="3.40"),
                # X2 priced same as draw — cluster-A pathology
            ],
        }
        detector = ValueDetector()
        skip = detector._bookmaker_coherence_flags(odds_by_market)
        # Both 1X2 and DC must be skipped
        assert MARKET_FULLTIME_RESULT in skip
        assert MARKET_DOUBLE_CHANCE in skip

    # ── Phase 7 — trends deep integration ──────────────────────────────

    def _trends_for(
        self, side_team_id: int, *, type_id: int,
        minute_value_pairs: list[tuple[int, int]],
    ):
        """Build cumulative Trend records: (minute, cumulative_value)."""
        from bip.sports.football.sportmonks.schemas import Trend
        return [
            Trend.model_validate({
                "id": id((m, side_team_id, type_id)),
                "fixture_id": 1, "type_id": type_id,
                "participant_id": side_team_id,
                "period_id": 1 if m <= 45 else 2,
                "minute": m, "value": v,
            })
            for m, v in minute_value_pairs
        ]

    def test_shot_acceleration_detects_rising_pace(self):
        """3 shots in last 3 min vs 2 shots in min 12-17 ago = accel ratio
        (3/3) / (2/5) = 1.0 / 0.4 = 2.5"""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Cumulative SHOTS_TOTAL for home_team_id=10 over time:
        # at min 15: 0; at min 17: 1; at min 20: 2 (=2 in min 15-20 → "5 min before recent")
        # at min 25: 5 (=3 in min 20-25 → "last 3 min" since current minute=25... wait
        # acceleration uses (minute-3, minute) for recent and (minute-8, minute-3) for prior.
        # At minute=20: recent=last 3 min (17-20); prior=min 12-17.
        trends = self._trends_for(10, type_id=StatType.SHOTS_TOTAL,
                                   minute_value_pairs=[
                                       (12, 0), (17, 2), (20, 5)
                                   ])
        # cumulative at min 20 = 5; at min 17 = 2; at min 12 = 0
        # recent: 5 - 2 = 3 in last 3 min → 1.0/min
        # prior: 2 - 0 = 2 in min 12-17 → 0.4/min
        # ratio = 1.0/0.4 = 2.5
        base = _make_state(minute=20, home_goals=0)
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        assert state.shot_acceleration("home") == pytest.approx(2.5, abs=0.01)

    def test_shot_acceleration_no_trends_returns_zero(self):
        """No trends → 0.0 (graceful degradation)."""
        from tests.evaluation.live.test_audit_hardening import _make_state
        state = _make_state(minute=30, home_goals=0)
        assert state.shot_acceleration("home") == 0.0

    def test_momentum_score_neutral_when_pace_matches_avg(self):
        """Recent rate == match-average rate → composite ≈ 1.0"""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # At minute=20 with cumulative shots=4 (avg 0.2/min) and last 5 min
        # = 1 shot (0.2/min) → ratio = 1.0
        trends = (
            self._trends_for(10, type_id=StatType.SHOTS_TOTAL,
                             minute_value_pairs=[(15, 3), (20, 4)])
            + self._trends_for(10, type_id=StatType.DANGEROUS_ATTACKS,
                               minute_value_pairs=[(15, 30), (20, 40)])
            + self._trends_for(10, type_id=StatType.KEY_PASSES,
                               minute_value_pairs=[(15, 6), (20, 8)])
        )
        base = _make_state(minute=20, home_goals=0)
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        # Each ratio: shots 0.2/0.2=1, DA 2.0/2.0=1, KP 0.4/0.4=1 → 1.0
        m = state.momentum_score("home", window=5)
        assert m == pytest.approx(1.0, abs=0.05)

    def test_momentum_score_high_when_accelerating(self):
        """Team accelerates: 4 shots in last 5 min, only 4 total → recent
        is 0.8/min vs match-avg 0.2/min → ratio 4.0 (clamped)."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # At minute 20, cumulative shots=4, prior at min 15 = 0
        # → recent 4 shots in 5 min = 0.8/min; avg 0.2/min → ratio 4.0
        trends = (
            self._trends_for(10, type_id=StatType.SHOTS_TOTAL,
                             minute_value_pairs=[(15, 0), (20, 4)])
            + self._trends_for(10, type_id=StatType.DANGEROUS_ATTACKS,
                               minute_value_pairs=[(15, 10), (20, 30)])
            + self._trends_for(10, type_id=StatType.KEY_PASSES,
                               minute_value_pairs=[(15, 1), (20, 5)])
        )
        base = _make_state(minute=20, home_goals=0)
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        m = state.momentum_score("home", window=5)
        # Composite is bounded but should be > 1.5 for clear acceleration
        assert m > 1.5

    def test_killing_clock_rejected_when_team_still_pushing(self):
        """Trend confirmation: instantaneous detector says clock-killing
        (high poss + low cumulative actions), but shots are still
        accelerating → reject (team isn't really killing clock)."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Configure state to instantaneously fire clock-killing:
        #   minute=70, poss=70, shots=4, KP=1 → poss_min=49, prod=5/49=0.10 < 0.20
        # But shots accelerating: 3 shots in last 3min vs 1 in prior 5min
        trends = self._trends_for(10, type_id=StatType.SHOTS_TOTAL,
                                   minute_value_pairs=[(62, 1), (67, 1), (70, 4)])
        base = _make_state(
            minute=70, home_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 70,
                StatType.SHOTS_TOTAL: 4, StatType.KEY_PASSES: 1,
            },
        )
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        # Without trends would be True (Phase 1a regression test confirms).
        # With trends showing acceleration, should be False.
        accel = state.shot_acceleration("home")
        assert accel > 0.9, f"test setup error: accel={accel:.2f}"
        assert state.is_killing_clock("home") is False

    def test_killing_clock_confirmed_when_actually_decelerating(self):
        """Sister test: instantaneous gate fires AND shots are flat /
        decelerating → confirmed clock-killing."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Most shots accumulated 20+ min ago, no recent activity
        trends = self._trends_for(10, type_id=StatType.SHOTS_TOTAL,
                                   minute_value_pairs=[(45, 4), (60, 4), (67, 4), (70, 4)])
        base = _make_state(
            minute=70, home_goals=1,
            home_stats={
                StatType.BALL_POSSESSION: 70,
                StatType.SHOTS_TOTAL: 4, StatType.KEY_PASSES: 1,
            },
        )
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        # Shot acceleration: recent (last 3 min) = 0; prior (3-8) = 0 → 0.0
        # 0.0 ≤ 0.9 → killing-clock confirmed
        assert state.is_killing_clock("home") is True

    def test_late_game_push_scaled_by_shot_acceleration(self):
        """Trailing team with high acceleration gets bigger nudge than
        trailing team with no recent shots."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Home trailing 0-1 at min 80
        pushing_trends = self._trends_for(
            10, type_id=StatType.SHOTS_TOTAL,
            minute_value_pairs=[(72, 4), (77, 5), (80, 9)],
        )
        # accel: recent (77-80) = 4; prior (72-77) = 1 → 4/3 ÷ 1/5 = 6.67 (capped 1.5)
        flat_trends = self._trends_for(
            10, type_id=StatType.SHOTS_TOTAL,
            minute_value_pairs=[(60, 4), (70, 4), (80, 4)],
        )
        # accel = 0 (no recent shots)
        base = _make_state(
            minute=80, home_goals=0, away_goals=1,
            home_stats={StatType.BALL_POSSESSION: 50, StatType.SHOTS_TOTAL: 9, StatType.KEY_PASSES: 3},
            home_pressure=[60, 65, 70, 70, 75],
            away_pressure=[40, 35, 30, 30, 25],
        )
        push_state = LiveMatchState(**{**base.__dict__, "trends": pushing_trends})
        flat_state = LiveMatchState(**{**base.__dict__, "trends": flat_trends})
        predictor = LiveMatchPredictor()
        push_probs = predictor.predict(push_state).by_market[MARKET_FULLTIME_RESULT]
        flat_probs = predictor.predict(flat_state).by_market[MARKET_FULLTIME_RESULT]
        # Pushing team gets bigger draw bump than flat team
        assert push_probs["draw"] > flat_probs["draw"] + 0.005

    # ── Bug 3 (post-jornada audit) — BTTS yes over-predicted at 0-0 ──

    def test_btts_yes_at_0_0_uses_joint_probability(self):
        """Bug discovered 2026-05-10 jornada: BTTS yes hit 1/33 (3%)
        because at 0-0 the predictor computed P(home scores) alone instead
        of the joint P(both teams score). Result: P=0.65-0.75 reported,
        true P ~0.30-0.40, picks fired against bookmaker odds at 1.7-2.0
        and lost almost all.

        Fix: joint probability assuming independence — matches the
        treatment in Phase 5 BTTS-2H derivation.
        """
        # 0-0 at min 25, both teams have ~70% per-team OU 0.5 prematch.
        # λ_pre per team = -ln(0.30) ≈ 1.20
        # λ_remaining (65 min left) ≈ 1.20 × 65/90 ≈ 0.87
        # P(team scores) ≈ 1 - exp(-0.87) ≈ 0.58
        # JOINT P(BTTS yes) ≈ 0.58 × 0.58 = 0.34   ← correct
        # OLD BUG: returned 0.58 (single team) — over by ~70%.
        state = _make_state(minute=25, home_goals=0, away_goals=0)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state).by_market[MARKET_BTTS]
        # New behavior: joint product → P(yes) ≈ 0.34. Allow tolerance for
        # the bivariate Poisson rho in the SM prior decoupling.
        assert probs["yes"] < 0.45, (
            f"BTTS yes at 0-0 should be joint-product (~0.34), got "
            f"{probs['yes']:.3f}. Single-team bug regressed."
        )
        # And no should reflect the rest of the mass.
        assert probs["no"] > 0.55

    def test_btts_yes_at_1_0_unchanged(self):
        """Sister test: when ONE team has scored, the existing logic
        (P(other scores)) stays correct and is not affected by the fix."""
        # 1-0 at min 25 — need away to score in 65 min. Sportmonks
        # AWAY_OU_0_5 yes=70 → λ ≈ 1.20 → λ_remaining 0.87 → P ≈ 0.58
        state = _make_state(minute=25, home_goals=1, away_goals=0)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state).by_market[MARKET_BTTS]
        # Single-team scoring prob (correct in this case, NOT the buggy path)
        assert 0.50 < probs["yes"] < 0.65

    def test_btts_yes_both_scored_returns_one(self):
        """Both teams already scored → yes is certain."""
        state = _make_state(minute=70, home_goals=1, away_goals=1)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state).by_market[MARKET_BTTS]
        assert probs["yes"] == 1.0
        assert probs["no"] == 0.0

    # ── Phase 8 — trends saturation (cards / cross-team / set-pieces) ──

    def test_cards_lambda_lifted_by_recent_foul_cluster(self):
        """Phase 8 (#A): rolling fouls in last 15 min materially above
        match-avg foul rate → cards λ lifts via foul_mult. Compare two
        states with same cumulative state but different recent foul
        trajectories."""
        from bip.evaluation.live.match_state import LiveMatchState
        from bip.evaluation.live.predictor import MARKET_CARDS_TOTAL_3_5
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Common state: minute=60, 2 yellows total, density-met
        base = _make_state(
            minute=60, home_goals=1,
            home_stats={
                StatType.SHOTS_TOTAL: 8, StatType.KEY_PASSES: 4,
                StatType.SHOTS_ON_TARGET: 3, StatType.CORNERS: 4,
                StatType.TACKLES: 12, StatType.INTERCEPTIONS: 8,
                StatType.DUELS_WON: 30, StatType.FOULS: 12,
            },
            away_stats={
                StatType.SHOTS_TOTAL: 6, StatType.KEY_PASSES: 3,
                StatType.SHOTS_ON_TARGET: 2, StatType.CORNERS: 3,
                StatType.TACKLES: 10, StatType.INTERCEPTIONS: 7,
                StatType.DUELS_WON: 28, StatType.FOULS: 10,
            },
            yellow_card_events=[(15, 10, 100), (35, 20, 200)],
        )
        # Trajectory A: fouls FLAT — match-avg holds
        flat_trends = (
            self._trends_for(10, type_id=StatType.FOULS,
                             minute_value_pairs=[(45, 10), (55, 11), (60, 12)])
            + self._trends_for(20, type_id=StatType.FOULS,
                               minute_value_pairs=[(45, 8), (55, 9), (60, 10)])
        )
        # Trajectory B: foul SURGE — 8 fouls in last 15 min vs 14 cumulative
        surge_trends = (
            self._trends_for(10, type_id=StatType.FOULS,
                             minute_value_pairs=[(45, 8), (55, 10), (60, 16)])
            + self._trends_for(20, type_id=StatType.FOULS,
                               minute_value_pairs=[(45, 6), (55, 8), (60, 14)])
        )
        flat_state = LiveMatchState(**{**base.__dict__, "trends": flat_trends})
        surge_state = LiveMatchState(**{**base.__dict__, "trends": surge_trends})
        predictor = LiveMatchPredictor()
        flat_p = predictor.predict(flat_state).by_market.get(MARKET_CARDS_TOTAL_3_5)
        surge_p = predictor.predict(surge_state).by_market.get(MARKET_CARDS_TOTAL_3_5)
        assert flat_p is not None and surge_p is not None
        # Surge state must have higher P(over 3.5)
        assert surge_p["over"] > flat_p["over"] + 0.01

    def test_cross_team_momentum_lifts_total_lambda(self):
        """Phase 8 (#D): when BOTH teams' momentum_score > 1.2, the
        joint product is high → _estimate_total_lambda lifts → OU 2.5
        over probability lifts. Compare to flat-momentum baseline."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Common base: 0-0 at minute 30, balanced sportmonks
        base = _make_state(minute=30, home_goals=0, away_goals=0)
        # Trajectory A: both teams flat (steady pace = momentum ~ 1.0)
        flat = []
        for tid in (StatType.SHOTS_TOTAL, StatType.DANGEROUS_ATTACKS, StatType.KEY_PASSES):
            for team_id in (10, 20):
                flat.extend(self._trends_for(
                    team_id, type_id=tid,
                    minute_value_pairs=[(15, 3), (25, 5), (30, 6)],
                ))
        # Trajectory B: both accelerating sharply (joint > 2.0)
        surge = []
        for tid, vals in [
            (StatType.SHOTS_TOTAL, [(15, 1), (25, 2), (30, 8)]),
            (StatType.DANGEROUS_ATTACKS, [(15, 5), (25, 10), (30, 30)]),
            (StatType.KEY_PASSES, [(15, 1), (25, 2), (30, 6)]),
        ]:
            for team_id in (10, 20):
                surge.extend(self._trends_for(team_id, type_id=tid,
                                               minute_value_pairs=vals))
        flat_state = LiveMatchState(**{**base.__dict__, "trends": flat})
        surge_state = LiveMatchState(**{**base.__dict__, "trends": surge})
        predictor = LiveMatchPredictor()
        flat_ou = predictor.predict(flat_state).by_market[MARKET_OU_25]
        surge_ou = predictor.predict(surge_state).by_market[MARKET_OU_25]
        # Surge should give materially higher P(over 2.5)
        assert surge_ou["over"] > flat_ou["over"] + 0.01

    def test_set_piece_intensity_helper(self):
        """Phase 8 (#F helper): combined corners + crosses rolling vs
        league baseline. Returns ratio."""
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Last 15 min: 3 corners + 8 crosses for home = 11 set-pieces
        # League baseline: 0.23/min × 15 = 3.45 expected
        # Ratio = 11/3.45 ≈ 3.19 (high pressure)
        trends = (
            self._trends_for(10, type_id=StatType.CORNERS,
                             minute_value_pairs=[(15, 1), (30, 1), (45, 4)])
            + self._trends_for(10, type_id=StatType.TOTAL_CROSSES,
                               minute_value_pairs=[(15, 2), (30, 4), (45, 12)])
        )
        base = _make_state(minute=45, home_goals=0)
        state = LiveMatchState(**{**base.__dict__, "trends": trends})
        # Last 15 min (min 30-45): corners delta = 4-1 = 3; crosses delta = 12-4 = 8 → 11
        sp = state.set_piece_intensity("home", window=15)
        assert sp > 2.5  # high pressure

    def test_set_piece_intensity_neutral_without_trends(self):
        from tests.evaluation.live.test_audit_hardening import _make_state
        state = _make_state(minute=45, home_goals=0)
        assert state.set_piece_intensity("home", window=15) == 1.0

    # ── Phase 6 — software completeness for backtest day ──────────────

    def test_stake_by_ci_scales_down_with_wide_interval(self):
        """Phase 6 (#3 follow-up): wide CI = uncertain → stake DOWN.
        With CI=0.10 stake multiplier = max(0.40, 1 - 0.40) = 0.60.
        With CI=0.02 stake multiplier = max(0.40, 1 - 0.08) = 0.92.
        """
        from bip.evaluation.live.match_state import LiveMatchState
        from tests.evaluation.live.test_audit_hardening import _make_state
        # Same edge, two different CI values — verify stake differs.
        state = _make_state(minute=30, home_goals=1)
        # Generous CI (wide) via custom probs.confidences
        narrow_ci_probs = _probs_with_conf(
            market_probs={
                MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
            },
            confidences={MARKET_FULLTIME_RESULT: 0.02},
        )
        wide_ci_probs = _probs_with_conf(
            market_probs={
                MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
            },
            confidences={MARKET_FULLTIME_RESULT: 0.10},
        )
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        # Disable CI gate to isolate the stake-by-CI scaler (the gate
        # would drop the wide-CI pick before stake computation runs).
        narrow = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(narrow_ci_probs, odds, home_team_name="A", away_team_name="B", state=state)
        wide = ValueDetector(
            min_edge_pct=3.0, drop_extreme=False, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(wide_ci_probs, odds, home_team_name="A", away_team_name="B", state=state)
        assert len(narrow) == 1 and len(wide) == 1
        # Wide CI stake should be substantially smaller
        assert wide[0].suggested_stake_pct < narrow[0].suggested_stake_pct - 0.05

    def test_cards_market_emits_with_yellow_cards(self):
        """Phase 6: NUMBER_OF_CARDS market is now predicted.

        With 4 yellows accumulated by minute 60 + adequate engagement,
        cards_total markets should emit.
        """
        from bip.evaluation.live.predictor import (
            MARKET_CARDS_TOTAL_3_5,
            MARKET_CARDS_TOTAL_4_5,
            MARKET_CARDS_TOTAL_5_5,
        )
        state = _make_state(
            minute=60, home_goals=1,
            home_stats={
                StatType.SHOTS_TOTAL: 8, StatType.KEY_PASSES: 4,
                StatType.SHOTS_ON_TARGET: 3, StatType.CORNERS: 4,
                StatType.TACKLES: 12, StatType.INTERCEPTIONS: 8,
                StatType.DUELS_WON: 30,
            },
            away_stats={
                StatType.SHOTS_TOTAL: 6, StatType.KEY_PASSES: 3,
                StatType.SHOTS_ON_TARGET: 2, StatType.CORNERS: 3,
                StatType.TACKLES: 10, StatType.INTERCEPTIONS: 7,
                StatType.DUELS_WON: 28,
            },
            yellow_card_events=[(15, 10, 100), (30, 20, 200), (40, 10, 101), (55, 20, 201)],
        )
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        # All three card lines should emit with reasonable probabilities
        assert MARKET_CARDS_TOTAL_3_5 in probs.by_market
        assert MARKET_CARDS_TOTAL_4_5 in probs.by_market
        assert MARKET_CARDS_TOTAL_5_5 in probs.by_market
        # 4 cards already → over 3.5 should be ~certain, over 5.5 less so
        assert probs.by_market[MARKET_CARDS_TOTAL_3_5]["over"] > 0.95
        assert (
            probs.by_market[MARKET_CARDS_TOTAL_5_5]["over"]
            < probs.by_market[MARKET_CARDS_TOTAL_4_5]["over"]
        )

    def test_cards_market_skipped_below_info_density(self):
        """Same density gate as corners — at min 15 with 1 yellow and
        no other stats, cards should not emit (Poisson rate from 1 sample
        is unreliable)."""
        from bip.evaluation.live.predictor import MARKET_CARDS_TOTAL_3_5
        state = _make_state(
            minute=15,
            yellow_card_events=[(10, 10, 100)],
            home_stats={}, away_stats={},  # no other accumulated stats
        )
        # Density should be below 0.40 (predictor's gate)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        assert MARKET_CARDS_TOTAL_3_5 not in probs.by_market

    def test_trends_consumer_shots_in_window(self):
        """Phase 6 trends-consumer: shots_in_last_window reads cumulative
        Trend records and computes a rolling delta."""
        from bip.sports.football.sportmonks.schemas import Trend
        # Trend records: SHOTS_TOTAL for home_team_id=10
        # at min 50: cum=4; at min 55: cum=7 → 3 shots in last 5 min
        trends = [
            Trend.model_validate({
                "id": 1, "fixture_id": 1, "type_id": StatType.SHOTS_TOTAL,
                "participant_id": 10, "period_id": 2, "minute": 50, "value": 4,
            }),
            Trend.model_validate({
                "id": 2, "fixture_id": 1, "type_id": StatType.SHOTS_TOTAL,
                "participant_id": 10, "period_id": 2, "minute": 55, "value": 7,
            }),
        ]
        from bip.evaluation.live.match_state import LiveMatchState
        # Build state directly with trends
        base = _make_state(minute=55, home_goals=1)
        state = LiveMatchState(
            **{**base.__dict__, "trends": trends}
        )
        assert state.shots_in_last_window("home", window=5) == 3
        assert state.shots_in_last_window("home", window=10) == 7

    def test_trends_consumer_no_data_returns_zero(self):
        """No trends data → returns 0 (graceful degradation)."""
        state = _make_state(minute=55, home_goals=1)
        # trends defaults to []
        assert state.shots_in_last_window("home", window=5) == 0

    def test_cross_cluster_correlation_collapses_dc_under(self):
        """Phase 6 (#5 cross-cluster): DC 1X + OU 2.5 under encode the
        same low-scoring-home-not-loses view. Bundle dedup should collapse
        them to one survivor (cross-cluster).
        """
        from bip.evaluation.live.value_detector import _bundle_dedup_picks
        # Two picks: same fixture, different clusters, but correlated
        dc_pick = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_DOUBLE_CHANCE, selection="1x",
            bookmaker_id=2, bookmaker_odd=1.40, our_probability=0.78,
            fair_odd=1.28, edge_pct=9.2,
            kelly_fraction_full=0.10, suggested_stake_pct=1.0,
        )
        ou_pick = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_OU_25, selection="under",
            bookmaker_id=2, bookmaker_odd=1.85, our_probability=0.58,
            fair_odd=1.72, edge_pct=7.3,
            kelly_fraction_full=0.07, suggested_stake_pct=0.6,
        )
        result = _bundle_dedup_picks([dc_pick, ou_pick])
        # Cross-cluster correlation must collapse to 1 survivor.
        assert len(result) == 1
        # DC pick has higher edge → wins
        assert result[0].market == MARKET_DOUBLE_CHANCE

    def test_cross_cluster_uncorrelated_pairs_both_survive(self):
        """Sister test: picks that are NOT in the correlated pairs set
        both survive cross-cluster pass."""
        from bip.evaluation.live.value_detector import _bundle_dedup_picks
        # FT home + BTTS yes — independent views
        ft_pick = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_FULLTIME_RESULT, selection="home",
            bookmaker_id=2, bookmaker_odd=1.80, our_probability=0.65,
            fair_odd=1.54, edge_pct=17.0,
            kelly_fraction_full=0.10, suggested_stake_pct=1.0,
        )
        btts_pick = LivePick(
            fixture_id=1, minute=40, home_team="A", away_team="B",
            market=MARKET_BTTS, selection="yes",
            bookmaker_id=2, bookmaker_odd=2.10, our_probability=0.55,
            fair_odd=1.82, edge_pct=15.5,
            kelly_fraction_full=0.10, suggested_stake_pct=1.0,
        )
        result = _bundle_dedup_picks([ft_pick, btts_pick])
        # Different clusters AND not in _CORRELATED_PAIRS → both survive
        assert len(result) == 2

    # ── #30 HTFT matcher fix — was bound to _ftr_matcher (incompatible) ─
    def test_htft_matcher_handles_word_labels(self):
        """Bug #30: predictor emits HTFT keys 'home_home', 'home_draw',
        etc. Before the fix, value_detector bound MARKET_HTFT to
        _ftr_matcher which only knew '1','X','2','Home','Draw','Away'.
        Result: HTFT odds NEVER matched → no picks emitted from HTFT
        despite the predictor computing probabilities for it.
        """
        from bip.evaluation.live.value_detector import _htft_matcher

        cases_word = [
            ("Home/Home", "home_home"),
            ("Home/Draw", "home_draw"),
            ("Home/Away", "home_away"),
            ("Draw/Home", "draw_home"),
            ("Draw/Draw", "draw_draw"),
            ("Draw/Away", "draw_away"),
            ("Away/Home", "away_home"),
            ("Away/Draw", "away_draw"),
            ("Away/Away", "away_away"),
        ]
        for label, expected in cases_word:
            o = _odd(market_id=MarketID.HALFTIME_FULLTIME, label=label, value="3.00")
            assert _htft_matcher(o) == expected, f"failed {label}"

    def test_htft_matcher_handles_numeric_labels(self):
        """Sportmonks may emit '1/X/2' style instead of word labels."""
        from bip.evaluation.live.value_detector import _htft_matcher
        cases = [
            ("1/1", "home_home"), ("1/X", "home_draw"), ("1/2", "home_away"),
            ("X/1", "draw_home"), ("X/X", "draw_draw"), ("X/2", "draw_away"),
            ("2/1", "away_home"), ("2/X", "away_draw"), ("2/2", "away_away"),
        ]
        for label, expected in cases:
            o = _odd(market_id=MarketID.HALFTIME_FULLTIME, label=label, value="3.00")
            assert _htft_matcher(o) == expected, f"failed {label}"

    def test_htft_matcher_rejects_garbage(self):
        from bip.evaluation.live.value_detector import _htft_matcher
        for bad in ("", "Home", "Home-Home", "1", "Foo/Bar", "Home/Bar"):
            o = _odd(market_id=MarketID.HALFTIME_FULLTIME, label=bad, value="3.00")
            assert _htft_matcher(o) is None, f"should reject {bad!r}"

    def test_htft_pick_actually_fires_against_odds_now(self):
        """End-to-end: predictor emits HTFT prob → detector matches it
        against an HTFT odd → pick generated. Before the fix, no HTFT
        odd ever matched."""
        state = _make_state(minute=20, home_goals=0)
        # HTFT 9-cell prob, with one cell strongly weighted
        probs = _probs_with_conf(market_probs={
            "htft": {
                "home_home": 0.50, "home_draw": 0.05, "home_away": 0.02,
                "draw_home": 0.10, "draw_draw": 0.15, "draw_away": 0.05,
                "away_home": 0.05, "away_draw": 0.05, "away_away": 0.03,
            },
        })
        odds = [_odd(
            market_id=MarketID.HALFTIME_FULLTIME, label="Home/Home", value="2.50",
        )]
        picks = ValueDetector(
            min_edge_pct=3.0, enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        ).evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        # Edge: 0.50 × 2.50 - 1 = 25% → above 3% min, pick must fire.
        assert len(picks) == 1
        assert picks[0].market == "htft"
        assert picks[0].selection == "home_home"

    # ── #21 league_id filter (allowlist / blocklist parsing) ────────────
    def test_league_filter_cli_parsing(self):
        """Bug #21: league_id was stored on state but never used. Fix:
        watch.py exposes --leagues-allowlist / --leagues-blocklist CLI
        flags. Validate parsing logic locally (don't run async loop).
        """
        # Replicate the inline parser from watch.main()
        def _parse_league_ids(s):
            s = (s or "").strip()
            if not s:
                return None
            try:
                return frozenset(int(x.strip()) for x in s.split(",") if x.strip())
            except ValueError:
                return None

        # Empty / None → no filter
        assert _parse_league_ids("") is None
        assert _parse_league_ids(None) is None
        # Valid CSV
        assert _parse_league_ids("8,564,82") == frozenset({8, 564, 82})
        # Whitespace tolerance
        assert _parse_league_ids("8, 564 , 82") == frozenset({8, 564, 82})
        # Trailing comma
        assert _parse_league_ids("8,564,") == frozenset({8, 564})
        # Invalid → graceful None
        assert _parse_league_ids("8,abc,82") is None

    # ── #20 corners gated by info-density ──────────────────────────────
    def test_corners_market_skipped_below_info_density_threshold(self):
        """Bug #20: previously corners emitted whenever minute >= 10 AND
        corners > 0, with NO info-density check — meaning a single corner
        at min 10 with no other stats produced a Poisson rate from one
        data point. Fix: gate corners on info_density_threshold (0.40).
        """
        # Min 10, 1 corner, no other accumulated stats → density very low.
        state = _make_state(
            minute=10,
            home_stats={StatType.CORNERS: 1},
            away_stats={},
        )
        assert state.informational_density < 0.40
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        # Corners markets must NOT appear
        for line_market in (
            MARKET_CORNERS_TOTAL_8_5, MARKET_CORNERS_TOTAL_9_5,
            MARKET_CORNERS_TOTAL_10_5,
        ):
            assert line_market not in probs.by_market, (
                f"{line_market} emitted at low density — bug #20 regressed."
            )

    def test_corners_market_emits_when_info_density_sufficient(self):
        """Sister test: with adequate stat accumulation at the same
        minute, corners SHOULD emit."""
        state = _make_state(
            minute=30,  # higher minute term
            home_stats={
                StatType.CORNERS: 4, StatType.SHOTS_TOTAL: 6,
                StatType.SHOTS_ON_TARGET: 3, StatType.KEY_PASSES: 4,
            },
            away_stats={
                StatType.CORNERS: 2, StatType.SHOTS_TOTAL: 4,
                StatType.SHOTS_ON_TARGET: 2, StatType.KEY_PASSES: 2,
            },
        )
        assert state.informational_density >= 0.40
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        # At least one corners line must be emitted
        emitted = sum(
            1 for m in (MARKET_CORNERS_TOTAL_8_5, MARKET_CORNERS_TOTAL_9_5,
                        MARKET_CORNERS_TOTAL_10_5)
            if m in probs.by_market
        )
        assert emitted >= 1

    # ── #17 BTTS factors are named constants, not magic numbers ─────────
    def test_btts_first_half_uses_named_constant(self):
        """Bug #17: previously the 0.45/0.55 factors were inline magic
        numbers. Fix: BTTS_FIRST_HALF_FRACTION + BTTS_SECOND_HALF_FRACTION
        are top-level constants with TODO calibration docstrings.
        """
        from bip.evaluation.live.predictor import (
            BTTS_FIRST_HALF_FRACTION,
            BTTS_SECOND_HALF_FRACTION,
        )
        # Constants are reasonable empirical values
        assert 0.30 <= BTTS_FIRST_HALF_FRACTION <= 0.55
        assert 0.45 <= BTTS_SECOND_HALF_FRACTION <= 0.70
        # Verify the predictor uses the constant — pre-HT BTTS-1H
        # output should equal full_yes × BTTS_FIRST_HALF_FRACTION
        state = _make_state(minute=20, home_goals=0)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        from bip.evaluation.live.predictor import MARKET_BTTS_FIRST_HALF
        btts1h = probs.by_market.get(MARKET_BTTS_FIRST_HALF)
        assert btts1h is not None
        # _BALANCED_PRE_MATCH has BTTS yes=60 → 0.6 × 0.45 = 0.27
        expected_yes = 0.60 * BTTS_FIRST_HALF_FRACTION
        # Normalised, the yes share is expected_yes / (expected_yes + (1-expected_yes))
        # = expected_yes since denominator = 1
        assert btts1h["yes"] == pytest.approx(expected_yes, abs=0.01)

    # ── #15 lambda silent default — None instead of 2.7 magic ───────────
    def test_no_ou_market_when_sportmonks_omits_ou_2_5(self):
        """Bug #15: previously _estimate_total_lambda returned 2.7 default
        when Sportmonks didn't emit OU 2.5 — leading every OU market
        (including BTTS-2H and 1H OU) to fabricate λ from a hard-coded
        prior. Fix returns None; callers skip the market.
        """
        # Strip OU 2.5 from sportmonks predictions
        sm = dict(_BALANCED_PRE_MATCH)
        sm.pop(PredictionType.OVER_UNDER_2_5_PROBABILITY, None)
        # Score 1-0 so OU 1.5 is not auto-resolved
        state = _make_state(minute=50, home_goals=1, sportmonks=sm)
        predictor = LiveMatchPredictor()
        probs = predictor.predict(state)
        # _ou_total uses _estimate_total_lambda → must skip OU markets
        # that depend on it. OU 1.5 directly uses Sportmonks's pre-built
        # OU 1.5, but OU 2.5 / 3.5 / first-half OU need the estimator.
        # With OU 2.5 prediction missing, the lambda inversion can't happen
        # for live recompute — these markets must NOT appear.
        assert MARKET_OU_25 not in probs.by_market
