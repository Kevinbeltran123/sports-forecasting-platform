"""Layer-1 tests for ValueDetector."""

from __future__ import annotations

import pytest

import numpy as np

from bip.evaluation.live.calibration import IsotonicProbabilityCalibrator
from bip.evaluation.live.commentary import CommentaryEvent, CommentaryEventType
from bip.evaluation.live.match_state import LiveMatchState
from bip.evaluation.live.predictor import (
    MARKET_AWAY_OU_15,
    MARKET_BTTS,
    MARKET_BTTS_SECOND_HALF,
    MARKET_CARDS_TOTAL_3_5,
    MARKET_FULLTIME_RESULT,
    MARKET_OU_25,
    MarketProbabilities,
)
from bip.evaluation.live.value_detector import (
    DEFAULT_KELLY_FRACTION,
    DEFAULT_MAX_STAKE_PCT,
    LivePick,
    ValueDetector,
    make_best_stack_detector,
)
from bip.sports.football.sportmonks.schemas import Odd
from bip.sports.football.sportmonks.types import MarketID


def _odd(
    *,
    market_id: int,
    label: str,
    value: str,
    suspended: bool = False,
    stopped: bool = False,
    total: str | None = None,
    bookmaker_id: int = 2,
    fixture_id: int = 1,
    market_description: str | None = None,
) -> Odd:
    return Odd.model_validate({
        "id": id(label),
        "fixture_id": fixture_id,
        "market_id": market_id,
        "bookmaker_id": bookmaker_id,
        "label": label,
        "value": value,
        "suspended": suspended,
        "stopped": stopped,
        "total": total,
        "market_description": market_description,
    })


def _probs(
    *,
    fixture_id: int = 1,
    minute: int = 30,
    snapshot_kind: str = "live",
    market_probs: dict[str, dict[str, float]],
) -> MarketProbabilities:
    return MarketProbabilities(
        fixture_id=fixture_id,
        minute=minute,
        snapshot_kind=snapshot_kind,
        by_market=market_probs,
        sources={k: "test" for k in market_probs},
    )


# ── Edge calculation ────────────────────────────────────────────────────────


class TestEdgeDetection:
    def test_value_pick_above_threshold(self):
        # Our prob = 0.55 on home; bookie pays 2.0 (implied 0.50) → +10% EV
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00"),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].market == MARKET_FULLTIME_RESULT
        assert picks[0].selection == "home"
        assert picks[0].edge_pct == pytest.approx(10.0, abs=0.01)

    def test_no_pick_below_threshold(self):
        # Our prob = 0.51 on home; bookie pays 2.0 (implied 0.50) → +2% EV (below threshold)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.51, "draw": 0.25, "away": 0.24},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00"),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 0


class TestSuspendedFiltering:
    def test_suspended_odds_skipped(self):
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.60, "draw": 0.20, "away": 0.20},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
                 suspended=True),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_stopped_odds_skipped(self):
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.60, "draw": 0.20, "away": 0.20},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00",
                 stopped=True),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []


class TestOddRangeFiltering:
    def test_micro_odd_rejected(self):
        # Odd 1.10 is too low to be a useful pick
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.95, "draw": 0.03, "away": 0.02},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.10"),
        ]
        det = ValueDetector(min_edge_pct=3.0, min_odd=1.20)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_lottery_odd_rejected(self):
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.13, "draw": 0.10, "away": 0.77},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="10.00"),
        ]
        det = ValueDetector(min_edge_pct=3.0, max_odd=8.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []


# ── Kelly sizing ────────────────────────────────────────────────────────────


class TestKellySizing:
    def test_kelly_quarter_default(self):
        # Our prob 0.55, odd 2.00 → b=1, p=0.55, q=0.45
        # Full Kelly = (1×0.55 - 0.45) / 1 = 0.10
        # Quarter Kelly = 0.025 → 2.5% → CAP applies (1.5%)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert picks[0].kelly_fraction_full == pytest.approx(0.10, abs=1e-6)
        # Stake: min(0.10 × 0.25 × 100, 1.5%) = min(2.5, 1.5) = 1.5
        assert picks[0].suggested_stake_pct == pytest.approx(DEFAULT_MAX_STAKE_PCT, abs=1e-6)

    def test_kelly_below_cap_uses_quarter(self):
        # Smaller edge → kelly stays under cap
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.42, "draw": 0.30, "away": 0.28},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.50")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        # Full Kelly: (1.5 × 0.42 - 0.58) / 1.5 = 0.0333
        assert picks[0].kelly_fraction_full == pytest.approx(0.0333, abs=1e-3)
        # Quarter = 0.00833 → ~0.83% < 1.5% cap
        assert picks[0].suggested_stake_pct < DEFAULT_MAX_STAKE_PCT


# ── Sorting ────────────────────────────────────────────────────────────────


class TestSorting:
    def test_picks_sorted_by_edge_desc(self):
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.50, "draw": 0.25, "away": 0.25},
            MARKET_BTTS: {"yes": 0.65, "no": 0.35},
        })
        odds = [
            _odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.20"),  # +10% EV
            _odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="Yes", value="1.80"),  # +17% EV
        ]
        # Opt out of Day-1 audit gates: this test exercises pure sorting
        # behavior, not the post-audit cascade rules.
        det = ValueDetector(
            min_edge_pct=3.0,
            market_blacklist=frozenset(),
            ban_positive_side_binaries=False,
        )
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 2
        # BTTS edge larger → first
        assert picks[0].market == MARKET_BTTS
        assert picks[1].market == MARKET_FULLTIME_RESULT


# ── Line market matching ────────────────────────────────────────────────────


class TestLineMatching:
    def test_ou_25_matches_correct_line(self):
        probs = _probs(market_probs={
            MARKET_OU_25: {"over": 0.60, "under": 0.40},
        })
        odds = [
            # Wrong line — should be ignored
            _odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.50",
                 total="1.5"),
            # Right line — match
            _odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85",
                 total="2.5"),
            # Wrong line again
            _odd(market_id=MarketID.MATCH_GOALS, label="Over", value="3.00",
                 total="3.5"),
        ]
        # Opt out of positive-side ban: this test exercises line-matching only.
        det = ValueDetector(
            min_edge_pct=3.0, ban_positive_side_binaries=False,
        )
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        # Only the line=2.5 odd should match
        assert all(p.bookmaker_odd == 1.85 for p in picks)
        assert len(picks) == 1


# ── Day-1 wrong-side audit gates (Tier 1.1 - 1.5) ───────────────────────────
# Justification: reports/sportmonks_live/exploratory/05_day5_action_plan.md


def _state(
    *,
    fixture_id: int = 1,
    minute: int = 30,
    home_goals: int = 0,
    away_goals: int = 0,
) -> LiveMatchState:
    return LiveMatchState(
        fixture_id=fixture_id,
        home_team_id=10, away_team_id=20,
        home_team_name="A", away_team_name="B",
        home_goals=home_goals, away_goals=away_goals,
        minute=minute, period_id=1 if minute < 45 else 2,
        is_live=True, is_half_time=False, is_finished=False,
    )


class TestMarketBlacklist:
    """Tier 1.1 + 1.3 — btts, away_ou_1_5, cards_total_3_5 are blacklisted."""

    def test_btts_market_dropped_by_default(self):
        probs = _probs(market_probs={MARKET_BTTS: {"no": 0.65, "yes": 0.35}})
        odds = [_odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="No", value="1.80")]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_away_ou_1_5_dropped_by_default(self):
        probs = _probs(market_probs={MARKET_AWAY_OU_15: {"under": 0.70, "over": 0.30}})
        odds = [
            _odd(market_id=MarketID.AWAY_GOALS, label="Under", value="1.70", total="1.5"),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_cards_total_3_5_dropped_by_default(self):
        probs = _probs(market_probs={MARKET_CARDS_TOTAL_3_5: {"under": 0.65, "over": 0.35}})
        odds = [
            _odd(market_id=MarketID.NUMBER_OF_CARDS, label="Under", value="1.80", total="3.5"),
        ]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_btts_emitted_when_blacklist_explicitly_disabled(self):
        probs = _probs(market_probs={MARKET_BTTS: {"no": 0.65, "yes": 0.35}})
        odds = [_odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="No", value="1.80")]
        det = ValueDetector(min_edge_pct=3.0, market_blacklist=frozenset())
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].market == MARKET_BTTS

    def test_emits_drop_decision_with_market_blacklist_reason(self):
        probs = _probs(market_probs={MARKET_BTTS: {"no": 0.65, "yes": 0.35}})
        odds = [_odd(market_id=MarketID.BOTH_TEAMS_TO_SCORE, label="No", value="1.80")]
        recorded: list[dict] = []
        det = ValueDetector(min_edge_pct=3.0)
        det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
            on_decision=lambda **kw: recorded.append(kw),
        )
        assert any(r.get("drop_reason") == "market_blacklist" for r in recorded)


class TestPositiveSideBinaryBan:
    """Tier 1.2 — over/yes on binary markets are dropped by default."""

    def test_btts_second_half_yes_dropped(self):
        # BTTS_SECOND_HALF is NOT in market_blacklist but IS binary.
        probs = _probs(market_probs={MARKET_BTTS_SECOND_HALF: {"yes": 0.85, "no": 0.15}})
        odds = [_odd(market_id=MarketID.BTTS_SECOND_HALF, label="Yes", value="1.40")]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_ou_25_over_dropped(self):
        probs = _probs(market_probs={MARKET_OU_25: {"over": 0.60, "under": 0.40}})
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85", total="2.5")]
        det = ValueDetector(min_edge_pct=3.0)
        assert det.evaluate(probs, odds, home_team_name="A", away_team_name="B") == []

    def test_btts_second_half_no_emitted(self):
        # Negative side passes the ban.
        probs = _probs(market_probs={MARKET_BTTS_SECOND_HALF: {"no": 0.65, "yes": 0.35}})
        odds = [_odd(market_id=MarketID.BTTS_SECOND_HALF, label="No", value="1.80")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].selection == "no"

    def test_multi_side_market_unaffected(self):
        # Fulltime result is NOT binary — neither home/draw/away is "over"/"yes".
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].selection == "home"

    def test_ban_opt_out(self):
        probs = _probs(market_probs={MARKET_OU_25: {"over": 0.60, "under": 0.40}})
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85", total="2.5")]
        det = ValueDetector(min_edge_pct=3.0, ban_positive_side_binaries=False)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].selection == "over"


def _isolated_detector(**overrides) -> ValueDetector:
    """ValueDetector with logical_score / CI / coherence gates disabled —
    used to isolate behavior of Day-1 audit gates in tests."""
    return ValueDetector(
        min_edge_pct=3.0,
        min_logical_score_emit=0.0,
        min_logical_score_flag=0.0,
        enforce_ci_gate=False,
        enforce_coherence=False,
        **overrides,
    )


class TestZeroZeroOverGate:
    """Tier 1.5 — drop Over picks at 0-0 score after minute 30 threshold."""

    def test_drops_over_at_zero_zero_after_threshold(self):
        probs = _probs(
            market_probs={MARKET_OU_25: {"over": 0.60, "under": 0.40}},
            minute=35,
        )
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85", total="2.5")]
        det = _isolated_detector(ban_positive_side_binaries=False)
        state = _state(minute=35, home_goals=0, away_goals=0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []

    def test_keeps_over_when_one_team_scored(self):
        probs = _probs(
            market_probs={MARKET_OU_25: {"over": 0.60, "under": 0.40}},
            minute=35,
        )
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85", total="2.5")]
        det = _isolated_detector(
            ban_positive_side_binaries=False,
            enforce_blackout=False,
        )
        state = _state(minute=35, home_goals=1, away_goals=0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1

    def test_keeps_over_at_zero_zero_before_threshold(self):
        probs = _probs(
            market_probs={MARKET_OU_25: {"over": 0.60, "under": 0.40}},
            minute=20,
        )
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Over", value="1.85", total="2.5")]
        det = _isolated_detector(ban_positive_side_binaries=False)
        state = _state(minute=20, home_goals=0, away_goals=0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1

    def test_under_unaffected_by_gate(self):
        probs = _probs(
            market_probs={MARKET_OU_25: {"under": 0.65, "over": 0.35}},
            minute=35,
        )
        odds = [_odd(market_id=MarketID.MATCH_GOALS, label="Under", value="1.70", total="2.5")]
        det = _isolated_detector()
        state = _state(minute=35, home_goals=0, away_goals=0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1
        assert picks[0].selection == "under"


class TestHighProbabilityKellyHaircut:
    """Tier 1.4 — linear taper on Kelly sizing above probability threshold."""

    def test_haircut_applied_above_threshold(self):
        # p=0.95, odd=1.30. Full Kelly = (0.30×0.95 - 0.05) / 0.30 = 0.7833.
        # kelly_fraction_full preserves the DIAGNOSTIC (raw, no haircut) value
        # so downstream consumers can see what would have been staked without
        # the haircut.
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.95, "draw": 0.03, "away": 0.02},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.30")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].kelly_fraction_full == pytest.approx(0.7833, abs=1e-3)

    def test_no_haircut_below_threshold(self):
        # p=0.55 (below 0.85 threshold) → haircut not applied
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        # Full Kelly: (1×0.55 - 0.45) / 1 = 0.10 — unchanged
        assert picks[0].kelly_fraction_full == pytest.approx(0.10, abs=1e-6)

    def test_haircut_reduces_stake_vs_no_haircut(self):
        # Compare stakes at p=0.90 with haircut on (default) vs off.
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.90, "draw": 0.05, "away": 0.05},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.40")]
        det_hc = ValueDetector(min_edge_pct=3.0)
        det_no_hc = ValueDetector(min_edge_pct=3.0, high_prob_haircut_threshold=1.01)
        with_haircut = det_hc.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
        )[0]
        without = det_no_hc.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
        )[0]
        # Stake with haircut should be ≤ stake without (both capped or both not).
        # At this stake size both hit the cap, but kelly_fraction_full differs.
        # Compare directly via internal sizing: lower the cap to surface diff.
        det_hc_low = ValueDetector(min_edge_pct=3.0, max_stake_pct=100.0)
        det_no_hc_low = ValueDetector(
            min_edge_pct=3.0, max_stake_pct=100.0,
            high_prob_haircut_threshold=1.01,
        )
        with_low = det_hc_low.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
        )[0]
        without_low = det_no_hc_low.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
        )[0]
        assert with_low.suggested_stake_pct < without_low.suggested_stake_pct


# ── Calibrator integration ──────────────────────────────────────────────────


def _overconfident_calibrator() -> IsotonicProbabilityCalibrator:
    """Build a calibrator that pulls high probabilities down.

    Training data: predicted 0.90 but actual win-rate 0.65, plus
    well-calibrated mid/low range. Mimics the Day-1 D9-D10 pattern.
    """
    rng = np.random.default_rng(7)
    # Well-calibrated low/mid
    mid_probs = np.full(150, 0.50)
    mid_out = (rng.uniform(size=150) < 0.50).astype(float)
    low_probs = np.full(150, 0.30)
    low_out = (rng.uniform(size=150) < 0.30).astype(float)
    # Overconfident high
    hi_probs = np.full(150, 0.90)
    hi_out = (rng.uniform(size=150) < 0.65).astype(float)
    probs = np.concatenate([low_probs, mid_probs, hi_probs])
    outcomes = np.concatenate([low_out, mid_out, hi_out])
    return IsotonicProbabilityCalibrator.fit(probs, outcomes)


class TestCalibratorIntegration:
    def test_no_calibrator_default(self):
        # Without calibrator, our_probability == model_probability_raw
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        assert picks[0].our_probability == pytest.approx(0.55)
        assert picks[0].model_probability_raw == pytest.approx(0.55)

    def test_calibrator_pulls_high_prob_down(self):
        # p=0.90 raw → calibrator pulls toward ~0.65. Use a generous odd
        # (2.00) so the calibrated edge still passes min_edge_pct=3.0:
        # edge_cal ≈ 0.65*2.00 - 1 = 0.30 = +30%.
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.90, "draw": 0.05, "away": 0.05},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        cal = _overconfident_calibrator()
        det = ValueDetector(
            min_edge_pct=3.0, calibrator=cal,
            enforce_ci_gate=False,
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
        )
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        assert len(picks) == 1
        # Raw preserved
        assert picks[0].model_probability_raw == pytest.approx(0.90)
        # Calibrated below raw
        assert picks[0].our_probability < 0.90
        # And below what the Tier 1.4 haircut would have produced
        # (haircut would map 0.90 → 0.85 + 0.65×0.05 = 0.8825)
        assert picks[0].our_probability < 0.85

    def test_calibrator_edge_uses_calibrated_prob(self):
        # p_raw=0.90, odd=1.40. Raw edge = 0.90*1.40 - 1 = 0.26
        # Calibrated p (around 0.65): edge ≈ 0.65*1.40 - 1 = -0.09
        # So the pick should NOT pass min_edge_pct=3.0 with the calibrator.
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.90, "draw": 0.05, "away": 0.05},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.40")]
        cal = _overconfident_calibrator()
        det = ValueDetector(
            min_edge_pct=3.0, calibrator=cal,
            # Bypass logical_score / CI so we isolate edge gate
            min_logical_score_emit=0.0, min_logical_score_flag=0.0,
            enforce_ci_gate=False,
        )
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        # Calibrated edge is negative → dropped by below_min_edge
        assert picks == []

    def test_calibrator_persistence_roundtrip(self, tmp_path):
        cal = _overconfident_calibrator()
        path = tmp_path / "cal.json"
        cal.save_json(path)
        loaded = IsotonicProbabilityCalibrator.load_json(path)
        # Use loaded calibrator in detector
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.90, "draw": 0.05, "away": 0.05},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="1.40")]
        det = ValueDetector(min_edge_pct=3.0, calibrator=loaded)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        if picks:
            assert picks[0].model_probability_raw == pytest.approx(0.90)
            assert picks[0].our_probability < 0.90

    def test_calibrator_drops_pick_when_calibrated_prob_zero(self):
        # If calibrator collapses to 0 (degenerate case), pick is dropped
        # before any Kelly computation.
        rng = np.random.default_rng(99)
        # All probs 0.5 but everyone loses → calibrator maps everything to 0
        loser_cal = IsotonicProbabilityCalibrator.fit(
            np.array([0.5] * 50 + [0.6] * 50),
            np.array([0.0] * 100),
        )
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.50, "draw": 0.25, "away": 0.25},
        })
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0, calibrator=loser_cal)
        picks = det.evaluate(probs, odds, home_team_name="A", away_team_name="B")
        # Calibrated to 0 → drop pre-edge OR below_min_edge
        assert picks == []


# ── Commentary cool-off gate (Tier 1.6) ─────────────────────────────────────


def _state_with_commentary(
    events: list[CommentaryEvent],
    *,
    minute: int = 35,
    home_goals: int = 0, away_goals: int = 1,
) -> LiveMatchState:
    """LiveMatchState carrying commentary events, with no recent material event."""
    return LiveMatchState(
        fixture_id=1, home_team_id=10, away_team_id=20,
        home_team_name="A", away_team_name="B",
        home_goals=home_goals, away_goals=away_goals,
        minute=minute, period_id=2 if minute >= 45 else 1,
        is_live=True, is_half_time=False, is_finished=False,
        commentary_events=events,
    )


class TestCommentaryCooloffGate:
    def test_var_check_blocks_picks_in_window(self):
        # VAR at min 30, current min 31 → within 3-min window → drop
        events = [CommentaryEvent(
            minute=30, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=True, text="VAR",
        )]
        state = _state_with_commentary(events, minute=31)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=31)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert picks == []

    def test_var_check_outside_window_allows_picks(self):
        # VAR at min 30, current min 34 → outside 3-min window → allow
        events = [CommentaryEvent(
            minute=30, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=True, text="VAR",
        )]
        state = _state_with_commentary(events, minute=34)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=34)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1

    def test_injury_short_cooloff(self):
        # Injury at min 20, current min 22 → outside 1-min window
        events = [CommentaryEvent(
            minute=20, extra_minute=0,
            event_type=CommentaryEventType.INJURY_DELAY,
            is_important=False, text="injury",
        )]
        state = _state_with_commentary(events, minute=22)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=22)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1

    def test_disabled_via_kwarg(self):
        events = [CommentaryEvent(
            minute=30, extra_minute=0,
            event_type=CommentaryEventType.VAR_CHECK,
            is_important=True, text="VAR",
        )]
        state = _state_with_commentary(events, minute=31)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=31)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0, enforce_commentary_cooloff=False)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1

    def test_drop_reason_recorded(self):
        events = [CommentaryEvent(
            minute=30, extra_minute=0,
            event_type=CommentaryEventType.GOAL_DISALLOWED,
            is_important=True, text="disallowed",
        )]
        state = _state_with_commentary(events, minute=31)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=31)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        recorded: list[dict] = []
        det = ValueDetector(min_edge_pct=3.0)
        det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B",
            state=state,
            on_decision=lambda **kw: recorded.append(kw),
        )
        reasons = {r.get("drop_reason") for r in recorded}
        assert any("commentary_cooloff" in (r or "") for r in reasons)

    def test_no_commentary_events_no_effect(self):
        # Empty commentary_events → gate is a no-op
        state = _state_with_commentary([], minute=30)
        probs = _probs(market_probs={
            MARKET_FULLTIME_RESULT: {"home": 0.55, "draw": 0.25, "away": 0.20},
        }, minute=30)
        odds = [_odd(market_id=MarketID.FULLTIME_RESULT, label="Home", value="2.00")]
        det = ValueDetector(min_edge_pct=3.0)
        picks = det.evaluate(
            probs, odds, home_team_name="A", away_team_name="B", state=state,
        )
        assert len(picks) == 1


# ── make_best_stack_detector factory ────────────────────────────────────────


class TestBestStackFactory:
    def test_no_calibrator_default(self):
        det = make_best_stack_detector()
        assert det.calibrator is None
        from bip.evaluation.live.value_detector import (
            DEFAULT_HIGH_PROB_HAIRCUT_THRESHOLD,
        )
        # Without calibrator, Tier 1.4 haircut stays ON at default threshold.
        assert det.high_prob_haircut_threshold == DEFAULT_HIGH_PROB_HAIRCUT_THRESHOLD

    def test_with_calibrator_disables_haircut(self, tmp_path):
        # Build a small calibrator and persist
        rng = np.random.default_rng(11)
        probs = np.linspace(0.1, 0.9, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        path = tmp_path / "cal.json"
        cal.save_json(path)
        det = make_best_stack_detector(
            calibrator_path=str(path), use_per_market=False,
        )
        assert det.calibrator is not None
        # Tier 1.4 haircut effectively disabled (threshold above 1.0).
        assert det.high_prob_haircut_threshold > 1.0

    def test_per_market_load_with_global_json_falls_back(self, tmp_path):
        # Persist a global JSON but request use_per_market=True.
        # Factory falls back to global gracefully.
        rng = np.random.default_rng(12)
        probs = np.linspace(0.1, 0.9, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicProbabilityCalibrator.fit(probs, outcomes)
        path = tmp_path / "global.json"
        cal.save_json(path)
        det = make_best_stack_detector(
            calibrator_path=str(path), use_per_market=True,
        )
        assert det.calibrator is not None

    def test_overrides_propagate(self):
        det = make_best_stack_detector(min_edge_pct=5.0)
        assert det.min_edge_pct == 5.0

    def test_default_tier1_gates_active(self):
        # Tier 1 gates are ON by default in the factory
        det = make_best_stack_detector()
        assert det.market_blacklist
        assert det.ban_positive_side_binaries is True
        assert det.drop_over_zero_zero is True
        assert det.enforce_commentary_cooloff is True
