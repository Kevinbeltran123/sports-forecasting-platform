"""Tests for cross_team_predictor — sanity + edge cases."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from bip.evaluation.tournaments.team_style_profiler.bettable_profile import (
    BettableProfile,
)
from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    derive_lambdas,
    predict_markets,
)
from bip.evaluation.tournaments.team_style_profiler.tsv_schema import (
    DistributionStat,
    SubProfile,
)


def _d(mean: float, width: float = 0.4, n: int = 20) -> DistributionStat:
    return DistributionStat(
        mean=mean, ci_low=mean - width / 2, ci_high=mean + width / 2, n=n
    )


def _profile(
    name: str,
    conf: str,
    goals_for: float = 1.8,
    goals_against: float = 1.0,
    btts: float = 0.55,
    over_25: float = 0.55,
    over_35: float = 0.30,
    corners_for: float = 6.0,
    corners_against: float = 4.0,
    yellow: float = 2.0,
    fouls: float = 12.0,
    sub: SubProfile | None = None,
) -> BettableProfile:
    return BettableProfile(
        team_name=name,
        confederation=conf,  # type: ignore[arg-type]
        source="own_tsv",
        goals_for_per_match=_d(goals_for),
        goals_against_per_match=_d(goals_against),
        btts_rate=_d(btts),
        over_25_rate=_d(over_25),
        over_35_rate=_d(over_35),
        corners_for_per_match=_d(corners_for),
        corners_against_per_match=_d(corners_against),
        yellow_cards_per_match=_d(yellow),
        fouls_per_match=_d(fouls),
        mean_total_goals=_d(goals_for + goals_against),
        possession_avg=_d(50.0),
        sub_profile_vs_opponent=sub,
    )


class TestDeriveLambdas:
    def test_symmetric_teams(self) -> None:
        h = _profile("A", "UEFA", goals_for=1.5, goals_against=1.5)
        a = _profile("B", "CONMEBOL", goals_for=1.5, goals_against=1.5)
        lam_h, lam_a = derive_lambdas(h, a)
        # Symmetric: λ_h ~ λ_a ~ 1.5
        assert lam_h == pytest.approx(1.5, abs=0.01)
        assert lam_a == pytest.approx(1.5, abs=0.01)

    def test_asymmetric_strong_home(self) -> None:
        h = _profile("Strong", "UEFA", goals_for=2.5, goals_against=0.8)
        a = _profile("Weak", "AFC", goals_for=0.8, goals_against=2.5)
        lam_h, lam_a = derive_lambdas(h, a)
        # Home λ should be high (strong attack + weak defense): ~2.5
        # Away λ should be low (weak attack + strong defense): ~0.8
        assert lam_h > lam_a
        assert lam_h > 2.0
        assert lam_a < 1.0

    def test_clamping_to_positive(self) -> None:
        # Pathological: 0 goals for / 0 against
        h = _profile("X", "UEFA", goals_for=0.0, goals_against=0.0)
        a = _profile("Y", "UEFA", goals_for=0.0, goals_against=0.0)
        lam_h, lam_a = derive_lambdas(h, a)
        assert lam_h >= 0.05
        assert lam_a >= 0.05


class TestPredictMarkets:
    def test_balanced_match_btts_around_55(self) -> None:
        # Both teams average 1.5 goals for, 1.0 against -> mid-scoring
        h = _profile("H", "UEFA", goals_for=1.5, goals_against=1.0)
        a = _profile("A", "UEFA", goals_for=1.5, goals_against=1.0)
        m = predict_markets(h, a)
        assert 0.40 < m.btts_yes.probability < 0.65
        # btts_yes + btts_no = 1
        assert m.btts_yes.probability + m.btts_no.probability == pytest.approx(1.0)

    def test_strong_vs_weak_lambda_asymmetry(self) -> None:
        strong = _profile("Strong", "CONMEBOL", goals_for=2.5, goals_against=0.6)
        weak = _profile("Weak", "AFC", goals_for=0.8, goals_against=2.5)
        m = predict_markets(strong, weak)
        assert m.lambda_home > m.lambda_away
        # AH home -1.5 (strong wins by 2+) should be plausible
        assert m.ah_home_minus_1_5.probability > 0.30

    def test_over_under_complementarity(self) -> None:
        h = _profile("H", "UEFA")
        a = _profile("A", "UEFA")
        m = predict_markets(h, a)
        assert m.over_2_5.probability + m.under_2_5.probability == pytest.approx(1.0, abs=0.001)

    def test_high_scoring_markers(self) -> None:
        # Both teams 3+ goals avg
        h = _profile("H", "UEFA", goals_for=3.0, goals_against=2.0)
        a = _profile("A", "CONMEBOL", goals_for=3.0, goals_against=2.0)
        m = predict_markets(h, a)
        assert m.over_2_5.probability > 0.70
        assert m.btts_yes.probability > 0.65

    def test_low_scoring_markers(self) -> None:
        h = _profile("H", "UEFA", goals_for=0.8, goals_against=0.5)
        a = _profile("A", "UEFA", goals_for=0.7, goals_against=0.5)
        m = predict_markets(h, a)
        assert m.over_2_5.probability < 0.40
        assert m.btts_yes.probability < 0.50

    def test_corners_emitted_when_data_available(self) -> None:
        h = _profile("H", "UEFA", corners_for=7.0, corners_against=3.5)
        a = _profile("A", "UEFA", corners_for=6.5, corners_against=4.0)
        m = predict_markets(h, a)
        assert m.corners_over_9_5 is not None
        # expected total corners ≈ (7+4)/2 + (6.5+3.5)/2 = 5.5 + 5 = 10.5
        # P(>9) should be moderate
        assert 0.40 < m.corners_over_9_5.probability < 0.85

    def test_corners_skipped_when_data_missing(self) -> None:
        h = _profile("H", "UEFA")
        # Zero out corners_against on home — usability check should fail
        h = replace(
            h,
            corners_against_per_match=DistributionStat(
                mean=0.0, ci_low=0.0, ci_high=0.0, n=0
            ),
        )
        a = _profile("A", "UEFA")
        m = predict_markets(h, a)
        assert m.corners_over_9_5 is None

    def test_cards_emitted(self) -> None:
        h = _profile("H", "UEFA", yellow=2.5)
        a = _profile("A", "CAF", yellow=2.8)
        m = predict_markets(h, a)
        assert m.cards_over_4_5 is not None
        # λ_total ~ 5.3 cards, P(>4) very high
        assert m.cards_over_4_5.probability > 0.55

    def test_ah_emitted_for_all_lines(self) -> None:
        h = _profile("H", "UEFA")
        a = _profile("A", "UEFA")
        m = predict_markets(h, a)
        assert m.ah_home_minus_0_5 is not None
        assert m.ah_home_minus_1_5 is not None
        assert m.ah_away_minus_0_5 is not None
        assert m.ah_away_minus_1_5 is not None

    def test_no_1x2_emitted(self) -> None:
        # Operator decision: TSP does NOT emit 1X2.
        h = _profile("H", "UEFA")
        a = _profile("A", "UEFA")
        m = predict_markets(h, a)
        markets = [mp.market for mp in m.all_markets]
        # No market should start with "1X2" or be 1X2-specific
        assert not any("1X2" in mkt for mkt in markets)
        assert not any(mkt in {"home_win", "draw", "away_win"} for mkt in markets)

    def test_sub_profile_pulls_lambda(self) -> None:
        # Argentina's general goals_for=2.0 (CI width 0.4);
        # against AFC sub-profile goals_for=3.0 with CI width 0.2 (narrower
        # -> twice the inverse-CI weight of general -> pulls mean toward sub).
        sub = SubProfile(
            vs_confederation="AFC",
            n_matches_vs_conf=5,
            goals_for_per_match=DistributionStat(mean=3.0, ci_low=2.9, ci_high=3.1, n=5),
            goals_against_per_match=_d(0.8),
            btts_rate=_d(0.5),
            over_25_rate=_d(0.6),
            corners_for_per_match=_d(6.0),
            yellow_cards_per_match=_d(2.0),
        )
        argentina = _profile(
            "Argentina", "CONMEBOL",
            goals_for=2.0, goals_against=0.9, sub=sub,
        )
        # Argentina vs Japan WITHOUT sub-profile (baseline)
        argentina_no_sub = _profile(
            "Argentina2", "CONMEBOL",
            goals_for=2.0, goals_against=0.9, sub=None,
        )
        japan = _profile("Japan", "AFC", goals_for=1.0, goals_against=1.5)
        lam_h_with_sub, _ = derive_lambdas(argentina, japan)
        lam_h_no_sub, _ = derive_lambdas(argentina_no_sub, japan)
        # Sub-profile (n=5, narrow CI) should pull λ_h higher than baseline
        assert lam_h_with_sub > lam_h_no_sub
        # And the lift should be material (>0.2) given sub's evidence
        assert (lam_h_with_sub - lam_h_no_sub) > 0.2

    def test_confidence_propagates(self) -> None:
        # Wide CI inputs -> LOW confidence
        h = _profile("H", "UEFA")
        h = replace(
            h,
            goals_for_per_match=DistributionStat(mean=1.5, ci_low=0.2, ci_high=2.8, n=10),
        )
        a = _profile("A", "UEFA")
        m = predict_markets(h, a)
        assert m.btts_yes.confidence == "LOW"
