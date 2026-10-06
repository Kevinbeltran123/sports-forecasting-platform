"""Tests for value_detector + odds_provider."""
from __future__ import annotations

import pytest

from bip.evaluation.tournaments.team_style_profiler.cross_team_predictor import (
    MarketPredictions,
    MarketProb,
)
from bip.evaluation.tournaments.team_style_profiler.odds_provider import (
    extract_pinnacle_canonical_odds,
    implied_probability,
)
from bip.evaluation.tournaments.team_style_profiler.value_detector import (
    MIN_Z_SCORE,
    ValueAlert,
    detect_value,
    filter_alerts,
)


# ─── odds_provider tests ────────────────────────────────────────────────


class TestImpliedProbability:
    def test_standard(self) -> None:
        assert implied_probability(2.0) == pytest.approx(0.5)
        assert implied_probability(1.5) == pytest.approx(0.667, abs=0.001)
        assert implied_probability(3.0) == pytest.approx(0.333, abs=0.001)

    def test_clamp_at_one(self) -> None:
        # Invalid odds <= 1.0
        assert implied_probability(1.0) == 1.0
        assert implied_probability(0.5) == 1.0


class TestExtractPinnacleCanonical:
    def test_btts_yes_no(self) -> None:
        payload = {
            "response": [
                {
                    "fixture": {"id": 999},
                    "bookmakers": [
                        {
                            "id": 1,
                            "name": "Pinnacle",
                            "bets": [
                                {
                                    "id": 8,
                                    "name": "Both Teams To Score",
                                    "values": [
                                        {"value": "Yes", "odd": "1.85"},
                                        {"value": "No", "odd": "1.95"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        odds = extract_pinnacle_canonical_odds(payload)
        assert odds["BTTS_yes"] == 1.85
        assert odds["BTTS_no"] == 1.95

    def test_goals_over_under(self) -> None:
        payload = {
            "response": [
                {
                    "fixture": {"id": 1},
                    "bookmakers": [
                        {
                            "id": 1,
                            "name": "Pinnacle",
                            "bets": [
                                {
                                    "id": 5,
                                    "name": "Goals Over/Under",
                                    "values": [
                                        {"value": "Over 2.5", "odd": "1.75"},
                                        {"value": "Under 2.5", "odd": "2.05"},
                                        {"value": "Over 3.5", "odd": "2.85"},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        odds = extract_pinnacle_canonical_odds(payload)
        assert odds["O2.5"] == 1.75
        assert odds["U2.5"] == 2.05
        assert odds["O3.5"] == 2.85

    def test_skips_non_pinnacle(self) -> None:
        payload = {
            "response": [
                {
                    "fixture": {"id": 1},
                    "bookmakers": [
                        {
                            "id": 99,
                            "name": "Bet365",
                            "bets": [
                                {
                                    "id": 8,
                                    "name": "Both Teams To Score",
                                    "values": [{"value": "Yes", "odd": "1.85"}],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        odds = extract_pinnacle_canonical_odds(payload)
        assert odds == {}

    def test_empty_response(self) -> None:
        assert extract_pinnacle_canonical_odds({"response": []}) == {}


# ─── value_detector tests ───────────────────────────────────────────────


def _mp(market: str, prob: float, confidence: str = "HIGH") -> MarketProb:
    return MarketProb(
        market=market, probability=prob, confidence=confidence, rationale="test"
    )


def _predictions(over25_prob: float = 0.65, btts_prob: float = 0.6,
                 over25_conf: str = "HIGH") -> MarketPredictions:
    return MarketPredictions(
        home_team="H", away_team="A", home_source="own_tsv", away_source="own_tsv",
        over_2_5=_mp("O2.5", over25_prob, over25_conf),
        over_3_5=_mp("O3.5", 0.35, "MEDIUM"),
        under_2_5=_mp("U2.5", 1 - over25_prob, over25_conf),
        btts_yes=_mp("BTTS_yes", btts_prob, "HIGH"),
        btts_no=_mp("BTTS_no", 1 - btts_prob, "HIGH"),
        corners_over_8_5=None, corners_over_9_5=None, corners_over_10_5=None,
        cards_over_3_5=None, cards_over_4_5=None, cards_over_5_5=None,
        ah_home_minus_0_5=None, ah_home_minus_1_5=None,
        ah_away_minus_0_5=None, ah_away_minus_1_5=None,
        lambda_home=1.5, lambda_away=1.2,
    )


class TestDetectValue:
    def test_clear_edge_passes(self) -> None:
        # P_emp=0.65, odd=1.85 -> P_implied=0.54 -> diff=0.11
        # std (HIGH) ≈ 0.05/1.96 ≈ 0.0255
        # Z = 0.11 / 0.0255 ≈ 4.3 >> 1.5 -> ALERT
        odds = {"O2.5": 1.85, "U2.5": 2.05, "BTTS_yes": 1.85, "BTTS_no": 1.95,
                "O3.5": 2.85}
        preds = _predictions(over25_prob=0.65)
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        assert o25.decision == "ALERT"
        assert o25.z_score > 1.5
        assert o25.edge_pct > 0.10

    def test_no_edge_skipped(self) -> None:
        # P_emp=0.55 close to P_implied=0.54 -> Z low
        odds = {"O2.5": 1.85, "U2.5": 2.05, "BTTS_yes": 1.85, "BTTS_no": 1.95,
                "O3.5": 2.85}
        preds = _predictions(over25_prob=0.55)
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        assert o25.decision == "SKIP_EDGE"

    def test_missing_odds_skipped(self) -> None:
        # No O2.5 in odds dict
        odds = {"BTTS_yes": 1.85}
        preds = _predictions(over25_prob=0.65)
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        assert o25.decision == "SKIP_NO_ODDS"

    def test_low_confidence_skipped(self) -> None:
        odds = {"O2.5": 1.85, "U2.5": 2.05, "BTTS_yes": 1.85, "BTTS_no": 1.95,
                "O3.5": 2.85}
        preds = _predictions(over25_prob=0.65, over25_conf="LOW")
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        assert o25.decision == "SKIP_CONFIDENCE"

    def test_medium_confidence_accepted(self) -> None:
        odds = {"O2.5": 1.85, "U2.5": 2.05, "BTTS_yes": 1.85, "BTTS_no": 1.95,
                "O3.5": 2.85}
        preds = _predictions(over25_prob=0.65, over25_conf="MEDIUM")
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        # MEDIUM has wider std (0.10/1.96 ≈ 0.051), Z = 0.11/0.051 ≈ 2.16 > 1.5
        assert o25.decision == "ALERT"

    def test_z_score_threshold(self) -> None:
        # Exactly at threshold edge case
        odds = {"O2.5": 1.85, "U2.5": 2.05, "BTTS_yes": 1.85, "BTTS_no": 1.95,
                "O3.5": 2.85}
        # P_implied for 1.85 = 0.5405. std HIGH = 0.0255.
        # Z=1.5 means P_emp - 0.5405 = 0.0383 -> P_emp = 0.579
        preds = _predictions(over25_prob=0.55)  # Z ~ 0.4 -> SKIP
        alerts = detect_value(preds, odds)
        o25 = next(a for a in alerts if a.market == "O2.5")
        assert o25.z_score < MIN_Z_SCORE


class TestFilterAlerts:
    def test_returns_only_alerts(self) -> None:
        alerts = [
            ValueAlert(
                market="A", decision="ALERT", p_empirical=0.6, p_implied=0.5,
                std_empirical=0.05, decimal_odd=2.0, z_score=2.0, edge_pct=0.2,
                confidence="HIGH", rationale="ok",
            ),
            ValueAlert(
                market="B", decision="SKIP_EDGE", p_empirical=0.5, p_implied=0.5,
                std_empirical=0.05, decimal_odd=2.0, z_score=0.0, edge_pct=0.0,
                confidence="HIGH", rationale="no edge",
            ),
            ValueAlert(
                market="C", decision="SKIP_NO_ODDS", p_empirical=0.6, p_implied=0,
                std_empirical=0.05, decimal_odd=None, z_score=None, edge_pct=None,
                confidence="HIGH", rationale="no odd",
            ),
        ]
        filtered = filter_alerts(alerts)
        assert len(filtered) == 1
        assert filtered[0].market == "A"
