"""Tests for the Telegram alert formatter."""

from __future__ import annotations

import pytest

from bip.evaluation.tournaments.predict.telegram_formatter import (
    _odd,
    _pct,
    _poisson_over,
    format_match_alert,
)
from tests.evaluation.conftest import make_match_prediction


# ── Utility helpers ───────────────────────────────────────────────────────────


def test_pct_formatting() -> None:
    assert _pct(0.423) == "42.3%"
    assert _pct(1.0) == "100.0%"
    assert _pct(0.0) == "0.0%"


def test_odd_formatting() -> None:
    assert _odd(0.5) == "2.00"
    assert _odd(0.333) == "3.00"
    assert _odd(0.0) == "∞"


def test_odd_rounds_to_two_decimals() -> None:
    assert _odd(0.4) == "2.50"


def test_poisson_over_zero() -> None:
    # P(Poisson(2) > 0) = 1 - P(X=0) = 1 - e^{-2} ≈ 0.865
    p = _poisson_over(2.0, 0)
    assert 0.86 < p < 0.87


def test_poisson_over_k_min_monotone() -> None:
    lam = 1.5
    assert _poisson_over(lam, 0) > _poisson_over(lam, 1) > _poisson_over(lam, 2)


# ── format_match_alert ────────────────────────────────────────────────────────


@pytest.fixture
def alert(make_match_prediction) -> str:
    pred = make_match_prediction()
    return format_match_alert(pred)


def test_alert_is_string(alert: str) -> None:
    assert isinstance(alert, str)
    assert len(alert) > 100


def test_alert_contains_team_names(alert: str) -> None:
    assert "Home FC" in alert
    assert "Away FC" in alert


def test_alert_contains_1x2_section(alert: str) -> None:
    assert "RESULTADO" in alert
    assert "Empate" in alert


def test_alert_contains_goals_section(alert: str) -> None:
    assert "GOLES" in alert
    assert "Over 2.5" in alert
    assert "BTTS" in alert


def test_alert_contains_handicap_section(alert: str) -> None:
    assert "HÁNDICAP" in alert
    assert "DNB" in alert


def test_alert_contains_first_half_section(alert: str) -> None:
    assert "PRIMER TIEMPO" in alert
    assert "Over 0.5" in alert


def test_alert_contains_scorers_section(alert: str) -> None:
    assert "P(MARCAR)" in alert


def test_alert_with_corners(make_match_prediction) -> None:
    from bip.evaluation.tournaments.predictors.corners_poisson import (
        CornersPoissonModel,
    )
    from tests.evaluation.conftest import make_blended_rates

    pred = make_match_prediction()
    model = CornersPoissonModel()
    corners = model.predict(
        make_blended_rates(team_id=1, corners=6.0),
        make_blended_rates(team_id=2, corners=5.0),
    )
    text = format_match_alert(pred, corners=corners)
    assert "CORNERS" in text
    assert "Over 9.5" in text
    assert "Más corners" in text


def test_alert_with_warnings(make_match_prediction) -> None:
    pred = make_match_prediction(warnings=("lineup uncertain",))
    text = format_match_alert(pred)
    assert "ADVERTENCIAS" in text
    assert "lineup uncertain" in text


def test_alert_without_corners_no_corners_section(make_match_prediction) -> None:
    pred = make_match_prediction()
    text = format_match_alert(pred, corners=None)
    assert "CORNERS" not in text


def test_odd_appears_in_output(alert: str) -> None:
    # The alert should contain at least one fair-odds value
    assert "odd justa" in alert
