"""Unit tests for the MarketKey StrEnum (G-MAINT-05).

RED-first TDD: these tests assert the contract before MarketKey exists in
bip.core.types. They are expected to fail with ImportError on the first
pytest run.

Contract under test:
  - StrEnum compatibility (MarketKey.ONEXTWO == "onextwo").
  - 5 canonical members matching markets.yaml keys.
  - from_str: case-insensitive, whitespace-insensitive, alias-aware,
    raises ValueError on unknown.
  - to_odds_api: maps to The Odds API market key (CORNERS -> "" sentinel).
  - to_threshold_attr: bridges to ModelParams attribute name (preserves
    legacy edge_threshold_1x2 spelling).
"""

from __future__ import annotations

import json

import pytest

from bip.core.types import MarketKey


def test_marketkey_is_strenum() -> None:
    """StrEnum guarantees: instance is also a str AND equals its string value."""
    assert isinstance(MarketKey.ONEXTWO, str)
    assert MarketKey.ONEXTWO == "onextwo"


def test_marketkey_members_match_yaml() -> None:
    """All 5 canonical keys present and match markets.yaml `key:` field exactly."""
    values = {m.value for m in MarketKey}
    assert values == {"onextwo", "btts", "ou", "ah", "corners"}


def test_from_str_canonical() -> None:
    """Canonical lowercase keys round-trip via from_str."""
    assert MarketKey.from_str("onextwo") is MarketKey.ONEXTWO
    assert MarketKey.from_str("btts") is MarketKey.BTTS
    assert MarketKey.from_str("ou") is MarketKey.OU
    assert MarketKey.from_str("ah") is MarketKey.AH
    assert MarketKey.from_str("corners") is MarketKey.CORNERS


def test_from_str_aliases_1x2() -> None:
    """Legacy + Odds-API aliases for the 1X2 market collapse to ONEXTWO."""
    assert MarketKey.from_str("1X2") is MarketKey.ONEXTWO
    assert MarketKey.from_str("1x2") is MarketKey.ONEXTWO
    assert MarketKey.from_str("h2h") is MarketKey.ONEXTWO


def test_from_str_ou_ah_aliases() -> None:
    """OU and AH have multiple external aliases that must resolve to the canonical member."""
    assert MarketKey.from_str("totals") is MarketKey.OU
    assert MarketKey.from_str("over_under") is MarketKey.OU
    assert MarketKey.from_str("alternate_spreads") is MarketKey.AH
    assert MarketKey.from_str("asian_handicap") is MarketKey.AH


def test_from_str_whitespace_insensitive() -> None:
    """Surrounding whitespace and mixed case both stripped/normalized."""
    assert MarketKey.from_str("  ONEXTWO  ") is MarketKey.ONEXTWO


def test_from_str_rejects_unknown() -> None:
    """Unknown alias raises ValueError with the raw input visible in the message."""
    with pytest.raises(ValueError) as exc_info:
        MarketKey.from_str("garbage")
    assert "garbage" in str(exc_info.value)


def test_to_odds_api_mapping() -> None:
    """to_odds_api maps to The Odds API market keys; CORNERS uses '' sentinel."""
    assert MarketKey.ONEXTWO.to_odds_api() == "h2h"
    assert MarketKey.BTTS.to_odds_api() == "btts"
    assert MarketKey.OU.to_odds_api() == "totals"
    assert MarketKey.AH.to_odds_api() == "alternate_spreads"
    assert MarketKey.CORNERS.to_odds_api() == ""


def test_to_threshold_attr() -> None:
    """to_threshold_attr bridges to ModelParams attribute names (1x2 spelling preserved)."""
    assert MarketKey.ONEXTWO.to_threshold_attr() == "edge_threshold_1x2"
    assert MarketKey.BTTS.to_threshold_attr() == "edge_threshold_btts"
    assert MarketKey.OU.to_threshold_attr() == "edge_threshold_ou"
    assert MarketKey.AH.to_threshold_attr() == "edge_threshold_ah"
    assert MarketKey.CORNERS.to_threshold_attr() == "edge_threshold_corners"


def test_json_serialization() -> None:
    """StrEnum serializes as its string value -- DB/JSON writes stay backwards compatible."""
    payload = json.dumps({"market": MarketKey.ONEXTWO})
    assert payload == '{"market": "onextwo"}'
