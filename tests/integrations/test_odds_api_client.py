"""Unit tests for OddsApiClient (Ola 2.D).

Uses httpx's built-in MockTransport — no respx dep, no actual network.
Real API is the operator's runtime concern (subscription + key);
these tests lock in protocol parsing + retry/error semantics so the
operator can trust the client when they switch on.
"""

from __future__ import annotations

import json

import httpx
import pytest

from bip.integrations.odds_api_client import (
    BETFAIR_EX_EU_KEY,
    H2H_MARKET,
    PINNACLE_KEY,
    SOCCER_WC_2026,
    SPREADS_MARKET,
    OddsApiClient,
    OddsApiError,
    OddsApiRateLimited,
    extract_pinnacle_h2h,
    shin_devig_h2h,
)


def _make_transport(handler):
    """Wrap a (request) → response handler as an httpx MockTransport."""

    def _handler(request: httpx.Request) -> httpx.Response:
        return handler(request)

    return httpx.MockTransport(_handler)


# ─────────────────────────────────────────────────────────────────
# Constructor + URL building
# ─────────────────────────────────────────────────────────────────


def test_constructor_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        OddsApiClient(api_key="")


def test_get_sports_url_includes_api_key(monkeypatch) -> None:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[])

    with OddsApiClient(api_key="testkey", transport=_make_transport(handler)) as c:
        c.get_sports()

    assert "apiKey=testkey" in captured["url"]
    assert "/sports" in captured["url"]


def test_get_sports_all_flag(monkeypatch) -> None:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[])

    with OddsApiClient(api_key="testkey", transport=_make_transport(handler)) as c:
        c.get_sports(all_sports=True)

    assert "all=true" in captured["url"]


def test_get_event_odds_packs_params(monkeypatch) -> None:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "evt1", "bookmakers": []})

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        c.get_event_odds(
            SOCCER_WC_2026,
            "evt1",
            bookmakers=(PINNACLE_KEY, BETFAIR_EX_EU_KEY),
            markets=(H2H_MARKET, SPREADS_MARKET),
        )

    url = captured["url"]
    assert f"/sports/{SOCCER_WC_2026}/events/evt1/odds" in url
    assert f"bookmakers={PINNACLE_KEY}%2C{BETFAIR_EX_EU_KEY}" in url
    assert f"markets={H2H_MARKET}%2C{SPREADS_MARKET}" in url
    assert "regions=eu" in url
    assert "oddsFormat=decimal" in url


# ─────────────────────────────────────────────────────────────────
# Rate limit + error handling
# ─────────────────────────────────────────────────────────────────


def test_rate_limit_headers_parsed_on_success() -> None:
    def handler(request):
        return httpx.Response(
            200,
            json={"ok": True},
            headers={
                "x-requests-used": "42",
                "x-requests-remaining": "458",
                "x-requests-last": "1",
            },
        )

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        resp = c.get_sports()

    assert resp.rate_limit.requests_used == 42
    assert resp.rate_limit.requests_remaining == 458
    assert resp.rate_limit.last_request_cost == 1


def test_rate_limit_headers_missing_safely() -> None:
    """The free-tier ping doesn't include rate-limit headers."""

    def handler(request):
        return httpx.Response(200, json={"ok": True})

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        resp = c.get_sports()

    assert resp.rate_limit.requests_used is None
    assert resp.rate_limit.requests_remaining is None


def test_http_429_raises_rate_limited_with_metadata() -> None:
    def handler(request):
        return httpx.Response(
            429,
            json={"error": "rate limited"},
            headers={"x-requests-used": "500", "x-requests-remaining": "0"},
        )

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        with pytest.raises(OddsApiRateLimited):
            c.get_sports()


def test_http_4xx_raises_odds_api_error() -> None:
    def handler(request):
        return httpx.Response(401, json={"error": "invalid key"})

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        with pytest.raises(OddsApiError, match="HTTP 401"):
            c.get_sports()


def test_http_500_raises_odds_api_error() -> None:
    def handler(request):
        return httpx.Response(500, json={"error": "internal"})

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        with pytest.raises(OddsApiError, match="HTTP 500"):
            c.get_sports()


# ─────────────────────────────────────────────────────────────────
# Retry on transient errors
# ─────────────────────────────────────────────────────────────────


def test_timeout_then_success_retries() -> None:
    state = {"attempts": 0}

    def handler(request):
        state["attempts"] += 1
        if state["attempts"] < 2:
            raise httpx.TimeoutException("synthetic timeout")
        return httpx.Response(200, json={"ok": True})

    with OddsApiClient(api_key="k", transport=_make_transport(handler)) as c:
        resp = c.get_sports()

    assert resp.status_code == 200
    assert state["attempts"] == 2


# ─────────────────────────────────────────────────────────────────
# Devig + extraction helpers
# ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "odds,expected_keys",
    [
        ({"home": 2.0, "draw": 3.5, "away": 4.0}, {"home", "draw", "away"}),
        ({"home": 1.5, "draw": 4.0, "away": 7.0}, {"home", "draw", "away"}),
    ],
)
def test_shin_devig_returns_probability_simplex(odds, expected_keys) -> None:
    probs = shin_devig_h2h(odds)
    assert set(probs.keys()) == expected_keys
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)
    assert all(0 < p < 1 for p in probs.values())


def test_shin_devig_low_overround_close_to_inverse() -> None:
    """For a near-fair book (low overround), de-vig ~= inverse / sum_inverse."""
    odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
    inv_sum = sum(1.0 / v for v in odds.values())
    expected = {k: (1.0 / v) / inv_sum for k, v in odds.items()}
    actual = shin_devig_h2h(odds)
    for k in odds:
        assert actual[k] == pytest.approx(expected[k], abs=1e-9)


def test_shin_devig_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        shin_devig_h2h({"home": 2.0, "draw": 3.5})  # missing away


def test_extract_pinnacle_h2h_happy_path() -> None:
    payload = {
        "id": "evt1",
        "home_team": "France",
        "away_team": "Brazil",
        "bookmakers": [
            {
                "key": PINNACLE_KEY,
                "markets": [
                    {
                        "key": H2H_MARKET,
                        "outcomes": [
                            {"name": "France", "price": 2.1},
                            {"name": "Brazil", "price": 3.4},
                            {"name": "Draw", "price": 3.5},
                        ],
                    }
                ],
            }
        ],
    }
    odds = extract_pinnacle_h2h(payload)
    assert odds == {"home": 2.1, "draw": 3.5, "away": 3.4}


def test_extract_pinnacle_h2h_missing_pinnacle() -> None:
    payload = {
        "id": "evt1",
        "home_team": "X",
        "away_team": "Y",
        "bookmakers": [{"key": "some_other_book", "markets": []}],
    }
    assert extract_pinnacle_h2h(payload) is None


def test_extract_pinnacle_h2h_pinnacle_no_h2h() -> None:
    """Pinnacle exists but didn't post h2h on this event."""
    payload = {
        "id": "evt1",
        "home_team": "X",
        "away_team": "Y",
        "bookmakers": [
            {
                "key": PINNACLE_KEY,
                "markets": [{"key": "spreads", "outcomes": []}],
            }
        ],
    }
    assert extract_pinnacle_h2h(payload) is None


def test_extract_pinnacle_h2h_handles_missing_team_name() -> None:
    """Defensive: payload schema drift shouldn't crash the extractor."""
    payload = {"id": "evt1", "bookmakers": []}
    assert extract_pinnacle_h2h(payload) is None
