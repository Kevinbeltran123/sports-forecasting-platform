"""ClvTrendChecker tests (CLV-03 — D-09 through D-12)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from bip.clv.trend_checker import ClvTrendChecker


def _make_checker(rows_global=None, rows_market=None, threshold=1.0, cooldown_hours=12):
    """Factory: ClvTrendChecker with pre-canned repo responses."""
    repo = MagicMock()

    def _last_n(n=50, market=None):
        if market is None:
            return rows_global or []
        return rows_market or []

    repo.last_n_settled = MagicMock(side_effect=_last_n)
    sender = MagicMock()
    sender.send_html = AsyncMock()
    return (
        ClvTrendChecker(
            repo=repo,
            ops_sender=sender,
            threshold=threshold,
            cooldown_hours=cooldown_hours,
        ),
        repo,
        sender,
    )


def _rows(values: list[float], market: str = "onextwo") -> list[dict]:
    return [
        {"clv_percentage": v, "market": market, "created_at": "2026-05-01T00:00:00Z"}
        for v in values
    ]


@pytest.mark.asyncio
async def test_alert_fires_below_threshold():
    """Global avg = 0.7% (< 1.0% threshold) AND >=50 settled rows → alert fires once."""
    rows = _rows([0.7] * 50)
    checker, repo, sender = _make_checker(rows_global=rows, rows_market=[])
    await checker.check()
    sender.send_html.assert_awaited_once()
    msg = sender.send_html.await_args.args[0]
    assert "0.7" in msg and "1%" in msg and "⚠" in msg


@pytest.mark.asyncio
async def test_no_alert_when_above_threshold():
    """Global avg = 2.5% (>= 1.0% threshold) → no alert."""
    rows = _rows([2.5] * 50)
    checker, repo, sender = _make_checker(rows_global=rows, rows_market=[])
    await checker.check()
    sender.send_html.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_insufficient_sample():
    """<50 settled rows → skip silently, no alert (CLV-03 — D-09)."""
    rows = _rows([0.0] * 30)
    checker, repo, sender = _make_checker(rows_global=rows, rows_market=[])
    await checker.check()
    sender.send_html.assert_not_awaited()


@pytest.mark.asyncio
async def test_cooldown_suppresses_repeat_alert():
    """First check fires alert; second check within cooldown does NOT fire (D-11)."""
    rows = _rows([0.5] * 50)
    checker, repo, sender = _make_checker(
        rows_global=rows, rows_market=[], cooldown_hours=12
    )
    await checker.check()
    assert sender.send_html.await_count == 1
    # Second call within cooldown
    await checker.check()
    assert sender.send_html.await_count == 1  # cooldown suppressed second alert


@pytest.mark.asyncio
async def test_per_market_skipped_below_50():
    """Per-market check with <50 rows → skip silently (D-10)."""
    rows_global = _rows([2.0] * 50)  # global is fine
    rows_market = _rows([0.1] * 30)  # market has <50
    checker, repo, sender = _make_checker(rows_global=rows_global, rows_market=rows_market)
    await checker.check()
    sender.send_html.assert_not_awaited()  # global ok; market <50 → no alert


@pytest.mark.asyncio
async def test_per_market_alert_appends_market_tag():
    """Per-market alert text appends '[market: 1X2]' (D-12)."""
    rows_global = _rows([2.0] * 50)  # global is fine — no alert
    rows_market = _rows([0.5] * 50)  # market < threshold, >=50
    checker, repo, sender = _make_checker(rows_global=rows_global, rows_market=rows_market)
    await checker.check()
    sender.send_html.assert_awaited_once()
    msg = sender.send_html.await_args.args[0]
    assert "[market: 1X2]" in msg
