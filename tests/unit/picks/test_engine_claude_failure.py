"""PickEngine D-01 branch: claude_failure_mode (filter vs skip)."""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _default_league_registry():
    """Mock LeagueRegistry returning permissive 1X2 threshold."""
    league_registry = MagicMock()
    params = MagicMock(
        spec=[
            "edge_threshold_1x2", "edge_threshold_btts", "edge_threshold_ou",
            "edge_threshold_ah", "edge_threshold_corners",
        ]
    )
    params.edge_threshold_1x2 = 0.05  # permissive — make sure simulate_pick fires
    params.edge_threshold_btts = 0.05
    params.edge_threshold_ou = 0.05
    params.edge_threshold_ah = 0.06
    params.edge_threshold_corners = 0.07
    cfg = MagicMock()
    cfg.model_params = params
    league_registry.get = MagicMock(return_value=cfg)
    return league_registry


def _make_engine(claude_failure_mode: str = "filter"):
    """Factory mirroring tests/picks/test_engine.py:_make_engine.

    PickEngine ctor (live signature, lines 54-68): pick_repo, validator, scheduler,
    sender, settings, league_registry. validator.validate returns None to simulate
    Claude API failure.
    """
    from bip.core.picks.engine import PickEngine

    settings = MagicMock()
    settings.claude_failure_mode = claude_failure_mode
    settings.max_kelly_fraction = 0.25

    pick_repo = MagicMock()
    pick_repo.insert = MagicMock(return_value={"id": 999})
    pick_repo.get_window_picks = MagicMock(return_value=[])

    validator = MagicMock()
    validator.validate = AsyncMock(return_value=None)  # simulate Claude API failure

    scheduler = MagicMock()
    scheduler.add_job = MagicMock()

    sender = MagicMock()

    engine = PickEngine(
        pick_repo=pick_repo,
        validator=validator,
        scheduler=scheduler,
        sender=sender,
        settings=settings,
        league_registry=_default_league_registry(),
    )
    return engine, pick_repo, scheduler


def _make_prediction():
    """Probabilities + opening odds chosen so simulate_pick yields a positive-edge index."""
    p = MagicMock()
    p.id = 100
    p.fixture_id = 12345
    p.league = "premier_league"
    p.sport = "football"
    p.market = "1X2"
    p.probabilities = {"1": 0.60, "X": 0.25, "2": 0.15}
    p.home_team = "Manchester United"
    p.away_team = "Chelsea"
    p.model_version = "v1"
    p.kickoff_utc = datetime(2026, 5, 2, 15, 0, 0, tzinfo=UTC)
    p.is_lineup_adjusted = False
    return p


@pytest.mark.asyncio
async def test_filter_mode_persists_filtered_with_reason_code():
    """D-01 default: claude_failure_mode='filter' → status='filtered', reason_code set, no send."""
    from bip.core.types import PickStatus
    engine, pick_repo, scheduler = _make_engine(claude_failure_mode="filter")
    prediction = _make_prediction()
    opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}
    pick = await engine.evaluate(prediction, opening_odds)
    assert pick is not None
    assert pick.status == PickStatus.filtered
    assert "claude_api_unavailable" in (pick.claude_reasoning or "")
    scheduler.add_job.assert_not_called()


@pytest.mark.asyncio
async def test_skip_mode_persists_pending_with_SKIPPED_validation():
    """D-01 opt-in: claude_failure_mode='skip' → status='pending', claude_validation='SKIPPED'."""
    from bip.core.types import PickStatus
    engine, pick_repo, scheduler = _make_engine(claude_failure_mode="skip")
    prediction = _make_prediction()
    opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}
    pick = await engine.evaluate(prediction, opening_odds)
    assert pick is not None
    assert pick.status == PickStatus.pending
    assert pick.claude_validation == "SKIPPED"


@pytest.mark.asyncio
async def test_skip_mode_schedules_send():
    """D-01 opt-in: skip mode triggers scheduler.add_job (so the pick reaches Telegram)."""
    engine, pick_repo, scheduler = _make_engine(claude_failure_mode="skip")
    prediction = _make_prediction()
    opening_odds = {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}
    await engine.evaluate(prediction, opening_odds)
    scheduler.add_job.assert_called_once()
