"""Shared test fixtures for betting-intelligence-platform."""

import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock

import pytest


LEAGUES_DIR = (
    Path(__file__).parent.parent
    / "src" / "bip" / "sports" / "football" / "config" / "leagues"
)


@pytest.fixture
def tmp_leagues_dir(tmp_path: Path) -> Path:
    """Temp directory with copies of the real league YAML files."""
    leagues_dest = tmp_path / "leagues"
    if LEAGUES_DIR.exists():
        shutil.copytree(LEAGUES_DIR, leagues_dest)
    else:
        leagues_dest.mkdir()
    return leagues_dest


@pytest.fixture
def league_registry(tmp_leagues_dir: Path):
    """LeagueRegistry loaded from tmp_leagues_dir."""
    from bip.sports.football.config.league_registry import LeagueRegistry
    return LeagueRegistry(tmp_leagues_dir)


@pytest.fixture
def settings(monkeypatch):
    """Settings with test env vars (all required fields present)."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-supabase-key-12345")
    monkeypatch.setenv("API_FOOTBALL_KEY", "test-api-football-key")
    monkeypatch.setenv("ODDS_API_KEY", "test-odds-api-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_CHANNEL_ID", "-1001234567890")
    monkeypatch.setenv("TELEGRAM_OPS_CHANNEL_ID", "-1009876543210")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    from bip.core.settings import Settings
    return Settings()


@pytest.fixture
def mock_client() -> MagicMock:
    """Supabase client mock with fluent API chain support."""
    client = MagicMock()
    return client


def setup_mock_chain(client: MagicMock, data: list | None = None) -> MagicMock:
    """Set up fluent Supabase method chain to return given data."""
    if data is None:
        data = [{"id": 1}]
    response = MagicMock()
    response.data = data
    builder = MagicMock()
    builder.execute.return_value = response
    builder.insert.return_value = builder
    builder.select.return_value = builder
    builder.update.return_value = builder
    builder.upsert.return_value = builder
    builder.eq.return_value = builder
    builder.order.return_value = builder
    builder.limit.return_value = builder
    client.table.return_value = builder
    return builder


@pytest.fixture
def sample_prediction():
    """Sample Prediction model instance with sport field."""
    from bip.core.storage.models import Prediction
    return Prediction(
        fixture_id=12345,
        league="premier_league",
        sport="football",
        market="btts",
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff_utc=datetime(2026, 4, 22, 15, 0, 0),
        probabilities={"yes": 0.65, "no": 0.35},
        model_version="v1.0",
    )


# ------------------------------------------------------------------
# Phase 2 fixtures — synthetic training data + model directory
# ------------------------------------------------------------------

@pytest.fixture
def synthetic_training_data():
    """50-row synthetic dataset with known temporal ordering.

    Returns (X, y, dates) where:
      X: (50, 5) float features from default_rng(42)
      y: (50,) int labels in {0, 1, 2}
      dates: (50,) datetime array, 7-day spacing starting 2024-01-01

    Use for fast ML unit tests (calibration, walk-forward, stacking).
    """
    import numpy as np
    from datetime import datetime, timedelta
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 3, size=n)
    dates = np.array([datetime(2024, 1, 1) + timedelta(days=i * 7) for i in range(n)])
    return X, y, dates


@pytest.fixture
def tmp_model_dir(tmp_path):
    """Temporary models/ directory for registry and loader tests.

    Layout:
      tmp_path/models/football/   <- returned parent (tmp_path/models)
    """
    model_dir = tmp_path / "models" / "football"
    model_dir.mkdir(parents=True)
    return tmp_path / "models"


# ------------------------------------------------------------------
# Phase 3 fixtures — Claude validator + Telegram delivery mocks
# ------------------------------------------------------------------

def _build_canned_message(verdict: str = "CONFIRM", reason_code: str = "ok",
                           reasoning: str = "Test reasoning", summary: str = "Test summary"):
    """Build a fake anthropic Message with content[0] as tool_use block."""
    from unittest.mock import MagicMock

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = {
        "verdict": verdict,
        "reason_code": reason_code,
        "reasoning": reasoning,
        "summary": summary,
    }
    msg = MagicMock()
    msg.content = [tool_block]
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 3500
    return msg


@pytest.fixture
def mock_anthropic_client():
    """Returns a MagicMock with .messages.create as AsyncMock returning a canned tool_use Message.

    Used by tests/claude/test_validator.py and tests/picks/test_engine.py to mock
    the AsyncAnthropic client. Default return: CONFIRM verdict.

    Override the return value per-test with:
        mock_anthropic_client.messages.create.return_value = _build_canned_message(verdict="REJECT")
    """
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_build_canned_message(verdict="CONFIRM"))
    return client


@pytest.fixture
def mock_telegram_bot():
    """Returns a MagicMock with .send_html as AsyncMock that records calls.

    Assert in tests:
        mock_telegram_bot.send_html.assert_awaited_once_with(<html_text>)
    """
    from unittest.mock import AsyncMock, MagicMock

    bot = MagicMock()
    bot.send_html = AsyncMock()
    return bot


# ------------------------------------------------------------------
# Phase 4 fixtures — production orchestration mocks
# ------------------------------------------------------------------

@pytest.fixture
def mock_sd_notify():
    """sdnotify.SystemdNotifier mock (D-07).

    Test pattern: HeartbeatTicker(path, mock_sd_notify).tick(); then
        mock_sd_notify.notify.assert_called_with("WATCHDOG=1")
    """
    from unittest.mock import MagicMock
    n = MagicMock()
    n.notify = MagicMock(return_value=True)
    return n


@pytest.fixture
def mock_ops_telegram_sender():
    """TelegramSender mock for the OPS channel (D-03).

    Distinct from `mock_telegram_bot` (picks channel). Tests for ClvTrendChecker,
    drift alerts, and any ops-bound message use this fixture to assert routing.
    """
    from unittest.mock import AsyncMock, MagicMock
    s = MagicMock()
    s.send_html = AsyncMock()
    return s


@pytest.fixture
def tmp_heartbeat_path(tmp_path):
    """tmp_path / 'heartbeat' (D-07). File does NOT pre-exist; HeartbeatTicker.tick()
    creates it via Path.touch(). Tests assert mtime advances on subsequent ticks.
    """
    return tmp_path / "heartbeat"
