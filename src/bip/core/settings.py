"""Global application settings loaded from environment variables."""

from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with env var loading via pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # SUPABASE_DB_PASSWORD is read directly by verify scripts, not by Settings
    )

    supabase_url: str
    supabase_key: str
    parquet_base_path: str = "data/cache"
    model_dir: str = "models"  # Phase 2: root for model artifacts (models/{sport}/{league}/{version}/)
    api_football_key: str
    odds_api_key: str
    telegram_bot_token: str = ""
    log_level: str = "INFO"

    # ─────────────────────────────────────────────────────
    # Phase 3 additions (D-13, RESEARCH §Standard Stack)
    # PATTERNS.md drift risk #3: append flat fields to existing class — NO per-feature settings classes.
    # ─────────────────────────────────────────────────────
    telegram_channel_id: str = ""              # D-13: -100<id> format; validated below (Pitfall 8)
    anthropic_api_key: str = ""                # D-05: Anthropic API key for Role C validator
    claude_model: str = "claude-sonnet-4-6"    # D-05: current Sonnet alias (verified May 2026)
    max_kelly_fraction: float = 0.25           # PICK-02 / D-10: quarter Kelly default

    # Orchestrator reconciliation tuning (G-MAINT-10 closeout in 260503-txj)
    # Defaults preserve historical behavior (4 retries x 30 minutes); ops can tune via env.
    reconcile_max_retries: int = 4             # max reschedule attempts on IN_PLAY/NOT_STARTED before abandoning
    reconcile_retry_minutes: int = 30          # backoff cadence between reschedule attempts

    # ─────────────────────────────────────────────────────
    # Phase 4 additions (D-01, D-03, D-07, D-09–D-12, D-14, RESEARCH §3 floor)
    # Flat fields only (PATTERNS.md drift risk #3 — no per-feature settings classes).
    # ─────────────────────────────────────────────────────
    # D-01: default preserves Phase 3 D-07 conservative behaviour (filter on validator failure)
    claude_failure_mode: Literal["filter", "skip"] = "filter"
    # D-03: separate ops channel; validated by _validate_ops_channel_id below
    telegram_ops_channel_id: str = ""
    # D-07: APScheduler IntervalTrigger touches this; systemd WatchdogSec reads mtime
    heartbeat_file_path: str = "/var/run/bip/heartbeat"
    # D-09: alert when rolling-50 avg CLV < threshold (percentage points)
    clv_trend_alert_threshold: float = 1.0
    # D-11: in-memory cooldown per scope between repeat alerts
    clv_trend_cooldown_hours: int = 12
    # D-14: skip drift comparison when either window has fewer settled picks than this
    drift_check_min_picks_per_window: int = 30
    # D-14 hard gate: absolute drop in mean CLV (percentage points) that triggers
    drift_absolute_threshold_pp: float = 2.0
    # D-14 soft gate: drop in mean CLV >= N x stdev triggers
    drift_stdev_multiplier: float = 1.5
    # RESEARCH §Drift Statistical Method: prevents zero-stdev tight gate (Pitfall 4)
    drift_stdev_floor_pp: float = 1.0
    # Markdown file consumed by ClaudeValidator's prompt-cache. Missing file is
    # tolerated (logged as warning); the cache becomes useless but the validator
    # still runs. Override via .env if learnings live outside the repo.
    learnings_path: str = "data/learnings/default.md"

    @field_validator("telegram_channel_id")
    @classmethod
    def _validate_channel_id(cls, v: str) -> str:
        """Pitfall 8: Telegram private channels are `-100<10-13 digits>`.

        Empty default allowed (tests / CI without secrets construct cleanly).
        When SET, must start with `-100` AND be at least 8 chars total.
        """
        if v == "":
            return v
        if not v.startswith("-100"):
            raise ValueError(
                f"telegram_channel_id must start with '-100' (got {v!r}). "
                "Forward a channel message to @userinfobot to read the correct ID."
            )
        if len(v) < 8:  # "-100" + at least 4 digits
            raise ValueError(
                f"telegram_channel_id appears truncated (got {v!r}, expected -100<10-13 digits>)."
            )
        return v

    @field_validator("telegram_ops_channel_id")
    @classmethod
    def _validate_ops_channel_id(cls, v: str) -> str:
        """T-4-01 mitigation: same shape as _validate_channel_id (D-13).

        Empty default allowed (tests / CI without ops bot construct cleanly). When SET,
        must start with `-100` AND be at least 8 chars. Catches misconfiguration where
        operator types picks-channel ID into ops slot (would silently route stake amounts
        to picks channel — secret leak per CONTEXT.md SC#2 reconciliation).
        """
        if v == "":
            return v
        if not v.startswith("-100"):
            raise ValueError(
                f"telegram_ops_channel_id must start with '-100' (got {v!r}). "
                "Forward an ops channel message to @userinfobot to read the correct ID."
            )
        if len(v) < 8:  # "-100" + at least 4 digits
            raise ValueError(
                f"telegram_ops_channel_id appears truncated (got {v!r}, "
                "expected -100<10-13 digits>)."
            )
        return v
