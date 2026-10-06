"""build_orchestrator factory + DriftChecker (D-05).

Inverse of PipelineOrchestrator.__init__: gathers settings, constructs every dep,
and returns the wired orchestrator + two TelegramBots + sd_notifier.

T-4-06 mitigation: structlog calls in this module log NON-SECRET attributes only
(heartbeat path, channel-id presence). The Settings instance is NEVER passed to
logger.info as a kwarg.

DriftChecker is constructed and passed to PipelineOrchestrator via the
drift_checker= kwarg added in 04-04. Without this wiring, the weekly_drift
cron job runs but the orchestrator's _check_drift body has no checker to
delegate to.

PickEngine and ClaudeValidator are constructed with REAL arguments — no None
placeholders. ClaudeValidator needs learnings_text loaded from disk (cached via
prompt-cache); PickEngine's live ctor takes (pick_repo, validator, scheduler,
sender, settings, league_registry).
"""
from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bip.clv.client import OddsApiClient
from bip.clv.recorder import ClvRecorder
from bip.clv.trend_checker import ClvTrendChecker
from bip.core.claude.validator import ClaudeValidator
from bip.core.metrics.aggregator import MetricsAggregator, compute_weekly_drift
from bip.core.picks.engine import PickEngine
from bip.core.scheduler.orchestrator import PipelineOrchestrator
from bip.core.settings import Settings
from bip.core.storage.repositories import (
    ClvRecordRepository,
    PerformanceMetricRepository,
    PickRepository,
)
from bip.core.telegram.bot import TelegramBot
from bip.core.telegram.sender import TelegramSender
from bip.production.heartbeat import HeartbeatTicker
from bip.sports.football.config.league_registry import LeagueRegistry
from bip.sports.football.plugin import FootballPlugin
from supabase import create_client

logger = structlog.get_logger(__name__)

# Anchor at the package, not the CWD: under systemd WorkingDirectory=/opt/bip the
# CWD-relative path resolved fine, but `python -m bip.production` from anywhere
# else (e.g. local-mode tmux from the user's home) silently pointed nowhere.
_LEAGUES_DIR = Path(__file__).resolve().parents[1] / "sports" / "football" / "config" / "leagues"


class DriftChecker:
    """D-14 weekly drift wrapper.

    Fetches CLV+pick rows, iterates tuples, alerts on triggered drift.
    """

    # v1: hardcoded tuples — matches MetricsAggregator._discover_tuples for consistency.
    _LEAGUES: tuple[str, ...] = (
        "premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1",
    )
    _MARKETS: tuple[str, ...] = ("onextwo",)
    _SPORT: str = "football"

    def __init__(
        self,
        clv_repo: ClvRecordRepository,
        ops_sender: Any,
        settings: Settings,
    ) -> None:
        self._clv_repo = clv_repo
        self._ops_sender = ops_sender
        self._settings = settings

    async def run(self, today: date | None = None) -> int:
        today = today or datetime.now(UTC).date()
        rows = self._clv_repo.last_n_settled(n=500, market=None) or []
        triggered_count = 0
        for league in self._LEAGUES:
            for market in self._MARKETS:
                tuple_rows = [
                    r for r in rows
                    if r.get("sport") == self._SPORT
                    and r.get("league") == league
                    and r.get("market") == market
                ]
                # Need >= 2 * min_picks for both prior and recent windows.
                if len(tuple_rows) < self._settings.drift_check_min_picks_per_window * 2:
                    continue
                result = compute_weekly_drift(
                    tuple_rows,
                    sport=self._SPORT, league=league, market=market, today=today,
                    min_picks=self._settings.drift_check_min_picks_per_window,
                    abs_threshold_pp=self._settings.drift_absolute_threshold_pp,
                    stdev_multiplier=self._settings.drift_stdev_multiplier,
                    stdev_floor_pp=self._settings.drift_stdev_floor_pp,
                )
                if result is None or not result.triggered:
                    continue
                msg = (
                    f"⚠️ DRIFT [league={league} market={market}] "
                    f"{result.delta_pp:+.1f}pp "
                    f"({result.prior_mean_pp:.1f}% → {result.recent_mean_pp:.1f}%)"
                )
                await self._ops_sender.send_html(msg)
                triggered_count += 1
                logger.info(
                    "drift_alert_fired",
                    league=league, market=market,
                    delta_pp=result.delta_pp,
                    prior_mean=result.prior_mean_pp,
                    recent_mean=result.recent_mean_pp,
                )
        logger.info("drift_check_complete", triggered=triggered_count)
        return triggered_count


def _load_learnings(settings: Settings) -> tuple[str, str]:
    """Load learnings markdown + compute SHA256 for prompt-cache key (D-06).

    Returns ("", "empty") gracefully if the file is missing — Phase 3 ClaudeValidator
    handles that case (the prompt-cache just becomes useless, not broken). Production
    deployments populate the file via Phase 5 learnings work.
    """
    path = Path(settings.learnings_path)
    if not path.exists():
        logger.warning("learnings_file_missing", path=str(path))
        return "", "empty"
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, sha


def build_orchestrator(
    settings: Settings,
) -> tuple[PipelineOrchestrator, TelegramBot, TelegramBot, Any]:
    """Build a fully-wired orchestrator + the two TelegramBots + the sd_notifier.

    Returned tuple is consumed by __main__.main() which owns the asyncio lifecycle.
    Tests construct with hand-built Settings and assert on the wiring.

    T-4-06: this function does NOT logger.info(settings, ...) — only specific
    non-secret attributes. Verify via grep before commit.
    """
    import sdnotify  # local import keeps dev unit tests clean on macOS

    client = create_client(settings.supabase_url, settings.supabase_key)
    pick_repo = PickRepository(client=client)
    clv_repo = ClvRecordRepository(client=client)
    perf_repo = PerformanceMetricRepository(client=client)

    # D-03 Option A: TWO TelegramBot instances, each bound to its channel_id at __init__.
    picks_bot = TelegramBot(
        token=settings.telegram_bot_token,
        channel_id=settings.telegram_channel_id,
    )
    ops_bot = TelegramBot(
        token=settings.telegram_bot_token,
        channel_id=settings.telegram_ops_channel_id,
    )

    odds_client = OddsApiClient(api_key=settings.odds_api_key)
    clv_recorder = ClvRecorder(client=client)

    learnings_text, learnings_sha = _load_learnings(settings)
    validator = ClaudeValidator(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        learnings_text=learnings_text,
        learnings_sha=learnings_sha,
    )

    league_registry = LeagueRegistry(_LEAGUES_DIR)
    plugin = FootballPlugin(settings=settings)

    # Shared scheduler — PickEngine schedules send jobs on the SAME AsyncIOScheduler
    # the orchestrator uses. Built once here and injected into BOTH constructors so
    # the binding is invariant by construction (no post-init reassignment).
    shared_scheduler = AsyncIOScheduler(timezone="UTC")

    picks_sender = TelegramSender(bot=picks_bot)
    pick_engine = PickEngine(
        pick_repo=pick_repo,
        validator=validator,
        scheduler=shared_scheduler,
        sender=picks_sender,
        settings=settings,
        league_registry=league_registry,
    )

    sd_notifier = sdnotify.SystemdNotifier()  # silently no-ops on macOS (NOTIFY_SOCKET unset)
    heartbeat_ticker = HeartbeatTicker(
        path=settings.heartbeat_file_path, notifier=sd_notifier
    )

    clv_trend_checker = ClvTrendChecker(
        repo=clv_repo,
        ops_sender=ops_bot,
        threshold=settings.clv_trend_alert_threshold,
        cooldown_hours=settings.clv_trend_cooldown_hours,
    )
    metrics_aggregator = MetricsAggregator(perf_repo=perf_repo, pick_repo=pick_repo)
    drift_checker = DriftChecker(clv_repo=clv_repo, ops_sender=ops_bot, settings=settings)

    orchestrator = PipelineOrchestrator(
        plugin=plugin,
        settings=settings,
        pick_engine=pick_engine,
        pick_repo=pick_repo,
        telegram_bot=picks_bot,
        odds_api_client=odds_client,
        clv_recorder=clv_recorder,
        league_registry=league_registry,
        # Phase 4
        heartbeat_ticker=heartbeat_ticker,
        clv_trend_checker=clv_trend_checker,
        metrics_aggregator=metrics_aggregator,
        drift_checker=drift_checker,
        ops_sender=ops_bot,
        scheduler=shared_scheduler,
    )
    # Hard invariant: PickEngine's DateTrigger send jobs and orchestrator's
    # CronTrigger jobs MUST live on the SAME AsyncIOScheduler. Anything that
    # silently breaks this (e.g. constructing two schedulers by mistake) makes
    # send jobs land in an orphan event-loop pool. Fail loud at startup.
    assert pick_engine._scheduler is orchestrator.scheduler, (
        "shared_scheduler invariant violated: PickEngine and PipelineOrchestrator "
        "do not share the same AsyncIOScheduler instance"
    )

    # T-4-06: log non-secret booleans only — DO NOT pass settings instance.
    logger.info(
        "build_orchestrator_complete",
        heartbeat_path=settings.heartbeat_file_path,
        ops_channel_configured=bool(settings.telegram_ops_channel_id),
        claude_failure_mode=settings.claude_failure_mode,
        learnings_sha=learnings_sha,
    )
    return orchestrator, picks_bot, ops_bot, sd_notifier
