"""APScheduler 3.x pipeline orchestrator.

CRITICAL: Uses APScheduler 3.x API (AsyncIOScheduler + add_job).
Do NOT use 4.x API (AsyncScheduler + add_schedule) — APScheduler 4.x is still alpha.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from bip.core.errors import SchedulerError
from bip.core.settings import Settings
from bip.core.storage.models import PickStatus, Prediction
from bip.core.types import MarketKey
from bip.sports import FixtureData, SportPlugin

logger = structlog.get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full betting intelligence pipeline using APScheduler 3.x."""

    # D-16 status code groups (RESEARCH "API-Football Status Codes" + Risk 9)
    _STATUS_SETTLED_REGULATION = frozenset({"FT", "AWD", "WO"})
    _STATUS_SETTLED_REGULATION_PLUS_ET = frozenset({"AET"})
    _STATUS_SETTLED_PEN = frozenset({"PEN"})
    _STATUS_VOID = frozenset({"PST", "CANC", "ABD"})
    _STATUS_IN_PLAY = frozenset({"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT"})
    _STATUS_NOT_STARTED = frozenset({"TBD", "NS"})
    # Defaults used when settings is None (test fixtures); production reads Settings.
    _RECONCILE_MAX_RETRIES_DEFAULT = 4
    _RECONCILE_RETRY_MINUTES_DEFAULT = 30

    def __init__(
        self,
        plugin: SportPlugin,
        settings: Settings | None = None,
        pick_engine: Any | None = None,
        pick_repo: Any | None = None,
        telegram_bot: Any | None = None,
        api_football_client: Any | None = None,
        odds_api_client: Any | None = None,
        clv_recorder: Any | None = None,
        league_registry: Any | None = None,
        # Phase 4 additions (D-03, D-07, D-09–D-12, D-13, D-14)
        heartbeat_ticker: Any | None = None,
        clv_trend_checker: Any | None = None,
        metrics_aggregator: Any | None = None,
        drift_checker: Any | None = None,
        ops_sender: Any | None = None,
        scheduler: AsyncIOScheduler | None = None,
    ) -> None:
        self.plugin = plugin
        self.settings = settings
        # Inject scheduler when one needs to be shared with PickEngine (DateTrigger
        # send jobs and CronTrigger orchestrator jobs MUST live on the same scheduler
        # instance). Default keeps the standalone-orchestrator path unchanged.
        self.scheduler = scheduler if scheduler is not None else AsyncIOScheduler(timezone="UTC")
        self._today_jobs_registered = False
        self.pick_engine = pick_engine
        self.pick_repo = pick_repo
        self.telegram_bot = telegram_bot
        self._api_client = api_football_client
        self.odds_api_client = odds_api_client
        self.clv_recorder = clv_recorder
        self.league_registry = league_registry
        # Phase 4
        self._heartbeat_ticker = heartbeat_ticker
        self._clv_trend_checker = clv_trend_checker
        self._metrics_aggregator = metrics_aggregator
        self._drift_checker = drift_checker
        self._ops_sender = ops_sender

    def start(self) -> None:
        self.scheduler.add_job(
            self._daily_orchestrator,
            trigger=CronTrigger(hour=6, minute=0, timezone="UTC"),
            id="daily_orchestrator",
            replace_existing=True,
        )

        self.scheduler.add_job(
            self._reconcile_clv,
            trigger=CronTrigger(hour=3, minute=0, timezone="UTC"),
            id="nightly_clv_reconciliation",
            replace_existing=True,
        )

        # ─── Phase 4 cron jobs (each conditional on its dep being wired) ───
        if self._heartbeat_ticker is not None:
            self.scheduler.add_job(
                self._heartbeat_ticker.tick,
                trigger=IntervalTrigger(minutes=5),
                id="heartbeat",
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=60,
            )

        if self._clv_trend_checker is not None:
            self.scheduler.add_job(
                self._check_clv_trend,
                trigger=CronTrigger(minute=0, timezone="UTC"),
                id="clv_trend",
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=600,
            )

        if self._metrics_aggregator is not None:
            self.scheduler.add_job(
                self._aggregate_metrics,
                trigger=CronTrigger(hour=23, minute=0, timezone="UTC"),
                id="metrics_aggregator",
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=600,
            )

        # Weekly drift cron registers when ops_sender is wired; the actual drift
        # body is owned by _drift_checker. _check_drift logs a skip when the
        # checker is None — never crashes.
        if self._ops_sender is not None:
            self.scheduler.add_job(
                self._check_drift,
                trigger=CronTrigger(day_of_week='mon', hour=6, minute=0, timezone="UTC"),
                id="weekly_drift",
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=600,
            )

        self.scheduler.start()
        logger.info("scheduler_started", jobs=len(self.scheduler.get_jobs()))

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._auto_recover())
        except RuntimeError:
            pass

    async def _auto_recover(self) -> None:
        """Auto-recovery on startup — re-register fixture jobs + re-queue lost send jobs (Pitfall 6)."""
        now = datetime.now(UTC)
        today_start = now.replace(hour=6, minute=0, second=0, microsecond=0)

        if now < today_start:
            logger.info("auto_recovery_skipped", reason="before_06:00_UTC")
        else:
            # Persistent (non-fixture) jobs MUST be excluded so a hot restart that
            # only kept these alive still triggers fixture-job recovery. Phase 4
            # added 4 jobs (heartbeat, clv_trend, metrics_aggregator, weekly_drift)
            # that previously fooled the filter into thinking fixtures existed.
            persistent_job_ids = {
                "daily_orchestrator",
                "nightly_clv_reconciliation",
                "heartbeat",
                "clv_trend",
                "metrics_aggregator",
                "weekly_drift",
            }
            existing_fixture_jobs = [
                j for j in self.scheduler.get_jobs()
                if j.id not in persistent_job_ids
            ]
            if not existing_fixture_jobs:
                logger.info("auto_recovery_triggered", reason="no_fixture_jobs_found")
                await self._daily_orchestrator()
            else:
                logger.info(
                    "auto_recovery_skipped",
                    reason="jobs_already_registered",
                    count=len(existing_fixture_jobs),
                )

        # Pitfall 6: re-queue lost send jobs
        if self.pick_repo is None or self.pick_engine is None:
            logger.info("auto_recovery_skipped_pending_sends", reason="pick_repo_or_engine_unwired")
            return

        try:
            pending = self.pick_repo.query_pending_sends(sport="football", max_age_minutes=30)
        except Exception as exc:
            logger.error("auto_recover_pending_sends_query_failed", error=str(exc))
            return

        requeued = 0
        existing_send_ids = {j.id for j in self.scheduler.get_jobs()
                             if j.id and j.id.startswith("send_pick_")}
        for row in pending:
            job_id = f"send_pick_{row['fixture_id']}_{row['market']}"
            if job_id in existing_send_ids:
                continue
            self.scheduler.add_job(
                self._send_recovered_pick,
                trigger=DateTrigger(run_date=datetime.now(UTC)),
                args=[row],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300,
            )
            requeued += 1
        logger.info("auto_recover_requeued", count=requeued, scanned=len(pending))
        # D-08: emit completion event so journal monitoring sees one terminal line per startup.
        logger.info("auto_recover_complete", jobs_re_queued=requeued, scanned=len(pending))

    async def _send_recovered_pick(self, row: dict) -> None:
        """Pitfall 6 recovery dispatch — re-build a Pick from the DB row and send."""
        if self.telegram_bot is None:
            logger.error("recovered_pick_send_failed", reason="telegram_bot_unwired")
            return
        from bip.core.storage.models import Pick
        from bip.core.telegram.sender import render_pick

        pick = Pick.model_validate(row)
        home = row.get("home_team", "Home")
        away = row.get("away_team", "Away")
        version = row.get("model_version", "unknown")
        text = render_pick(pick, home, away, version, reason_code=None)
        try:
            await self.telegram_bot.send_html(text)
            logger.info("recovered_pick_sent", fixture_id=pick.fixture_id, market=pick.market)
        except Exception as exc:
            logger.error("recovered_pick_send_failed", fixture_id=pick.fixture_id,
                         market=pick.market, error=str(exc))

    async def _daily_orchestrator(self) -> None:
        now = datetime.now(UTC)
        logger.info("daily_orchestrator_started", date=now.date().isoformat())

        try:
            fixtures = await self.plugin.get_fixtures(date=now)
        except Exception as exc:
            logger.error("get_fixtures_failed", error=str(exc))
            raise SchedulerError(f"Daily orchestrator failed: {exc}") from exc

        # Each fixture gets: T-2h, T-30min, T-1min (CLV snapshot), T+150min (reconcile) DateTrigger jobs
        for fixture in fixtures:
            self._register_fixture_jobs(fixture, now)

        logger.info("daily_orchestrator_completed", fixture_count=len(fixtures))

    def _register_fixture_jobs(self, fixture: object, now: datetime) -> None:
        kickoff = fixture.kickoff_utc  # type: ignore[attr-defined]
        fixture_id = fixture.fixture_id  # type: ignore[attr-defined]

        t_minus_2h = kickoff - timedelta(hours=2)
        if t_minus_2h > now:
            self.scheduler.add_job(
                self._run_pipeline,
                trigger=DateTrigger(run_date=t_minus_2h),
                args=[fixture, "t_minus_2h"],
                id=f"pipeline_{fixture_id}_t_minus_2h",
                replace_existing=True,
            )

        t_minus_30m = kickoff - timedelta(minutes=30)
        if t_minus_30m > now:
            self.scheduler.add_job(
                self._run_pipeline,
                trigger=DateTrigger(run_date=t_minus_30m),
                args=[fixture, "t_minus_30m"],
                id=f"pipeline_{fixture_id}_t_minus_30m",
                replace_existing=True,
            )

        # CLV snapshot at kickoff - 1 minute (Pinnacle h2h freezes at kickoff;
        # capturing 1 min before is the closing line we benchmark against).
        t_minus_1m = kickoff - timedelta(minutes=1)
        if t_minus_1m > now:
            self.scheduler.add_job(
                self._record_clv,
                trigger=DateTrigger(run_date=t_minus_1m),
                args=[fixture],
                id=f"clv_{fixture_id}_t_minus_1m",
                replace_existing=True,
            )

        # D-16: result reconciliation at kickoff + 150min
        t_plus_150 = kickoff + timedelta(minutes=150)
        if t_plus_150 > now:
            self.scheduler.add_job(
                self._reconcile_results,
                trigger=DateTrigger(run_date=t_plus_150),
                args=[fixture],
                id=f"reconcile_{fixture_id}",
                replace_existing=True,
                misfire_grace_time=600,
            )

    async def _run_pipeline(self, fixture: FixtureData, stage: str) -> None:
        """Run prediction pipeline for a fixture at the specified stage.

        Strategy 2 wiring (G-CODE-01/02/03 + G-MAINT-04):
          - orchestrator constructs the typed Prediction;
          - plugin.predict() stays pure (returns ProbabilityMap);
          - PickEngine.evaluate is the only entry into the Pick lifecycle.

        D-01: T-30min skips evaluate when a pending 1X2 pick already exists for the
        fixture (prevents dup-alert from successful T-2h pick).
        """
        fixture_id = fixture.fixture_id
        logger.info("pipeline_started", fixture_id=fixture_id, stage=stage)
        try:
            features = await self.plugin.build_features(fixture)

            if self.pick_engine is None or stage not in ("t_minus_2h", "t_minus_30m"):
                logger.info("pipeline_completed", fixture_id=fixture_id, stage=stage)
                return

            if stage == "t_minus_30m" and self.pick_repo is not None:
                # D-01 dup-alert guard -- repo method is SYNC, returns list[dict].
                # G-MAINT-05: normalize raw market via MarketKey so legacy "1X2"
                # rows AND canonical "onextwo" rows are both recognized.
                existing = self.pick_repo.get_pending_for_fixture(fixture_id)
                blocking = []
                for p in existing:
                    raw = p.get("market")
                    if raw is None:
                        continue
                    try:
                        if (
                            MarketKey.from_str(raw) == MarketKey.ONEXTWO
                            and p.get("status") == PickStatus.pending.value
                        ):
                            blocking.append(p)
                    except ValueError:
                        continue
                if blocking:
                    logger.info(
                        "t30_skipped_pending_already",
                        fixture_id=fixture_id,
                        t2h_pick_id=blocking[0].get("id"),
                    )
                    return

            prob_map = await self.plugin.predict(features, market=MarketKey.ONEXTWO)
            if prob_map is None:
                logger.info(
                    "pipeline_skip_evaluate_no_prediction",
                    fixture_id=fixture_id, stage=stage,
                )
                logger.info("pipeline_completed", fixture_id=fixture_id, stage=stage)
                return

            opening_odds = await self.plugin.get_opening_odds(fixture_id)

            prediction = Prediction(
                fixture_id=fixture_id,
                league=features.league,
                sport="football",
                market=MarketKey.ONEXTWO,
                home_team=fixture.home_team,
                away_team=fixture.away_team,
                kickoff_utc=fixture.kickoff_utc,
                probabilities=prob_map.probabilities,
                model_version=prob_map.model_version,
                is_lineup_adjusted=(stage == "t_minus_30m"),
            )

            try:
                await self.pick_engine.evaluate(prediction, opening_odds)
            except Exception as exc:
                logger.error(
                    "pipeline_evaluate_failed",
                    fixture_id=fixture_id, stage=stage, error=str(exc),
                )

            logger.info("pipeline_completed", fixture_id=fixture_id, stage=stage)
        except Exception as exc:
            logger.error("pipeline_failed", fixture_id=fixture_id, stage=stage, error=str(exc))

    async def _record_clv(self, fixture: object) -> None:
        """Snapshot Pinnacle closing odds at kickoff - 1m and persist via ClvRecorder.

        Wiring contract:
          - clv_recorder + odds_api_client + pick_repo are all required.
          - league_registry is optional (falls back to "soccer_epl" sport_key).
          - One ClvRecord is written per pending 1X2 pick on the fixture.
        """
        fixture_id = fixture.fixture_id  # type: ignore[attr-defined]

        if (
            self.clv_recorder is None
            or self.odds_api_client is None
            or self.pick_repo is None
        ):
            logger.info(
                "clv_skip_unwired",
                fixture_id=fixture_id,
                missing=[
                    name for name, obj in (
                        ("clv_recorder", self.clv_recorder),
                        ("odds_api_client", self.odds_api_client),
                        ("pick_repo", self.pick_repo),
                    )
                    if obj is None
                ],
            )
            return

        try:
            pending = self.pick_repo.get_pending_for_fixture(fixture_id)
        except Exception as exc:
            logger.error("clv_pending_query_failed", fixture_id=fixture_id, error=str(exc))
            return

        # G-MAINT-05: normalize raw market via MarketKey to recognize legacy
        # "1X2" rows AND canonical "onextwo" rows.
        pending_1x2 = []
        for p in pending:
            raw = p.get("market")
            if raw is None:
                continue
            try:
                if MarketKey.from_str(raw) == MarketKey.ONEXTWO:
                    pending_1x2.append(p)
            except ValueError:
                continue
        if not pending_1x2:
            logger.info("clv_skip_no_pending", fixture_id=fixture_id)
            return

        league = getattr(fixture, "league", None)
        sport_key = "soccer_epl"
        if self.league_registry is not None and league:
            try:
                sport_key = self.league_registry.get(league).api_mappings.odds_api_sport_key
            except Exception as exc:
                logger.warning(
                    "clv_sport_key_lookup_failed",
                    fixture_id=fixture_id, league=league, error=str(exc),
                    fallback=sport_key,
                )

        try:
            event_id = await self.odds_api_client.find_event_by_fixture(
                sport_key=sport_key,
                home_team=fixture.home_team,  # type: ignore[attr-defined]
                away_team=fixture.away_team,  # type: ignore[attr-defined]
                kickoff_utc=fixture.kickoff_utc,  # type: ignore[attr-defined]
            )
        except Exception as exc:
            logger.error("clv_find_event_failed", fixture_id=fixture_id, error=str(exc))
            return

        if event_id is None:
            logger.info("clv_skip_no_event", fixture_id=fixture_id, sport_key=sport_key)
            return

        try:
            bookmaker = await self.odds_api_client.fetch_pinnacle_closing_odds(
                sport_key=sport_key,
                event_id=event_id,
                market_key=MarketKey.ONEXTWO.to_odds_api(),
            )
        except Exception as exc:
            logger.error("clv_fetch_pinnacle_failed", fixture_id=fixture_id, error=str(exc))
            return

        if bookmaker is None:
            logger.info("clv_skip_no_pinnacle", fixture_id=fixture_id, event_id=event_id)
            return

        # Project Pinnacle outcomes into {"1": ..., "X": ..., "2": ...}
        markets = bookmaker.get("markets") or []
        if not markets:
            logger.info("clv_skip_no_markets", fixture_id=fixture_id, event_id=event_id)
            return
        outcomes = markets[0].get("outcomes") or []

        home_lc = (fixture.home_team or "").strip().lower()  # type: ignore[attr-defined]
        away_lc = (fixture.away_team or "").strip().lower()  # type: ignore[attr-defined]
        closing: dict[str, float] = {}
        for outcome in outcomes:
            name_lc = (outcome.get("name") or "").strip().lower()
            price = outcome.get("price")
            if price is None:
                continue
            if name_lc == "draw":
                closing["X"] = float(price)
            elif home_lc and (home_lc in name_lc or name_lc in home_lc):
                closing["1"] = float(price)
            elif away_lc and (away_lc in name_lc or name_lc in away_lc):
                closing["2"] = float(price)

        if set(closing.keys()) != {"1", "X", "2"}:
            logger.info(
                "clv_skip_incomplete_market",
                fixture_id=fixture_id,
                event_id=event_id,
                got=sorted(closing.keys()),
            )
            return

        recorded = 0
        for pick in pending_1x2:
            try:
                self.clv_recorder.record(
                    pick_id=pick["id"],
                    fixture_id=fixture_id,
                    sport="football",
                    market=MarketKey.ONEXTWO,
                    odds_at_pick=pick["odds_at_pick"],
                    closing_odds_dict=closing,
                    selection=pick["selection"],
                )
                recorded += 1
            except Exception as exc:
                logger.error(
                    "clv_record_failed",
                    fixture_id=fixture_id,
                    pick_id=pick.get("id"),
                    error=str(exc),
                )

        logger.info(
            "clv_recorded_count",
            fixture_id=fixture_id,
            event_id=event_id,
            count=recorded,
        )

    async def _reconcile_clv(self) -> None:
        # Phase 4 implemented rolling-50 trend via ClvTrendChecker. Snapshot
        # back-fill for fixtures that missed the T-1m window remains intentionally
        # deferred (Pinnacle moves post-match make backfill non-true closing lines
        # — see ROADMAP Phase 4 SC#2). Logged at info level so it doesn't pollute
        # the journal with false-warning noise every night.
        logger.info(
            "clv_reconciliation_skipped",
            reason="snapshot_backfill_intentionally_deferred",
        )

    async def _reconcile_results(self, fixture: object, retries: int = 0) -> None:
        """D-16: settle pending picks based on API-Football status.

        All 16 status codes handled (Risk 9). Pitfall 7 reschedule on in-play, max 4 retries.
        Q3 RESOLVED: after 4 retries, log structlog.ERROR event=reconcile_abandoned.
        """
        fixture_id = fixture.fixture_id  # type: ignore[attr-defined]
        if self._api_client is None or self.pick_repo is None:
            logger.error("reconcile_unwired", fixture_id=fixture_id,
                         missing="api_client" if self._api_client is None else "pick_repo")
            return

        try:
            raw = await self._api_client.get_fixture(fixture_id)
        except Exception as exc:
            logger.error("reconcile_api_failed", fixture_id=fixture_id, error=str(exc))
            return

        item = raw["response"][0] if raw and raw.get("response") else None
        if item is None:
            logger.error("reconcile_no_response", fixture_id=fixture_id)
            return

        status = item["fixture"]["status"]["short"]
        logger.info("reconcile_started", fixture_id=fixture_id, status=status, retries=retries)

        if status in self._STATUS_VOID:
            n = self.pick_repo.update_status_by_fixture(fixture_id, "void")
            logger.info("reconcile_voided", fixture_id=fixture_id, status=status, count=n)
            return

        if status in self._STATUS_IN_PLAY or status in self._STATUS_NOT_STARTED:
            max_retries = (
                self.settings.reconcile_max_retries
                if self.settings is not None
                else self._RECONCILE_MAX_RETRIES_DEFAULT
            )
            retry_minutes = (
                self.settings.reconcile_retry_minutes
                if self.settings is not None
                else self._RECONCILE_RETRY_MINUTES_DEFAULT
            )
            if retries >= max_retries:
                logger.error(
                    "reconcile_abandoned",
                    fixture_id=fixture_id,
                    status=status,
                    retries=retries,
                    max_retries=max_retries,
                    note="max reschedule attempts reached — leaving picks pending for manual review (Q3)",
                )
                return
            self.scheduler.add_job(
                self._reconcile_results,
                trigger=DateTrigger(run_date=datetime.now(UTC) + timedelta(minutes=retry_minutes)),
                args=[fixture, retries + 1],
                id=f"reconcile_{fixture_id}",
                replace_existing=True,
                misfire_grace_time=600,
            )
            logger.info("reconcile_rescheduled", fixture_id=fixture_id, status=status,
                        retries=retries + 1)
            return

        home = item["goals"]["home"]
        away = item["goals"]["away"]

        if home is None or away is None:
            logger.error("reconcile_missing_goals", fixture_id=fixture_id, status=status,
                         home=home, away=away)
            return

        if status in self._STATUS_SETTLED_PEN:
            actual = "X"
            push_market = True
        elif status in (self._STATUS_SETTLED_REGULATION | self._STATUS_SETTLED_REGULATION_PLUS_ET):
            actual = "1" if home > away else ("2" if away > home else "X")
            push_market = False
        else:
            logger.error("reconcile_unknown_status", fixture_id=fixture_id, status=status,
                         note="leaving picks pending for manual review (Risk 9)")
            return

        pending = self.pick_repo.get_pending_for_fixture(fixture_id)
        settled = 0
        for pick in pending:
            raw = pick.get("market")
            if raw is None:
                continue
            try:
                if MarketKey.from_str(raw) != MarketKey.ONEXTWO:
                    continue
            except ValueError:
                continue
            if push_market:
                new_status = "push"
            else:
                new_status = "won" if pick["selection"] == actual else "lost"
            self.pick_repo.update_status(pick["id"], new_status)
            settled += 1
        logger.info("reconcile_settled", fixture_id=fixture_id, status=status, count=settled)

    # ─────────────────────────────────────────────────────────
    # Phase 4 cron wrappers (D-09–D-12, D-13, D-14)
    # ─────────────────────────────────────────────────────────

    async def _check_clv_trend(self) -> None:
        """Wrapper: delegate to ClvTrendChecker.check (D-09–D-12)."""
        if self._clv_trend_checker is None:
            return
        try:
            await self._clv_trend_checker.check()
        except Exception as exc:
            logger.error("clv_trend_check_failed", error=str(exc))

    async def _aggregate_metrics(self) -> None:
        """Wrapper: delegate to MetricsAggregator.run (D-13)."""
        if self._metrics_aggregator is None:
            return
        try:
            await self._metrics_aggregator.run()
        except Exception as exc:
            logger.error("metrics_aggregation_failed", error=str(exc))

    async def _check_drift(self) -> None:
        """Wrapper: weekly drift check (D-14).

        Delegates to self._drift_checker (constructed in 04-05 builder.py). When
        _drift_checker is None (test fixtures or partial wiring), log a skip
        and return — never crashes.
        """
        if self._drift_checker is None:
            logger.info("drift_check_skipped", reason="no_drift_checker_provided")
            return
        try:
            logger.info("drift_check_started", date=date.today().isoformat())
            await self._drift_checker.run()
        except Exception as exc:
            logger.error("drift_check_failed", error=str(exc))

    def get_registered_jobs(self) -> list:
        return self.scheduler.get_jobs()

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=True)
        logger.info("scheduler_stopped")
