"""Entry point: ``python -m bip.pipeline --dry-run``.

Sprint 3 Ola C. Smoke-runs the v4 stack with stubs so the wiring can be
verified without any external dependency. Fires one cycle of each
worker via PipelineScheduler.fire_now and prints a structured summary.

Usage:
    uv run python -m bip.pipeline --dry-run
    uv run python -m bip.pipeline --dry-run --date 2026-06-11
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta

import structlog

from bip.pipeline.factory import build_pipeline
from bip.pipeline.scheduler import (
    JOB_CLV,
    JOB_DELIVERY,
    JOB_ORCHESTRATOR,
)

log = structlog.get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bip.pipeline")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use in-memory stubs for Supabase / Claude / Telegram (default).",
    )
    p.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD date for the hydrator pass (default: today UTC).",
    )
    p.add_argument(
        "--no-mundial",
        action="store_true",
        help="Skip MundialModel registration.",
    )
    p.add_argument(
        "--no-ligas",
        action="store_true",
        help="Skip LigasModel registration.",
    )
    args = p.parse_args(argv)
    # Default to dry-run if neither flag is set (no live mode yet)
    args.dry_run = True
    return args


async def run_dry_smoke(args: argparse.Namespace) -> dict:
    """Wire the pipeline + fire one cycle of each worker; return a summary."""
    pipeline = build_pipeline(
        dry_run=True,
        include_ligas=not args.no_ligas,
        include_mundial=not args.no_mundial,
    )

    date = args.date or datetime.now(UTC).date().isoformat()

    async def _orchestrator_cycle():
        fixtures, hyd_summary = await pipeline.hydrator.hydrate_for_date(date)
        # Augment with a synthetic Mundial fixture so the dry-run shows
        # the registry actually routing into predictions_raw (the stub
        # ApiFootballClient returns []).
        if not args.no_mundial and pipeline.registry.__contains__("mundial_wc2026"):
            synth_fixture_id = pipeline.registry.get("mundial_wc2026").fixture_ids[0]
            fixtures.append(
                {
                    "fixture_id": synth_fixture_id,
                    "match_id": synth_fixture_id,
                    "competition": "WC2026",
                    "home_team": "Synthetic A",
                    "away_team": "Synthetic B",
                    "match_datetime": datetime.now(UTC) + timedelta(hours=2),
                    "features": None,
                }
            )
        orch_summary = await pipeline.orchestrator.run_for_fixtures(fixtures)
        return {
            "hydrator": {
                "n_fixtures": hyd_summary.n_fixtures_total,
                "n_leagues": hyd_summary.n_leagues_queried,
            },
            "orchestrator": {
                "n_predictions_emitted": orch_summary.n_predictions_emitted,
                "n_inserted": orch_summary.n_inserted,
                "n_errors": orch_summary.n_errors,
            },
        }

    async def _delivery_cycle():
        summary = await pipeline.delivery_worker.run_once()
        return {
            "n_pending_rows": summary.n_pending_rows,
            "n_shadow": summary.n_shadow,
            "n_sent": summary.n_sent,
            "n_killed": summary.n_killed,
        }

    async def _clv_cycle():
        summary = await pipeline.clv_worker.run_once()
        return {
            "n_candidate_rows": summary.n_candidate_rows,
            "n_measured": summary.n_measured,
        }

    pipeline.scheduler.register_orchestrator_daily(_orchestrator_cycle)
    pipeline.scheduler.register_delivery_worker_interval(_delivery_cycle)
    pipeline.scheduler.register_clv_worker_interval(_clv_cycle)

    log.info(
        "pipeline_dry_run_start",
        date=date,
        jobs=[j["id"] for j in pipeline.scheduler.jobs],
    )

    orchestrator_result = await pipeline.scheduler.fire_now(JOB_ORCHESTRATOR)
    delivery_result = await pipeline.scheduler.fire_now(JOB_DELIVERY)
    clv_result = await pipeline.scheduler.fire_now(JOB_CLV)

    return {
        "date": date,
        "dry_run": True,
        "models_registered": pipeline.registry.names(),
        "cycles": {
            JOB_ORCHESTRATOR: orchestrator_result,
            JOB_DELIVERY: delivery_result,
            JOB_CLV: clv_result,
        },
        "in_memory_rows": len(
            pipeline.in_memory_client._rows.get("predictions_raw", [])
        ),
        "telegram_messages_sent": len(pipeline.sender.sent),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = asyncio.run(run_dry_smoke(args))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
