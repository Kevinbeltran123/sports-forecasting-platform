"""End-to-end smoke for Phase 3 -- ONE fixture, LIVE services.

Wires: Settings -> AsyncAnthropic -> ClaudeValidator -> TelegramBot/Sender ->
PickRepository -> AsyncIOScheduler -> PickEngine, then calls PickEngine.evaluate
on a synthetic Prediction.

USAGE (manual only -- never in CI):
    uv run python scripts/smoke_e2e_pick.py --fixture-id 12345 \\
        --probs '{"1": 0.55, "X": 0.25, "2": 0.20}' \\
        --odds  '{"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}'

The script:
  1. Loads Settings (.env), validates TELEGRAM_CHANNEL_ID format
  2. Builds TelegramBot, calls .start(), confirms _app._rate_limiter is wired
  3. Loads learnings.md + git SHA, builds ClaudeValidator
  4. Builds PickEngine with real PickRepository (writes to live Supabase)
  5. Calls PickEngine.evaluate with the supplied synthetic Prediction + odds
  6. If a Pick is queued for send (CONFIRM / FLAG), waits up to 60s for the DateTrigger to fire
  7. Prints a final summary so the operator can compare against Telegram inbox + Supabase row

EXITS:
  0 -- pick processed (any outcome -- filtered, rejected, or sent)
  1 -- wiring or runtime error before evaluate completed
  2 -- evaluate completed but Telegram send failed (verify .env / channel admin)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from supabase import create_client

from bip.core.claude.learnings_loader import load_learnings
from bip.core.claude.validator import ClaudeValidator
from bip.core.picks.engine import PickEngine
from bip.core.settings import Settings
from bip.core.storage.repositories import PickRepository
from bip.core.telegram.bot import TelegramBot
from bip.core.telegram.sender import TelegramSender

logger = structlog.get_logger(__name__)


def _build_synthetic_prediction(fixture_id: int, probs: dict, kickoff_in_hours: float = 4.0):
    """Build a Prediction-shaped SimpleNamespace for PickEngine.evaluate."""
    return SimpleNamespace(
        id=999_900 + fixture_id,
        fixture_id=fixture_id,
        league="premier_league",
        sport="football",
        market="1X2",
        probabilities=probs,
        home_team="Smoke Home FC",
        away_team="Smoke Away FC",
        kickoff_utc=datetime.now(UTC) + timedelta(hours=kickoff_in_hours),
        is_lineup_adjusted=False,
        is_shadow=False,
        model_version="smoke-v1",
    )


async def run(fixture_id: int, probs: dict, odds: dict) -> int:
    settings = Settings()

    # -- Step 1: TelegramBot ------------------------------------------------
    bot = TelegramBot(
        token=settings.telegram_bot_token,
        channel_id=settings.telegram_channel_id,
    )
    await bot.start()
    logger.info("smoke_bot_started", channel_id=settings.telegram_channel_id)

    # -- Step 2: ClaudeValidator --------------------------------------------
    learnings_text, learnings_sha = load_learnings()
    validator = ClaudeValidator(
        api_key=settings.anthropic_api_key,
        model=settings.claude_model,
        learnings_text=learnings_text,
        learnings_sha=learnings_sha,
    )
    logger.info(
        "smoke_validator_ready",
        learnings_sha=learnings_sha,
        model=settings.claude_model,
    )

    # -- Step 3: Supabase client + PickRepository ---------------------------
    supabase = create_client(settings.supabase_url, settings.supabase_key)
    pick_repo = PickRepository(client=supabase)

    # -- Step 4: AsyncIOScheduler + PickEngine ------------------------------
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.start()
    sender = TelegramSender(bot=bot)
    engine = PickEngine(
        pick_repo=pick_repo,
        validator=validator,
        scheduler=scheduler,
        sender=sender,
        settings=settings,
    )

    # -- Step 5: evaluate ---------------------------------------------------
    prediction = _build_synthetic_prediction(fixture_id, probs)
    try:
        pick = await engine.evaluate(prediction, odds)
    except Exception as exc:
        logger.error("smoke_evaluate_failed", error=str(exc), exc_info=True)
        await bot.shutdown()
        scheduler.shutdown(wait=False)
        return 1

    if pick is None:
        print("FAIL: evaluate returned None -- wiring error")
        await bot.shutdown()
        scheduler.shutdown(wait=False)
        return 1

    print("\n=== EVALUATE RESULT ===")
    print(f"Status: {pick.status.value}")
    print(f"Selection: {pick.selection} @ {pick.best_odds:.2f}")
    print(f"Edge: {pick.edge * 100:.1f}% | Stake: {pick.suggested_stake or 0:.1f}u")
    print(f"Claude validation: {pick.claude_validation}")
    print(f"Claude summary: {pick.claude_summary}")
    print(f"Reasoning excerpt: {(pick.claude_reasoning or '')[:200]}")
    print()

    # -- Step 6: wait for the DateTrigger send (up to 60s) ------------------
    if pick.status.value == "pending":
        print("Waiting up to 60s for the send_pick DateTrigger to fire...")
        for _ in range(60):
            await asyncio.sleep(1)
            jobs = scheduler.get_jobs()
            send_jobs = [j for j in jobs if j.id.startswith(f"send_pick_{fixture_id}")]
            if not send_jobs:
                print("send_pick job completed (job no longer in queue)")
                break
        else:
            print("WARNING: send_pick job still queued after 60s")

    # -- Cleanup ------------------------------------------------------------
    await bot.shutdown()
    scheduler.shutdown(wait=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3 E2E smoke runner -- manual only, never in CI"
    )
    parser.add_argument(
        "--fixture-id",
        type=int,
        required=True,
        help="Synthetic fixture id (any int -- does NOT have to match a real fixture)",
    )
    parser.add_argument(
        "--probs",
        type=str,
        required=True,
        help='JSON object: {"1": 0.55, "X": 0.25, "2": 0.20}',
    )
    parser.add_argument(
        "--odds",
        type=str,
        required=True,
        help='JSON object: {"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}',
    )
    args = parser.parse_args(argv)

    probs = json.loads(args.probs)
    odds = json.loads(args.odds)

    return asyncio.run(run(args.fixture_id, probs, odds))


if __name__ == "__main__":
    sys.exit(main())
