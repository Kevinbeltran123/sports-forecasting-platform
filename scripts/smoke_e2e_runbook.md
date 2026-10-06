# Phase 3 -- End-to-End Smoke Runbook

**Owner:** project maintainer
**Frequency:** Once per Phase 3 verification (and any time live wiring changes)
**Cost:** ~$0.01 Anthropic + 1 Telegram message
**Companion script:** `scripts/smoke_e2e_pick.py`

---

## Pre-flight checklist

Before running the smoke, confirm the following:

- [ ] Migration 004 applied to live Supabase (run `uv run python scripts/verify_migration_004.py` -- expect exit 0)
- [ ] `.env` populated with TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID (with `-100` prefix), ANTHROPIC_API_KEY, CLAUDE_MODEL
- [ ] Telegram bot is ADMIN of the channel (not just a member); confirm by sending one manual message via the bot -- if it fails with `Forbidden: bot is not a member of the channel chat`, fix permissions before continuing
- [ ] Anthropic API key is valid and has available credit
- [ ] All Phase 3 unit tests GREEN (`uv run pytest tests/picks/ tests/claude/ tests/telegram/ tests/scheduler/ tests/scripts/ -x -q`)

---

## Smoke invocation

```bash
uv run python scripts/smoke_e2e_pick.py \
    --fixture-id 99999 \
    --probs '{"1": 0.60, "X": 0.25, "2": 0.15}' \
    --odds  '{"1": 1.85, "X": 3.40, "2": 4.20, "bookmaker": "betano"}'
```

The probs above (1=60%, X=25%, 2=15%) versus odds 1.85/3.40/4.20 give:
- Implied prob 1: 1/1.85 ~= 54%; model 60% -> edge ~= +11% (well above 5% threshold)
- Should produce a CONFIRM or FLAG verdict from Claude (REJECT possible if learnings flag the synthetic context)

---

## Verification -- what to confirm AFTER the script exits

### 1. Console output (immediate)

Expect a printed `=== EVALUATE RESULT ===` block listing:
- `Status: pending` (CONFIRM/FLAG path) OR `filtered` (no_edge / market_cap / claude_api_unavailable) OR `rejected` (REJECT)
- `Selection: 1 @ 1.85` (or X / 2)
- `Edge: +11.4%` (approximately)
- `Stake: 0.5u` or `1.0u` etc (quarter-Kelly, rounded to 0.5u)
- `Claude validation: CONFIRM` or `FLAG` or `REJECT`

If `Status: pending`, the script then waits up to 60s for the DateTrigger send to fire.

### 2. Telegram channel (manual visual check)

- [ ] Open the private Telegram channel
- [ ] Confirm ONE new message arrived
- [ ] Headline: `<b>Smoke Home FC vs Smoke Away FC</b>` (bold)
- [ ] If FLAG: leading WARNING emoji + `Claude flagged: <reason_code>` line
- [ ] Market line: `1X2: 1 @ 1.85` (selection + odds)
- [ ] Edge + Stake line
- [ ] Up to 3 reasoning bullets from Claude's `summary`
- [ ] Footer: model version + Claude verdict tag
- [ ] Total length under ~600 chars (mobile-readable, no truncation)

If REJECT or filtered: NO Telegram message should appear (correct behaviour per D-03 / D-14).

### 3. Supabase picks table

```bash
psql "$(uv run python -c 'from bip.core.settings import Settings; from scripts.verify_migration_004 import derive_db_url; import os; print(derive_db_url(Settings().supabase_url, os.environ["SUPABASE_DB_PASSWORD"]))')" \
    -c "select id, fixture_id, market, status, claude_validation, claude_summary, claude_validated_at from picks where fixture_id = 99999 order by id desc limit 1;"
```

- [ ] Exactly 1 row for fixture_id=99999
- [ ] status matches console verdict (`pending`, `filtered`, or `rejected`)
- [ ] claude_validation matches (`CONFIRM` / `FLAG` / `REJECT` / NULL for filtered/no-edge)
- [ ] claude_summary present (CONFIRM/FLAG/REJECT paths)
- [ ] claude_validated_at is a recent ISO timestamp

### 4. Learnings SHA audit (CLAUDE-01 audit)

The smoke run logs `learnings_sha=<40-char-hex>` via structlog. Cross-check against:

```bash
git log -1 --format=%H -- src/bip/core/claude/prompts/football-learnings.md
```

The two SHAs MUST match. If they differ, learnings.md was modified after Claude validated -- investigate.

### 5. Reconciliation (optional -- only if you want to test the post-match flow)

The smoke uses a kickoff 4 hours in the future, so reconciliation will not fire automatically. To test reconciliation:

1. Manually update the picks row to set `created_at` 3+ hours ago (so it qualifies for the next reconcile cycle)
2. Trigger `_reconcile_results` manually:
   ```bash
   uv run python -c "
   import asyncio
   from datetime import UTC, datetime, timedelta
   from types import SimpleNamespace
   from supabase import create_client
   from bip.core.settings import Settings
   from bip.core.storage.repositories import PickRepository
   from bip.core.scheduler.orchestrator import PipelineOrchestrator
   from bip.sports.football.client import ApiFootballClient
   # ... wire pipeline orchestrator with pick_repo + api_football_client, call _reconcile_results
   "
   ```
3. Confirm pick row's status updated to `won`/`lost`/`void`/`push`

---

## Failure modes & next steps

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `FAIL: evaluate returned None` | Wiring or runtime error before persistence | Check console traceback; ensure all `uv sync` deps installed |
| Telegram message never arrives | Bot not admin OR channel_id wrong | Re-check channel admin permissions; verify channel_id with @userinfobot |
| Claude returns None (filtered/claude_api_unavailable) | Anthropic API outage or insufficient credit | Check status.anthropic.com; check credit balance |
| Supabase row missing | Settings.supabase_url/key wrong, OR migration 004 not applied | Run verify_migration_004.py |
| `claude_validation` value violates CHECK constraint | Migration 004 picks_claude_validation_check missing or older | Re-apply migration 004 |
| Multiple rows for same fixture_id | Idempotency broken (specifics §195) | File a bug -- engine should produce ONE row per (fixture_id, market, prediction_id) |

---

## Sign-off

Once steps 1-4 above pass, mark Phase 3 as verified. Reconciliation (step 5) can be skipped if running close to the wire -- the unit tests in tests/scheduler/test_reconcile.py already cover the settlement logic.

Date smoked: _(YYYY-MM-DD)_
Outcome: _(pass / fail)_
Notes: _(free-form)_
