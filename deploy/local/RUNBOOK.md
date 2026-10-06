# Local Run Mode (tmux)

This runbook is the **default operating mode** until the models have a validated out-of-sample track record. Until then, the pipeline runs on your laptop in a `tmux` session — no VPS, no systemd, no monthly bill.

## Why this mode?

- **No infra cost** while model performance is unproven. ~€0/mo vs €5-10/mo for Hetzner.
- **Zero deployment ceremony.** Pull, run, iterate.
- **Same code, same Settings.** `python -m bip.production` runs end-to-end on macOS/Linux without modification — `sdnotify` no-ops, signal handlers work, AsyncIOScheduler is portable.
- **Cost: ~30% missed coverage.** You don't run while sleeping or working on other things. Early CLV recordings during weekday-evening fixtures may be missed. Acceptable trade-off during evaluation.

## Prerequisites

- macOS or Linux laptop with `tmux` installed (`brew install tmux` / `apt install tmux`)
- Repo cloned, `uv sync` run, `.venv` healthy
- Supabase project + migrations applied (same as VPS path)
- Two Telegram channels created + bot added; bot token in env
- API keys: API-Football Pro, Odds API Rookie, Anthropic Sonnet

## One-time setup

Create a local `.env` at the repo root (NOT `/etc/bip/.env` — that path is for the VPS install):

```bash
cp env.example .env
chmod 600 .env
```

Required keys (same as VPS install):

```
SUPABASE_URL=...
SUPABASE_KEY=...
API_FOOTBALL_KEY=...
ODDS_API_KEY=...
ANTHROPIC_API_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHANNEL_ID=...
TELEGRAM_OPS_CHANNEL_ID=...

# Local-mode overrides (NOT the VPS defaults)
HEARTBEAT_FILE_PATH=/tmp/bip_heartbeat
```

> The `HEARTBEAT_FILE_PATH` override matters: the VPS default `/var/run/bip/heartbeat` requires root or membership in the `bip` group, neither of which exists on your laptop. `/tmp/bip_heartbeat` is writable by your user.

## Starting a session

```bash
cd <repo-root>
tmux new -s bip
uv run python -m bip.production
```

Detach (keep running): **Ctrl-b d**.
Re-attach later: `tmux attach -t bip`.

Expected log lines within ~30s:
```
build_orchestrator_complete heartbeat_path=/tmp/bip_heartbeat ...
production_started ...
scheduler_started jobs=N
```

## Health checks (run from a separate shell)

```bash
# Process alive?
pgrep -fl "bip.production"

# Heartbeat advancing? (mtime should increase by ~300s on each check, 5 min apart)
stat -f '%m' /tmp/bip_heartbeat       # macOS
stat -c '%Y' /tmp/bip_heartbeat       # Linux

# Telegram bot reachable? Send a test from the picks channel — bot should ack.
```

## Stopping cleanly

Inside the tmux session, press **Ctrl-c** once. The asyncio signal handler catches SIGINT and runs the same graceful-shutdown path the VPS uses:
```
shutdown_started signal=SIGINT
shutdown_complete
```

If the process hangs (>30s), **Ctrl-c** again or `pkill -TERM -f bip.production` from another shell.

## What you GIVE UP vs the VPS path

| Capability | VPS (systemd) | Local tmux |
|---|---|---|
| Auto-restart on crash | ✅ `Restart=always` | ❌ — you re-run manually |
| Auto-start on host reboot | ✅ `WantedBy=multi-user.target` | ❌ — laptop reboot loses session |
| Watchdog (kill hung process) | ✅ `WatchdogSec=600` + sd_notify | ❌ — `sdnotify` no-ops on macOS |
| Coverage when you sleep / close laptop | ✅ 100% | ❌ ~70% |
| Auto-recovery of in-flight send jobs | ✅ Yes (via `_auto_recover`) | ⚠️ Only on next manual start |

The first three matter only **after** the system is producing real value. Until then, every restart you do manually is also a chance to read the journal output and notice issues.

## When to graduate from this mode

Move to the systemd path once rolling CLV and calibration metrics are stable over a sustained evaluation window.

## Operational tips

- **Run during high-fixture windows.** Weekend afternoons (EU football) and evening matchdays are when the pipeline actually has work. Mid-week Tuesday morning at 09:00 = no fixtures, nothing to do, save your laptop battery.
- **`tmux logging`.** Inside the session: `Ctrl-b :pipe-pane -o "cat >> ~/bip-session.log"` to mirror stdout to a file you can grep later.
- **Kill switch.** If a Telegram alert says something looks wrong: `Ctrl-c` in the tmux pane stops everything immediately. Cleaner than `kill -9`.
- **The `weekly_drift` job runs Mondays 06:00 UTC.** If you're never running on Monday mornings, you'll miss it. Acceptable in local mode — drift detection is a tail-risk alarm, not a hot path.
