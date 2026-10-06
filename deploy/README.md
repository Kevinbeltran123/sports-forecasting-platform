# Production Deployment (Linux VPS)

Deployment artifacts for the platform (Phase 4). The pipeline runs as a single systemd service (`bip.service`) on a Linux VPS.

## Architecture (one-line)

`systemd (Type=notify, WatchdogSec=600) → python -m bip.production → AsyncIOScheduler (8 jobs: 4 from Phases 1-3 + 4 from Phase 4) → Supabase + Telegram + APIs`

## Prerequisites

- Ubuntu/Debian 22.04+ (systemd ≥ 219; supports `Type=notify` + `WatchdogSec=`)
- Python 3.12 + `uv` installed
- Repo cloned to `/opt/bip` (or override via `REPO_ROOT=/path bash deploy/install.sh`)
- `cd /opt/bip && uv sync` to build `.venv`
- Supabase project linked + migrations applied (including 005 — `supabase db push`)
- Two Telegram channels created: picks channel + ops channel; bot added to both
- API keys obtained: API-Football Pro, Odds API Rookie, Anthropic Sonnet

## Install

```bash
cd /opt/bip
sudo bash deploy/install.sh
```

The script is idempotent — re-run to validate after editing `/etc/bip/.env`.

It creates:
- `bip` system user (no shell, home /opt/bip)
- `/var/run/bip/` (mode 0750, owner bip:bip — heartbeat file lives here)
- `/etc/bip/.env` (mode 600, owner bip:bip — NEVER overwritten on re-run)
- `/etc/systemd/system/bip.service`
- `/etc/tmpfiles.d/bip.conf`

It does NOT auto-start the service — the operator confirms `.env` first.

## .env contents

After install creates an empty `.env`, populate:

```
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<supabase_key>
API_FOOTBALL_KEY=<api_football_pro_key>
ODDS_API_KEY=<odds_api_rookie_key>
ANTHROPIC_API_KEY=<anthropic_key>
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_CHANNEL_ID=-100<picks_channel_id>
TELEGRAM_OPS_CHANNEL_ID=-100<ops_channel_id>
```

Optional Phase 4 tuning (defaults are sensible):

```
CLAUDE_FAILURE_MODE=filter            # default; "skip" once CLV track record justifies
CLV_TREND_ALERT_THRESHOLD=1.0
DRIFT_CHECK_MIN_PICKS_PER_WINDOW=30
DRIFT_ABSOLUTE_THRESHOLD_PP=2.0
DRIFT_STDEV_FLOOR_PP=1.0
HEARTBEAT_FILE_PATH=/var/run/bip/heartbeat
```

## Start

```bash
sudo systemctl start bip.service
journalctl -u bip.service -f
```

Expect within 30 seconds:
```
production_started heartbeat_path=/var/run/bip/heartbeat ops_channel_configured=True
scheduler_started jobs=N
```

## Health checks

```bash
systemctl status bip.service             # active (running)
stat -c '%Y' /var/run/bip/heartbeat      # mtime advances every 5 min
journalctl -u bip.service | grep heartbeat | tail -5
journalctl -u bip.service | grep auto_recover_complete
```

## Stop / restart

```bash
sudo systemctl stop bip.service          # SIGTERM → graceful shutdown (~30s)
sudo systemctl restart bip.service       # auto-recover on next boot via _auto_recover
```

## Operational alerts

| Channel | Source | Examples |
|---------|--------|----------|
| Picks (TELEGRAM_CHANNEL_ID) | PickEngine | qualified predictions that passed the edge and validation gates |
| Ops (TELEGRAM_OPS_CHANNEL_ID) | ClvTrendChecker, DriftChecker | `⚠️ CLV +0.7% < +1% threshold`, `⚠️ DRIFT [league=PL market=onextwo] -2.3pp (3.5% → 1.2%)` |

When ops alerts fire: pause alert delivery, audit recent predictions, run `pytest tests/integration/` for regression sanity.

## Troubleshooting

**Hardening — `MemoryDenyWriteExecute`** — disabled by default in `bip.service` because CatBoost/XGBoost/LightGBM use `mmap(PROT_EXEC)` for CPU-dispatch JIT and the directive blocks them with `mmap: Permission denied`. Enable only after running `deploy/SMOKE_TEST.md` Check 5 against the same host with all ML wheels actually imported. (RESEARCH §Pitfall 8)

**`WatchdogSec` triggers restarts every 10 min** — `Type=notify` is paired with `WatchdogSec` but `sd_notify("WATCHDOG=1")` isn't reaching systemd. Check: `journalctl -u bip.service | grep -i notify`; verify the `sdnotify` Python package is installed (`uv pip show sdnotify`).

**Empty heartbeat file but service is running** — APScheduler heartbeat job not registered. Check `journalctl -u bip.service | grep scheduler_started` for the job count; should be 8 (Phases 1-3 = 2 + Phase 4 = 4 + per-fixture jobs as fixtures are added).

**`/etc/bip/.env permission denied`** — file mode is 600 owned by bip:bip; only bip can read it. To edit: `sudo -u root vim /etc/bip/.env` (root can read 600 files), then `chown bip:bip /etc/bip/.env && chmod 600 /etc/bip/.env`.
