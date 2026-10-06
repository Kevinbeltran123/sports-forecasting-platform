# BIP Phase 4 — Staging Smoke Test

Run this once on a fresh Hetzner VPS before declaring Phase 4 verifiable.
Each check has explicit expected output. If any check fails, fix before continuing.

## Prerequisites

- Hetzner Ubuntu/Debian 22.04+ with `/opt/bip` cloned + `.venv` built
- `bash deploy/install.sh` completed without error
- `/etc/bip/.env` populated with all 8 required keys (see `deploy/README.md`)
- Supabase migration 005 already pushed (verified in 04-00-PLAN Task 4)

## Smoke-Test Checklist

### Check 1: systemd unit syntax (T-4-05 verification)

```bash
systemd-analyze verify deploy/systemd/bip.service && echo OK
```

**Expected:** `OK` (no errors). If `systemd-analyze verify` outputs warnings about deprecated directives, those are fine; ERRORS are not.

Also check `User=bip` is set:
```bash
systemctl show bip.service -p User
```

**Expected:** `User=bip` (NOT `User=root` — T-4-05).

### Check 2: First-run `.env` permissions (T-4-02 verification)

```bash
stat -c '%a %U:%G' /etc/bip/.env
```

**Expected:** `600 bip:bip`.

### Check 3: Re-run install.sh preserves `.env` (T-4-02 regression check)

```bash
EXISTING_HASH=$(md5sum /etc/bip/.env | cut -d' ' -f1)
sudo bash deploy/install.sh   # idempotent re-run
NEW_HASH=$(md5sum /etc/bip/.env | cut -d' ' -f1)
[ "$EXISTING_HASH" = "$NEW_HASH" ] && echo "OK — env preserved" || echo "FAIL — env was modified"
stat -c '%a %U:%G' /etc/bip/.env
```

**Expected:** `OK — env preserved` AND `600 bip:bip` (mode unchanged).

### Check 4: Heartbeat directory permissions (T-4-04 verification)

```bash
stat -c '%a %U:%G' /var/run/bip
```

**Expected:** `750 bip:bip` (NOT `755`, NOT root-owned).

### Check 5: Service starts and signals READY=1 (Type=notify verification)

```bash
sudo systemctl start bip.service
sleep 5
journalctl -u bip.service -n 50 | grep -E "Got notification message from PID|production_started"
```

**Expected:** at least 2 lines:
- `... systemd[1]: bip.service: Got notification message from PID NNNN ...` (sd_notify READY=1 received)
- `... bip.production: production_started heartbeat_path=/var/run/bip/heartbeat ops_channel_configured=True`

If `Got notification message` is missing → `sdnotify` not reaching systemd → check `Type=notify` and `NotifyAccess=main` in unit.

### Check 6: Heartbeat file mtime advances every 5 min (D-07 verification)

```bash
stat -c '%Y' /var/run/bip/heartbeat   # initial mtime (seconds since epoch)
sleep 360                              # wait 6 min (one full tick + buffer)
stat -c '%Y' /var/run/bip/heartbeat   # new mtime
```

**Expected:** Second mtime is ≥ 300s greater than first (one tick at minute interval). If file doesn't exist or mtime didn't advance: heartbeat job not registered or `HeartbeatTicker.tick` raising.

### Check 7: No API keys in journal (T-4-06 verification)

```bash
journalctl -u bip.service --since "10 minutes ago" | grep -cE "sk-ant-|sk-or-|Bearer [a-zA-Z0-9]{20,}|TELEGRAM_BOT_TOKEN=" || echo "0"
```

**Expected:** `0` (no API key prefixes anywhere in the journal).

### Check 8: Graceful shutdown completes within 30s (D-05 verification)

```bash
START=$(date +%s)
sudo systemctl stop bip.service
END=$(date +%s)
echo "Shutdown took $((END - START)) seconds"
journalctl -u bip.service --since "1 minute ago" | grep -E "shutdown_started|shutdown_complete"
```

**Expected:**
- Shutdown duration < 30s
- `shutdown_started signal=SIGTERM` and `shutdown_complete` both present in journal
- `systemctl status bip.service` shows `inactive (dead)` and exit code 0.

## After all 8 checks pass

1. Restart service: `sudo systemctl start bip.service`
2. Wait 24h with operator-monitoring journal output via `journalctl -u bip.service -f`
3. Verify daily metrics aggregator runs at 23:00 UTC: `journalctl -u bip.service | grep metrics_aggregation_complete`
4. Run `pytest tests/integration/ -x` against the live database to verify the RPC contract

## Optional: Auto-restart verification (manual)

To confirm `WatchdogSec=600` triggers restart on hung process:

```bash
# Pause the python process to simulate a hang (TSTP suspends without crashing).
sudo pkill -STOP -f "bip.production"
# Wait 10+ min; systemd should kill + restart.
sudo journalctl -u bip.service -f | grep -E "WATCHDOG|RestartSec"
```

**Expected within 10 min:** `Watchdog timeout` or similar + `Service restart triggered` + new PID in `systemctl status`. If nothing happens: `Type=notify` not actually wired (check Pitfall 1).

## Optional: Host reboot recovery (manual)

```bash
sudo systemctl reboot
# After VPS reboots, log back in:
systemctl is-active bip.service       # should be 'active'
journalctl -u bip.service | grep auto_recover_complete  # should appear with jobs_re_queued count
```

## Sign-off

When all 8 numbered checks pass + optional auto-restart and host-reboot succeed, Phase 4 is verifiable.
Record the results as the formal verification handoff.
