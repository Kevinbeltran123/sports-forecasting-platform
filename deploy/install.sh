#!/usr/bin/env bash
# deploy/install.sh — idempotent BIP production setup (Phase 4 D-06).
#
# Steps (RESEARCH §Deployment Runbook):
#   1. Pre-flight checks (root, systemd, .venv exists)
#   2. Create bip system user (idempotent)
#   3. Create /var/run/bip via tmpfiles.d
#   4. Create /etc/bip/.env (NEVER overwrite — operator-managed secrets)
#   5. Validate .env required keys (grep -q only — never echoes values)
#   6. Install systemd unit + reload
#   7. Print next-step instructions (no auto-start)
#
# T-4-02 mitigation: never overwrites existing /etc/bip/.env (preserves mode 600).
# T-4-05 mitigation: creates dedicated `bip` system user.
# T-4-06 mitigation: env validation uses `grep -q` (silent) — values never echoed.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/opt/bip}"
SYSTEMD_DIR="/etc/systemd/system"
TMPFILES_DIR="/etc/tmpfiles.d"
ENV_DIR="/etc/bip"
ENV_FILE="${ENV_DIR}/.env"

# 1. Pre-flight checks
[ "$EUID" -eq 0 ] || { echo "ERROR: must run as root (sudo bash deploy/install.sh)"; exit 1; }
command -v systemctl >/dev/null 2>&1 || { echo "ERROR: systemd required"; exit 1; }
[ -f "${REPO_ROOT}/.venv/bin/python" ] || {
    echo "ERROR: ${REPO_ROOT}/.venv not built — cd ${REPO_ROOT} && uv sync first"
    exit 1
}

# 2. Create bip system user (idempotent)
if ! id -u bip >/dev/null 2>&1; then
    useradd -r -s /sbin/nologin -d "${REPO_ROOT}" bip
    echo "Created system user 'bip'."
else
    echo "System user 'bip' already exists — skipping."
fi

# 3. Install tmpfiles.d snippet + create /var/run/bip immediately
install -m 644 deploy/tmpfiles.d/bip.conf "${TMPFILES_DIR}/bip.conf"
systemd-tmpfiles --create "${TMPFILES_DIR}/bip.conf"
# Belt-and-braces: ensure /var/run/bip exists right now (in case tmpfiles is lazy).
install -d -o bip -g bip -m 750 /var/run/bip

# 4. Create /etc/bip and /etc/bip/.env (T-4-02: do NOT overwrite if exists)
install -d -o root -g root -m 755 "${ENV_DIR}"
if [ ! -f "${ENV_FILE}" ]; then
    install -m 600 -o bip -g bip /dev/null "${ENV_FILE}"
    echo ""
    echo "=================================================================="
    echo "  Created empty ${ENV_FILE}. Edit it now to set required keys:"
    echo "    SUPABASE_URL, SUPABASE_KEY"
    echo "    API_FOOTBALL_KEY, ODDS_API_KEY"
    echo "    ANTHROPIC_API_KEY"
    echo "    TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID, TELEGRAM_OPS_CHANNEL_ID"
    echo "  Then re-run this script to validate keys."
    echo "=================================================================="
    exit 0
else
    echo "${ENV_FILE} already exists — preserving (mode 600 owner bip)."
fi

# 5. Validate .env required keys have NON-EMPTY values (T-4-06: never echo values)
# Regex requires `KEY=` followed by at least one non-whitespace char so commented
# (`#KEY=...`) and empty (`KEY=`) lines both fail validation early.
REQUIRED_KEYS=(
    SUPABASE_URL SUPABASE_KEY
    API_FOOTBALL_KEY ODDS_API_KEY
    ANTHROPIC_API_KEY
    TELEGRAM_BOT_TOKEN TELEGRAM_CHANNEL_ID TELEGRAM_OPS_CHANNEL_ID
)
MISSING=()
for KEY in "${REQUIRED_KEYS[@]}"; do
    if ! grep -qE "^${KEY}=[[:space:]]*[^[:space:]#]" "${ENV_FILE}"; then
        MISSING+=("${KEY}")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: ${ENV_FILE} missing or empty required keys: ${MISSING[*]}"
    exit 1
fi
echo "All required keys present and non-empty in ${ENV_FILE}."

# 6. Install systemd unit + reload
install -m 644 deploy/systemd/bip.service "${SYSTEMD_DIR}/bip.service"
systemctl daemon-reload
systemctl enable bip.service

# 7. Print next-step instructions (no auto-start — operator confirms .env first)
echo ""
echo "=================================================================="
echo "  Install complete."
echo "  Next steps:"
echo "    1. Review ${ENV_FILE} (mode 600, owner bip)."
echo "    2. systemctl start bip.service"
echo "    3. journalctl -u bip.service -f"
echo ""
echo "  Health checks:"
echo "    systemctl status bip.service       # active (running)?"
echo "    stat -c '%Y' /var/run/bip/heartbeat  # mtime advances every 5 min"
echo "    journalctl -u bip.service | grep production_started"
echo "=================================================================="
