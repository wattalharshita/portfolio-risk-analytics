#!/usr/bin/env bash
#
# refresh_data.sh — scheduled market-data refresh.
# Activates the venv, runs ingestion, timestamps all output to a rotating log,
# and exits with a meaningful status code so cron/monitoring can detect failure.
#
# Cron (weekdays at 18:30, after US market close):
#   30 18 * * 1-5 /path/to/portfolio-risk-analytics/scripts/refresh_data.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/refresh_$(date +%Y%m%d).log"
ERR_FILE="${LOG_DIR}/refresh_errors.log"
SOURCE="${1:-yfinance}"

mkdir -p "${LOG_DIR}"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_FILE}"; }

on_error() {
    local exit_code=$?
    local line=$1
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] FAILED at line ${line} (exit ${exit_code})" \
        | tee -a "${ERR_FILE}" >&2
    exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

log "=== refresh start (source=${SOURCE}) ==="
cd "${PROJECT_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    log "virtualenv activated: $(python --version 2>&1)"
else
    log "WARNING: no .venv found, using system python"
fi

log "running ingestion..."
python src/ingest.py --source "${SOURCE}" 2>&1 | tee -a "${LOG_FILE}"

ROWS=$(python - <<'PYEOF'
import sqlite3
conn = sqlite3.connect("data/portfolio.db")
print(conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0])
PYEOF
)
log "warehouse now holds ${ROWS} price rows"

# Rotate: keep 30 days of logs
find "${LOG_DIR}" -name 'refresh_*.log' -mtime +30 -delete 2>/dev/null || true

log "=== refresh complete ==="
exit 0
