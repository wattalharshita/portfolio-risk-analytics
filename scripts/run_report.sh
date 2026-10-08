#!/usr/bin/env bash
#
# run_report.sh — regenerate the full analysis and archive a dated report artefact.
#
# Cron (Mondays at 07:00):
#   0 7 * * 1 /path/to/portfolio-risk-analytics/scripts/run_report.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${REPORT_DIR}/risk_report_${STAMP}.txt"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

trap 'echo "[$(date -u "+%Y-%m-%dT%H:%M:%SZ")] report FAILED at line ${LINENO}" \
    | tee -a "${LOG_DIR}/report_errors.log" >&2' ERR

cd "${PROJECT_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

log "generating analysis report -> ${REPORT_FILE}"
{
    echo "Portfolio Risk & Returns Analytics — generated $(date -u '+%Y-%m-%d %H:%M:%SZ')"
    echo
    python scripts/run_analysis.py
} > "${REPORT_FILE}"

log "running test suite"
python -m pytest tests/ -q >> "${REPORT_FILE}" 2>&1

# Keep the 12 most recent reports
ls -1t "${REPORT_DIR}"/risk_report_*.txt 2>/dev/null | tail -n +13 | xargs -r rm --

log "report archived: ${REPORT_FILE} ($(wc -l < "${REPORT_FILE}") lines)"
exit 0
