#!/usr/bin/env bash
# start_worker.sh
# Start the beyondLDA2 Celery worker on the compute server.
#
# Prerequisites:
#   1. SSH tunnel to cloud server is active (tunnel.sh start)
#   2. Python venv with GPAW + worker deps is activated
#
# Usage:
#   source /path/to/gpaw-venv/bin/activate
#   ./start_worker.sh
#
# Or with env vars:
#   DB_PASSWORD=xxx MPI_NPROCS=8 CONCURRENCY=4 ./start_worker.sh

set -euo pipefail

# ── Configuration (override via environment) ────────────────────────────
# These default to localhost because the SSH tunnel forwards cloud ports.
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-beyondlda2}"
DB_USER="${DB_USER:-beyondlda2}"
DB_PASSWORD="${DB_PASSWORD:?Set DB_PASSWORD}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6381}"

WORK_DIR="${WORK_DIR:-/tmp/beyondlda2-jobs}"
CONCURRENCY="${CONCURRENCY:-1}"

# ── Validate tunnel ──────────────────────────────────────────────────────
echo "Checking SSH tunnel ..."
if command -v nc &>/dev/null; then
  nc -z "${DB_HOST}" "${DB_PORT}" 2>/dev/null \
    && echo "  PostgreSQL: OK (${DB_HOST}:${DB_PORT})" \
    || { echo "  ERROR: Cannot reach PostgreSQL at ${DB_HOST}:${DB_PORT} — is tunnel running?"; exit 1; }
  nc -z "${REDIS_HOST}" "${REDIS_PORT}" 2>/dev/null \
    && echo "  Redis:       OK (${REDIS_HOST}:${REDIS_PORT})" \
    || { echo "  ERROR: Cannot reach Redis at ${REDIS_HOST}:${REDIS_PORT} — is tunnel running?"; exit 1; }
fi

# ── Ensure work dir exists ───────────────────────────────────────────────
mkdir -p "${WORK_DIR}"

# ── Export for Celery worker ─────────────────────────────────────────────
export DB_HOST DB_PORT DB_NAME DB_USER DB_PASSWORD
export REDIS_HOST REDIS_PORT
export WORK_DIR
export PYTHON_PATH="${PYTHON_PATH:-/home/clackyuser/clacky_workspace/beyondLDA2}"

# ── Start worker ─────────────────────────────────────────────────────────
echo ""
echo "Starting Celery worker (concurrency=${CONCURRENCY}) ..."
echo "  Broker:  redis://${REDIS_HOST}:${REDIS_PORT}/0"
echo "  DB:      postgresql://${DB_USER}:**@${DB_HOST}:${DB_PORT}/${DB_NAME}"
echo "  Workdir: ${WORK_DIR}"
echo ""

exec celery -A server.worker worker \
  --loglevel=info \
  --concurrency="${CONCURRENCY}"
