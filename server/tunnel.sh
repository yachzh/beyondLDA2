#!/usr/bin/env bash
# tunnel.sh
# SSH tunnel from compute server → cloud server.
# Forwards PostgreSQL (5432) and Redis (6381) ports so the Celery worker
# can connect to the cloud server's services via localhost.
#
# Usage:
#   ./tunnel.sh start        # Establish tunnel (background)
#   ./tunnel.sh status       # Check if tunnel is running
#   ./tunnel.sh stop         # Kill the tunnel
#   ./tunnel.sh restart      # Restart the tunnel
#
# Environment variables (set in .env or export):
#   CLOUD_SSH_USER           SSH user (default: root)
#   CLOUD_SSH_HOST           Cloud server IP or domain (required)
#   CLOUD_SSH_PORT           SSH port (default: 22)
#   CLOUD_SSH_KEY            Path to SSH private key (optional)
#
# The tunnel forwards:
#   localhost:5432 → cloud:5432   (PostgreSQL)
#   localhost:6381 → cloud:6381   (Redis)

set -euo pipefail

CLOUD_SSH_USER="${CLOUD_SSH_USER:-root}"
CLOUD_SSH_PORT="${CLOUD_SSH_PORT:-22}"
CLD="${CLOUD_SSH_USER}@${CLOUD_SSH_HOST:?Set CLOUD_SSH_HOST}"

TUNNEL_NAME="beyondlda2-tunnel"
PID_FILE="/tmp/${TUNNEL_NAME}.pid"

# ── Build SSH options ──────────────────────────────────────────────
SSH_OPTS=(
  -N                          # no remote commands
  -o ServerAliveInterval=30   # keep alive
  -o ServerAliveCountMax=3
  -o ExitOnForwardFailure=yes
  -p "${CLOUD_SSH_PORT}"
  -L 5432:localhost:5432      # PostgreSQL
  -L 6381:localhost:6381      # Redis
)

if [ -n "${CLOUD_SSH_KEY:-}" ]; then
  SSH_OPTS+=(-i "${CLOUD_SSH_KEY}")
fi

# ── Commands ────────────────────────────────────────────────────────────

start() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Tunnel already running (PID $(cat "${PID_FILE}"))"
    exit 0
  fi

  echo "Starting SSH tunnel to ${CLD} ..."
  ssh "${SSH_OPTS[@]}" "${CLD}" &
  SSH_PID=$!
  echo "${SSH_PID}" > "${PID_FILE}"

  # Wait a moment then verify
  sleep 2
  if kill -0 "${SSH_PID}" 2>/dev/null; then
    echo "Tunnel established (PID ${SSH_PID})"
    echo "  localhost:5432 → ${CLD}:5432 (PostgreSQL)"
    echo "  localhost:6381 → ${CLD}:6381 (Redis)"
  else
    echo "ERROR: Tunnel failed to establish"
    rm -f "${PID_FILE}"
    exit 1
  fi
}

stop() {
  if [ ! -f "${PID_FILE}" ]; then
    echo "No tunnel PID file found"
    # Try to find and kill any matching SSH processes
    PIDS=$(pgrep -f "ssh.*-L 5432:localhost:5432.*-L 6381:localhost:6381" 2>/dev/null || true)
    if [ -n "${PIDS}" ]; then
      echo "Found orphaned tunnel(s): ${PIDS} — killing..."
      kill ${PIDS} 2>/dev/null || true
    fi
    exit 0
  fi

  PID=$(cat "${PID_FILE}")
  if kill -0 "${PID}" 2>/dev/null; then
    echo "Stopping tunnel (PID ${PID}) ..."
    kill "${PID}"
    sleep 1
  fi
  rm -f "${PID_FILE}"
  echo "Tunnel stopped"
}

status() {
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Tunnel is RUNNING (PID $(cat "${PID_FILE}"))"
    # Quick connectivity check
    if command -v nc &>/dev/null; then
      nc -z localhost 5432 2>/dev/null && echo "  PostgreSQL: connected" || echo "  PostgreSQL: NOT reachable"
      nc -z localhost 6381 2>/dev/null && echo "  Redis:       connected" || echo "  Redis:       NOT reachable"
    fi
  else
    echo "Tunnel is NOT running"
    exit 1
  fi
}

# ── Dispatch ────────────────────────────────────────────────────────────
case "${1:-help}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}"
    echo ""
    echo "Environment:"
    echo "  CLOUD_SSH_HOST  (required)  Cloud server IP or domain"
    echo "  CLOUD_SSH_USER  (optional)  SSH user (default: root)"
    echo "  CLOUD_SSH_PORT  (optional)  SSH port (default: 22)"
    echo "  CLOUD_SSH_KEY   (optional)  Path to SSH private key"
    exit 1
    ;;
esac
