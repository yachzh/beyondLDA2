#!/usr/bin/env bash
# Start the beyondLDA2 API server for local development
# Usage: bash server/run.sh

export DB_NAME=mydb
export DB_USER=myuser
export DB_PASSWORD=mypassword
export GPAW_SETUP_PATH="${GPAW_SETUP_PATH:-$HOME/.local/lib/python3.12/site-packages/gpaw_data/setups}"

cd "$(dirname "$0")/.."
uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload
