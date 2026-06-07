# beyondLDA2 API Server

Async REST API for GPAW+U calculations, built with FastAPI + Celery + PostgreSQL.

## Architecture

```
Client ──HTTP──►  FastAPI API Server     Celery Worker
                   │                     │
                   │  ┌──────────────┐   │  ┌───────────────────┐
                   │  │  PostgreSQL  │   │  │  mpirun gpaw      │
                   │  │  (job queue) │◄──┼──┤  python script.py │
                   │  └──────────────┘   │  └───────────────────┘
                   │  ┌──────────────┐   │  ┌───────────────────┐
                   │  │  Redis       │───┼──│  Celery broker    │
                   │  └──────────────┘   │  └───────────────────┘
```

- **API server** (FastAPI) validates requests, writes to PostgreSQL, enqueues Celery tasks
- **Worker** (Celery) picks up jobs, generates a GPAW Python script, runs it via `mpirun gpaw python`, and writes structured results back
- **PostgreSQL** stores job metadata (status, payload, results, error logs)
- **Redis** brokers tasks between API server and workers
- **ase.db** files (SQLite) are used optionally for permanent scientific result storage

## Endpoints

### `GET /api/v1/status`

Server health and GPAW version info.

### `POST /api/v1/calculate`

Submit a calculation. Returns `202 Accepted` with a `job_id`.

**Request body (JSON):**

```json
{
  "calculation_type": "electronic_energy",
  "structure": {
    "atoms": [
      {"symbol": "Fe", "x": 0.0, "y": 0.0, "z": 0.0},
      {"symbol": "C",  "x": 1.2, "y": 0.0, "z": 0.0}
    ],
    "cell": [[10,0,0],[0,10,0],[0,0,10]],
    "pbc": [true, true, true]
  },
  "parameters": {
    "xc": "LDA",
    "hubbard_u": 4.0,
    "magnetic_center": "Fe",
    "spin_state": "HS",
    "planewave": false,
    "kmesh": [1, 1, 1]
  },
  "database": "results.db",
  "label": "fe-calc-1"
}
```

**Supported `calculation_type` values:**

| Type | Method called | Returns |
|---|---|---|
| `electronic_energy` | `get_electronic_energy()` | energy in eV |
| `ks_gap` | `get_ksgap()` | KS HOMO–LUMO gap |
| `gllbsc_gap` | `get_gllbscgap()` | GLLBSC gap (E_ks + D_xc) |
| `g0w0_gap` | `get_g0w0gap()` | Quasiparticle gap |
| `local_opt` | `local_opt()` | Energy + optimized geometry |
| `absorption` | `get_absorption()` | BSE absorption spectrum |
| `phonon` | `phonon()` | Vibrational frequencies |

### `GET /api/v1/jobs/:id`

Poll job status. Returns `completed`, `failed`, `processing`, or `queued`.

When completed, `results` contains the calculation output.

### `POST /api/v1/input`

Dry-run: generates the Python script without executing it. Useful for verification.

### `GET /api/v1/jobs`

List recent jobs with their status and calculation type.

## Quick Start (Docker)

```bash
cd server
docker compose up -d
curl http://localhost:8000/api/v1/status
```

## Quick Start (manual)

```bash
# Install dependencies
pip install -r server/requirements.txt

# Start PostgreSQL and Redis (docker or native)
# Set environment variables:
export DB_HOST=localhost
export REDIS_HOST=localhost

# Start Celery worker (in terminal 1)
celery -A server.worker worker --loglevel=info --concurrency=1

# Start API server (in terminal 2)
uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | API server bind address |
| `API_PORT` | `8000` | API server port |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `beyondlda2` | PostgreSQL database name |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `MPI_NPROCS` | `4` | Number of MPI processes for GPAW |
| `GPAW_SETUP_PATH` | — | Path to PAW dataset directory |
| `WORK_DIR` | `/tmp/beyondlda2-jobs` | Scratch directory for job scripts |
