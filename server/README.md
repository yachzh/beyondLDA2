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

## End-to-End Example: HS–LS Energy Splitting of Ferrocene

The most common use case is computing the high-spin (HS) vs. low-spin (LS)
energy splitting of a magnetic molecule. Below is a complete walkthrough.

### 1. Check server health

```bash
curl http://localhost:8000/api/v1/status
```

Returns:

```json
{
  "status": "ok",
  "gpaw_version": "25.7.0",
  "ase_version": "3.28.0",
  "api_version": "0.1.0"
}
```

### 2. Submit the HS calculation

The structure is ferrocene (FeC₁₀H₁₀) with an HS-optimized geometry. Positions
are given in **scaled (fractional) coordinates** with respect to the cell vectors.

```bash
curl -s -X POST http://localhost:8000/api/v1/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "calculation_type": "electronic_energy",
    "structure": {
      "atoms": [
        {"symbol": "C",  "x": 0.3895, "y": 0.4086, "z": 0.5825},
        {"symbol": "C",  "x": 0.6107, "y": 0.5914, "z": 0.4176},
        {"symbol": "C",  "x": 0.4337, "y": 0.5652, "z": 0.5793},
        {"symbol": "C",  "x": 0.5925, "y": 0.6490, "z": 0.5725},
        {"symbol": "C",  "x": 0.5665, "y": 0.4350, "z": 0.4205},
        {"symbol": "C",  "x": 0.4075, "y": 0.3510, "z": 0.4267},
        {"symbol": "C",  "x": 0.4780, "y": 0.6042, "z": 0.4215},
        {"symbol": "C",  "x": 0.3505, "y": 0.4532, "z": 0.4270},
        {"symbol": "C",  "x": 0.5220, "y": 0.3958, "z": 0.5779},
        {"symbol": "C",  "x": 0.6495, "y": 0.5468, "z": 0.5720},
        {"symbol": "H",  "x": 0.2725, "y": 0.3124, "z": 0.5867},
        {"symbol": "H",  "x": 0.7275, "y": 0.6876, "z": 0.4133},
        {"symbol": "H",  "x": 0.3564, "y": 0.6114, "z": 0.5801},
        {"symbol": "H",  "x": 0.6610, "y": 0.7729, "z": 0.5674},
        {"symbol": "H",  "x": 0.6439, "y": 0.3887, "z": 0.4199},
        {"symbol": "H",  "x": 0.3392, "y": 0.2276, "z": 0.4316},
        {"symbol": "H",  "x": 0.4731, "y": 0.7098, "z": 0.4215},
        {"symbol": "H",  "x": 0.2311, "y": 0.4230, "z": 0.4323},
        {"symbol": "H",  "x": 0.5269, "y": 0.2902, "z": 0.5780},
        {"symbol": "H",  "x": 0.7689, "y": 0.5770, "z": 0.5664},
        {"symbol": "Fe", "x": 0.5000, "y": 0.5000, "z": 0.5000}
      ],
      "cell": [[10.2245, 0.0, 0.0],
               [-5.11225, 8.85467, 0.0],
               [0.0, 0.0, 21.2621]],
      "pbc": [true, true, true]
    },
    "parameters": {
      "xc": "LDA",
      "hubbard_u": 4.0,
      "magnetic_center": "Fe",
      "spin_state": "HS",
      "spin_pol": true,
      "fixspin": false,
      "conv_default": false
    },
    "label": "fe-ferrocene-hs"
  }'
```

Response (HTTP 202):

```json
{
  "job_id": 14,
  "status": "queued",
  "check_status_url": "/api/v1/jobs/14"
}
```

### 3. Submit the LS calculation

The LS calculation uses a **different geometry** — the structure relaxed in the
low-spin state. Using the HS geometry for an LS calculation will produce an
incorrect energy because the two spin states have different equilibrium
geometries.

```bash
curl -s -X POST http://localhost:8000/api/v1/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "calculation_type": "electronic_energy",
    "structure": {
      "atoms": [
        {"symbol": "C",  "x": 0.3892, "y": 0.4077, "z": 0.5825},
        {"symbol": "C",  "x": 0.6111, "y": 0.5923, "z": 0.4172},
        {"symbol": "C",  "x": 0.4326, "y": 0.5641, "z": 0.5790},
        {"symbol": "C",  "x": 0.5934, "y": 0.6500, "z": 0.5725},
        {"symbol": "C",  "x": 0.5676, "y": 0.4359, "z": 0.4206},
        {"symbol": "C",  "x": 0.4067, "y": 0.3501, "z": 0.4269},
        {"symbol": "C",  "x": 0.4771, "y": 0.6032, "z": 0.4213},
        {"symbol": "C",  "x": 0.3507, "y": 0.4535, "z": 0.4273},
        {"symbol": "C",  "x": 0.5232, "y": 0.3969, "z": 0.5781},
        {"symbol": "C",  "x": 0.6494, "y": 0.5466, "z": 0.5719},
        {"symbol": "H",  "x": 0.2724, "y": 0.3119, "z": 0.5865},
        {"symbol": "H",  "x": 0.7279, "y": 0.6881, "z": 0.4135},
        {"symbol": "H",  "x": 0.3553, "y": 0.6104, "z": 0.5799},
        {"symbol": "H",  "x": 0.6620, "y": 0.7740, "z": 0.5673},
        {"symbol": "H",  "x": 0.6449, "y": 0.3897, "z": 0.4200},
        {"symbol": "H",  "x": 0.3380, "y": 0.2260, "z": 0.4318},
        {"symbol": "H",  "x": 0.4723, "y": 0.7089, "z": 0.4213},
        {"symbol": "H",  "x": 0.2313, "y": 0.4234, "z": 0.4326},
        {"symbol": "H",  "x": 0.5281, "y": 0.2913, "z": 0.5781},
        {"symbol": "H",  "x": 0.7687, "y": 0.5767, "z": 0.5662},
        {"symbol": "Fe", "x": 0.4998, "y": 0.4999, "z": 0.4998}
      ],
      "cell": [[10.2245, 0.0, 0.0],
               [-5.11225, 8.85467, 0.0],
               [0.0, 0.0, 21.2621]],
      "pbc": [true, true, true]
    },
    "parameters": {
      "xc": "LDA",
      "hubbard_u": 4.0,
      "magnetic_center": "Fe",
      "spin_state": "LS",
      "spin_pol": true,
      "fixspin": false,
      "conv_default": false
    },
    "label": "fe-ferrocene-ls"
  }'
```

### 4. Poll for results

Jobs run asynchronously. Poll the status endpoint with the returned `job_id`:

```bash
curl http://localhost:8000/api/v1/jobs/14
```

While running:

```json
{
  "id": 14,
  "status": "processing",
  "created_at": "2026-06-07T17:35:52+00:00",
  "updated_at": "2026-06-07T17:36:18+00:00",
  "results": null,
  "error": null
}
```

On completion:

```json
{
  "id": 14,
  "status": "completed",
  "created_at": "2026-06-07T17:35:52+00:00",
  "updated_at": "2026-06-07T17:36:54+00:00",
  "results": {
    "label": "fe-ferrocene-hs",
    "calculation_type": "electronic_energy",
    "gpaw_version": "25.7.0",
    "energy": -135.34551300067145,
    "final_formula": "C10H10Fe",
    "final_positions": [
      [ ... 21×3 atom positions in Cartesian (Å) ... ]
    ]
  },
  "error": null
}
```

On failure:

```json
{
  "id": 10,
  "status": "failed",
  "results": null,
  "error": "GridBoundsError: Atom Fe at position [0, 0, 0] is too close to a grid boundary. Center the atoms in the cell."
}
```

### 5. Compute the splitting

```bash
# Save job IDs
HS_JOB=14
LS_JOB=16

# Extract energies
HS_E=$(curl -s http://localhost:8000/api/v1/jobs/$HS_JOB | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['results']['energy'])")
LS_E=$(curl -s http://localhost:8000/api/v1/jobs/$LS_JOB | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['results']['energy'])")

echo "E(HS) = $HS_E eV"
echo "E(LS) = $LS_E eV"
echo "ΔE(HS-LS) = $(python3 -c "print($HS_E - $LS_E)") eV"
echo "ΔE(HS-LS) = $(python3 -c "print(($HS_E - $LS_E) * 96.485)") kJ/mol"
```

Output:

```
E(HS) = -135.34551300067145 eV
E(LS) = -136.2061235660317 eV
ΔE(HS-LS) = 0.860611 eV
ΔE(HS-LS) = 83.04 kJ/mol
```

A positive ΔE means **LS is more stable**, which is the correct ground state for
ferrocene (Fe²⁺ in a strong cyclopentadienyl ligand field).

### 6. Key pitfalls

| Pitfall | Consequence | How to avoid |
|---|---|---|
| Using the same structure for HS and LS | Wrong relative energy | Pre-relax each spin state independently, then use the relaxed geometry from that spin state for the single-point energy |
| Placing atoms at cell corners | `GridBoundsError` | Use scaled positions inside (0, 1), or call `atoms.center()` in the calculation script |
| Forgetting `magnetic_center` | GPAW uses default (non-magnetic) setup | Always set `magnetic_center` when using Hubbard U |
| `conv_default: false` without tight SCF params | SCF converges too loosely | Use the defaults shown above (`etol=1e-6`, `maxcycl=500`, `beta=0.07`) |

### 7. Automation script

For a production workflow, wrap this in a script that submits both jobs and
waits for completion:

```bash
#!/bin/bash
# hs_ls_pipeline.sh — submit HS & LS, wait, compare
API="http://localhost:8000/api/v1"

submit_and_poll() {
  local label=$1 spin=$2
  local resp=$(curl -s -X POST "$API/calculate" \
    -H "Content-Type: application/json" \
    -d "$(cat payload_${spin}.json)")
  local jid=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
  echo "Submitted $label → job $jid"
  for i in $(seq 1 120); do
    sleep 10
    local st=$(curl -s "$API/jobs/$jid" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
    echo "[$i] $label: $st"
    if [ "$st" = "completed" ] || [ "$st" = "failed" ]; then
      break
    fi
  done
  echo "$jid"
}

HS_JOB=$(submit_and_poll "HS" "hs")
LS_JOB=$(submit_and_poll "LS" "ls")

# Compare
HS_E=$(curl -s "$API/jobs/$HS_JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['results']['energy'])")
LS_E=$(curl -s "$API/jobs/$LS_JOB" | python3 -c "import sys,json; print(json.load(sys.stdin)['results']['energy'])")
python3 -c "
d = float('$HS_E') - float('$LS_E')
print(f'E(HS) = {$HS_E} eV')
print(f'E(LS) = {$LS_E} eV')
print(f'ΔE   = {d:.6f} eV = {d*96.485:.2f} kJ/mol')
"
```

### 8. Full response schema

A completed job returns the following fields in `results`:

| Field | Type | Description |
|---|---|---|
| `label` | string | User-assigned label from the request |
| `calculation_type` | string | Matches the requested type |
| `gpaw_version` | string | GPAW version used for the run |
| `energy` | float | Total energy in eV (electronic_energy / local_opt) |
| `final_formula` | string | Chemical formula from the final Atoms object |
| `final_positions` | array[float] | Final Cartesian positions in Å — 21×3 for ferrocene |
| `ks_gap` | float | KS HOMO–LUMO gap in eV (ks_gap type) |
| `gllbsc_gap` | float | GLLBSC gap in eV (gllbsc_gap type) |
| `g0w0_gap` | float | Quasiparticle gap in eV (g0w0_gap type) |
| `opt_geometry` | array[float] | Optimized positions after geometry relaxation (local_opt type) |
| `absorption_energies` | array[float] | Photon energies in eV (absorption type) |
| `absorption_spectrum` | array[float] | Oscillator strengths (absorption type) |
| `error` | string | Error message on failure |

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
