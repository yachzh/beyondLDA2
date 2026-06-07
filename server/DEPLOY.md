# beyondLDA2 — Cloud + Compute Deployment

Two-server architecture:
- **Cloud server** (public IP, domain): FastAPI + PostgreSQL + Redis
- **Compute server** (private, heavy hardware): Celery worker with GPAW

SSH tunnel connects them — no open firewall ports needed.

```
Internet → [Cloud server]
              ├── FastAPI (port 8000)          ← serves REST API
              ├── PostgreSQL (port 5432)       ← host-native
              └── Redis (port 6381)            ← Docker, localhost-only

                          │ SSH tunnel
                          ▼

             [Compute server]
              └── Celery worker (GPAW)         ← runs in your Python venv
```

---

## 1. Cloud Server Setup

### Prerequisites
- Docker & Docker Compose
- PostgreSQL 16 running natively on port 5432
- A domain pointing to the server (e.g., `api.beyondlda2.com`)
- SSH access (for compute server tunnel)

### Step 1: Create the PostgreSQL database

```bash
sudo -u postgres psql -c "CREATE USER beyondlda2 WITH PASSWORD 'beyondlda2';"
sudo -u postgres psql -c "CREATE DATABASE beyondlda2 OWNER beyondlda2;"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE beyondlda2 TO beyondlda2;"
```

Allow local TCP connections (SSH tunnel uses localhost):

```bash
# /etc/postgresql/16/main/pg_hba.conf — ensure this line is present:
# host    beyondlda2    beyondlda2    127.0.0.1/32    md5
```

### Step 2: Clone and configure

```bash
git clone <your-repo> /opt/beyondLDA2
cd /opt/beyondLDA2/server

cp .env.cloud.example .env.cloud
# Edit DB_PASSWORD to match your host PostgreSQL
nano .env.cloud
```

### Step 3: Build and start API + Redis

```bash
docker compose -f docker-compose.cloud.yml --env-file .env.cloud build
docker compose -f docker-compose.cloud.yml --env-file .env.cloud up -d
```

Verify:
```bash
curl http://localhost:8000/api/v1/status
# → {"gpaw_version":null,"ase_version":null}
# (GPAW is None — correct, this container only has FastAPI)
```

### Step 4: Reverse proxy with TLS (Nginx)

```nginx
# /etc/nginx/sites-available/api.beyondlda2.com
server {
    listen 443 ssl;
    server_name api.beyondlda2.com;
    ssl_certificate /etc/letsencrypt/live/api.beyondlda2.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.beyondlda2.com/privkey.pem;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
server {
    listen 80;
    server_name api.beyondlda2.com;
    return 301 https://$server_name$request_uri;
}
```

```bash
sudo certbot --nginx -d api.beyondlda2.com
```

---

## 2. Compute Server Setup

The compute server runs the Celery worker **in your existing GPAW virtual environment** — no Docker needed. It connects to the cloud server through an SSH tunnel.

### Step 1: Install worker dependencies

Activate your GPAW venv and add only the Celery/DB client packages:

```bash
source /path/to/your/gpaw-venv/bin/activate
pip install -r /opt/beyondLDA2/server/requirements.worker.txt
```

That's it — GPAW, ASE, numpy are already in your venv.

### Step 2: Configure and start the SSH tunnel

```bash
cd /opt/beyondLDA2/server

# Copy and edit the compute env file
cp .env.compute.example .env.compute
# Set CLOUD_SSH_HOST to your cloud server's IP/domain
nano .env.compute

# Source env vars and start the tunnel
set -a; source .env.compute; set +a
./tunnel.sh start
```

**Auto-start on boot** (systemd):

```ini
# /etc/systemd/system/beyondlda2-tunnel.service
[Unit]
Description=SSH tunnel to beyondLDA2 cloud server
After=network-online.target

[Service]
Type=forking
ExecStart=/opt/beyondLDA2/server/tunnel.sh start
ExecStop=/opt/beyondLDA2/server/tunnel.sh stop
EnvironmentFile=/opt/beyondLDA2/server/.env.compute
Restart=on-failure
User=root

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now beyondlda2-tunnel
```

### Step 3: Start the worker

```bash
cd /opt/beyondLDA2

# Activate your GPAW venv and start the worker
source /path/to/gpaw-venv/bin/activate

DB_PASSWORD=beyondlda2 \
MPI_NPROCS=8 \
CONCURRENCY=1 \
./server/start_worker.sh
```

Or use your own supervisor (tmux, screen, supervisord, systemd):

```bash
celery -A server.worker worker \
  --loglevel=info \
  --concurrency=1
```

The worker automatically reads `DB_HOST`, `REDIS_HOST`, etc. from the environment (defaulting to `localhost` for tunnel).

---

## 3. End-to-End Test

From any machine:

```bash
curl -X POST https://api.beyondlda2.com/api/v1/calculate \
  -H 'Content-Type: application/json' \
  -d '{
    "calculation_type": "ks_gap",
    "structure": {
      "atoms": [
        {"symbol": "Si", "x": 0.0, "y": 0.0, "z": 0.0},
        {"symbol": "Si", "x": 0.25, "y": 0.25, "z": 0.25}
      ],
      "cell": [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    },
    "parameters": {
      "xc": "LDA", "hubbard_u": 0.0, "kmesh": [4, 4, 4],
      "ecut": 150, "pwcut": 450, "spin_pol": false, "symmetry": false
    }
  }'
# → {"job_id":1,"check_status_url":"/api/v1/jobs/1"}

curl -s https://api.beyondlda2.com/api/v1/jobs/1 | jq .
```

---

## 4. File Reference

| File | Purpose |
|------|---------|
| `Dockerfile.api` | Lightweight API image (no GPAW) |
| `docker-compose.cloud.yml` | Cloud server: API + Redis |
| `requirements.worker.txt` | Celery worker pip deps (adds to your GPAW venv) |
| `start_worker.sh` | One-command worker launcher |
| `tunnel.sh` | SSH tunnel manager (`start`/`stop`/`status`) |
| `.env.cloud.example` | Cloud server env template |
| `.env.compute.example` | Compute server env template |
| `requirements.api.txt` | API-only pip deps (for Docker build) |

## 5. Security Notes

- Redis is bound to `127.0.0.1:6381` — no public access
- PostgreSQL accessible only via localhost (SSH tunnel)
- SSH tunnel is the only link between servers — no open firewall
- TLS terminates at Nginx on the cloud server
- API has no built-in auth yet — add API keys or OAuth if exposing publicly
