# syntax = docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libxc-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# ── Builder stage: compile GPAW (takes a while) ─────────────────────────
FROM base AS builder

WORKDIR /build
RUN pip install --no-cache-dir pip setuptools wheel numpy

# Download and build GPAW
ARG GPAW_VERSION=24.6.0
RUN pip install --no-cache-dir gpaw==${GPAW_VERSION} ase

RUN python -c "from gpaw import GPAW; print(f'GPAW {GPAW.__version__} built OK')"

# ── Runtime stage ──────────────────────────────────────────────────────
FROM base AS runtime

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install PAW setup data
RUN gpaw install-data /usr/share/gpaw --register
ENV GPAW_SETUP_PATH=/usr/share/gpaw

# Install API server dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy beyondLDA2 source
COPY beyondLDA2.py beyondLDA2_db.py ./
COPY server/ ./server/

# Default: run Celery worker
CMD ["celery", "-A", "server.worker", "worker", "--loglevel=info", "--concurrency=1"]
