"""
Celery worker for beyondLDA2 — runs GPAW calculations asynchronously.

The worker polls Redis for jobs, generates a temp directory with the
calculation script, shells out via ``mpirun gpaw python script.py``,
and writes results back to PostgreSQL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SASession

from server.config import settings
from server.models import Base, Simulation
from server.runner import write_job_dir
from server.schemas import CalculateRequest

# ── Celery app ─────────────────────────────────────────────────────────

app = Celery(
    "beyondlda2",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ── Database ───────────────────────────────────────────────────────────

_sync_engine = create_engine(settings.DATABASE_URL_SYNC, pool_pre_ping=True)
Base.metadata.create_all(_sync_engine)


def _update_job(job_id: int, **kwargs):
    """Update a simulation row by id."""
    with SASession(_sync_engine) as session:
        sim = session.query(Simulation).filter_by(id=job_id).first()
        if sim:
            for k, v in kwargs.items():
                setattr(sim, k, v)
            session.commit()


# ── Task ───────────────────────────────────────────────────────────────

@app.task(bind=True, max_retries=1)
def run_gpaw_calculation(self, job_id: int, payload: dict):
    """Celery task: run a GPAW calculation for the given job.

    Parameters
    ----------
    job_id : int
        Primary key in the ``simulations`` PostgreSQL table.
    payload : dict
        The calculation request payload (validated by Pydantic on submission).
    """
    # --- Mark processing ---
    _update_job(job_id, status="processing")

    # --- Reconstruct request ---
    req = CalculateRequest(**payload)

    # --- Create work directory ---
    label = req.label or f"job-{job_id}"
    work_dir = Path(settings.WORK_DIR) / f"{label}-{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Write job files ---
    job_dir = write_job_dir(req, str(work_dir), label=label)
    script_path = job_dir / "script.py"

    # --- Prepare environment ---
    env = os.environ.copy()
    if settings.GPAW_SETUP_PATH:
        env["GPAW_SETUP_PATH"] = settings.GPAW_SETUP_PATH

    # --- Build command ---
    nprocs = settings.MPI_NPROCS
    if nprocs > 1:
        cmd = ["mpirun", "--allow-run-as-root", "-np", str(nprocs), "gpaw", "python", str(script_path)]
    else:
        cmd = ["gpaw", "python", str(script_path)]

    # --- Run ---
    try:
        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=86400,  # 24h hard limit
            env=env,
        )

        # Read output.json
        output_path = job_dir / "output.json"
        if output_path.exists():
            with open(output_path) as f:
                parsed = json.load(f)
        else:
            parsed = {"error": "No output.json produced"}

        # Check for errors
        if "error" in parsed and parsed["error"]:
            _update_job(
                job_id,
                status="failed",
                error_log=parsed.get("traceback", parsed["error"]),
            )
            return {"job_id": job_id, "status": "failed", "error": parsed["error"]}

        # Success
        _update_job(
            job_id,
            status="completed",
            parsed_results=parsed,
            fdf_output=result.stdout,
        )
        return {"job_id": job_id, "status": "completed", "results": parsed}

    except subprocess.TimeoutExpired:
        error_msg = "Calculation timed out (24h limit)"
        _update_job(job_id, status="failed", error_log=error_msg)
        return {"job_id": job_id, "status": "failed", "error": error_msg}

    except Exception as e:
        tb = traceback.format_exc()
        _update_job(job_id, status="failed", error_log=f"{type(e).__name__}: {e}\n{tb}")
        raise
