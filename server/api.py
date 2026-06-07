"""
FastAPI application — async REST API for beyondLDA2 GPAW calculations.

Endpoints mirror the ruby_siesta design:
  GET  /api/v1/status       → Server / GPAW info
  POST /api/v1/calculate    → Submit calculation (async, returns 202)
  GET  /api/v1/jobs/:id     → Poll job status / results
  POST /api/v1/input        → Preview generated script (dry-run)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from server.config import settings
from server.models import Base, Simulation
from server.runner import generate_script
from server.schemas import (
    CalculateRequest,
    InputPreview,
    JobSubmitted,
    JobStatus,
    StatusResponse,
)
from server.worker import run_gpaw_calculation

# ── FastAPI app ────────────────────────────────────────────────────────

app = FastAPI(
    title="beyondLDA2 API",
    description="Async GPAW calculations via REST API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Async database ─────────────────────────────────────────────────────

_async_engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
_async_session = sessionmaker(_async_engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with _async_session() as session:
        yield session


@app.on_event("startup")
async def startup():
    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Helpers ────────────────────────────────────────────────────────────

def _gpaw_version() -> str | None:
    """Try to import gpaw and return its version string."""
    try:
        import gpaw
        return gpaw.__version__
    except ImportError:
        return None


def _ase_version() -> str | None:
    try:
        import ase
        return ase.__version__
    except ImportError:
        return None


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/v1/status", response_model=StatusResponse)
async def status():
    """Return server and GPAW environment info."""
    return StatusResponse(
        gpaw_version=_gpaw_version(),
        ase_version=_ase_version(),
    )


@app.post("/api/v1/calculate", status_code=202, response_model=JobSubmitted)
async def submit_calculation(req: CalculateRequest):
    """Submit a GPAW calculation asynchronously.

    The request is validated, stored in PostgreSQL as ``queued``, and a
    Celery task is enqueued. Poll ``/api/v1/jobs/:id`` for completion.
    """
    payload = req.model_dump(mode="json")

    async with _async_session() as session:
        sim = Simulation(
            status="queued",
            payload=payload,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        session.add(sim)
        await session.commit()
        await session.refresh(sim)
        job_id = sim.id

    # Enqueue the Celery task (fire-and-forget)
    run_gpaw_calculation.delay(job_id, payload)

    return JobSubmitted(
        job_id=job_id,
        check_status_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: int):
    """Poll the status and results of an async calculation."""
    async with _async_session() as session:
        result = await session.execute(
            select(Simulation).where(Simulation.id == job_id)
        )
        sim = result.scalar_one_or_none()

    if sim is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = JobStatus(
        id=sim.id,
        status=sim.status,
        created_at=sim.created_at.isoformat() if sim.created_at else None,
        updated_at=sim.updated_at.isoformat() if sim.updated_at else None,
    )

    if sim.status == "completed":
        response.results = sim.parsed_results
        response.script = sim.fdf_output
    elif sim.status == "failed":
        response.error = sim.error_log

    return response


@app.post("/api/v1/input", response_model=InputPreview)
async def preview_input(req: CalculateRequest):
    """Generate the Python calculation script without running it.

    Useful for verification before submitting a full job.
    """
    script = generate_script(req)
    atoms = None
    struct = req.structure
    if struct.atoms:
        atoms = struct.atoms

    return InputPreview(
        script=script,
        calculation_type=req.calculation_type.value,
        label=req.label,
        num_atoms=len(atoms) if atoms else 0,
    )


@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 20, offset: int = 0):
    """List recent jobs."""
    async with _async_session() as session:
        result = await session.execute(
            select(Simulation)
            .order_by(Simulation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        rows = result.scalars().all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "calculation_type": r.payload.get("calculation_type"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


# ── Entrypoint ─────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run("server.api:app", host=settings.HOST, port=settings.PORT, reload=True)


if __name__ == "__main__":
    main()
