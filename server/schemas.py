"""
Pydantic schemas for request/response validation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class CalcType(str, Enum):
    electronic_energy = "electronic_energy"
    ks_gap = "ks_gap"
    gllbsc_gap = "gllbsc_gap"
    g0w0_gap = "g0w0_gap"
    local_opt = "local_opt"
    absorption = "absorption"
    phonon = "phonon"


class XCFunctional(str, Enum):
    LDA = "LDA"
    PBE = "PBE"
    revPBE = "revPBE"
    RPBE = "RPBE"
    PBE0 = "PBE0"
    B3LYP = "B3LYP"
    vdW_DF = "vdW-DF"
    vdW_DF2 = "vdW-DF2"
    GLLBSC = "GLLBSC"


class SpinState(str, Enum):
    HS = "HS"
    LS = "LS"
    IS = "IS"


# ── Request schemas ────────────────────────────────────────────────────

class AtomPosition(BaseModel):
    symbol: str = Field(..., description="Element symbol, e.g. 'Fe'")
    x: float
    y: float
    z: float


class StructureInput(BaseModel):
    """Atomic structure can be provided as a list of atoms + cell, or
    as a raw XYZ/CIF/STRUCT string with a format hint."""

    # Option A: explicit positions
    atoms: list[AtomPosition] | None = None
    cell: list[list[float]] | None = None
    pbc: list[bool] | list[int] | None = None

    # Option B: raw file content
    content: str | None = None
    format: str | None = Field(
        None, description="ASE read format, e.g. 'xyz', 'cif', 'struct'"
    )


class CalcParameters(BaseModel):
    """Parameters forwarded to ``lda_plus_u``."""

    xc: XCFunctional = XCFunctional.LDA
    hubbard_u: float = 0.0
    magnetic_center: str = "Fe"
    spin_state: SpinState = SpinState.LS
    planewave: bool = False
    pwcut: int = 450
    kmesh: list[int] = [1, 1, 1]
    beta: float = 0.07
    maxcycl: int = 500
    etol: float = 1.0e-6
    dentol: float = 1.0e-6
    temperature: float = 300.0
    charge: float = 0.0
    fixspin: bool = True
    isMol: bool = False
    nbands: int | None = None

    # Type-specific extras
    force_convergence: float | None = Field(
        0.02, description="local_opt: force convergence in eV/Å"
    )
    ecut: int | None = Field(
        150, description="g0w0: plane-wave cutoff for self-energy"
    )
    ppa: bool = Field(False, description="g0w0: use plasmon-pole approximation")
    nbands_gw: int | None = Field(None, description="g0w0: number of bands")


class CalculateRequest(BaseModel):
    """Full request payload for submitting a GPAW calculation."""

    calculation_type: CalcType
    structure: StructureInput
    parameters: CalcParameters = CalcParameters()
    database: str | None = Field(
        None,
        description=(
            "Path to an ase.db file for storing scientific results. "
            "Relative paths are resolved against the server's ASE_DB_DIR."
        ),
    )
    label: str | None = None


# ── Response schemas ───────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status: str = "ok"
    gpaw_version: str | None = None
    ase_version: str | None = None
    api_version: str = "0.1.0"


class JobSubmitted(BaseModel):
    job_id: int
    status: str = "queued"
    check_status_url: str


class JobStatus(BaseModel):
    id: int
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    results: dict[str, Any] | None = None
    script: str | None = None  # populated on /input endpoint
    error: str | None = None


class InputPreview(BaseModel):
    script: str
    calculation_type: str
    label: str | None
    num_atoms: int
