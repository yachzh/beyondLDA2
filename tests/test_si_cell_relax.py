#!/usr/bin/env python3
"""
Cell relaxation test on bulk Si (diamond) using planewave mode.

Relaxes both atomic positions and cell parameters via ExpCellFilter.
All outputs go to a scratch directory outside the repo.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_si_cell_relax.py
"""
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)

from ase.build import bulk
from beyondLDA2 import lda_plus_u

# --- Scratch directory setup ---
SCRATCH = os.environ.get(
    'BEYOND_LDA2_SCRATCH',
    os.path.expanduser('~/scratch/beyondLDA2/si-cell-relax'),
)
os.makedirs(SCRATCH, exist_ok=True)
os.chdir(SCRATCH)

# --- Create bulk Si (diamond structure, conventional 2-atom cell) ---
si = bulk('Si', 'diamond', a=5.43, cubic=False)
si.set_pbc((True, True, True))

print(f"Initial structure: {len(si)} atoms")
print(f"Initial cell: {si.cell.lengths()}")
print(f"Initial volume: {si.get_volume():.3f} Å³")

# --- Set up LDA planewave calculator ---
dft = lda_plus_u(
    atoms=si,
    xc='LDA',
    planewave=True,
    pwcut=300,
    kmesh=(4, 4, 4),
    spin_pol=False,
    fname='si-cell-relax',
)

# --- Run cell relaxation ---
energy = dft.local_opt(
    force_convergence=0.05,
    maxstep=0.02,
    varcell=True,
)

# --- Report results ---
print(f"\n{'='*60}")
print(f"Cell relaxation completed!")
print(f"Final energy: {energy:.6f} eV")
print(f"Final cell:   {si.cell.lengths()}")
print(f"Final angles: {si.cell.angles()}")
print(f"Final volume: {si.get_volume():.3f} Å³")
print(f"Outputs in:   {SCRATCH}")
print(f"{'='*60}")
