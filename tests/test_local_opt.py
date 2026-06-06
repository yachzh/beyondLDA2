#!/usr/bin/env python3
"""
Test local_opt on the ls structure (LCAO mode, PBC imposed).

All calculation outputs go to a scratch directory outside the repo.
Set BEYOND_LDA2_SCRATCH env var to override the default path.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_local_opt.py
"""
import sys
import os
import shutil

# Ensure import works from repo root
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)

from ase.io import read
from beyondLDA2 import lda_plus_u

# --- Scratch directory setup ---
SCRATCH = os.environ.get(
    'BEYOND_LDA2_SCRATCH',
    os.path.expanduser('~/scratch/beyondLDA2/ls-test'),
)
os.makedirs(SCRATCH, exist_ok=True)

# Copy input structure to scratch (beyondLDA2 outputs are relative to CWD)
struct_src = os.path.join(_repo_root, 'ls.STRUCT_OUT')
struct_dst = os.path.join(SCRATCH, 'ls.STRUCT_OUT')
if not os.path.exists(struct_dst):
    shutil.copy2(struct_src, struct_dst)

# Change to scratch dir so all output files land there
os.chdir(SCRATCH)

# Read the ls structure
atoms = read('ls.STRUCT_OUT', format='struct_out')
atoms.set_pbc((True, True, True))
atoms.center()

# LCAO mode with Hubbard U
dft = lda_plus_u(
    atoms=atoms,
    magnetic_center='Fe',
    spin_state='LS',
    xc='LDA',
    hubbard_u=4.0,
    fname='ls-test',
)

# Run local geometry optimization
energy = dft.local_opt(
    force_convergence=0.05,
    maxstep=0.02,
)

print(f"\n{'='*60}")
print(f"local_opt completed successfully!")
print(f"Final energy: {energy:.6f} eV")
print(f"Outputs in: {SCRATCH}")
print(f"{'='*60}")
