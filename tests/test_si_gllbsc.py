#!/usr/bin/env python3
"""
Compute the GLLBSC band gap for bulk Si (diamond) using planewave mode.

The GLLBSC functional includes a derivative discontinuity (Δ_xc) correction
on top of the Kohn-Sham gap, giving improved band gaps compared to LDA.
All outputs go to a scratch directory outside the repo.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_si_gllbsc.py
"""
import sys
import os
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)

from ase.build import bulk
from beyondLDA2 import lda_plus_u, sec2time

# --- Scratch directory setup ---
SCRATCH = os.environ.get(
    'BEYOND_LDA2_SCRATCH',
    os.path.expanduser('~/scratch/beyondLDA2/si-gllbsc'),
)
os.makedirs(SCRATCH, exist_ok=True)
os.chdir(SCRATCH)

# --- Create bulk Si (diamond, 2-atom primitive cell) ---
si = bulk('Si', 'diamond', a=5.43, cubic=False)
si.set_pbc((True, True, True))

print(f"Structure: {len(si)} atoms, volume = {si.get_volume():.3f} Å³")
print(f"Cell: {si.cell.lengths()}")
print()

# --- Set up planewave calculator (xc will be overridden to GLLBSC) ---
dft = lda_plus_u(
    atoms=si,
    xc='LDA',
    planewave=True,
    pwcut=300,
    kmesh=(4, 4, 4),
    spin_pol=False,
    fname='si-gllbsc',
)

# --- Compute GLLBSC gap ---
print("-" * 60)
print("Computing GLLBSC gap ...")
start = time.time()
gap = dft.get_gllbscgap()
wall = sec2time(time.time() - start)
print(f"  GLLBSC gap = {gap:.3f} eV  ({wall})")

# Read back the breakdown from the result file
result_file = f"result-{dft.get_label()}.txt"
if os.path.exists(result_file):
    with open(result_file) as f:
        content = f.read()
    for line in content.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            print(f"    {key.strip()}: {val.strip()}")

print()
print("=" * 60)
print("Bulk Si GLLBSC gap completed!")
print(f"Outputs in: {SCRATCH}")
print("=" * 60)
