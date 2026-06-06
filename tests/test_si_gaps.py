#!/usr/bin/env python3
"""
Compute KS and G0W0 band gaps for bulk Si (diamond) using planewave mode.

Reports the Kohn-Sham LDA gap and the G0W0 quasi-particle correction.
All outputs go to a scratch directory outside the repo.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_si_gaps.py
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
    os.path.expanduser('~/scratch/beyondLDA2/si-gaps'),
)
os.makedirs(SCRATCH, exist_ok=True)
os.chdir(SCRATCH)

# --- Create bulk Si (diamond, 2-atom primitive cell) ---
si = bulk('Si', 'diamond', a=5.43, cubic=False)
si.set_pbc((True, True, True))

print(f"Structure: {len(si)} atoms, volume = {si.get_volume():.3f} Å³")
print(f"Cell: {si.cell.lengths()}")
print()

# --- Set up LDA planewave calculator ---
# Use modest settings: G0W0 is expensive even for 2 atoms
dft = lda_plus_u(
    atoms=si,
    xc='LDA',
    planewave=True,
    pwcut=300,
    kmesh=(4, 4, 4),
    spin_pol=False,
    fname='si-gaps',
)

# --- 1. Kohn-Sham gap ---
print("-" * 60)
print("Computing Kohn-Sham gap ...")
start = time.time()
ks_gap = dft.get_ksgap()
wall_ks = sec2time(time.time() - start)
print(f"  KS gap = {ks_gap:.3f} eV  ({wall_ks})")
print()

# --- 2. G0W0 quasi-particle gap ---
print("-" * 60)
print("Computing G0W0 gap (ecut=100, relbands=(-1,1)) ...")
start = time.time()
qp_gap = dft.get_g0w0gap(
    ecut=100,
    relbands=(-1, 1),
)
wall_gw = sec2time(time.time() - start)
print(f"  G0W0 gap = {qp_gap:.3f} eV  ({wall_gw})")
print()

# --- Report ---
print("=" * 60)
print("Bulk Si band gaps")
print(f"  LDA  (KS)  = {ks_gap:.3f} eV")
print(f"  G0W0       = {qp_gap:.3f} eV")
print(f"  Δ (GW - KS) = {qp_gap - ks_gap:+.3f} eV")
print(f"Outputs in: {SCRATCH}")
print("=" * 60)
