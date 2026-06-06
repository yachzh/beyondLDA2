#!/usr/bin/env python3
"""
HS-LS energy splitting test using the README quick-start example.

Reads hs.STRUCT_OUT / ls.STRUCT_OUT, runs single-point LDA+U, and
reports the high-spin / low-spin energy difference in kJ/mol.
All outputs go to a scratch directory outside the repo.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_hs_ls_splitting.py
"""
import sys
import os
import shutil
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)

from ase.io import read
from ase.units import kJ, mol
from beyondLDA2 import lda_plus_u, sec2time

# --- Scratch directory setup ---
SCRATCH = os.environ.get(
    'BEYOND_LDA2_SCRATCH',
    os.path.expanduser('~/scratch/beyondLDA2/hs-ls-splitting'),
)
os.makedirs(SCRATCH, exist_ok=True)

# Copy input structures to scratch
for spin in ['hs', 'ls']:
    src = os.path.join(_script_dir, f'{spin}.STRUCT_OUT')
    dst = os.path.join(SCRATCH, f'{spin}.STRUCT_OUT')
    if not os.path.exists(dst):
        shutil.copy2(src, dst)

# Change to scratch dir so all output files land there
os.chdir(SCRATCH)

# --- Parameters ---
magcenter = 'fe'
U = 4.0
xc = 'LDA'

energies = {}

for spin in ['hs', 'ls']:
    atoms = read(f'{spin}.STRUCT_OUT')
    atoms.set_pbc((True, True, True))
    atoms.center()

    start = time.time()
    dft = lda_plus_u(atoms=atoms,
                     magnetic_center=magcenter,
                     spin_state=spin,
                     xc=xc,
                     hubbard_u=U,
                     fname=f'{xc}-U{U:.1f}-{spin}')
    energy = dft.get_electronic_energy()
    energies[spin] = energy

    wall = sec2time(time.time() - start)
    print(f'{spin} {energy:14.6f} eV  Walltime: {wall}')

e_hl = (energies['hs'] - energies['ls']) / (1 * kJ / mol)

print(f"\n{'='*60}")
print(f"HS-LS splitting test completed!")
print(f"E_HS = {energies['hs']:.6f} eV")
print(f"E_LS = {energies['ls']:.6f} eV")
print(f"E_HL = {e_hl:5.1f} kJ/mol")
print(f"Outputs in: {SCRATCH}")
print(f"{'='*60}")
