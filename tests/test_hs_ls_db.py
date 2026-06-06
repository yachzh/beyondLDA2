#!/usr/bin/env python3
"""
HS-LS energy splitting test with auto database storage.

Same workflow as test_hs_ls_splitting.py but stores every calculation
result into an ase.db database. Demonstrates the full analysis
pipeline: compute → auto-store → query → aggregate.

The database persists in the scratch directory so results can be
re-examined or merged later without re-running calculations.

Usage:
    cd /path/to/beyondLDA2
    python tests/test_hs_ls_db.py

Outputs:
    <scratch>/hs-ls-db/           scratch directory
    <scratch>/hs-ls-db/hs-ls.db   ASE database with all rows
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
from beyondLDA2_db import LDAPlusUDatabase


def main():
    # --- Scratch directory setup ---
    SCRATCH = os.environ.get(
        'BEYOND_LDA2_SCRATCH',
        os.path.expanduser('~/scratch/beyondLDA2/hs-ls-db'),
    )
    os.makedirs(SCRATCH, exist_ok=True)
    db_path = os.path.join(SCRATCH, 'hs-ls.db')
    os.chdir(SCRATCH)

    # Copy input structures to scratch
    for spin in ['hs', 'ls']:
        src = os.path.join(_script_dir, f'{spin}.STRUCT_OUT')
        dst = os.path.join(SCRATCH, f'{spin}.STRUCT_OUT')
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # --- Create database ---
    db = LDAPlusUDatabase(db_path)
    print(f"Database: {db_path}")
    print(f"  Existing rows: {db.nrows}")

    # --- Parameters ---
    magcenter = 'fe'
    U = 4.0
    xc = 'LDA'

    energies = {}

    # --- Run HS + LS with auto DB storage ---
    for spin in ['hs', 'ls']:
        atoms = read(f'{spin}.STRUCT_OUT')
        atoms.set_pbc((True, True, True))
        atoms.center()

        start = time.time()
        dft = lda_plus_u(
            atoms=atoms,
            magnetic_center=magcenter,
            spin_state=spin,
            xc=xc,
            hubbard_u=U,
            fname=f'{xc}-U{U:.1f}-{spin}',
            conv_default=True,
            database=db,
        )
        energy = dft.get_electronic_energy()
        energies[spin] = energy

        wall = time.time() - start
        print(f'{spin}  energy={energy:14.6f} eV  walltime={sec2time(wall)}')

    # --- Query DB + compute splitting ---
    print(f"\n{'='*60}")
    print("Database query: HS-LS splitting analysis")
    print(f"{'='*60}")

    hs_rows = db.select(method='electronic_energy', spin_state='hs')
    ls_rows = db.select(method='electronic_energy', spin_state='ls')

    if not hs_rows or not ls_rows:
        print("ERROR: Could not find HS/LS rows in database")
        sys.exit(1)

    e_hs = hs_rows[0].result_value
    e_ls = ls_rows[0].result_value
    e_hl = (e_hs - e_ls) / (1 * kJ / mol)

    print(f"\nE_HS (from DB) = {e_hs:.6f} eV")
    print(f"E_LS (from DB) = {e_ls:.6f} eV")
    print(f"E_HL (from DB) = {e_hl:6.1f} kJ/mol")

    # Direct comparison with in-memory result
    assert abs(e_hs - energies['hs']) < 1e-10, "DB value mismatch (HS)"
    assert abs(e_ls - energies['ls']) < 1e-10, "DB value mismatch (LS)"

    # --- Print full DB contents ---
    print(f"\n{'='*60}")
    print(f"All rows in database ({db.nrows} total)")
    print(f"{'='*60}")
    for row in db.select():
        data = row.data or {}
        wt = data.get('walltime', '?')
        print(f"  id={row.id:2d}  {row.formula:6s}  {row.method:20s}  "
              f"{row.result_value:10.4f} eV  spin={row.spin_state:4s}  "
              f"wall={sec2time(wt) if isinstance(wt, (int, float)) else wt}")

    # --- Structure round-trip verification ---
    print(f"\n{'='*60}")
    print("Structure round-trip check")
    print(f"{'='*60}")
    for row in db.select():
        atoms_rt = row.toatoms()
        formula = atoms_rt.get_chemical_formula()
        n_atoms = len(atoms_rt)
        print(f"  id={row.id}: {formula} ({n_atoms} atoms) — "
              f"reconstructed OK")

    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"E_HL = {e_hl:6.1f} kJ/mol")
    print(f"Database: {db_path}")
    print(f"Outputs in: {SCRATCH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
