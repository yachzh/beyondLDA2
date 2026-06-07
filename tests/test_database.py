#!/usr/bin/env python3
"""
Test the beyondLDA2 database integration.

Demonstrates the full workflow:
1. Create a database and store calculation results (simulated or real)
2. Query the database for analysis
3. Aggregate results (e.g., HS-LS splitting, gap comparison)

Two modes:
  - ``python tests/test_database.py``    → simulated data (fast, no GPAW needed)
  - ``python tests/test_database.py --real`` → real lda_plus_u calculations

Usage:
    cd /path/to/beyondLDA2
    python tests/test_database.py [--real]
"""
import sys
import os
import tempfile
import json
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
sys.path.insert(0, _repo_root)

import numpy as np


# ============================================================
# Part 1: Standalone DB API (works without GPAW)
# ============================================================
def test_db_api():
    """Demonstrate LDAPlusUDatabase with simulated data."""
    print("=" * 70)
    print("PART 1: Standalone LDAPlusUDatabase API (simulated data)")
    print("=" * 70)

    from beyondLDA2_db import LDAPlusUDatabase
    from ase import Atoms

    # --- 1. Create database (temp file) ---
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = LDAPlusUDatabase(db_path)
    print(f"\nDatabase created: {db.filename}")

    # --- 2. Store simulated calculations ---
    # Simulate a molecule: Fe-O dimer
    mol = Atoms('FeO',
                positions=[(0, 0, 0), (1.6, 0, 0)],
                pbc=(True, True, True))

    print("\n--- Storing simulated calculations ---")
    for spin, label in [('HS', 'feo-hs'), ('LS', 'feo-ls')]:
        db.store(
            atoms=mol,
            method='electronic_energy',
            result_value=-15.0 if spin == 'HS' else -14.2,
            label=label,
            xc='LDA',
            spin_state=spin,
            hubbard_u=4.0,
            walltime=3600.0,
            input_args={
                'xc': 'LDA', 'hubbard_u': 4.0, 'spin_state': spin,
                'planewave': False, 'spin_pol': True,
            },
        )
        print(f"  Stored {label}: energy = {'HS' if spin == 'HS' else 'LS'}")

    # Also store gap calculations
    db.store(
        atoms=mol,
        method='ks_gap',
        result_value=2.5,
        label='feo-hs',
        xc='LDA',
        spin_state='HS',
        hubbard_u=4.0,
        walltime=3700.0,
    )
    print(f"  Stored feo-hs: KS gap = 2.5 eV")

    db.store(
        atoms=mol,
        method='gllbsc_gap',
        result_value=5.2,
        label='feo-hs',
        xc='GLLBSC',
        spin_state='HS',
        hubbard_u=4.0,
        gllbsc_ks_gap=2.5,
        gllbsc_dxc=2.7,
        walltime=3800.0,
    )
    print(f"  Stored feo-hs: GLLBSC gap = 5.2 eV (Eks=2.5, Dxc=2.7)")

    # Store a second structure
    mol2 = Atoms('Fe2O3',
                 positions=[(0, 0, 0), (1.6, 0, 0), (0, 1.6, 0), (0.8, 0.8, 0), (1.6, 1.6, 0)],
                 pbc=(True, True, True))
    db.store(
        atoms=mol2,
        method='electronic_energy',
        result_value=-25.0,
        label='fe2o3-hs',
        xc='PBE',
        spin_state='HS',
        hubbard_u=4.0,
        walltime=7200.0,
    )
    print(f"  Stored fe2o3-hs: energy = -25.0 eV")

    # --- 3. Query the database ---
    print("\n--- Querying the database ---")
    print(f"  Total rows: {db.nrows}")

    # Query by formula
    rows = db.select(formula="FeO")
    print(f"\n  Rows for FeO: {len(rows)} rows")
    for row in rows:
        print(f"    id={row.id}  method={row.method}  result={row.result_value:.4f}  "
              f"spin={row.spin_state}  label={row.label}")

    # Query by method + spin
    rows = db.select(method="electronic_energy", spin_state="HS")
    print(f"\n  HS energies: {len(rows)} rows")
    for row in rows:
        print(f"    id={row.id}  formula={row.formula}  energy={row.result_value:.4f} eV  "
              f"label={row.label}")

    # --- 4. Analysis: HS-LS splitting ---
    print("\n--- Analysis: HS-LS splitting (simulated) ---")
    hs_rows = db.select(method="electronic_energy", spin_state="HS", formula="FeO")
    ls_rows = db.select(method="electronic_energy", spin_state="LS", formula="FeO")

    if hs_rows and ls_rows:
        e_hs = hs_rows[0].result_value
        e_ls = ls_rows[0].result_value
        e_hl = (e_hs - e_ls) / (1 * 0.001)  # eV → kJ/mol (approx)
        print(f"  E_HS = {e_hs:.6f} eV")
        print(f"  E_LS = {e_ls:.6f} eV")
        print(f"  E_HL = {e_hl:.1f} kJ/mol")

    # --- 5. Export to JSON for external analysis ---
    print("\n--- Export to JSON ---")
    all_rows = db.select()
    export = []
    for row in all_rows:
        export.append({
            'id': row.id,
            'formula': row.formula,
            'method': row.method,
            'result_value': row.result_value,
            'spin_state': getattr(row, 'spin_state', None),
            'xc': getattr(row, 'xc', None),
            'hubbard_u': getattr(row, 'hubbard_u', None),
            'label': getattr(row, 'label', None),
        })
    print(f"  Exported {len(export)} rows to JSON format")
    print(json.dumps(export, indent=2))

    # --- 6. Round-trip: reconstruct atoms from DB ---
    print("\n--- Structure round-trip ---")
    first_row = all_rows[0]
    atoms_rt = first_row.toatoms()
    print(f"  Reconstructed: {atoms_rt.get_chemical_formula()}"
          f"  ({len(atoms_rt)} atoms)")

    # --- 7. Verify auto-added fields ---
    print("\n--- Auto-added fields (created_utc, creator) ---")
    row = all_rows[0]
    print(f"  created_utc = {row.created_utc}")
    print(f"  creator = {row.creator}")
    assert row.created_utc.endswith('Z'), "created_utc should be ISO 8601 UTC"
    assert isinstance(row.creator, str) and len(row.creator) > 0
    print("  ✓ Both fields present and valid")

    # Filter by creator
    rows_by_me = db.select(creator=row.creator)
    print(f"  Rows by '{row.creator}': {len(rows_by_me)}/{db.nrows}")
    assert len(rows_by_me) == db.nrows

    assert len(rows_by_me) == db.nrows

    # Cleanup
    os.unlink(db_path)
    print(f"\n  Cleaned up temp database.")
    return True


# ============================================================
# Part 2: Integration with lda_plus_u (requires GPAW)
# ============================================================
def test_real_calculation():
    """Run a real lda_plus_u calculation with database storage.

    This demonstrates the integration with a light calculation.
    """
    print("=" * 70)
    print("PART 2: Real lda_plus_u + database integration")
    print("=" * 70)

    from beyondLDA2_db import LDAPlusUDatabase
    from beyondLDA2 import lda_plus_u
    from ase.build import bulk

    # --- Create database ---
    SCRATCH = os.environ.get(
        'BEYOND_LDA2_SCRATCH',
        os.path.expanduser('~/scratch/beyondLDA2/db-test'),
    )
    os.makedirs(SCRATCH, exist_ok=True)
    db_path = os.path.join(SCRATCH, "calculations.db")
    db = LDAPlusUDatabase(db_path)
    print(f"\nDatabase: {db_path}")

    # --- Si bulk gap calculation with DB ---
    si = bulk('Si', 'diamond', a=5.43, cubic=False)
    si.set_pbc((True, True, True))

    print("\n--- Running KS gap (planewave, LDA) ---")
    dft = lda_plus_u(
        atoms=si,
        xc='LDA',
        planewave=True,
        pwcut=300,
        kmesh=(4, 4, 4),
        spin_pol=False,
        fname='si-gaps',
        database=db,
    )
    ks_gap = dft.get_ksgap()

    print(f"  KS gap = {ks_gap:.3f} eV")

    # --- Query DB ---
    print("\n--- Database contents ---")
    for row in db.select():
        print(f"  id={row.id}  method={row.method}  "
              f"result={row.result_value:.4f}  formula={row.formula}")
        if hasattr(row, 'walltime') and row.data.get('walltime'):
            print(f"      walltime={row.data['walltime']:.1f}s")

    print(f"\n  Total rows: {db.nrows}")
    return True


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test beyondLDA2 database integration")
    parser.add_argument('--real', action='store_true',
                        help="Run real lda_plus_u calculations (requires GPAW)")
    args = parser.parse_args()

    success = test_db_api()

    if args.real:
        success &= test_real_calculation()

    print("\n" + "=" * 70)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed.")
        sys.exit(1)
