# beyondLDA2

DFT+U and beyond for molecular magnetism with GPAW and ASE.

Designed for atomistic simulations of extended molecular systems with magnetic
centers (Mn, Fe, Co, Ni). Provides a unified interface to single-point energies,
band gaps (KS, GLLBSC, G0W0), optical spectra (BSE/RPA), geometry optimization,
and phonons — all through the `lda_plus_u` class.

## Capabilities

| Method | Class method | Description |
|---|---|---|
| **DFT+U energy** | `get_electronic_energy()` | Single-point SCF with Hubbard U correction on the magnetic center |
| **Kohn-Sham gap** | `get_ksgap()` | KS HOMO–LUMO gap after SCF |
| **GLLBSC gap** | `get_gllbscgap()` | Derivative discontinuity gap (Δ_xc correction) |
| **G0W0 gap** | `get_g0w0gap()` | Quasiparticle band gap from one-shot GW |
| **BSE spectrum** | `get_absorption()` | Optical absorption from Bethe-Salpeter equation (supports TDHF, RPA, BSE) |
| **RPA response** | `setup_rpa()` | Dielectric function in the random-phase approximation |
| **Local opt.** | `local_opt()` | Geometry relaxation (BFGS / QuasiNewton), optionally with variable cell |
| **Global opt.** | `global_opt()` | Minima hopping for global structure search |
| **Phonons** | `phonon()` | Vibrational frequencies via finite displacements |

## Supported XC functionals

`LDA`, `PBE`, `revPBE`, `RPBE`, `PBE0`, `B3LYP`, `vdW-DF`, `vdW-DF2`, `GLLBSC`

Set via the `xc` parameter (case-insensitive).

## Magnetic centers

| Element | HS moment (μ_B) | LS moment (μ_B) |
|---|---|---|
| Mn | 5.0 | 1.0 |
| Fe | 4.0 | 0.0 |
| Co | 3.0 | 1.0 |
| Ni | 2.0 | 2.0 |

Magnetic moments are assigned automatically to the element matching
`magnetic_center`. Custom moments can be passed via `magmoms`.

## Requirements

- **GPAW** ≥ 20.1.0 (tested with 20.1.0, 24.6.0, 25.7.0)
- **ASE** ≥ 3.20.0 (tested with 3.20.0, 3.25.0, 3.28.0)
- **libxc** ≥ 5.2.3 (required by GPAW)
- numpy, scipy

## Installation

```bash
git clone https://github.com/yachzh/beyondLDA2.git
cd beyondLDA2
pip install -r requirements.txt
```

GPAW must be compiled with libxc support. See [GPAW install docs](https://wiki.fysik.dtu.dk/gpaw/install.html)
for details. A working build with libxc 5.2.3 and BLAS/LAPACK is required.

PAW setup data must be installed after GPAW:
```bash
gpaw install-data $HOME/.local
```

Ensure `GPAW_SETUP_PATH` points to the setup directory if it is not in a
standard location (e.g., inside a virtual environment).

## Quick start

Compute the high-spin / low-spin energy splitting for an Fe-porphyrin-like
molecule:

```python
import time
from ase.io import read
from ase.units import kJ, mol
from beyondLDA2 import lda_plus_u, sec2time

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
    print(f'{spin} {energy:14.6f}  Walltime: {wall}')

e_hl = (energies['hs'] - energies['ls']) / (1 * kJ / mol)
print(f'E_HL = {e_hl:5.1f} kJ/mol')
```

Example output (Fe center, LDA+U=4.0 eV, single k-point, LCAO-dzp):
```
hs     -135.345739  Walltime: 0:09:32
ls     -136.206124  Walltime: 0:01:21
E_HL =  83.0 kJ/mol
```

## Constructor reference

```python
lda_plus_u(
    atoms,                              # ASE Atoms object
    magnetic_center='fe',               # Element symbol (case-insensitive)
    spin_state='LS',                    # 'hs', 'ls', or 'is'
    xc='LDA',                           # Exchange-correlation functional
    planewave=False,                    # Use PW mode (default: LCAO)
    pwcut=450,                          # Plane-wave cutoff (eV, PW mode only)
    hubbard_u=0.0,                      # U value for DFT+U (eV)
    kmesh=(1, 1, 1),                    # k-point mesh
    beta=0.07,                          # Mixing parameter
    nmaxold=5,                          # Number of old densities in mixer
    weight=10,                          # Mixer weight
    maxcycl=500,                        # Max SCF iterations
    etol=1e-6,                          # Energy convergence (eV/electron)
    dentol=1e-6,                        # Density convergence (electrons/electron)
    eigentol=1e-8,                      # Eigenstate convergence (eV²/electron)
    symmetry=False,                     # Use point-group symmetry
    temperature=300,                    # Electronic temperature (K)
    vdw=False,                          # Dispersion correction (DFT-D3)
    charge=0.0,                         # Total charge
    efield=0.0,                         # Electric field (eV/Å)
    high_ox=False,                      # High oxidation state (+1 μ_B)
    spin_pol=True,                      # Spin-polarized calculation
    fixspin=True,                       # Fix magnetic moment (FM)
    isMol=False,                        # Non-periodic (PBC = False)
    magmoms=None,                       # Custom magnetic moments (array)
    nbands=None,                        # Override number of bands
    conv_bands=None,                    # Bands for convergence check
    dipcorr=False,                      # Dipole correction (slab)
    conv_default=False,                 # Use GPAW default convergence
    semicore=False,                     # Semi-core pv setups
    kp_shift=False,                     # Gamma-centered k-points
    domain_parallel=False,              # Domain parallelization
    fname=None                          # Output file basename
)
```

## Output

Each calculation writes a `.txt` file with the full GPAW log (SCF iterations,
timing breakdown, memory). The filename is controlled by the `fname` parameter
(default: `gpaw-{xc}.txt`).

## Notes

- **LCAO mode** (default) uses a DZP basis. PW mode is required for G0W0, BSE,
  and RPA calculations.
- **Structural input** is read from external files (`.STRUCT_OUT`, .xyz, etc.)
  via ASE's `read()`. The repository includes example `hs.STRUCT_OUT` and
  `ls.STRUCT_OUT` files in the `tests/` directory.
- The module is **parallel-safe**: uses `paropen` / `parprint` for MPI-aware
  I/O. Domain parallelization can be enabled for G0W0/BSE.
- **Version compatibility**: tested with GPAW 20.1.0–25.7.0 and ASE
  3.20.0–3.28.0 (LCAO-LDA+U on a single core). Results are bitwise identical
  across versions. The `ExpCellFilter` import adapts to ASE ≥ 3.26.0 vs. earlier
  releases automatically.

## License

MIT
