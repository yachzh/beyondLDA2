# beyondLDA2
---------------

A Python application created to perform LDA+U and G0W0 computations utilizing Atomic Simulation Environment (ASE) and GPAW. 
It is designed to facilitate the simulation of extended molecular materials with magnetic centers, 
specifically those incorporating elements such as Mn, Fe, Co, and Ni.

- Total electronic energy
- Global optimization of atomic structures
- Kohn-Sham gap
- GLLBSC gap
- G0W0 gap
- Optical spectrum based on BSE and RPA
- Excitons
- Phonons

## Introduction
---------------

This script provides a simple way to perform LDA+U calculations using GPAW. The LDA+U method is a widely used approach for 
treating strongly correlated systems, 
and this program aims to make it easy to use this method within the ASE framework.

## Installation
---------------

To use the package, simply clone the repository and install the required dependencies:
```bash
git clone https://github.com/yachzh/beyondLDA2.git
cd beyondLDA2
pip install -r requirements.txt
```

Successfully compiling GPAW might require the installation of the libxc library:
```bash
wget https://gitlab.com/libxc/libxc/-/archive/6.2.2/libxc-6.2.2.tar.bz2
tar -jxvf libxc-6.2.2.tar.bz2
cd libxc-6.2.2
autoreconf -i configure.ac
./configure --enable-shared --disable-fortran --prefix=$HOME/.local/apps/libxc-6.2.2
make
make install

export xc_src=$HOME/.local/apps/libxc-6.2.2
export C_INCLUDE_PATH=$xc_src/include
export LIBRARY_PATH=$xc_src/lib
export LD_LIBRARY_PATH=$xc_src/lib
```

## Example Usage
---------------

Here is a sample to demonstrate the usage of the project:
```python
import time
from ase.parallel import paropen
from ase.io import read
from ase.units import kJ, mol
from beyondLDA2 import sec2time, lda_plus_u

# Define the magnetic center and Hubbard U value
magcenter = 'fe'
U = 4.0
xc = 'LDA'

# Define the output file
resultfile = paropen('result.txt', 'w')

# Initialize an empty dictionary to store the energies
energies = {}

# Loop over the spin states
for spin in ['hs', 'ls']:
    # Read the structure file
    atoms = read(f'{spin}.STRUCT_OUT')
    atoms.set_pbc((True, True, True))
    atoms.center()

    # Start the timer
    start_time = time.time()

    # Set up the LDA+U calculator
    dft = lda_plus_u(atoms=atoms, magnetic_center=magcenter,
                     spin_state=spin, xc=xc, hubbard_u=U,
                     fname=f'{xc}-U{U:.1f}-{spin}')

    # Get the electronic energy
    energy = dft.get_electronic_energy()

    # Store the energy in the dictionary
    energies[spin] = energy

    # End the timer
    end_time = time.time()

    # Calculate the walltime
    walltime = sec2time(end_time - start_time)

    # Write the result to the output file
    resultfile.write('%s %14.6f Walltime: %10s\n' % (spin, energy, walltime))

# Calculate the energy difference between the high-spin and low-spin states
e_hl = (energies['hs'] - energies['ls']) / (1 * kJ / mol)

# Write the result to the output file
resultfile.write('E_hl = %5.1f kJ/mol\n' % e_hl) # => E_hl =  83.0 kJ/mol

# Close the output file
resultfile.close()
```
