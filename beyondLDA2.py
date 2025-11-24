"""
---------------------------------------------------------------------
Single-point calculations and structural relaxations of
molecular materials containing magnetic centers (Mn, Fe, Co, and Ni)
based on GPAW and ASE
---------------------------------------------------------------------
@author Yachao Zhang
@email yachao.zhang@pku.edu.cn
@date April 30, 2022 11:21 PM
"""

import os
import pickle
import time
import numpy as np
import gpaw
from gpaw.response.df import DielectricFunction
from gpaw.response.bse import BSE
from gpaw.response.g0w0 import G0W0
from ase.optimize.minimahopping import MinimaHopping
from ase.optimize import QuasiNewton, BFGS
from gpaw.wavefunctions.pw import PW
from ase.io import read, write
from ase.units import kB
from gpaw import GPAW, FermiDirac
from gpaw.mixer import MixerSum, Mixer
from gpaw.poisson import PoissonSolver
from gpaw.dipole_correction import DipoleCorrection
from gpaw.utilities import h2gpts
from ase.constraints import FixAtoms, ExpCellFilter
from gpaw.external import ConstantElectricField
from ase.calculators.dftd3 import DFTD3
from ase.vibrations import Vibrations
from ase.parallel import paropen, parprint


def version_compare(current_version, min_version):
    """
    Compare two version strings (e.g., '22.8' and '22.7').

    Returns:
        - Negative number if current_version is less than min_version,
        - Zero if they are equal,
        - Positive number if current_version is greater than min_version.
    """
    def parse_version(version_str):
        try:
            return list(map(int, str(version_str).split('.')))
        except ValueError:
            raise ValueError(f"Invalid version string: {version_str}")

    current = parse_version(current_version)
    minimum = parse_version(min_version)

    # Pad the shorter version with zeros for comparison
    max_length = max(len(current), len(minimum))
    current += [0] * (max_length - len(current))
    minimum += [0] * (max_length - len(minimum))

    for c, m in zip(current, minimum):
        if c < m:
            return -1
        elif c > m:
            return 1

    # All parts are equal up to the length of both versions
    return 0


def qpbandgap(result):
    ks_skn = result['eps']
    ks_cbmin = np.amin(ks_skn[0, :, 1])
    ks_vbmax = np.amax(ks_skn[0, :, 0])
    ks_gap = ks_cbmin - ks_vbmax

    qp_skn = result['qp']
    qp_cbmin = np.amin(qp_skn[0, :, 1])
    qp_vbmax = np.amax(qp_skn[0, :, 0])
    qp_gap = qp_cbmin - qp_vbmax
    return ks_gap, qp_gap


def band_gap(calc):
    ef = calc.get_fermi_level()
    Nb = calc.wfs.bd.nbands
    w_k = calc.wfs.kd.weight_k
    x = 0
    nspin = calc.get_number_of_spins()
    energies = np.empty(len(w_k) * Nb * nspin)
    for spin in np.arange(nspin):
        for k, _ in enumerate(w_k):
            energies[x:x + Nb] = calc.get_eigenvalues(k, spin)
            x += Nb
    index1 = np.where(energies - ef <= 0)
    index2 = np.where(energies - ef > 0)

    Vb = max(energies[index1[0]])
    Cb = min(energies[index2[0]])
    return Cb - Vb


def sec2time(wall_time):
    total_seconds = round(wall_time)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    return f'{hours}:{minutes}:{seconds}'


class lda_plus_u:
    """perform LDA+U calculations using gpaw"""
    spin = {
        'Mn': (5.0, 1.0),
        'Fe': (4.0, 0.0, 2.0),
        'Co': (3.0, 1.0),
        'Ni': (2.0, 2.0)
    }
    gpv = gpaw.__version__

    def __init__(
            self,
            atoms=None,
            magnetic_center=None,
            spin_state='LS',
            xc='LDA',
            planewave=False,
            pwcut=450,  # eV
            hubbard_u=0.0,
            kmesh=(1, 1, 1),
            beta=0.07,
            nmaxold=5,
            weight=10,
            maxcycl=500,
            etol=1.e-6,
            dentol=1.e-6,
            eigentol=1.e-8,
            symmetry=False,
            temperature=300,  # kB
            vdw=False,
            charge=0.0,
            efield=0.0,
            high_ox=False,
            spin_pol=True,
            fixspin=True,
            isMol=False,
            magmoms=None,
            nbands=None,
            conv_bands=None,
            dipcorr=False,
            conv_default=False,
            kp_shift=False,
            domain_parallel=False,
            fname=None):
        parameters = locals()
        parameters.pop('self')
        self.args = parameters

    def get_label(self):
        label = self.args['fname']
        if label is None:
            label = 'gpaw-%s' % self.args['xc']
        return label

    def get_calc(self):
        if self.args['isMol']:
            self.args['atoms'].set_pbc((False, False, False))
        if self.args['conv_bands'] is None:
            conv_bands_label = 'occupied'
        else:
            conv_bands_label = self.args['conv_bands']
        xc_label = {
            'lda': 'LDA',
            'pbe': 'PBE',
            'revpbe': 'revPBE',
            'rpbe': 'RPBE',
            'pbe0': 'PBE0',
            'b3lyp': 'B3LYP',
            'vdw-df': 'vdW-DF',
            'vdw-df2': 'vdW-DF2',
            'gllbsc': 'GLLBSC'
        }
        inic = {
            'convergence': {
                'energy': self.args['etol'],
                'density': self.args['dentol'],
                'eigenstates': self.args['eigentol'],
                'bands': conv_bands_label
            },
            'maxiter': self.args['maxcycl'],
            'kpts': self.args['kmesh'],
            'xc': xc_label[self.args['xc'].lower()],
            'spinpol': self.args['spin_pol'],
            'txt': self.get_label() + '.txt'
        }
        if self.args['planewave']:
            inic['mode'] = PW(self.args['pwcut'])
            inic['eigensolver'] = 'rmm-diis'
            self.args['conv_default'] = True
            self.args['kp_shift'] = True
            self.args['symmetry'] = True
        else:
            inic['mode'] = 'lcao'
            inic['basis'] = 'dzp'
        inic['symmetry'] = {
            'point_group': self.args['symmetry'],
            'time_reversal': self.args['symmetry']
        }
        if self.args['kp_shift']:
            inic['kpts'] = {'size': self.args['kmesh'], 'gamma': True}
        if self.args['conv_default']:
            inic['convergence'] = {
                'energy': 0.0005,  # eV / electron
                'density': 1.0e-4,  # electrons / electron
                'eigenstates': 4.0e-8,  # eV^2 / electron
                'bands': conv_bands_label
            }
        if self.args['domain_parallel']:
            inic['parallel'] = {'domain': 1}
        if self.args['spin_pol']:
            mixer = MixerSum(beta=self.args['beta'],
                             nmaxold=self.args['nmaxold'],
                             weight=self.args['weight'])
        else:
            mixer = Mixer(beta=self.args['beta'],
                          nmaxold=self.args['nmaxold'],
                          weight=self.args['weight'])
            self.args['fixspin'] = False
        inic['occupations'] = FermiDirac(width=self.args['temperature'] * kB,
                                         fixmagmom=self.args['fixspin'])
        inic['mixer'] = mixer

        Ueff = self.args['hubbard_u']
        if abs(Ueff) > 1.e-6:
            metal = self.args['magnetic_center']
            inic['setups'] = {metal.capitalize(): ':d,%.4f' % Ueff}

        if abs(self.args['charge']) > 1.e-6:
            inic['charge'] = self.args['charge']

        efield = self.args['efield']
        if abs(efield) > 1.e-6:
            inic['external'] = ConstantElectricField(efield, [0, 0, 1],
                                                     tolerance=1e-07)

        if self.args['dipcorr']:
            self.args['atoms'].set_pbc((True, True, False))
            if version_compare(self.gpv, 22.8) < 0:
                poissonsolver = PoissonSolver()
                correction = DipoleCorrection(poissonsolver, 2)
                cell = self.args['atoms'].cell
                inic['poissonsolver'] = correction
                inic['gpts'] = h2gpts(0.2, cell, idiv=16)
            else:
                correction = {'dipolelayer': 'xy'}
                inic['poissonsolver'] = correction

        if self.args['nbands'] is not None:
            inic['nbands'] = self.args['nbands']

        if self.args['magnetic_center'] is not None and self.args['spin_pol']:
            self.set_magnetic_moment()

        calc = GPAW(**inic)
        if self.args['vdw']:
            d3 = DFTD3(dft=calc)
            return d3
        return calc

    def set_magnetic_moment(self):
        magnetic_moments = self.args['magmoms']
        if magnetic_moments is None:
            magcenter = self.args['magnetic_center'].capitalize()
            spinstate = self.args['spin_state'].lower()
            number_of_atoms = self.args['atoms'].get_global_number_of_atoms()
            local_magetic_moment = np.zeros(number_of_atoms)
            if spinstate == 'hs':
                local_spin = self.spin[magcenter][0]
            elif spinstate == 'is':
                local_spin = self.spin[magcenter][2]
            else:
                local_spin = self.spin[magcenter][1]
            if self.args['high_ox']:
                local_spin += 1

            for atom in self.args['atoms']:
                if atom.symbol == magcenter:
                    local_magetic_moment[atom.index] = local_spin

            self.args['atoms'].set_initial_magnetic_moments(
                magmoms=local_magetic_moment)
        else:
            self.args['atoms'].set_initial_magnetic_moments(
                magmoms=magnetic_moments)

    def get_electronic_energy(self, write_wave=False):
        calc = self.get_calc()
        self.args['atoms'].calc = calc
        energy = self.args['atoms'].get_potential_energy()
        if write_wave:
            gpwfile = '%s.gpw' % self.get_label()
            calc.write(gpwfile, mode='all')

        return energy

    def get_ksgap(self, write_wave=False):
        fname = self.get_label()
        f = paropen(f'result-{fname}.txt', 'w')
        start_time = time.time()
        self.get_electronic_energy(write_wave=write_wave)
        end_time = time.time()
        walltime = sec2time(end_time - start_time)
        gap = band_gap(self.args['atoms'].calc)
        f.write(f'Kohn-Sham gap = {gap:.3f} eV\n')
        f.write(f'Walltime: {walltime}\n')
        f.close()
        return gap

    def get_gllbscgap(self, write_wave=False):
        if not self.args['xc'].upper() == 'GLLBSC':
            self.args['xc'] = 'GLLBSC'
        fname = self.get_label()
        f = paropen(f'result-{fname}.txt', 'w')
        start_time = time.time()
        self.get_electronic_energy(write_wave=write_wave)
        calc = self.args['atoms'].calc
        response = calc.hamiltonian.xc.xcs['RESPONSE']
        response.calculate_delta_xc()
        Eks, Dxc = response.calculate_delta_xc_perturbation()
        gap = Eks + Dxc
        end_time = time.time()
        walltime = sec2time(end_time - start_time)
        f.write(f'Kohn-Sham gap: {Eks:.3f} eV\n')
        f.write(f'Delta_xc: {Dxc:.3f} eV\n')
        f.write(f'Calculated band gap: {gap:.3f} eV\n')
        f.write(f'Walltime: {walltime}\n')
        f.close()
        return gap

    def get_g0w0gap(self,
                    ecut=150,
                    ppa=False,
                    extpol=False,
                    ksbands=None,
                    nbands=None,
                    bands=None,
                    relbands=(-1, 1),
                    domega0=0.025,
                    omega2=10.0,
                    is2D=False):
        if version_compare(self.gpv, 22.8) >= 0:
            self.args['domain_parallel'] = True
        fname = self.get_label()
        f = paropen(f'result-{fname}.txt', 'w')
        start_time = time.time()
        self.get_electronic_energy()
        calc = self.args['atoms'].calc
        if ksbands is None:
            calc.diagonalize_full_hamiltonian()
        else:
            calc.diagonalize_full_hamiltonian(nbands=ksbands)
        calc.write(f'{fname}.gpw', 'all')

        freq = {'type': 'nonlinear', 'domega0': domega0, 'omega2': omega2}
        if bands is not None:
            relbands = None
        if is2D:
            truncation = '2D'
            q0_correction = True
        else:
            truncation = None
            q0_correction = False
        if ppa:
            nblocksmax = False
        else:
            nblocksmax = True
        if version_compare(self.gpv, 22.8) < 0:
            gw = G0W0(
                calc=f'{fname}.gpw',
                ecut=ecut,  # plane-wave cutoff for self-energy
                ecut_extrapolation=extpol,
                nbands=nbands,
                bands=bands,
                relbands=relbands,
                ppa=ppa,
                truncation=truncation,
                filename=f'{fname}-g0w0',
                nblocksmax=nblocksmax,
                q0_correction=q0_correction,
                domega0=domega0,
                omega2=omega2,
                savepckl=True)
        else:
            gw = G0W0(
                calc=f'{fname}.gpw',
                ecut=ecut,  # plane-wave cutoff for self-energy
                ecut_extrapolation=extpol,
                nbands=nbands,
                bands=bands,
                relbands=relbands,
                ppa=ppa,
                truncation=truncation,
                filename=f'{fname}-g0w0',
                nblocksmax=nblocksmax,
                q0_correction=q0_correction,
                frequencies=freq)
        result = gw.calculate()
        end_time = time.time()
        walltime = sec2time(end_time - start_time)
        if relbands is not None:
            ks_gap, qp_gap = qpbandgap(result)
            f.write(f'Kohn-Sham gap = {ks_gap:.3f} eV\n')
            f.write(f'G0W0 gap = {qp_gap:.3f} eV\n')
        else:
            ks_gap = 0
            qp_gap = 0  # not implemented yet
            f.write('Gap not available!\n')
        f.write(f'Walltime: {walltime}\n')
        f.close()
        return qp_gap

    def setup_bse(
            self,
            ecut=50,
            eshift=0.8,
            diag_full_ham=False,
            nbands=100,
            vb_list=None,
            cb_list=None,
            spin_orbit_coup=True,
            write_v=False,
            mode='BSE',  # TDHF, RPA, BSE
            is2D=False):
        self.args['domain_parallel'] = True
        fname = self.get_label()
        self.get_electronic_energy()
        calc = self.args['atoms'].calc
        if diag_full_ham:
            if nbands is None:
                calc.diagonalize_full_hamiltonian()
            else:
                calc.diagonalize_full_hamiltonian(nbands=nbands)
        calc.write(f'{fname}.gpw', 'all')

        if is2D:
            truncation = '2D'
            integrate_gamma = 1
        else:
            truncation = None
            integrate_gamma = 0

        bse = BSE(f'{fname}.gpw',
                  spinors=spin_orbit_coup,
                  ecut=ecut,
                  valence_bands=vb_list,
                  conduction_bands=cb_list,
                  truncation=truncation,
                  nbands=nbands,
                  eshift=eshift,
                  mode=mode,
                  write_v=write_v,
                  integrate_gamma=integrate_gamma,
                  txt=f'bse_{fname}.txt')
        return bse

    def setup_rpa(self,
                  ecut=50,
                  eshift=0.8,
                  diag_full_ham=False,
                  nbands=100,
                  frequencies=np.linspace(0, 5, 1001),
                  intraband=False,
                  hilbert=False,
                  eta=0.2):
        self.args['domain_parallel'] = True
        fname = self.get_label()
        self.get_electronic_energy()
        calc = self.args['atoms'].calc
        if diag_full_ham:
            if nbands is None:
                calc.diagonalize_full_hamiltonian()
            else:
                calc.diagonalize_full_hamiltonian(nbands=nbands)
        calc.write(f'{fname}.gpw', 'all')

        rpa = DielectricFunction(f'{fname}.gpw',
                                 ecut=ecut,
                                 frequencies=frequencies,
                                 nbands=nbands,
                                 intraband=intraband,
                                 hilbert=hilbert,
                                 eta=eta,
                                 eshift=eshift,
                                 txt=f'rpa_{fname}.txt')

        return rpa

    def get_absorption(
            self,
            ecut=50,
            eshift=0.8,
            diag_full_ham=False,
            nbands=100,
            vb_list=None,
            cb_list=None,
            spin_orbit_coup=True,
            mode='BSE',  # TDHF, RPA, BSE
            write_v=False,
            energy_range=np.linspace(0, 5, 5001),
            eta=0.05):
        fname = self.get_label()
        bse = self.setup_bse(ecut=ecut,
                             eshift=eshift,
                             diag_full_ham=diag_full_ham,
                             nbands=nbands,
                             vb_list=vb_list,
                             cb_list=cb_list,
                             mode=mode,
                             spin_orbit_coup=spin_orbit_coup,
                             write_v=write_v,
                             is2D=True)
        hw, abs_w = bse.get_2d_absorption(w_w=energy_range,
                                          eta=eta,
                                          filename=f'abs_bse_{fname}.csv',
                                          write_eig=f'bse_eig_{fname}.dat')
        return hw, abs_w

    def local_opt(self,
                  force_convergence=0.02,
                  optimizer=BFGS,
                  maxstep=0.02,
                  fix_atoms=False,
                  fix_index_set=None,
                  varcell=False):
        calc = self.get_calc()

        fn_traj = f'qn-{self.get_label()}.traj'
        restart_needed = False

        if os.path.exists(fn_traj) and os.path.getsize(fn_traj) > 0:
            parprint(f"Found existing trajectory: {fn_traj}")
            parprint("Reading latest positions...")
            self.args['atoms'] = read(fn_traj, index=-1)
            restart_needed = True

        self.args['atoms'].calc = calc

        if fix_atoms:
            constraint = FixAtoms(indices=fix_index_set)
            self.args['atoms'].set_constraint(constraint)

        if varcell and self.args['planewave']:
            opt = optimizer(ExpCellFilter(self.args['atoms']),
                            trajectory=fn_traj,
                            maxstep=maxstep,
                            append_trajectory=True)
        else:
            opt = optimizer(self.args['atoms'],
                            trajectory=fn_traj,
                            maxstep=maxstep,
                            append_trajectory=True)
        if restart_needed:
            opt.replay_trajectory(fn_traj)

        opt.run(fmax=force_convergence)
        write('%s-opt.xyz' % self.get_label(), self.args['atoms'])
        write('%s-opt.struct' % self.get_label(), self.args['atoms'])
        write('%s-opt.vasp' % self.get_label(), self.args['atoms'])
        energy = self.args['atoms'].get_potential_energy()
        return energy

    def global_opt(
            self,
            force_convergence=0.04,
            optimizer=QuasiNewton,
            fix_atoms=False,
            fix_index_set=None,
            ini_temperature=1000,  # in K
            max_temperature=1250,  # in K
            totstep=20):
        calc = self.get_calc()
        self.args['atoms'].calc = calc

        if fix_atoms:
            constraint = FixAtoms(indices=fix_index_set)
            self.args['atoms'].set_constraint(constraint)

        opt = MinimaHopping(atoms=self.args['atoms'],
                            optimizer=optimizer,
                            fmax=force_convergence,
                            T0=ini_temperature)
        opt(totalsteps=totstep, maxtemp=max_temperature)

    def phonon(self, vibIndices=None):
        calc = self.get_calc()
        self.args['atoms'].calc = calc

        label = self.get_label()
        if vibIndices is None:
            vib = Vibrations(self.args['atoms'], name=f'{label}-vibmode')
        else:
            vib = Vibrations(self.args['atoms'],
                             indices=vibIndices,
                             name=f'{label}-vibmode')
        vib.run()
        vib_energies = vib.get_energies()
        vib.summary(log=label + '.vib')

        with open(f'{label}vib.pickle', 'wb') as f:
            pickle.dump(vib_energies, f)
        vib.write_dos(out=label + 'vib-dos.dat', start=0, width=25, npts=5000)
        vib.write_jmol()
