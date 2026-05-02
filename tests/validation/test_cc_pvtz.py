"""
cc-pVTZ Validation Tests (RHF + MP2)
======================================
Validate cc-pVTZ basis set with f-orbitals against PySCF.
First f-orbital (L=3) computation in QENEX LAB.

PySCF 2.12.1 cart=False cross-validation (regenerated 2026-04-21
for the spherical-d basis fix).
"""

import pytest
import numpy as np

# PySCF cart=False references (spherical d- and f-functions).
PYSCF_RHF = {
    "He": -2.8611533448,
    "H2_1.4": -1.1329605255,
    "H2O": -76.0571701271,
}
PYSCF_MP2 = {
    "He": {"corr": -0.0331375618, "tot": -2.8942909065},
    "H2_1.4": {"corr": -0.0316790935, "tot": -1.1646396190},
    "H2O": {"corr": -0.2750736106, "tot": -76.3322437377},
}


class TestCCpVTZRHF:
    """RHF/cc-pVTZ energy validation."""

    def test_he_rhf(self):
        """He RHF/cc-pVTZ < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvtz"
        )
        E, _ = HartreeFockSolver().compute_energy(mol, verbose=False)
        assert abs(E - PYSCF_RHF["He"]) < 1e-6

    def test_h2_rhf(self):
        """H2 RHF/cc-pVTZ < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvtz",
        )
        E, _ = HartreeFockSolver().compute_energy(mol, verbose=False)
        assert abs(E - PYSCF_RHF["H2_1.4"]) < 1e-6

    def test_h2o_rhf(self):
        """H2O RHF/cc-pVTZ (65 bf, f-orbitals) < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvtz",
        )
        E, _ = HartreeFockSolver().compute_energy(mol, verbose=False)
        assert abs(E - PYSCF_RHF["H2O"]) < 1e-6

    def test_basis_count_he(self):
        """He cc-pVTZ: 15 Cartesian AOs."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvtz"
        )
        assert len(build_basis(mol)) == 15

    def test_basis_count_h2o(self):
        """H2O cc-pVTZ: 65 Cartesian AOs (35 O + 15 H + 15 H)."""
        from molecule import Molecule
        from integrals import build_basis

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvtz",
        )
        assert len(build_basis(mol)) == 65

    def test_cc_pvtz_lower_than_cc_pvdz(self):
        """cc-pVTZ He should give lower energy than cc-pVDZ (variational)."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        hf = HartreeFockSolver()
        E_dz, _ = hf.compute_energy(
            Molecule(
                [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
            ),
            verbose=False,
        )
        E_tz, _ = hf.compute_energy(
            Molecule(
                [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvtz"
            ),
            verbose=False,
        )
        assert E_tz < E_dz


class TestCCpVTZMP2:
    """MP2/cc-pVTZ correlation energy validation."""

    def test_he_mp2(self):
        """He MP2/cc-pVTZ corr < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvtz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        E_tot, E_corr = MP2Solver().compute_correlation(hf, mol, verbose=False)
        assert abs(E_corr - PYSCF_MP2["He"]["corr"]) < 1e-6
        assert abs(E_tot - PYSCF_MP2["He"]["tot"]) < 1e-6

    def test_h2_mp2(self):
        """H2 MP2/cc-pVTZ corr < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvtz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        E_tot, E_corr = MP2Solver().compute_correlation(hf, mol, verbose=False)
        assert abs(E_corr - PYSCF_MP2["H2_1.4"]["corr"]) < 1e-6

    def test_h2o_mp2(self):
        """H2O MP2/cc-pVTZ (65 bf, f-orbitals) corr < 1e-6 Eh vs PySCF."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvtz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        E_tot, E_corr = MP2Solver().compute_correlation(hf, mol, verbose=False)
        assert abs(E_corr - PYSCF_MP2["H2O"]["corr"]) < 1e-6

    def test_cc_pvtz_more_correlation_than_cc_pvdz(self):
        """cc-pVTZ should recover more MP2 correlation than cc-pVDZ."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        hf_dz = HartreeFockSolver()
        mol_dz = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf_dz.compute_energy(mol_dz, verbose=False)
        _, E_corr_dz = MP2Solver().compute_correlation(hf_dz, mol_dz, verbose=False)

        hf_tz = HartreeFockSolver()
        mol_tz = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvtz"
        )
        hf_tz.compute_energy(mol_tz, verbose=False)
        _, E_corr_tz = MP2Solver().compute_correlation(hf_tz, mol_tz, verbose=False)

        assert E_corr_tz < E_corr_dz, (
            f"cc-pVTZ ({E_corr_tz:.6f}) should recover more corr than cc-pVDZ ({E_corr_dz:.6f})"
        )
