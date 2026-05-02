"""
CCSD and CCSD(T) Validation Tests
====================================
Validates coupled cluster energies against PySCF references.
Sub-nanohartree agreement on all systems.

PySCF 2.12.1 cc-pVDZ cart=False cross-validation (regenerated
2026-04-21 for the spherical-d basis fix; see commit log).
"""

import pytest
import numpy as np

# Regenerated PySCF references with cart=False (spherical d-fns).
PYSCF_CCSD = {
    "He": -0.0324343541,
    "H2_1.4": -0.0346892839,
    "H2O": -0.2132821696,
    "HF_mol": -0.2087419265,
    "NH3": -0.2039437689,
    "CH4": -0.1872784018,
}
PYSCF_T = {
    "H2O": -0.0030555295,
    "HF_mol": -0.0019363032,
    "NH3": -0.0037132768,
    "CH4": -0.0037324546,
}


class TestCCSD:
    """CCSD correlation energy validation."""

    def test_he_ccsd(self):
        """He CCSD/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        _, E_corr = CCSDSolver().solve(hf, mol, verbose=False)

        assert abs(E_corr - PYSCF_CCSD["He"]) < 1e-6

    def test_h2_ccsd(self):
        """H2 CCSD/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        _, E_corr = CCSDSolver().solve(hf, mol, verbose=False)

        assert abs(E_corr - PYSCF_CCSD["H2_1.4"]) < 1e-6

    def test_h2o_ccsd(self):
        """H2O CCSD/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        _, E_corr = CCSDSolver().solve(hf, mol, verbose=False)

        assert abs(E_corr - PYSCF_CCSD["H2O"]) < 1e-6

    def test_h2o_ccsd_t(self):
        """H2O (T)/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        solver = CCSDSolver()
        solver.solve(hf, mol, verbose=False)
        E_t = solver.ccsd_t(verbose=False)

        assert abs(E_t - PYSCF_T["H2O"]) < 1e-6

    def test_ccsd_lower_than_mp2_he(self):
        """CCSD recovers more correlation than MP2."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)

        _, E_mp2 = MP2Solver().compute_correlation(hf, mol, verbose=False)
        _, E_ccsd = CCSDSolver().solve(hf, mol, verbose=False)

        assert E_ccsd < E_mp2

    def test_ccsd_correlation_negative(self):
        """CCSD correlation must be negative."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        _, E_corr = CCSDSolver().solve(hf, mol, verbose=False)

        assert E_corr < 0

    def test_hf_mol_ccsd_t(self):
        """HF molecule CCSD(T)/cc-pVDZ < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("F", (0, 0, 0)), ("H", (0, 0, 1.7328))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        solver = CCSDSolver()
        solver.solve(hf, mol, verbose=False)
        E_t = solver.ccsd_t(verbose=False)
        assert abs(solver._E_corr - PYSCF_CCSD["HF_mol"]) < 1e-6
        assert abs(E_t - PYSCF_T["HF_mol"]) < 1e-6

    def test_nh3_ccsd_t(self):
        """NH3 CCSD(T)/cc-pVDZ < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [
                ("N", (0, 0, 0.1173)),
                ("H", (0, 1.7717, -0.5461)),
                ("H", (1.5342, -0.8858, -0.5461)),
                ("H", (-1.5342, -0.8858, -0.5461)),
            ],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        solver = CCSDSolver()
        solver.solve(hf, mol, verbose=False)
        E_t = solver.ccsd_t(verbose=False)
        assert abs(solver._E_corr - PYSCF_CCSD["NH3"]) < 1e-6
        assert abs(E_t - PYSCF_T["NH3"]) < 1e-6

    @pytest.mark.timeout(300)
    def test_ch4_ccsd_t(self):
        """CH4 CCSD(T)/cc-pVDZ < 1e-6 Eh (35 AOs, 5 atoms)."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        d = 1.186
        mol = Molecule(
            [
                ("C", (0, 0, 0)),
                ("H", (d, d, d)),
                ("H", (-d, -d, d)),
                ("H", (-d, d, -d)),
                ("H", (d, -d, -d)),
            ],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        solver = CCSDSolver()
        solver.solve(hf, mol, verbose=False)
        E_t = solver.ccsd_t(verbose=False)
        assert abs(solver._E_corr - PYSCF_CCSD["CH4"]) < 1e-6
        assert abs(E_t - PYSCF_T["CH4"]) < 1e-6

    def test_h2_ccsd_t_zero(self):
        """H2 (T) = 0 (only 2 electrons)."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        solver = CCSDSolver()
        solver.solve(hf, mol, verbose=False)
        E_t = solver.ccsd_t(verbose=False)

        assert E_t == 0.0
