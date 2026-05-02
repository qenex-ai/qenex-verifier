"""
cc-pVDZ DFT (B3LYP) Validation Tests
=======================================
Validate B3LYP/cc-pVDZ energies against PySCF reference values.
First DFT calculation with a correlation-consistent basis in QENEX LAB.

Cross-validated against PySCF 2.12.1 (cart=True) on 2026-03-15.
"""

import pytest
import numpy as np


# PySCF-verified reference energies (B3LYP/cc-pVDZ, cart=True)
PYSCF_B3LYP_CC_PVDZ = {
    "He": -2.9070540939,
    "H2_1.4": -1.1733062237,  # R(H-H) = 1.4 bohr
    "H2O": -76.4215375499,  # Expt geometry
}


class TestB3LYPCCpVDZ:
    """Validate B3LYP/cc-pVDZ energies against PySCF references."""

    def test_he_b3lyp(self):
        """He B3LYP/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from dft import DFTSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        E = DFTSolver(mol, functional="B3LYP").solve()

        ref = PYSCF_B3LYP_CC_PVDZ["He"]
        error = abs(E - ref)
        assert error < 1e-6, (
            f"He B3LYP/cc-pVDZ: E={E:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_h2_b3lyp(self):
        """H2 B3LYP/cc-pVDZ matches PySCF to < 1e-4 Eh."""
        from molecule import Molecule
        from dft import DFTSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        E = DFTSolver(mol, functional="B3LYP").solve()

        ref = PYSCF_B3LYP_CC_PVDZ["H2_1.4"]
        error = abs(E - ref)
        assert error < 1e-4, (
            f"H2 B3LYP/cc-pVDZ: E={E:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    @pytest.mark.timeout(600)
    def test_h2o_b3lyp(self):
        """H2O B3LYP/cc-pVDZ matches PySCF to < 5e-4 Eh.

        Grid-dependent: our 75-radial/110-angular grid is smaller than
        PySCF's default, so ~0.05 kcal/mol error is expected.
        Timeout: 600s because 25 bf x d-orbitals x numerical grid is slow.
        """
        from molecule import Molecule
        from dft import DFTSolver

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
        E = DFTSolver(mol, functional="B3LYP").solve()

        ref = PYSCF_B3LYP_CC_PVDZ["H2O"]
        error = abs(E - ref)
        assert error < 5e-4, (
            f"H2O B3LYP/cc-pVDZ: E={E:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_b3lyp_lower_than_hf_he(self):
        """B3LYP/cc-pVDZ should give lower energy than RHF/cc-pVDZ for He."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from dft import DFTSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )

        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(mol, verbose=False)
        E_dft = DFTSolver(mol, functional="B3LYP").solve()

        assert E_dft < E_hf, (
            f"B3LYP ({E_dft:.10f}) should be lower than RHF ({E_hf:.10f})"
        )

    @pytest.mark.timeout(600)
    def test_b3lyp_lower_than_hf_h2o(self):
        """B3LYP/cc-pVDZ should give lower energy than RHF/cc-pVDZ for H2O."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from dft import DFTSolver

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
        E_hf, _ = hf.compute_energy(mol, verbose=False)
        E_dft = DFTSolver(mol, functional="B3LYP").solve()

        assert E_dft < E_hf, (
            f"B3LYP ({E_dft:.10f}) should be lower than RHF ({E_hf:.10f})"
        )
