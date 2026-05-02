"""
cc-pVDZ MP2 Validation Tests
===============================
Validate MP2/cc-pVDZ correlation energies against PySCF reference values.
The cc-pVDZ basis was designed specifically for correlated methods (Dunning 1989).

Cross-validated against PySCF 2.12.1 (cart=False, regenerated
2026-04-21 for the spherical-d basis fix).
"""

import pytest
import numpy as np


# PySCF-verified reference energies (MP2/cc-pVDZ, cart=False).
# He and H2 are unchanged from cart=True since those systems have no
# d-functions in cc-pVDZ.  H2O's references shift by ~3.6 mHa (the
# known spurious s-contaminant effect being removed).
PYSCF_MP2_CC_PVDZ = {
    "He": {"E_corr": -0.0258283396, "E_tot": -2.8809888168},
    "H2_1.4": {"E_corr": -0.0263792393, "E_tot": -1.1550886883},
    "H2O": {"E_corr": -0.2039582807, "E_tot": -76.2307580899},
}


class TestMP2CCpVDZ:
    """Validate MP2/cc-pVDZ energies against PySCF references."""

    def test_he_mp2_correlation(self):
        """He MP2/cc-pVDZ correlation matches PySCF to < 1e-8 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        mp2 = MP2Solver()
        E_tot, E_corr = mp2.compute_correlation(hf, mol, verbose=False)

        ref = PYSCF_MP2_CC_PVDZ["He"]
        err_corr = abs(E_corr - ref["E_corr"])
        err_tot = abs(E_tot - ref["E_tot"])
        assert err_corr < 1e-8, (
            f"He MP2 corr: {E_corr:.10f}, ref={ref['E_corr']:.10f}, err={err_corr:.2e}"
        )
        assert err_tot < 1e-8, (
            f"He MP2 total: {E_tot:.10f}, ref={ref['E_tot']:.10f}, err={err_tot:.2e}"
        )

    def test_h2_mp2_correlation(self):
        """H2 MP2/cc-pVDZ correlation matches PySCF to < 1e-8 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        mp2 = MP2Solver()
        E_tot, E_corr = mp2.compute_correlation(hf, mol, verbose=False)

        ref = PYSCF_MP2_CC_PVDZ["H2_1.4"]
        err_corr = abs(E_corr - ref["E_corr"])
        err_tot = abs(E_tot - ref["E_tot"])
        assert err_corr < 1e-8, (
            f"H2 MP2 corr: {E_corr:.10f}, ref={ref['E_corr']:.10f}, err={err_corr:.2e}"
        )
        assert err_tot < 1e-8, (
            f"H2 MP2 total: {E_tot:.10f}, ref={ref['E_tot']:.10f}, err={err_tot:.2e}"
        )

    def test_h2o_mp2_correlation(self):
        """H2O MP2/cc-pVDZ correlation matches PySCF to < 1e-6 Eh."""
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
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        mp2 = MP2Solver()
        E_tot, E_corr = mp2.compute_correlation(hf, mol, verbose=False)

        ref = PYSCF_MP2_CC_PVDZ["H2O"]
        err_corr = abs(E_corr - ref["E_corr"])
        err_tot = abs(E_tot - ref["E_tot"])
        assert err_corr < 1e-6, (
            f"H2O MP2 corr: {E_corr:.10f}, ref={ref['E_corr']:.10f}, err={err_corr:.2e}"
        )
        assert err_tot < 1e-6, (
            f"H2O MP2 total: {E_tot:.10f}, ref={ref['E_tot']:.10f}, err={err_tot:.2e}"
        )

    def test_he_correlation_negative(self):
        """MP2 correlation energy must be negative (by 2nd-order PT theorem)."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        mp2 = MP2Solver()
        _, E_corr = mp2.compute_correlation(hf, mol, verbose=False)

        assert E_corr < 0, f"MP2 correlation must be negative, got {E_corr}"

    def test_h2o_correlation_significant(self):
        """H2O should have significant correlation (~0.2 Eh with cc-pVDZ)."""
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
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        mp2 = MP2Solver()
        _, E_corr = mp2.compute_correlation(hf, mol, verbose=False)

        assert -0.3 < E_corr < -0.1, (
            f"H2O MP2 corr should be ~-0.2 Eh, got {E_corr:.6f}"
        )

    def test_cc_pvdz_more_correlation_than_sto3g(self):
        """cc-pVDZ should recover more correlation energy than STO-3G."""
        from molecule import Molecule
        from solver import HartreeFockSolver, MP2Solver

        mol_sto = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf_sto = HartreeFockSolver()
        hf_sto.compute_energy(mol_sto, verbose=False)
        mp2_sto = MP2Solver()
        _, E_corr_sto = mp2_sto.compute_correlation(hf_sto, mol_sto, verbose=False)

        mol_cc = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf_cc = HartreeFockSolver()
        hf_cc.compute_energy(mol_cc, verbose=False)
        mp2_cc = MP2Solver()
        _, E_corr_cc = mp2_cc.compute_correlation(hf_cc, mol_cc, verbose=False)

        assert E_corr_cc < E_corr_sto, (
            f"cc-pVDZ corr ({E_corr_cc:.6f}) should be more negative "
            f"than STO-3G ({E_corr_sto:.6f})"
        )
