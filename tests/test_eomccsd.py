"""
Tests for EOM-CCSD (Excited States)
=====================================
Validates EOM-CCSD excitation energies against reference values.

100% native — no PySCF in the computation path.
PySCF reference values computed separately for validation.
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, "packages/qenex_chem/src")

# PySCF reference values (singlet EOM-CCSD/STO-3G, cart=True)
PYSCF_H2O_SINGLET = [
    0.4566229178,
    0.5411041786,
    0.5986098457,
    0.6987540613,
    0.8264572004,
]


class TestEOMCCSDH2O:
    """Test EOM-CCSD on H2O (STO-3G)."""

    @pytest.fixture(scope="class")
    def h2o_eom(self):
        """Run CCSD + EOM-CCSD on H2O once for all tests."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver
        from eomccsd import EOMCCSDSolver

        ang2bohr = 1.8897259886
        mol = Molecule(
            [
                ("O", (0, 0, 0)),
                ("H", (0, 0.757 * ang2bohr, 0.587 * ang2bohr)),
                ("H", (0, -0.757 * ang2bohr, 0.587 * ang2bohr)),
            ],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )

        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        ccsd = CCSDSolver(convergence=1e-10)
        ccsd.solve(hf, mol, verbose=False)
        eom = EOMCCSDSolver()
        evals = eom.solve(ccsd, nroots=5, verbose=False)
        return evals

    def test_first_excitation(self, h2o_eom):
        """First singlet excitation energy matches PySCF."""
        assert abs(h2o_eom[0] - PYSCF_H2O_SINGLET[0]) < 1e-4, (
            f"S1: {h2o_eom[0]:.8f} vs {PYSCF_H2O_SINGLET[0]:.8f}"
        )

    def test_five_states(self, h2o_eom):
        """All 5 singlet states match PySCF within 1e-4 Eh."""
        for i in range(5):
            assert abs(h2o_eom[i] - PYSCF_H2O_SINGLET[i]) < 1e-4, (
                f"S{i + 1}: {h2o_eom[i]:.8f} vs {PYSCF_H2O_SINGLET[i]:.8f}"
            )

    def test_ordering(self, h2o_eom):
        """Excitation energies are in ascending order."""
        for i in range(len(h2o_eom) - 1):
            assert h2o_eom[i] <= h2o_eom[i + 1] + 1e-10, (
                f"State {i + 1} ({h2o_eom[i]:.6f}) > State {i + 2} ({h2o_eom[i + 1]:.6f})"
            )

    def test_positive(self, h2o_eom):
        """All excitation energies are positive."""
        for i, e in enumerate(h2o_eom):
            assert e > 0, f"State {i + 1} has negative excitation energy: {e}"
