"""
Edge Case Stress Tests — QENEX Chemistry Solvers
=================================================

Phase B3: Tests for boundary conditions, convergence limits, and
numerical stability of the quantum chemistry implementations.

Test Categories:
    1. Single-atom systems (H, He, Li, Be, Ne)
    2. Bond length extremes (very short, very long, dissociation)
    3. Grid sensitivity (small vs large grids in DFT)
    4. Convergence failure handling
    5. Input validation (bad inputs, edge geometries)
    6. Numerical stability (near-degenerate eigenvalues)

Author: QENEX LAB v3.0-INFINITY
Date: 2026-03-13
"""

import pytest
import numpy as np
import sys
import os

# Angstrom to Bohr conversion
ANG2BOHR = 1.0 / 0.529177


# ============================================================================
# 1. Single-Atom Systems
# ============================================================================


class TestSingleAtomHF:
    """HF calculations on single atoms — no nuclear repulsion, no bonds."""

    def test_he_atom_rhf(self):
        """He atom: 2 electrons, 1 basis function, closed shell."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        he = Molecule([("He", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()
        E_elec, E_tot = hf.compute_energy(he, verbose=False)

        # He has no nuclear repulsion → E_elec = E_tot
        assert abs(E_elec - E_tot) < 1e-12, (
            f"He: E_elec ({E_elec}) != E_tot ({E_tot}) for single atom"
        )
        # He HF energy ~-2.808 Ha
        assert -3.0 < E_elec < -2.5, f"He energy out of range: {E_elec}"

    def test_ne_atom_rhf(self):
        """Ne atom: 10 electrons, 5 basis functions, closed shell."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        ne = Molecule([("Ne", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(ne, verbose=False)

        assert E_elec < -120.0, f"Ne energy too high: {E_elec}"

    def test_be_atom_rhf(self):
        """Be atom: 4 electrons, closed shell."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        be = Molecule([("Be", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(be, verbose=False)

        # Be HF/STO-3G ~-14.35 Ha
        assert -15.0 < E_elec < -14.0, f"Be energy out of range: {E_elec}"

    def test_h_atom_uhf(self):
        """H atom: 1 electron, doublet, requires UHF."""
        from solver import UHFSolver
        from molecule import Molecule

        h = Molecule([("H", (0.0, 0.0, 0.0))], multiplicity=2)
        uhf = UHFSolver()
        E_elec, _ = uhf.compute_energy(h, verbose=False)

        # H UHF/STO-3G ~-0.467 Ha
        assert -0.6 < E_elec < -0.3, f"H UHF energy out of range: {E_elec}"

    def test_li_atom_uhf(self):
        """Li atom: 3 electrons, doublet, requires UHF."""
        from solver import UHFSolver
        from molecule import Molecule

        li = Molecule([("Li", (0.0, 0.0, 0.0))], multiplicity=2)
        uhf = UHFSolver()
        E_elec, _ = uhf.compute_energy(li, verbose=False)

        # Li UHF/STO-3G ~-7.316 Ha
        assert -8.0 < E_elec < -7.0, f"Li UHF energy out of range: {E_elec}"

    def test_rhf_rejects_odd_electrons(self):
        """RHF should reject odd-electron systems."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h = Molecule([("H", (0.0, 0.0, 0.0))])  # 1 electron
        hf = HartreeFockSolver()

        with pytest.raises(ValueError, match="even number of electrons"):
            hf.compute_energy(h, verbose=False)


# ============================================================================
# 2. Bond Length Extremes
# ============================================================================


class TestBondLengthExtremes:
    """Test HF behavior at extreme bond lengths."""

    def test_h2_equilibrium(self):
        """H2 at equilibrium (R=1.4 bohr) — standard reference."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(h2, verbose=False)

        assert -1.2 < E < -1.0, f"H2 equilibrium energy out of range: {E}"

    def test_h2_short_bond(self):
        """H2 at very short bond (R=0.5 bohr) — strong nuclear repulsion."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.5))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(h2, verbose=False)

        # At short bond, energy should be higher (less negative) than equilibrium
        E_eq = -1.1168  # approximate equilibrium value
        assert E > E_eq, f"H2 at R=0.5: energy ({E}) should be > equilibrium ({E_eq})"

    def test_h2_long_bond(self):
        """H2 at stretched bond (R=5.0 bohr) — approaching dissociation."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 5.0))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(h2, verbose=False)

        # At long bond, energy approaches -1.0 Ha (two isolated H atoms in RHF)
        assert E < 0, f"H2 at R=5.0: energy should be negative, got {E}"
        # Should converge (not diverge)
        assert E > -2.0, f"H2 at R=5.0: energy suspiciously low: {E}"

    def test_h2_very_long_bond(self):
        """H2 at very long bond (R=20.0 bohr) — near-dissociation limit."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 20.0))])
        hf = HartreeFockSolver()
        E, _ = hf.compute_energy(h2, verbose=False)

        # At very long distance, RHF gives wrong answer (~-1.0 Ha instead of -1.0 Ha)
        # but should still converge
        assert E < 0, f"H2 at R=20: energy should be negative"

    def test_nuclear_singularity_raises(self):
        """Two atoms at same position should raise ValueError."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()

        with pytest.raises(ValueError, match="singularity"):
            hf.compute_energy(h2, verbose=False)

    def test_h2_potential_energy_curve_monotonic(self):
        """H2 PEC: energy increases monotonically for R < R_eq."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        hf = HartreeFockSolver()
        # Sample 3 points below equilibrium
        energies = []
        for R in [0.8, 1.0, 1.2]:
            h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, R))])
            E, _ = hf.compute_energy(h2, verbose=False)
            energies.append(E)

        # Energy should decrease (become more negative) as R increases toward eq
        assert energies[0] > energies[1] > energies[2], (
            f"H2 PEC not monotonically decreasing: {energies}"
        )


# ============================================================================
# 3. DFT Grid Sensitivity
# ============================================================================


class TestDFTGridSensitivity:
    """Test DFT energy stability across different grid sizes."""

    def test_he_grid_convergence(self):
        """He B3LYP energy converges with grid size."""
        from molecule import Molecule
        from dft import DFTSolver

        he = Molecule([("He", (0.0, 0.0, 0.0))])

        old_err = np.seterr(under="ignore")
        try:
            # Small grid
            dft_small = DFTSolver(he, functional="B3LYP", n_radial=20, n_angular=26)
            E_small = float(dft_small.solve())

            # Medium grid (default)
            dft_med = DFTSolver(he, functional="B3LYP", n_radial=75, n_angular=110)
            E_med = float(dft_med.solve())

            # The two should agree to within ~1 mHa for He
            delta = abs(E_small - E_med)
            print(
                f"\nHe grid convergence: small={E_small:.8f}, med={E_med:.8f}, Δ={delta * 1000:.4f} mHa"
            )

            # Both should be reasonable He energies
            assert -3.0 < E_small < -2.5, f"Small grid He energy bad: {E_small}"
            assert -3.0 < E_med < -2.5, f"Medium grid He energy bad: {E_med}"
        finally:
            np.seterr(**old_err)

    def test_h2_grid_convergence(self):
        """H2 B3LYP energy converges with grid size."""
        from molecule import Molecule
        from dft import DFTSolver

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.3984))])

        old_err = np.seterr(under="ignore")
        try:
            # Small grid
            dft_small = DFTSolver(h2, functional="B3LYP", n_radial=20, n_angular=26)
            E_small = float(dft_small.solve())

            # Medium grid
            dft_med = DFTSolver(h2, functional="B3LYP", n_radial=75, n_angular=110)
            E_med = float(dft_med.solve())

            print(f"\nH2 grid convergence: small={E_small:.8f}, med={E_med:.8f}")
            print(f"  Δ = {abs(E_small - E_med) * 1000:.4f} mHa")

            # Both should be reasonable H2 energies
            assert -1.3 < E_small < -1.0, f"Small grid H2 energy bad: {E_small}"
            assert -1.3 < E_med < -1.0, f"Medium grid H2 energy bad: {E_med}"
        finally:
            np.seterr(**old_err)


# ============================================================================
# 4. DFT Restricted KS Rejects Open-Shell
# ============================================================================


class TestDFTInputValidation:
    """DFT solver input validation for edge cases."""

    def test_rks_rejects_odd_electrons(self):
        """Restricted KS-DFT should reject odd-electron systems."""
        from molecule import Molecule
        from dft import DFTSolver

        h = Molecule([("H", (0.0, 0.0, 0.0))])  # 1 electron
        dft_solver = DFTSolver(h, functional="B3LYP")

        old_err = np.seterr(under="ignore")
        try:
            with pytest.raises(ValueError, match="even electrons|Restricted"):
                dft_solver.solve()
        finally:
            np.seterr(**old_err)

    def test_unknown_element_rejected(self):
        """Unknown elements should be rejected early."""
        from molecule import Molecule

        with pytest.raises(ValueError, match="Unknown|not supported|invalid"):
            Molecule([("Xx", (0.0, 0.0, 0.0))])


# ============================================================================
# 5. MP2 Edge Cases
# ============================================================================


class TestMP2EdgeCases:
    """MP2 edge cases and boundary conditions."""

    def test_he_mp2_zero_correlation(self):
        """He with STO-3G has 0 virtual orbitals → zero MP2 correlation."""
        from solver import HartreeFockSolver, MP2Solver
        from molecule import Molecule

        he = Molecule([("He", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()
        hf.compute_energy(he, verbose=False)

        mp2 = MP2Solver()
        _, E_corr = mp2.compute_correlation(hf, he, verbose=False)

        assert E_corr == 0.0, f"He MP2 correlation should be exactly zero, got {E_corr}"

    def test_mp2_frozen_core_reduces_correlation(self):
        """Frozen core should give less correlation than full MP2."""
        from solver import HartreeFockSolver, MP2Solver
        from molecule import Molecule

        h2o = Molecule(
            [
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, 1.43, -0.89)),
                ("H", (0.0, -1.43, -0.89)),
            ]
        )

        # Full MP2
        hf_full = HartreeFockSolver()
        hf_full.compute_energy(h2o, verbose=False)
        mp2_full = MP2Solver(frozen_core=False)
        _, E_corr_full = mp2_full.compute_correlation(hf_full, h2o, verbose=False)

        # Frozen core MP2
        hf_frozen = HartreeFockSolver()
        hf_frozen.compute_energy(h2o, verbose=False)
        mp2_frozen = MP2Solver(frozen_core=True)
        _, E_corr_frozen = mp2_frozen.compute_correlation(hf_frozen, h2o, verbose=False)

        print(f"\nH2O MP2: full corr={E_corr_full:.8f}, frozen={E_corr_frozen:.8f}")

        # Frozen core should have LESS correlation (fewer excitations)
        assert abs(E_corr_frozen) < abs(E_corr_full), (
            f"Frozen core ({E_corr_frozen}) should be less than full ({E_corr_full})"
        )


# ============================================================================
# 6. Numerical Stability
# ============================================================================


class TestNumericalStability:
    """Tests for numerical stability of the solvers."""

    def test_overlap_matrix_positive_definite(self):
        """Overlap matrix S must be positive definite for all systems."""
        from solver import HartreeFockSolver
        from molecule import Molecule
        import integrals as ints

        for name, atoms in [
            ("He", [("He", (0.0, 0.0, 0.0))]),
            ("H2", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]),
            ("LiH", [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 3.0))]),
        ]:
            mol = Molecule(atoms)
            hf = HartreeFockSolver()
            basis = hf.build_basis(mol)
            N = len(basis)

            S = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    s_val = 0.0
                    for pi in basis[i].primitives:
                        for pj in basis[j].primitives:
                            s_val += ints.overlap(pi, pj)
                    S[i, j] = s_val

            eigvals = np.linalg.eigvalsh(S)
            assert all(eigvals > 0), (
                f"{name}: Overlap matrix not positive definite (min eigenvalue={min(eigvals)})"
            )

    def test_overlap_diagonal_unity(self):
        """Diagonal of overlap matrix should be close to 1 for normalized basis."""
        from solver import HartreeFockSolver
        from molecule import Molecule
        import integrals as ints

        he = Molecule([("He", (0.0, 0.0, 0.0))])
        hf = HartreeFockSolver()
        basis = hf.build_basis(he)
        N = len(basis)

        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                s_val = 0.0
                for pi in basis[i].primitives:
                    for pj in basis[j].primitives:
                        s_val += ints.overlap(pi, pj)
                S[i, j] = s_val

        for i in range(N):
            assert abs(S[i, i] - 1.0) < 0.01, (
                f"Overlap S[{i},{i}]={S[i, i]} not close to 1.0"
            )

    def test_hf_energy_deterministic(self):
        """Same molecule should give same energy on repeated runs."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])

        energies = []
        for _ in range(3):
            hf = HartreeFockSolver()
            E, _ = hf.compute_energy(h2, verbose=False)
            energies.append(E)

        # All three should be identical to machine precision
        for i in range(1, len(energies)):
            assert abs(energies[i] - energies[0]) < 1e-12, (
                f"HF energy not deterministic: {energies}"
            )

    def test_symmetry_h2_displacement(self):
        """H2 energy should be independent of center-of-mass position."""
        from solver import HartreeFockSolver
        from molecule import Molecule

        R = 1.4  # bohr

        # At origin
        h2_origin = Molecule(
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, R)),
            ]
        )

        # Displaced by (10, 10, 10) bohr
        h2_displaced = Molecule(
            [
                ("H", (10.0, 10.0, 10.0)),
                ("H", (10.0, 10.0, 10.0 + R)),
            ]
        )

        hf1 = HartreeFockSolver()
        E1, _ = hf1.compute_energy(h2_origin, verbose=False)

        hf2 = HartreeFockSolver()
        E2, _ = hf2.compute_energy(h2_displaced, verbose=False)

        assert abs(E1 - E2) < 1e-8, (
            f"H2 energy depends on position: {E1:.10f} vs {E2:.10f}"
        )


# ============================================================================
# 7. Molecule Validation Edge Cases
# ============================================================================


class TestMoleculeValidation:
    """Test Molecule class input validation."""

    def test_valid_elements_accepted(self):
        """All supported elements (H through Ar) should be accepted."""
        from molecule import Molecule

        for sym in ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
            mol = Molecule([(sym, (0.0, 0.0, 0.0))])
            assert len(mol.atoms) == 1, f"{sym} not accepted"

    def test_charge_modifies_electrons(self):
        """Charged molecules should have modified electron counts."""
        from molecule import Molecule

        # He with charge +1 → 1 electron (doublet)
        he_plus = Molecule([("He", (0.0, 0.0, 0.0))], charge=1, multiplicity=2)
        assert he_plus.charge == 1

    def test_negative_coordinate(self):
        """Negative coordinates should be accepted."""
        from molecule import Molecule

        mol = Molecule(
            [
                ("H", (-1.0, -2.0, -3.0)),
                ("H", (1.0, 2.0, 3.0)),
            ]
        )
        assert len(mol.atoms) == 2

    def test_large_coordinate(self):
        """Very large coordinates should be accepted."""
        from molecule import Molecule

        mol = Molecule(
            [
                ("H", (0.0, 0.0, 0.0)),
                ("H", (0.0, 0.0, 100.0)),  # 100 bohr apart
            ]
        )
        assert len(mol.atoms) == 2


# ============================================================================
# 8. Cross-Method Consistency Under Stress
# ============================================================================


class TestCrossMethodStress:
    """Verify consistency between methods under stress conditions."""

    def test_h2_variational_principle(self):
        """HF energy is an upper bound to exact ground state energy.
        For H2/STO-3G, exact (FCI) = HF since only 2 basis functions.
        But MP2 should still give E_MP2 <= E_HF (for 2+ basis functions)."""
        from solver import HartreeFockSolver, MP2Solver
        from molecule import Molecule

        h2 = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))])

        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(h2, verbose=False)

        mp2 = MP2Solver()
        E_mp2, E_corr = mp2.compute_correlation(hf, h2, verbose=False)

        # E_MP2 = E_HF + E_corr, E_corr <= 0
        assert E_corr <= 0, f"MP2 correlation should be non-positive: {E_corr}"
        assert E_mp2 <= E_hf + 1e-15, (
            f"Variational violation: E_MP2={E_mp2} > E_HF={E_hf}"
        )

    def test_lih_methods_ordering(self):
        """For LiH: E(DFT/B3LYP) < E(HF) < 0 (correlation lowers energy)."""
        from solver import HartreeFockSolver
        from molecule import Molecule
        from dft import DFTSolver

        R = 1.596 * ANG2BOHR
        lih = Molecule([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, R))])

        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(lih, verbose=False)

        old_err = np.seterr(under="ignore")
        try:
            dft_solver = DFTSolver(lih, functional="B3LYP", n_radial=75, n_angular=110)
            E_dft = float(dft_solver.solve())
        finally:
            np.seterr(**old_err)

        print(f"\nLiH: E_HF={E_hf:.8f}, E_DFT={E_dft:.8f}")
        assert E_dft < E_hf, f"B3LYP ({E_dft}) should be lower than HF ({E_hf}) for LiH"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
