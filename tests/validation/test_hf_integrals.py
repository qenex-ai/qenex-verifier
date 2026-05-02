"""
Hartree-Fock & Integrals Validation Tests — Proper Pytest Conversion
=====================================================================

Converts the 12 script-style validation files into proper pytest test functions
that are automatically discovered and run by the CI test suite.

Converted from:
  test_he.py, test_o.py, test_ne.py, test_h_atom.py, test_eri.py,
  test_h2_dissoc.py, test_simple_kinetic.py, test_p_kinetic.py,
  test_p_nuc.py, test_nuc_att.py, test_boys_recurrence.py,
  test_integrals.py (partial — Boys function and kinetic tests)

All reference values verified by running the original scripts.
Coordinates in BOHR throughout.

Author: QENEX LAB v3.0-INFINITY
"""

import numpy as np
import pytest

from solver import HartreeFockSolver
from molecule import Molecule
import integrals as ints


# ============================================================================
# Section 1: Primitive Integral Tests (overlap, kinetic, nuclear attraction)
# ============================================================================


class TestPrimitiveOverlap:
    """Validate overlap integrals for primitive Gaussian basis functions."""

    def test_ss_overlap_normalized(self):
        """Normalized (s|s) overlap should be 1.0."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        S = ints.overlap(bf, bf)
        assert abs(S - 1.0) < 1e-6, f"(s|s) overlap = {S:.8f}, expected 1.0"

    def test_pp_overlap_normalized(self):
        """Normalized (p|p) overlap should be 1.0."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
        S = ints.overlap(bf, bf)
        assert abs(S - 1.0) < 1e-6, f"(px|px) overlap = {S:.8f}, expected 1.0"

    def test_py_overlap_normalized(self):
        """Normalized (py|py) overlap should be 1.0."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 1, 0))
        S = ints.overlap(bf, bf)
        assert abs(S - 1.0) < 1e-6, f"(py|py) overlap = {S:.8f}, expected 1.0"


class TestPrimitiveKinetic:
    """Validate kinetic energy integrals.

    For a primitive Gaussian x^l * exp(-α r²), the kinetic energy integral
    follows (2l+3)/2 * α for normalized functions with same center & exponent.
    """

    def test_ss_kinetic(self):
        """s-orbital kinetic energy: T_ss(α=1) = 1.5 (= 3/2 * α)."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        T = ints.kinetic(bf, bf)
        assert abs(T - 1.5) < 1e-4, f"T_ss = {T:.6f}, expected 1.5"

    def test_pp_kinetic(self):
        """p-orbital kinetic energy: T_pp(α=1) = 2.5 (= 5/2 * α)."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
        T = ints.kinetic(bf, bf)
        assert abs(T - 2.5) < 1e-4, f"T_pp(px) = {T:.6f}, expected 2.5"

    def test_py_kinetic(self):
        """py-orbital kinetic energy: T_pp(α=1) = 2.5."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 1, 0))
        T = ints.kinetic(bf, bf)
        assert abs(T - 2.5) < 1e-4, f"T_pp(py) = {T:.6f}, expected 2.5"

    def test_dd_kinetic(self):
        """d(xy)-orbital kinetic energy: T_dd(α=1) = 3.5 (= 7/2 * α)."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 1, 0))
        T = ints.kinetic(bf, bf)
        assert abs(T - 3.5) < 1e-4, f"T_dd(xy) = {T:.6f}, expected 3.5"


class TestNuclearAttraction:
    """Validate nuclear attraction integrals <φ|Z/r|φ>."""

    def test_ss_nuclear_attraction(self):
        """(s|1/r|s) for Z=1, center at origin: 2*sqrt(2/π) ≈ 1.595769."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        # nuclear_attraction returns negative (attractive potential)
        V = ints.nuclear_attraction(bf, bf, np.array([0.0, 0.0, 0.0]), 1.0)
        assert abs(-V - 1.595769) < 1e-3, f"(s|1/r|s) = {-V:.6f}, expected 1.595769"

    def test_pp_nuclear_attraction(self):
        """(px|1/r|px) for Z=1, center at origin: ≈ 1.0638."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
        V = ints.nuclear_attraction(bf, bf, np.array([0.0, 0.0, 0.0]), 1.0)
        assert abs(-V - 1.0638) < 0.01, f"(p|1/r|p) = {-V:.6f}, expected ~1.064"

    def test_ps_nuclear_attraction_zero_by_symmetry(self):
        """(px|1/r|s) at origin should be zero by symmetry."""
        bf_p = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
        bf_s = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        V = ints.nuclear_attraction(bf_p, bf_s, np.array([0.0, 0.0, 0.0]), 1.0)
        assert abs(V) < 1e-10, f"(p|1/r|s) at origin = {V:.2e}, expected 0.0"


class TestERI:
    """Validate electron repulsion integrals (ERI)."""

    def test_ssss_raw_eri(self):
        """Raw (ss|ss) ERI for α=1: 2π^2.5 / (2·2·√4) ≈ 4.37."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        eri_val = ints.eri(bf, bf, bf, bf)
        raw = eri_val / (bf.N**4)
        assert abs(raw - 4.37) < 0.01, f"Raw (ss|ss) = {raw:.4f}, expected ~4.37"

    def test_ssss_normalized_eri(self):
        """Normalized (ss|ss) ERI for α=1: 2/√π ≈ 1.128379."""
        bf = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 0))
        eri_val = ints.eri(bf, bf, bf, bf)
        assert abs(eri_val - 1.128379) < 1e-3, (
            f"Normalized (ss|ss) = {eri_val:.6f}, expected 1.128379"
        )


# ============================================================================
# Section 2: Boys Function Tests
# ============================================================================


class TestBoysFunction:
    """Validate Boys function F_n(t) using upward recurrence vs scipy reference.

    The Boys function: F_n(t) = ∫₀¹ u^(2n) exp(-t·u²) du
    Used in molecular integral evaluation.
    """

    @staticmethod
    def _reference_boys(n, t):
        """Reference Boys function via scipy incomplete gamma."""
        from scipy.special import gammainc, gamma

        if t < 1e-12:
            return 1.0 / (2 * n + 1)
        return 0.5 * (t ** (-n - 0.5)) * (gamma(n + 0.5) * gammainc(n + 0.5, t))

    @staticmethod
    def _upward_recurrence(n_target, t):
        """Compute F_n(t) via upward recurrence from F_0."""
        import math

        if t < 1e-8:
            return 1.0 / (2 * n_target + 1)
        f_curr = 0.5 * np.sqrt(np.pi / t) * math.erf(np.sqrt(t))
        if n_target == 0:
            return f_curr
        exp_t = np.exp(-t)
        for n in range(n_target):
            f_next = ((2 * n + 1) * f_curr - exp_t) / (2 * t)
            f_curr = f_next
        return f_curr

    @pytest.mark.parametrize(
        "t,n_max",
        [(0.1, 4), (1.0, 4), (5.0, 10), (10.0, 10), (20.0, 10), (50.0, 10)],
    )
    def test_upward_recurrence_accuracy(self, t, n_max):
        """Upward recurrence matches scipy reference for moderate n."""
        for n in range(1, n_max + 1):
            ref = self._reference_boys(n, t)
            up = self._upward_recurrence(n, t)
            # Known instability: upward recurrence loses precision for small t, large n
            if t >= 1.0 or n <= 4:
                rel_err = abs(ref - up) / max(abs(ref), 1e-30)
                assert rel_err < 1e-6, (
                    f"Boys F_{n}({t}): ref={ref:.6e}, up={up:.6e}, "
                    f"rel_err={rel_err:.2e}"
                )

    def test_boys_small_t_limit(self):
        """F_n(0) = 1/(2n+1) exactly."""
        for n in range(5):
            ref = 1.0 / (2 * n + 1)
            up = self._upward_recurrence(n, 0.0)
            assert abs(up - ref) < 1e-12, f"F_{n}(0) = {up}, expected {ref}"


# ============================================================================
# Section 3: Hydrogen Atom STO-3G Integrals
# ============================================================================


class TestHydrogenAtomIntegrals:
    """Validate STO-3G integrals for the hydrogen atom.

    STO-3G H atom 1s: composed of 3 primitive Gaussians.
    Reference values from analytical evaluation.
    """

    def _get_h_basis(self):
        """Build H atom STO-3G basis function."""
        mol = Molecule([("H", (0.0, 0.0, 0.0))], multiplicity=2)
        solver = HartreeFockSolver()
        basis = solver.build_basis(mol)
        return basis[0]

    def test_h_overlap_unity(self):
        """H 1s STO-3G self-overlap should be 1.0."""
        bf = self._get_h_basis()
        S = 0.0
        for p1 in bf.primitives:
            for p2 in bf.primitives:
                S += ints.overlap(p1, p2)
        assert abs(S - 1.0) < 1e-4, f"H 1s overlap = {S:.6f}, expected 1.0"

    def test_h_kinetic_energy(self):
        """H 1s STO-3G kinetic energy: ~0.760 Hartree."""
        bf = self._get_h_basis()
        T = 0.0
        for p1 in bf.primitives:
            for p2 in bf.primitives:
                T += ints.kinetic(p1, p2)
        assert abs(T - 0.760) < 0.01, f"H 1s kinetic = {T:.6f}, expected ~0.760"

    def test_h_nuclear_potential(self):
        """H 1s STO-3G nuclear attraction: ~-1.227 Hartree."""
        bf = self._get_h_basis()
        V = 0.0
        for p1 in bf.primitives:
            for p2 in bf.primitives:
                V += ints.nuclear_attraction(p1, p2, np.array([0.0, 0.0, 0.0]), 1.0)
        assert abs(V - (-1.227)) < 0.01, f"H 1s potential = {V:.6f}, expected ~-1.227"

    def test_h_hcore_energy(self):
        """H 1s STO-3G H_core (T+V): ~-0.467 Hartree."""
        bf = self._get_h_basis()
        T = 0.0
        V = 0.0
        for p1 in bf.primitives:
            for p2 in bf.primitives:
                T += ints.kinetic(p1, p2)
                V += ints.nuclear_attraction(p1, p2, np.array([0.0, 0.0, 0.0]), 1.0)
        E_core = T + V
        assert abs(E_core - (-0.467)) < 0.01, (
            f"H 1s H_core = {E_core:.6f}, expected ~-0.467"
        )


# ============================================================================
# Section 4: Hartree-Fock Total Energies (atoms)
# ============================================================================


class TestHFAtomEnergies:
    """Validate RHF/STO-3G total energies for closed-shell atoms.

    Reference values from Szabo & Ostlund, Table 3.1 and verified by
    running the QENEX HartreeFockSolver.
    """

    def test_he_rhf(self):
        """He RHF/STO-3G: -2.8078 Hartree (Szabo ref: -2.808)."""
        mol = Molecule([("He", (0.0, 0.0, 0.0))])
        solver = HartreeFockSolver()
        E_elec, E_tot = solver.compute_energy(mol)
        # STO-3G gives -2.8078 vs exact HF limit -2.8617
        assert abs(E_tot - (-2.8078)) < 0.01, (
            f"He RHF E = {E_tot:.6f}, expected ~-2.808"
        )

    def test_ne_rhf(self):
        """Ne RHF/STO-3G: ~-126.60 Hartree (exact HF: -128.547)."""
        mol = Molecule([("Ne", (0.0, 0.0, 0.0))])
        solver = HartreeFockSolver()
        E_elec, E_tot = solver.compute_energy(mol)
        assert E_tot < -125.0, f"Ne RHF E = {E_tot:.4f}, expected < -125"
        assert E_tot > -130.0, f"Ne RHF E = {E_tot:.4f}, expected > -130"

    def test_o_uhf(self):
        """O atom UHF/STO-3G: ~-73.66 Hartree (triplet, 8 electrons)."""
        mol = Molecule([("O", (0.0, 0.0, 0.0))], multiplicity=3)
        solver = HartreeFockSolver()
        E_elec, E_tot = solver.compute_energy(mol)
        assert E_tot < -72.0, f"O UHF E = {E_tot:.4f}, expected < -72"
        assert E_tot > -75.0, f"O UHF E = {E_tot:.4f}, expected > -75"


# ============================================================================
# Section 5: H₂ Dissociation Limit
# ============================================================================


class TestH2Dissociation:
    """Validate H₂ at large separation approaches 2 × E(H).

    At R=20 Bohr, H₂ should dissociate to two independent H atoms.
    RHF gives the wrong limit (due to restricted wave function) but
    the energy should still be physical.
    """

    def test_h2_dissociation_energy_physical(self):
        """H₂ at R=20 Bohr: energy should be between -0.5 and -1.0 Hartree.

        RHF gives ~-0.621 Ha (wrong dissociation limit: H⁺ + H⁻ mixture).
        UHF would give ~-0.933 Ha (correct: 2 × -0.467).
        """
        mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (20.0, 0.0, 0.0))])
        solver = HartreeFockSolver()
        E_elec, E_tot = solver.compute_energy(mol)
        # RHF dissociation limit is higher than 2*E(H)
        assert E_tot < -0.5, f"H₂(R=20) RHF E = {E_tot:.4f}, expected < -0.5"
        assert E_tot > -1.0, f"H₂(R=20) RHF E = {E_tot:.4f}, expected > -1.0"

    def test_h2_nuclear_repulsion_small_at_large_r(self):
        """Nuclear repulsion at R=20 Bohr should be ~0.05 Hartree."""
        mol = Molecule([("H", (0.0, 0.0, 0.0)), ("H", (20.0, 0.0, 0.0))])
        solver = HartreeFockSolver()
        E_nuc = solver.compute_nuclear_repulsion(mol)
        assert abs(E_nuc - 0.05) < 0.01, f"H₂(R=20) E_nuc = {E_nuc:.6f}, expected ~0.05"


# ============================================================================
# Section 6: Ruthless Chaos Suite (error handling & edge cases)
# ============================================================================


class TestRuthlessSuite:
    """Cross-domain edge case and error handling tests.

    Validates that the scientific modules correctly reject invalid inputs:
    negative simulation steps, fake elements, spin parity mismatches, etc.

    Converted from the script-style test_ruthless_suite.py which tested
    10 cross-domain edge cases.
    """

    def test_invalid_element_rejected(self):
        """Molecule with fake element 'X' should raise an error."""
        with pytest.raises((ValueError, KeyError)):
            mol = Molecule(atoms=[("X", (0, 0, 0))])
            solver = HartreeFockSolver()
            solver.compute_energy(mol)

    def test_spin_parity_mismatch(self):
        """H₂ (2 electrons) with multiplicity=2 (doublet) should fail.

        Even electrons require odd multiplicity (singlet, triplet, ...),
        odd electrons require even multiplicity (doublet, quartet, ...).
        """
        with pytest.raises((ValueError, Exception)):
            mol = Molecule(atoms=[("H", (0, 0, 0)), ("H", (0, 0, 1))], multiplicity=2)
            solver = HartreeFockSolver()
            solver.compute_energy(mol)

    def test_negative_simulation_steps(self):
        """LatticeSimulator should reject negative step count."""
        try:
            from lattice import LatticeSimulator
        except ImportError:
            pytest.skip("LatticeSimulator not available")
        with pytest.raises(ValueError):
            sim = LatticeSimulator(dimensions=2, size=10)
            sim.run_simulation(steps=-100, temperature=1.0)

    def test_trivial_lattice_size(self):
        """LatticeSimulator should reject size=1 (no connectivity)."""
        try:
            from lattice import LatticeSimulator
        except ImportError:
            pytest.skip("LatticeSimulator not available")
        with pytest.raises(ValueError):
            sim = LatticeSimulator(dimensions=2, size=1)

    def test_stop_codon_injection(self):
        """ProteinFolder should reject sequences with internal stop codons."""
        try:
            from folding import ProteinFolder
        except ImportError:
            pytest.skip("ProteinFolder not available")
        with pytest.raises((ValueError, Exception)):
            folder = ProteinFolder()
            folder.fold_sequence("MET*ALA")

    def test_qlang_rejects_reserved_constant_rebinding(self):
        """Q-Lang v0.4 rejects rebinding of any bound name with
        ``RebindingError`` (all bindings are immutable).  A fundamental
        constant is just another binding once registered, so the
        constant-protection story is subsumed by v0.4's
        immutable-bindings invariant."""
        import qlang_v04

        interp = qlang_v04.QLangInterpreter()
        interp.run("let c = 2.99792458e8")
        with pytest.raises(qlang_v04.RebindingError):
            interp.run("let c = 10")

    def test_qlang_syntax_reject_blocks_code_injection(self):
        """Q-Lang v0.4's grammar (see SPEC §4) accepts only:
        let / print / experiment blocks + simple expressions.
        Python builtin access like ``__import__`` is rejected at
        parse time with ``QLangSyntaxError``.  No interpreter runtime
        is reached."""
        import qlang_v04

        interp = qlang_v04.QLangInterpreter()
        with pytest.raises(qlang_v04.QLangSyntaxError):
            interp.run('let hack = __import__("os").name')
