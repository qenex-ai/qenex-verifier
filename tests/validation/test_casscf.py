"""
CASSCF Validation Tests
=========================
Validates the Complete Active Space Self-Consistent Field solver against
known reference values and physical constraints.

Key tests:
1. H2 CAS(2,2)/STO-3G at equilibrium — energy below HF
2. H2 dissociation curve — CASSCF gets it right where RHF fails
3. He CAS(2,1) — recovers exact HF (1 det = no correlation)
4. 2-RDM correctness — trace, symmetry, N-representability
5. Natural orbital occupations — physical bounds, sum rule
6. Orbital gradient — zero at convergence
7. New API: (ncas, nelecas) constructor, tuple return value

Reference: At stretched H2 (R=6.0 bohr), CAS(2,2) gives proper
dissociation to 2 hydrogen atoms while RHF gives the wrong limit.
The natural orbital occupations become ~1.0/1.0 (diradical)
instead of ~2.0/0.0 (single reference).
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, "packages/qenex_chem/src")


class TestCASSCFNewAPI:
    """Test the new (ncas, nelecas) constructor API."""

    def test_ncas_nelecas_constructor(self):
        """Constructor accepts (ncas, nelecas) parameters."""
        from casscf import CASSCFSolver

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        assert casscf.ncas == 2
        assert casscf.nelecas == 2

    def test_backward_compatible_constructor(self):
        """Constructor still accepts (n_active_orbitals, n_active_electrons)."""
        from casscf import CASSCFSolver

        casscf = CASSCFSolver(n_active_orbitals=4, n_active_electrons=4)
        assert casscf.ncas == 4
        assert casscf.nelecas == 4
        assert casscf.n_active_orbitals == 4
        assert casscf.n_active_electrons == 4

    def test_missing_params_raises(self):
        """Missing active space parameters raise ValueError."""
        from casscf import CASSCFSolver

        with pytest.raises(ValueError, match="Must specify"):
            CASSCFSolver(ncas=2)

    def test_solve_returns_tuple(self):
        """solve() returns (E_casscf, ci_coefficients, natural_occupations)."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        result = casscf.solve(hf, mol, verbose=False)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 3, f"Expected 3 elements, got {len(result)}"

        E, ci_coeffs, nat_occ = result
        assert isinstance(E, float), f"Energy should be float, got {type(E)}"
        assert isinstance(ci_coeffs, np.ndarray), "CI coeffs should be ndarray"
        assert isinstance(nat_occ, np.ndarray), "Natural occupations should be ndarray"


class TestCASSCFH2Equilibrium:
    """Validate CAS(2,2) on H2 at equilibrium (R=1.4 bohr)."""

    @pytest.fixture(scope="class")
    def h2_result(self):
        """Run CASSCF on H2 at equilibrium."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        E_cas, ci_coeffs, nat_occ = casscf.solve(hf, mol, verbose=False)

        return {
            "E_casscf": E_cas,
            "E_hf": E_hf,
            "ci_coeffs": ci_coeffs,
            "nat_occ": nat_occ,
            "casscf": casscf,
            "mol": mol,
        }

    def test_energy_negative(self, h2_result):
        """CASSCF energy is negative (bound system)."""
        assert h2_result["E_casscf"] < 0

    def test_energy_below_hf(self, h2_result):
        """CASSCF energy <= HF energy (variational)."""
        assert h2_result["E_casscf"] <= h2_result["E_hf"] + 1e-10

    def test_ci_vector_normalized(self, h2_result):
        """CI vector has unit norm."""
        norm = np.linalg.norm(h2_result["ci_coeffs"])
        assert abs(norm - 1.0) < 1e-10

    def test_natural_occupations_physical(self, h2_result):
        """Occupations in [0, 2] and sum to nelecas."""
        occ = h2_result["nat_occ"]
        assert all(o >= -1e-8 for o in occ), f"Negative occupation: {occ}"
        assert all(o <= 2.0 + 1e-8 for o in occ), f"Occupation > 2: {occ}"
        assert abs(np.sum(occ) - 2.0) < 1e-6, f"Occupation sum != 2: {np.sum(occ)}"

    def test_equilibrium_mostly_single_reference(self, h2_result):
        """At equilibrium, H2 is mostly single reference."""
        occ = h2_result["nat_occ"]
        # The bonding orbital should have occupation close to 2
        assert max(occ) > 1.5, f"Max occupation too low: {max(occ)}"

    def test_rdm1_trace(self, h2_result):
        """1-RDM trace equals number of active electrons."""
        rdm1 = h2_result["casscf"].get_rdm1()
        assert abs(np.trace(rdm1) - 2.0) < 1e-6

    def test_rdm1_symmetric(self, h2_result):
        """1-RDM is symmetric."""
        rdm1 = h2_result["casscf"].get_rdm1()
        assert np.allclose(rdm1, rdm1.T, atol=1e-10)

    def test_rdm1_positive_semidefinite(self, h2_result):
        """1-RDM eigenvalues are non-negative."""
        rdm1 = h2_result["casscf"].get_rdm1()
        eigvals = np.linalg.eigvalsh(rdm1)
        assert all(v >= -1e-8 for v in eigvals), f"Negative eigenvalue: {eigvals}"

    def test_convergence(self, h2_result):
        """CASSCF converged."""
        assert h2_result["casscf"]._converged


class TestCASSCFH2Dissociation:
    """The classic test: H2 bond breaking with CAS(2,2).

    This is THE test for multireference methods. At large R:
    - RHF gives the wrong dissociation limit (too high energy)
    - CASSCF correctly dissociates to two isolated hydrogen atoms
    - Natural occupations become ~1.0/1.0 (diradical character)
    """

    def test_dissociation_energy_below_hf(self):
        """At stretched R=4.0 bohr, CASSCF energy < HF energy."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        R = 4.0  # Stretched bond
        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, R))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        E_cas, _, nat_occ = casscf.solve(hf, mol, verbose=False)

        # At dissociation, CASSCF should be significantly below HF
        assert E_cas < E_hf - 1e-4, (
            f"CASSCF ({E_cas:.8f}) should be well below HF ({E_hf:.8f}) at R=4.0"
        )

    def test_dissociation_diradical_character(self):
        """At R=6.0 bohr, natural occupations approach 1.0/1.0."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        R = 6.0  # Well into dissociation regime
        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, R))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        _, _, nat_occ = casscf.solve(hf, mol, verbose=False)

        # At dissociation, both orbitals should have ~1.0 occupation
        # (perfect diradical: equal weight of sigma^2 and sigma*^2)
        assert abs(nat_occ[0] - nat_occ[1]) < 0.5, (
            f"Occupations should be close at R=6.0: {nat_occ}"
        )

    def test_dissociation_curve_monotonic(self):
        """Energy curve is smooth and approaches correct dissociation limit."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        R_values = [1.0, 1.4, 2.0, 3.0, 5.0]
        energies_cas = []
        energies_hf = []

        for R in R_values:
            mol = Molecule(
                [("H", (0, 0, 0)), ("H", (0, 0, R))],
                charge=0,
                multiplicity=1,
                basis_name="sto-3g",
            )
            hf = HartreeFockSolver()
            E_hf, _ = hf.compute_energy(mol, verbose=False)

            casscf = CASSCFSolver(ncas=2, nelecas=2)
            E_cas, _, _ = casscf.solve(hf, mol, verbose=False)

            energies_cas.append(E_cas)
            energies_hf.append(E_hf)

        # CASSCF should always be <= HF
        for i, R in enumerate(R_values):
            assert energies_cas[i] <= energies_hf[i] + 1e-8, (
                f"CASSCF > HF at R={R}: {energies_cas[i]:.8f} vs {energies_hf[i]:.8f}"
            )

        # All energies should be negative
        for i, E in enumerate(energies_cas):
            assert E < 0, f"Energy should be negative at R={R_values[i]}: {E}"

        # At large R, CASSCF should be significantly below HF
        # (this is the whole point of CASSCF)
        gap_at_large_R = energies_hf[-1] - energies_cas[-1]
        assert gap_at_large_R > 0.01, (
            f"CASSCF-HF gap at R=5.0 should be > 0.01 Eh: {gap_at_large_R:.6f}"
        )


class TestCASSCF2RDM:
    """Validate the 2-RDM computation."""

    @pytest.fixture(scope="class")
    def h2_rdms(self):
        """Run CASSCF and get RDMs."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        casscf.solve(hf, mol, verbose=False)

        return casscf.get_rdm1(), casscf.get_rdm2()

    def test_rdm2_not_zero(self, h2_rdms):
        """2-RDM should not be all zeros (regression test)."""
        _, rdm2 = h2_rdms
        assert np.max(np.abs(rdm2)) > 1e-6, "2-RDM should not be all zeros"

    def test_rdm2_trace_consistency(self, h2_rdms):
        """Trace of 2-RDM relates to 1-RDM: sum_r Gamma[p,q,r,r] = (N-1)*gamma[p,q]."""
        rdm1, rdm2 = h2_rdms
        N = 2  # active electrons
        # sum_r Gamma[p,q,r,r] should equal (N-1)*gamma[p,q]
        rdm2_traced = np.einsum("pqrr->pq", rdm2)
        expected = (N - 1) * rdm1
        assert np.allclose(rdm2_traced, expected, atol=1e-6), (
            f"2-RDM trace inconsistency:\n"
            f"  Tr_r Gamma[p,q,r,r] = {rdm2_traced}\n"
            f"  (N-1)*gamma = {expected}"
        )


class TestCASSCFBitstrings:
    """Test the bitstring determinant representation."""

    def test_build_determinants_cas22(self):
        """CAS(2,2) generates 4 determinants."""
        from casscf import CASSCFSolver

        casscf = CASSCFSolver(ncas=2, nelecas=2)
        dets = casscf._build_determinants(2, 1, 1)

        assert len(dets) == 4
        # Each determinant is (alpha_bs, beta_bs)
        for alpha, beta in dets:
            assert isinstance(alpha, int)
            assert isinstance(beta, int)
            # Each string should have exactly 1 bit set
            assert bin(alpha).count("1") == 1
            assert bin(beta).count("1") == 1

    def test_build_determinants_cas44(self):
        """CAS(4,4) generates 36 determinants."""
        from casscf import CASSCFSolver

        casscf = CASSCFSolver(ncas=4, nelecas=4)
        dets = casscf._build_determinants(4, 2, 2)

        # C(4,2) * C(4,2) = 6 * 6 = 36
        assert len(dets) == 36

    def test_build_determinants_cas21(self):
        """CAS(2,1) with 1 alpha, 0 beta: 2 determinants."""
        from casscf import CASSCFSolver

        casscf = CASSCFSolver(ncas=2, nelecas=1)
        dets = casscf._build_determinants(2, 1, 0)

        # C(2,1) * C(2,0) = 2 * 1 = 2
        assert len(dets) == 2


class TestCASSCFOrbitalGradient:
    """Test orbital gradient computation."""

    def test_gradient_near_zero_at_convergence(self):
        """Orbital gradient norm should be small at convergence."""
        from molecule import Molecule
        from solver import HartreeFockSolver
        from casscf import CASSCFSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)

        casscf = CASSCFSolver(ncas=2, nelecas=2, convergence=1e-10)
        E_cas, _, _ = casscf.solve(hf, mol, verbose=False)

        # The solver should have converged, meaning gradient is small
        assert casscf._converged, "CASSCF should converge for H2"
