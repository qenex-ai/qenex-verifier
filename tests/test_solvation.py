"""
Tests for PCM Solvation Module (CPCM)
======================================
Validates the Conductor-like Polarizable Continuum Model:
cavity construction, surface charges, and solvation energies.

All geometries in Bohr. Uses STO-3G basis for speed.
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, "packages/qenex_chem/src")

HARTREE_TO_KCAL = 627.5094740631
ANG2BOHR = 1.8897259886


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2o_mol():
    """Water molecule in Bohr coordinates, STO-3G."""
    from molecule import Molecule

    return Molecule(
        [
            ("O", (0, 0, 0)),
            ("H", (0, 0.757 * ANG2BOHR, 0.587 * ANG2BOHR)),
            ("H", (0, -0.757 * ANG2BOHR, 0.587 * ANG2BOHR)),
        ],
        charge=0,
        multiplicity=1,
        basis_name="sto-3g",
    )


@pytest.fixture(scope="module")
def he_mol():
    """Helium atom, STO-3G."""
    from molecule import Molecule

    return Molecule(
        [("He", (0, 0, 0))],
        charge=0,
        multiplicity=1,
        basis_name="sto-3g",
    )


@pytest.fixture(scope="module")
def h2_mol():
    """H2 at 1.4 bohr, STO-3G."""
    from molecule import Molecule

    return Molecule(
        [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
        charge=0,
        multiplicity=1,
        basis_name="sto-3g",
    )


@pytest.fixture(scope="module")
def hf_solver():
    """Shared HartreeFockSolver."""
    from solver import HartreeFockSolver

    return HartreeFockSolver()


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class TestPCMSolver:
    """Tests for the CPCM implicit solvation model."""

    def test_vacuum_zero(self, h2_mol, hf_solver):
        """Solvation energy in vacuum (eps=1) should be exactly 0.

        When the dielectric constant equals 1, the screening factor
        f(eps) = (eps-1)/eps = 0, so all surface charges are zero and
        E_solv = 0.5 * q . V = 0.
        """
        from solvation import PCMSolver

        hf_solver.compute_energy(h2_mol, verbose=False)
        pcm = PCMSolver(solvent="vacuum")
        E_solv = pcm.compute_solvation_energy(h2_mol, hf_solver, verbose=False)

        assert E_solv == 0.0, (
            f"Vacuum solvation energy should be exactly 0, got {E_solv}"
        )

    def test_water_negative(self, h2o_mol, hf_solver):
        """H2O solvation in water should be negative (stabilizing).

        A polar molecule in a high-dielectric solvent is always stabilized
        by the reaction field: E_solv < 0.
        """
        from solvation import PCMSolver

        hf_solver.compute_energy(h2o_mol, verbose=False)
        pcm = PCMSolver(solvent="water")
        E_solv = pcm.compute_solvation_energy(h2o_mol, hf_solver, verbose=False)

        assert E_solv < 0, (
            f"Water solvation energy should be negative, got {E_solv:.8f} Eh"
        )

    def test_water_reasonable(self, h2o_mol, hf_solver):
        """H2O solvation in water should be between -1 and -20 kcal/mol.

        Experimental solvation free energy of water is about -6.3 kcal/mol.
        CPCM with STO-3G and post-SCF approach will differ, but should
        remain in a physically plausible range.
        """
        from solvation import PCMSolver

        hf_solver.compute_energy(h2o_mol, verbose=False)
        pcm = PCMSolver(solvent="water")
        E_solv = pcm.compute_solvation_energy(h2o_mol, hf_solver, verbose=False)
        E_kcal = E_solv * HARTREE_TO_KCAL

        assert -20.0 < E_kcal < -1.0, (
            f"H2O solvation = {E_kcal:.4f} kcal/mol, expected [-20, -1]"
        )

    def test_higher_epsilon_more_negative(self, h2o_mol, hf_solver):
        """Higher dielectric constant should give more negative solvation energy.

        The CPCM screening factor f(eps) = (eps-1)/eps is monotonically
        increasing in eps. Since E_solv = -0.5*f(eps)*V^T D^{-1} V,
        a larger f means a more negative E_solv.
        """
        from solvation import PCMSolver

        hf_solver.compute_energy(h2o_mol, verbose=False)

        # Low dielectric: benzene (eps=2.27)
        pcm_low = PCMSolver(solvent="benzene")
        E_low = pcm_low.compute_solvation_energy(h2o_mol, hf_solver, verbose=False)

        # High dielectric: water (eps=78.39)
        pcm_high = PCMSolver(solvent="water")
        E_high = pcm_high.compute_solvation_energy(h2o_mol, hf_solver, verbose=False)

        assert E_high < E_low, (
            f"Water (eps=78.39) E_solv={E_high:.8f} should be more negative "
            f"than benzene (eps=2.27) E_solv={E_low:.8f}"
        )

    def test_cavity_surface_area(self, h2o_mol):
        """Cavity should have a positive total surface area.

        The solvent-excluded surface built from vdW spheres must have
        a finite positive area. For H2O with default scaling (1.2),
        the total area should be tens of bohr^2.
        """
        from solvation import PCMSolver

        pcm = PCMSolver(solvent="water")
        cavity = pcm.build_cavity(h2o_mol, verbose=False)

        total_area = np.sum(cavity.areas)
        assert total_area > 0, (
            f"Total surface area should be positive, got {total_area}"
        )
        assert cavity.n_tesserae > 0, "Should have at least one tessera"

        # Each individual area element should also be positive
        assert np.all(cavity.areas > 0), "All tessera areas must be positive"

        # Surface area should be physically reasonable (> 10 bohr^2 for water)
        assert total_area > 10.0, (
            f"Total area {total_area:.2f} bohr^2 seems too small for H2O"
        )

    def test_he_near_zero(self, he_mol, hf_solver):
        """He atom solvation should be near zero (nonpolar, spherically symmetric).

        A closed-shell noble gas atom has no permanent dipole or significant
        multipole moments. The electrostatic solvation energy should be
        very small in magnitude (< 1 kcal/mol).
        """
        from solvation import PCMSolver

        hf_solver.compute_energy(he_mol, verbose=False)
        pcm = PCMSolver(solvent="water")
        E_solv = pcm.compute_solvation_energy(he_mol, hf_solver, verbose=False)
        E_kcal = E_solv * HARTREE_TO_KCAL

        assert abs(E_kcal) < 1.0, (
            f"He solvation = {E_kcal:.4f} kcal/mol, expected |E| < 1 kcal/mol"
        )
