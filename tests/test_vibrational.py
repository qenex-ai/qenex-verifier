"""
Tests for Vibrational Frequency Analysis Module
=================================================
Validates harmonic vibrational frequencies, normal mode counts,
zero-point energy, and thermochemistry from the VibrationalAnalysis class.

All geometries in Bohr. Uses STO-3G basis for speed.
"""

import sys
import pytest
import numpy as np

sys.path.insert(0, "packages/qenex_chem/src")


# ---------------------------------------------------------------------------
# Fixtures — molecules and shared solvers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_mol():
    """H2 at equilibrium-ish geometry (1.4 bohr), STO-3G."""
    from molecule import Molecule

    return Molecule(
        [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
        charge=0,
        multiplicity=1,
        basis_name="sto-3g",
    )


@pytest.fixture(scope="module")
def hf_solver():
    """Shared HartreeFockSolver instance."""
    from solver import HartreeFockSolver

    return HartreeFockSolver()


@pytest.fixture(scope="module")
def vib_analysis():
    """Shared VibrationalAnalysis (quiet mode)."""
    from vibrational import VibrationalAnalysis

    return VibrationalAnalysis(verbose=False)


@pytest.fixture(scope="module")
def h2_freq_result(h2_mol, hf_solver, vib_analysis):
    """Pre-computed frequency result for H2 (reused across tests)."""
    return vib_analysis.compute_frequencies(h2_mol, hf_solver)


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


class TestVibrationalAnalysis:
    """Tests for VibrationalAnalysis on H2 (STO-3G)."""

    def test_h2_frequency(self, h2_freq_result):
        """H2 stretching frequency should be 4000-6000 cm^-1 for STO-3G.

        STO-3G overestimates the harmonic frequency compared to experiment
        (~4401 cm^-1), but 4000-6000 is a safe range for the minimal basis.
        """
        freqs = h2_freq_result["frequencies_cm1"]
        real_freqs = freqs[freqs > 0]

        assert len(real_freqs) >= 1, "Should have at least 1 real frequency"

        # The single stretch mode
        stretch = np.max(real_freqs)
        assert 4000 < stretch < 6000, (
            f"H2 stretch {stretch:.1f} cm^-1 outside [4000, 6000]"
        )

    def test_h2_one_real_mode(self, h2_freq_result):
        """H2 (linear, 2 atoms) should have exactly 1 real vibrational mode.

        Degrees of freedom: 3N - 5 = 6 - 5 = 1 for a linear molecule.
        """
        freqs = h2_freq_result["frequencies_cm1"]
        real_freqs = freqs[freqs > 0]

        assert len(real_freqs) == 1, (
            f"Expected 1 real mode for H2, got {len(real_freqs)}"
        )

    def test_h2_zpe_positive(self, h2_freq_result):
        """Zero-point energy must be positive (sum of 0.5*h*nu > 0)."""
        zpe = h2_freq_result["zpe_hartree"]
        assert zpe > 0, f"ZPE should be positive, got {zpe}"

        # Also check it's physically reasonable (a few kcal/mol)
        HARTREE_TO_KCAL = 627.5094740631
        zpe_kcal = zpe * HARTREE_TO_KCAL
        assert 0.1 < zpe_kcal < 20.0, (
            f"ZPE = {zpe_kcal:.4f} kcal/mol seems unreasonable"
        )

    def test_h2_no_imaginary(self, h2_freq_result):
        """At equilibrium geometry, there should be no imaginary frequencies.

        Imaginary frequencies (n_imag > 0) indicate a saddle point.
        The near-equilibrium H2 geometry should be a true minimum.
        """
        n_imag = h2_freq_result["n_imag"]
        assert n_imag == 0, (
            f"Expected 0 imaginary frequencies at equilibrium, got {n_imag}"
        )

    def test_thermochemistry(self, h2_mol, hf_solver, vib_analysis, h2_freq_result):
        """Thermochemistry at 298 K: G < H and S > 0.

        Physical constraints from statistical mechanics:
        - Gibbs free energy G = H - TS, so G < H when T > 0 and S > 0
        - Total entropy must be positive (S_trans + S_rot + S_vib + S_elec > 0)
        """
        thermo = vib_analysis.compute_thermochemistry(
            h2_mol,
            hf_solver,
            temperature=298.15,
            freq_result=h2_freq_result,
        )

        G = thermo["G"]
        H = thermo["H"]
        S = thermo["S_total"]

        assert S > 0, f"Total entropy should be positive, got {S}"
        assert G < H, (
            f"Gibbs energy ({G:.8f}) should be less than enthalpy ({H:.8f}) at 298 K"
        )

        # Sanity: TS should be a small correction (< 0.1 Eh for small molecules)
        TS = thermo["temperature"] * S
        assert 0 < TS < 0.1, f"T*S = {TS:.8f} Eh seems unreasonable"

        # Translational entropy should dominate for H2
        assert thermo["S_trans"] > 0, "S_trans must be positive"
        assert thermo["S_rot"] > 0, "S_rot must be positive for diatomic"
        assert thermo["S_vib"] >= 0, "S_vib must be non-negative"
