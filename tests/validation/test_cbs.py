"""
Complete Basis Set (CBS) Extrapolation Tests
==============================================
Validates CBS extrapolation using Helgaker 2-point formula
with cc-pVDZ/cc-pVTZ pair.

References:
  Helgaker et al., J. Chem. Phys. 106, 9639 (1997)
  Feller, J. Chem. Phys. 96, 6104 (1992)
"""

import pytest
import numpy as np


class TestCBSFormulas:
    """Test the CBS extrapolation formulas directly."""

    def test_corr_extrapolation_formula(self):
        """Helgaker X^-3 formula gives correct CBS limit."""
        from cbs import extrapolate_corr_2point

        # With known values: E(2) = -0.1, E(3) = -0.15
        # CBS = (8 * (-0.1) - 27 * (-0.15)) / (8 - 27) = (-0.8 + 4.05) / -19
        E_cbs = extrapolate_corr_2point(-0.1, -0.15, X=2, Y=3)
        expected = (8 * (-0.1) - 27 * (-0.15)) / (8 - 27)
        assert abs(E_cbs - expected) < 1e-12

    def test_corr_cbs_more_negative_than_tz(self):
        """CBS correlation should be more negative than TZ."""
        from cbs import extrapolate_corr_2point

        E_corr_dz = -0.0258  # He-like
        E_corr_tz = -0.0333
        E_cbs = extrapolate_corr_2point(E_corr_dz, E_corr_tz)
        assert E_cbs < E_corr_tz, "CBS should be more negative than TZ"

    def test_hf_cbs_lower_than_tz(self):
        """CBS HF energy should be lower than TZ."""
        from cbs import extrapolate_hf_2point

        E_hf_dz = -2.855
        E_hf_tz = -2.861
        E_cbs = extrapolate_hf_2point(E_hf_dz, E_hf_tz)
        assert E_cbs < E_hf_tz, "CBS HF should be lower than TZ"


class TestCBSEnergies:
    """Validate CBS-extrapolated energies against known limits."""

    def test_he_cbs(self):
        """He MP2/CBS total energy is reasonable (-2.89 to -2.91 Eh)."""
        from molecule import Molecule
        from cbs import compute_cbs

        mol = Molecule([("He", (0, 0, 0))], charge=0, multiplicity=1)
        result = compute_cbs(mol, verbose=False)

        assert -2.91 < result["E_total_cbs"] < -2.89, (
            f"He CBS total {result['E_total_cbs']:.6f} out of expected range"
        )

    def test_he_cbs_lower_than_tz(self):
        """He CBS total should be lower than cc-pVTZ total."""
        from molecule import Molecule
        from cbs import compute_cbs

        mol = Molecule([("He", (0, 0, 0))], charge=0, multiplicity=1)
        result = compute_cbs(mol, verbose=False)

        assert result["E_total_cbs"] < result["E_total_tz"]

    def test_h2_cbs(self):
        """H2 MP2/CBS total energy is reasonable."""
        from molecule import Molecule
        from cbs import compute_cbs

        mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 1.4))], charge=0, multiplicity=1)
        result = compute_cbs(mol, verbose=False)

        assert -1.18 < result["E_total_cbs"] < -1.16

    def test_h2o_cbs(self):
        """H2O MP2/CBS total energy is reasonable (-76.35 to -76.40 Eh)."""
        from molecule import Molecule
        from cbs import compute_cbs

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)], charge=0, multiplicity=1
        )
        result = compute_cbs(mol, verbose=False)

        assert -76.40 < result["E_total_cbs"] < -76.35, (
            f"H2O CBS total {result['E_total_cbs']:.6f} out of range"
        )

    def test_h2o_cbs_lower_than_tz(self):
        """H2O CBS should be lower than cc-pVTZ."""
        from molecule import Molecule
        from cbs import compute_cbs

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)], charge=0, multiplicity=1
        )
        result = compute_cbs(mol, verbose=False)

        assert result["E_total_cbs"] < result["E_total_tz"]

    def test_h2o_cbs_correction_significant(self):
        """CBS correction from TZ should be 10-30 kcal/mol for H2O."""
        from molecule import Molecule
        from cbs import compute_cbs

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)], charge=0, multiplicity=1
        )
        result = compute_cbs(mol, verbose=False)

        delta_kcal = result["delta_tz"] * 627.5094
        assert -40 < delta_kcal < -10, (
            f"CBS correction from TZ = {delta_kcal:.1f} kcal/mol (expected -10 to -40)"
        )

    def test_he_correlation_recovery(self):
        """He MP2/CBS should recover 80-95% of exact correlation energy."""
        from molecule import Molecule
        from cbs import compute_cbs

        mol = Molecule([("He", (0, 0, 0))], charge=0, multiplicity=1)
        result = compute_cbs(mol, verbose=False)

        # He exact non-relativistic: -2.903724 Eh, HF limit: -2.861680
        E_corr_exact = -2.903724 - (-2.861680)  # -0.042044
        E_corr_cbs = result["E_corr_cbs"]
        recovery = E_corr_cbs / E_corr_exact * 100

        assert 80 < recovery < 95, (
            f"He MP2/CBS recovers {recovery:.1f}% correlation (expected 80-95%)"
        )
