"""
QENEX LAB — Universal Physical Constants Module
================================================

Single source of truth for ALL physical constants across every domain.
Values from CODATA 2018/2022 recommended values with full uncertainties.

Reference: https://physics.nist.gov/cuu/Constants/
CODATA 2018: Rev. Mod. Phys. 93, 025010 (2021)

Every domain package MUST import constants from here.
No domain package should hardcode any physical constant.

Usage:
    from qenex_core.constants import CODATA
    c = CODATA.c                  # speed of light
    G = CODATA.G                  # gravitational constant
    hbar = CODATA.hbar            # reduced Planck constant
    k_B = CODATA.k_B             # Boltzmann constant
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class PhysicalConstant:
    """A physical constant with value, uncertainty, and units."""

    value: float
    uncertainty: float  # Standard uncertainty (1σ)
    units: str
    name: str
    symbol: str
    source: str = "CODATA 2018"

    @property
    def relative_uncertainty(self) -> float:
        """Relative standard uncertainty."""
        if self.value == 0:
            return 0.0
        return abs(self.uncertainty / self.value)

    def __float__(self) -> float:
        """Return the constant's numerical value as a Python float."""
        return self.value

    def __repr__(self) -> str:
        """Format as 'name (symbol) = value ± uncertainty units'."""
        """Format as 'name (symbol) = value ± uncertainty units'."""
        if self.uncertainty > 0:
            return f"{self.name} ({self.symbol}) = {self.value} ± {self.uncertainty} {self.units}"
        return f"{self.name} ({self.symbol}) = {self.value} {self.units} (exact)"


class CODATAConstants:
    """
    CODATA 2018 Recommended Values of the Fundamental Physical Constants.

    Constants marked (exact) have zero uncertainty after the 2019 SI redefinition.
    All values in SI units unless otherwise noted.
    """

    # ==========================================
    # EXACT DEFINING CONSTANTS (SI 2019)
    # ==========================================

    # Speed of light in vacuum (exact)
    c = PhysicalConstant(
        value=299792458.0,
        uncertainty=0.0,
        units="m/s",
        name="Speed of light in vacuum",
        symbol="c",
    )

    # Planck constant (exact)
    h = PhysicalConstant(
        value=6.62607015e-34,
        uncertainty=0.0,
        units="J·s",
        name="Planck constant",
        symbol="h",
    )

    # Reduced Planck constant (exact, derived: h / (2π))
    # Phase 26c fix C-4: Full float64 precision (was truncated to 9 sig figs).
    # Exact: 6.62607015e-34 / (2π) = 1.0545718176461565e-34
    hbar = PhysicalConstant(
        value=1.0545718176461565e-34,
        uncertainty=0.0,
        units="J·s",
        name="Reduced Planck constant",
        symbol="ℏ",
    )

    # Elementary charge (exact)
    e = PhysicalConstant(
        value=1.602176634e-19,
        uncertainty=0.0,
        units="C",
        name="Elementary charge",
        symbol="e",
    )

    # Boltzmann constant (exact)
    k_B = PhysicalConstant(
        value=1.380649e-23,
        uncertainty=0.0,
        units="J/K",
        name="Boltzmann constant",
        symbol="k_B",
    )

    # Avogadro constant (exact)
    N_A = PhysicalConstant(
        value=6.02214076e23,
        uncertainty=0.0,
        units="1/mol",
        name="Avogadro constant",
        symbol="N_A",
    )

    # Luminous efficacy (exact)
    K_cd = PhysicalConstant(
        value=683.0,
        uncertainty=0.0,
        units="lm/W",
        name="Luminous efficacy",
        symbol="K_cd",
    )

    # ==========================================
    # ELECTROMAGNETIC CONSTANTS
    # ==========================================

    # Vacuum electric permittivity
    epsilon_0 = PhysicalConstant(
        value=8.8541878128e-12,
        uncertainty=1.3e-21,
        units="F/m",
        name="Vacuum electric permittivity",
        symbol="ε₀",
    )

    # Vacuum magnetic permeability
    mu_0 = PhysicalConstant(
        value=1.25663706212e-6,
        uncertainty=1.9e-16,
        units="N/A²",
        name="Vacuum magnetic permeability",
        symbol="μ₀",
    )

    # Fine-structure constant
    alpha = PhysicalConstant(
        value=7.2973525693e-3,
        uncertainty=1.1e-12,
        units="",
        name="Fine-structure constant",
        symbol="α",
    )

    # ==========================================
    # ATOMIC AND NUCLEAR CONSTANTS
    # ==========================================

    # Electron mass
    m_e = PhysicalConstant(
        value=9.1093837015e-31,
        uncertainty=2.8e-40,
        units="kg",
        name="Electron mass",
        symbol="m_e",
    )

    # Proton mass
    m_p = PhysicalConstant(
        value=1.67262192369e-27,
        uncertainty=5.1e-37,
        units="kg",
        name="Proton mass",
        symbol="m_p",
    )

    # Neutron mass
    m_n = PhysicalConstant(
        value=1.67492749804e-27,
        uncertainty=9.5e-37,
        units="kg",
        name="Neutron mass",
        symbol="m_n",
    )

    # Atomic mass unit
    u = PhysicalConstant(
        value=1.66053906660e-27,
        uncertainty=5.0e-37,
        units="kg",
        name="Atomic mass unit",
        symbol="u",
    )

    # Bohr radius
    a_0 = PhysicalConstant(
        value=5.29177210903e-11,
        uncertainty=8.0e-21,
        units="m",
        name="Bohr radius",
        symbol="a₀",
    )

    # Hartree energy
    E_h = PhysicalConstant(
        value=4.3597447222071e-18,
        uncertainty=8.5e-30,
        units="J",
        name="Hartree energy",
        symbol="E_h",
    )

    # Rydberg constant
    R_inf = PhysicalConstant(
        value=10973731.568160,
        uncertainty=0.000021,
        units="1/m",
        name="Rydberg constant",
        symbol="R∞",
    )

    # Stefan-Boltzmann constant
    sigma_SB = PhysicalConstant(
        value=5.670374419e-8,
        uncertainty=0.0,
        units="W/(m²·K⁴)",
        name="Stefan-Boltzmann constant",
        symbol="σ",
    )

    # Gas constant
    R = PhysicalConstant(
        value=8.314462618,
        uncertainty=0.0,
        units="J/(mol·K)",
        name="Molar gas constant",
        symbol="R",
    )

    # Faraday constant
    F = PhysicalConstant(
        value=96485.33212,
        uncertainty=0.0,
        units="C/mol",
        name="Faraday constant",
        symbol="F",
    )

    # ==========================================
    # GRAVITATIONAL CONSTANT
    # ==========================================

    G = PhysicalConstant(
        value=6.67430e-11,
        uncertainty=1.5e-15,
        units="m³/(kg·s²)",
        name="Newtonian constant of gravitation",
        symbol="G",
    )

    # ==========================================
    # ASTROPHYSICAL CONSTANTS (IAU 2015)
    # ==========================================

    # Solar mass
    M_sun = PhysicalConstant(
        value=1.98841e30,
        uncertainty=2.5e25,
        units="kg",
        name="Solar mass",
        symbol="M☉",
        source="IAU 2015",
    )

    # Solar radius
    R_sun = PhysicalConstant(
        value=6.957e8,
        uncertainty=6.5e4,
        units="m",
        name="Solar radius",
        symbol="R☉",
        source="IAU 2015",
    )

    # Solar luminosity
    L_sun = PhysicalConstant(
        value=3.828e26,
        uncertainty=0.0,
        units="W",
        name="Solar luminosity",
        symbol="L☉",
        source="IAU 2015",
    )

    # Solar effective temperature
    T_sun = PhysicalConstant(
        value=5772.0,
        uncertainty=0.8,
        units="K",
        name="Solar effective temperature",
        symbol="T☉",
        source="IAU 2015",
    )

    # Astronomical unit (exact by definition)
    AU = PhysicalConstant(
        value=1.495978707e11,
        uncertainty=0.0,
        units="m",
        name="Astronomical unit",
        symbol="AU",
        source="IAU 2012",
    )

    # Parsec
    pc = PhysicalConstant(
        value=3.0856775814913673e16,
        uncertainty=0.0,
        units="m",
        name="Parsec",
        symbol="pc",
        source="IAU 2015",
    )

    # Light-year
    ly = PhysicalConstant(
        value=9.4607304725808e15,
        uncertainty=0.0,
        units="m",
        name="Light-year",
        symbol="ly",
    )

    # ==========================================
    # COSMOLOGICAL PARAMETERS (Planck 2018)
    # ==========================================

    H_0 = PhysicalConstant(
        value=67.4,
        uncertainty=0.5,
        units="km/s/Mpc",
        name="Hubble constant",
        symbol="H₀",
        source="Planck 2018",
    )

    Omega_m = PhysicalConstant(
        value=0.315,
        uncertainty=0.007,
        units="",
        name="Matter density parameter",
        symbol="Ωₘ",
        source="Planck 2018",
    )

    Omega_Lambda = PhysicalConstant(
        value=0.685,
        uncertainty=0.007,
        units="",
        name="Dark energy density parameter",
        symbol="Ω_Λ",
        source="Planck 2018",
    )

    Omega_b = PhysicalConstant(
        value=0.0493,
        uncertainty=0.0006,
        units="",
        name="Baryon density parameter",
        symbol="Ω_b",
        source="Planck 2018",
    )

    T_CMB = PhysicalConstant(
        value=2.7255,
        uncertainty=0.0006,
        units="K",
        name="CMB temperature",
        symbol="T_CMB",
        source="Fixsen 2009",
    )

    # ==========================================
    # CLIMATE SCIENCE CONSTANTS
    # ==========================================

    S_0 = PhysicalConstant(
        value=1361.0,
        uncertainty=0.5,
        units="W/m²",
        name="Total solar irradiance",
        symbol="S₀",
        source="Kopp & Lean 2011",
    )

    R_Earth = PhysicalConstant(
        value=6.371e6,
        uncertainty=1e3,
        units="m",
        name="Earth mean radius",
        symbol="R_⊕",
        source="IERS",
    )

    M_Earth = PhysicalConstant(
        value=5.97217e24,
        uncertainty=1.3e20,
        units="kg",
        name="Earth mass",
        symbol="M_⊕",
        source="IERS",
    )

    # ==========================================
    # CHEMISTRY CONSTANTS
    # ==========================================

    # Bohr magneton
    mu_B = PhysicalConstant(
        value=9.2740100783e-24,
        uncertainty=2.8e-33,
        units="J/T",
        name="Bohr magneton",
        symbol="μ_B",
    )

    # Nuclear magneton
    mu_N = PhysicalConstant(
        value=5.0507837461e-27,
        uncertainty=1.5e-36,
        units="J/T",
        name="Nuclear magneton",
        symbol="μ_N",
    )

    # ==========================================
    # NEUROSCIENCE CONSTANTS
    # ==========================================

    # Membrane capacitance (typical biological membrane)
    C_membrane = PhysicalConstant(
        value=1.0e-2,
        uncertainty=0.1e-2,
        units="F/m²",
        name="Membrane capacitance",
        symbol="C_m",
        source="Hodgkin-Huxley 1952",
    )

    # ==========================================
    # UNIT CONVERSION FACTORS
    # ==========================================

    # Energy conversions
    eV_to_J = PhysicalConstant(
        value=1.602176634e-19,
        uncertainty=0.0,
        units="J/eV",
        name="eV to Joule",
        symbol="eV→J",
    )

    Hartree_to_eV = PhysicalConstant(
        value=27.211386245988,
        uncertainty=5.3e-11,
        units="eV/E_h",
        name="Hartree to eV",
        symbol="E_h→eV",
    )

    Hartree_to_kJ_mol = PhysicalConstant(
        value=2625.4996394799,
        uncertainty=0.0,
        units="kJ/(mol·E_h)",
        name="Hartree to kJ/mol",
        symbol="E_h→kJ/mol",
    )

    Hartree_to_kcal_mol = PhysicalConstant(
        value=627.5094740631,
        uncertainty=0.0,
        units="kcal/(mol·E_h)",
        name="Hartree to kcal/mol",
        symbol="E_h→kcal/mol",
    )

    kcal_mol_to_Hartree = PhysicalConstant(
        value=1.5936011e-3,
        uncertainty=0.0,
        units="E_h/(kcal/mol)",
        name="kcal/mol to Hartree",
        symbol="kcal/mol→E_h",
    )

    # Chemical accuracy threshold
    CHEMICAL_ACCURACY_HARTREE = 1.5936e-3  # 1 kcal/mol in Hartree
    CHEMICAL_ACCURACY_EV = 0.0434  # 1 kcal/mol in eV

    # Length conversions
    Angstrom_to_Bohr = PhysicalConstant(
        value=1.8897259886,
        uncertainty=3.6e-10,
        units="a₀/Å",
        name="Angstrom to Bohr",
        symbol="Å→a₀",
    )

    Bohr_to_Angstrom = PhysicalConstant(
        value=0.529177210903,
        uncertainty=8.0e-12,
        units="Å/a₀",
        name="Bohr to Angstrom",
        symbol="a₀→Å",
    )

    # ==========================================
    # MATHEMATICAL CONSTANTS
    # ==========================================

    pi = math.pi
    euler_e = math.e
    golden_ratio = (1 + math.sqrt(5)) / 2

    @classmethod
    def get_all(cls) -> Dict[str, PhysicalConstant]:
        """Return all physical constants as a dictionary."""
        result = {}
        for name, value in vars(cls).items():
            if isinstance(value, PhysicalConstant):
                result[name] = value
        return result

    @classmethod
    def validate_against_nist(
        cls, name: str, value: float, tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Validate a computed/measured value against NIST reference.

        Returns:
            (is_valid, relative_error)
        """
        ref = getattr(cls, name, None)
        if ref is None:
            raise ValueError(f"Unknown constant: {name}")

        if ref.value == 0:
            return abs(value) < tolerance, abs(value)

        rel_error = abs(value - ref.value) / abs(ref.value)
        return rel_error < tolerance, rel_error

    @classmethod
    def summary(cls) -> str:
        """Generate a summary table of all constants."""
        lines = [
            "=" * 90,
            "QENEX LAB — CODATA 2018 Physical Constants",
            "=" * 90,
            f"{'Name':<35} {'Symbol':<8} {'Value':<25} {'Units':<15}",
            "-" * 90,
        ]
        for name, const in cls.get_all().items():
            lines.append(
                f"{const.name:<35} {const.symbol:<8} {const.value:<25.15g} {const.units:<15}"
            )
        lines.append("=" * 90)
        return "\n".join(lines)


# Convenience alias
CODATA = CODATAConstants

# Commonly used values as module-level floats for performance
c = 299792458.0
h = 6.62607015e-34
hbar = 1.0545718176461565e-34  # Phase 26c C-4: full float64 precision
e_charge = 1.602176634e-19
k_B = 1.380649e-23
N_A = 6.02214076e23
G = 6.67430e-11
m_e = 9.1093837015e-31
m_p = 1.67262192369e-27
a_0 = 5.29177210903e-11
E_h = 4.3597447222071e-18
alpha_fs = 7.2973525693e-3
sigma_SB = 5.670374419e-8
R_gas = 8.314462618

__all__ = [
    "PhysicalConstant",
    "CODATAConstants",
    "CODATA",
    "c",
    "h",
    "hbar",
    "e_charge",
    "k_B",
    "N_A",
    "G",
    "m_e",
    "m_p",
    "a_0",
    "E_h",
    "alpha_fs",
    "sigma_SB",
    "R_gas",
]
