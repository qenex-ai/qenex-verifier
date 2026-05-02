"""
QENEX LAB — Physical Constants (CODATA 2018)
=============================================

Single source of truth for all physical constants used across the platform.
Every module should import from here rather than redefining inline.

All values from CODATA 2018 recommended values unless noted:
    https://physics.nist.gov/cuu/Constants/

Usage:
    from constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM
    # or namespace-style:
    from constants import PhysConst
    x = PhysConst.HARTREE_TO_EV
"""

__all__ = [
    # Energy conversions
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL",
    "HARTREE_TO_KJ",
    "HARTREE_TO_CM1",
    "HARTREE_TO_J",
    "EV_TO_HARTREE",
    "KCAL_TO_HARTREE",
    "EV_TO_KCAL",
    "EV_TO_NM",
    "EV_TO_CM1",
    # Length conversions
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_M",
    "BOHR_TO_NM",
    # Mass conversions
    "AMU_TO_KG",
    "AMU_TO_ME",
    "ME_TO_AMU",
    # Fundamental constants (SI)
    "AVOGADRO",
    "BOLTZMANN_SI",
    "BOLTZMANN_EH",
    "PLANCK_SI",
    "PLANCK_EV",
    "SPEED_OF_LIGHT_SI",
    "SPEED_OF_LIGHT_CM",
    "ELECTRON_CHARGE",
    "VACUUM_PERMITTIVITY",
    "BOHR_MAGNETON",
    # Derived / spectroscopy
    "GAS_CONSTANT",
    "STANDARD_TEMPERATURE",
    "STANDARD_PRESSURE",
    "FREQ_AU_TO_CM1",
    # Namespace alias
    "PhysConst",
]

# ---------------------------------------------------------------------------
# Energy conversions  (CODATA 2018)
# ---------------------------------------------------------------------------

HARTREE_TO_EV: float = 27.211386245988  # 1 Eh → eV
HARTREE_TO_KCAL: float = 627.5094740631  # 1 Eh → kcal/mol
HARTREE_TO_KJ: float = 2625.4996394799  # 1 Eh → kJ/mol
HARTREE_TO_CM1: float = 219474.6313632  # 1 Eh → cm⁻¹  (wavenumbers)
HARTREE_TO_J: float = 4.3597447222071e-18  # 1 Eh → Joules

EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV
KCAL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCAL
EV_TO_KCAL: float = HARTREE_TO_KCAL / HARTREE_TO_EV  # ≈ 23.0605
EV_TO_NM: float = 1239.8419843320  # E(eV) → λ(nm)  via hc
EV_TO_CM1: float = 8065.543937  # 1 eV → cm⁻¹

# ---------------------------------------------------------------------------
# Length conversions  (CODATA 2018)
# ---------------------------------------------------------------------------

BOHR_TO_ANGSTROM: float = 0.529177210903  # 1 a₀ → Å
ANGSTROM_TO_BOHR: float = 1.8897259886  # 1 Å  → a₀  (= 1/BOHR_TO_ANGSTROM)
BOHR_TO_M: float = 5.29177210903e-11  # 1 a₀ → m
BOHR_TO_NM: float = BOHR_TO_M * 1e9  # 1 a₀ → nm

# ---------------------------------------------------------------------------
# Mass conversions  (CODATA 2018)
# ---------------------------------------------------------------------------

AMU_TO_KG: float = 1.66053906660e-27  # 1 Da → kg
AMU_TO_ME: float = 1822.888486209  # 1 Da → mₑ  (electron masses)
ME_TO_AMU: float = 1.0 / AMU_TO_ME

# ---------------------------------------------------------------------------
# Fundamental constants (SI)
# ---------------------------------------------------------------------------

AVOGADRO: float = 6.02214076e23  # mol⁻¹
BOLTZMANN_SI: float = 1.380649e-23  # J K⁻¹
BOLTZMANN_EH: float = 3.1668115634556e-6  # Eh K⁻¹  (kB in atomic units)
PLANCK_SI: float = 6.62607015e-34  # J s
PLANCK_EV: float = 4.135667696e-15  # eV s
SPEED_OF_LIGHT_SI: float = 299792458.0  # m s⁻¹  (exact)
SPEED_OF_LIGHT_CM: float = 2.99792458e10  # cm s⁻¹
ELECTRON_CHARGE: float = 1.602176634e-19  # C  (exact)
VACUUM_PERMITTIVITY: float = 8.8541878128e-12  # F m⁻¹
BOHR_MAGNETON: float = 9.2740100783e-24  # J T⁻¹

# ---------------------------------------------------------------------------
# Derived / spectroscopy
# ---------------------------------------------------------------------------

GAS_CONSTANT: float = AVOGADRO * BOLTZMANN_SI  # J mol⁻¹ K⁻¹ = 8.31446…
STANDARD_TEMPERATURE: float = 298.15  # K  (25 °C)
STANDARD_PRESSURE: float = 101325.0  # Pa (1 atm)

# Frequency conversion factor: sqrt(force_constant / reduced_mass) → cm⁻¹
# force_constant in Eh/bohr², reduced_mass in Dalton → cm⁻¹
# Factor = (1/(2π)) * sqrt(Eh/bohr² / (Da * bohr²)) * (1/c_cm)
# Pre-computed for convenience (same as used in vibrational.py)
FREQ_AU_TO_CM1: float = (
    (1.0 / (2.0 * 3.141592653589793))
    * (HARTREE_TO_J / (AMU_TO_KG * BOHR_TO_M**2)) ** 0.5
    / SPEED_OF_LIGHT_CM
)


# ---------------------------------------------------------------------------
# Namespace alias for dot-notation access
# ---------------------------------------------------------------------------


class PhysConst:
    """
    Namespace alias so callers can write ``PhysConst.HARTREE_TO_EV`` instead
    of the bare name.  All attributes are the module-level constants above.
    """

    HARTREE_TO_EV = HARTREE_TO_EV
    HARTREE_TO_KCAL = HARTREE_TO_KCAL
    HARTREE_TO_KJ = HARTREE_TO_KJ
    HARTREE_TO_CM1 = HARTREE_TO_CM1
    HARTREE_TO_J = HARTREE_TO_J
    EV_TO_HARTREE = EV_TO_HARTREE
    KCAL_TO_HARTREE = KCAL_TO_HARTREE
    EV_TO_KCAL = EV_TO_KCAL
    EV_TO_NM = EV_TO_NM
    EV_TO_CM1 = EV_TO_CM1
    BOHR_TO_ANGSTROM = BOHR_TO_ANGSTROM
    ANGSTROM_TO_BOHR = ANGSTROM_TO_BOHR
    BOHR_TO_M = BOHR_TO_M
    BOHR_TO_NM = BOHR_TO_NM
    AMU_TO_KG = AMU_TO_KG
    AMU_TO_ME = AMU_TO_ME
    ME_TO_AMU = ME_TO_AMU
    AVOGADRO = AVOGADRO
    BOLTZMANN_SI = BOLTZMANN_SI
    BOLTZMANN_EH = BOLTZMANN_EH
    PLANCK_SI = PLANCK_SI
    PLANCK_EV = PLANCK_EV
    SPEED_OF_LIGHT_SI = SPEED_OF_LIGHT_SI
    SPEED_OF_LIGHT_CM = SPEED_OF_LIGHT_CM
    ELECTRON_CHARGE = ELECTRON_CHARGE
    VACUUM_PERMITTIVITY = VACUUM_PERMITTIVITY
    BOHR_MAGNETON = BOHR_MAGNETON
    GAS_CONSTANT = GAS_CONSTANT
    STANDARD_TEMPERATURE = STANDARD_TEMPERATURE
    STANDARD_PRESSURE = STANDARD_PRESSURE
    FREQ_AU_TO_CM1 = FREQ_AU_TO_CM1
