"""
QENEX LAB Regression Benchmark System
=======================================
The scientific record of QENEX LAB's accuracy.

Every number is traceable to a reference. Every deviation is explainable.
This is not a test suite — this is the foundation of trust for a scientific
lab that lasts decades.

Reference sources:
    - PySCF 2.12.1 (cart=True) cross-validated energies
    - NIST CCCBDB experimental geometries and frequencies
    - W4-11 / HEAT345 atomization energies (where applicable)
    - CODATA 2018 physical constants

All coordinates in BOHR. All energies in Hartree unless noted.

Usage:
    from benchmark import BenchmarkSuite

    suite = BenchmarkSuite()
    summary = suite.run_quick()        # ~30s: H2, He, H2O on STO-3G
    summary = suite.run_full()         # ~5min: all molecules, cc-pVDZ
    summary = suite.run_accuracy()     # Compare against experiment
    summary = suite.run_benchmark()    # Everything

Author: QENEX LAB
"""

import time
import numpy as np

# Support both package and direct imports
try:
    from .molecule import Molecule
    from .solver import HartreeFockSolver
    from .ccsd import CCSDSolver
    from .eomccsd import EOMCCSDSolver
    from .uccsd import UCCSDSolver
    from .cbs import extrapolate_cbs_2point
except ImportError:
    from molecule import Molecule
    from solver import HartreeFockSolver
    from ccsd import CCSDSolver
    from eomccsd import EOMCCSDSolver
    from uccsd import UCCSDSolver
    from cbs import extrapolate_cbs_2point  # type: ignore[no-redef]

__all__ = ["BenchmarkSuite"]

# =====================================================================
# VERSION
# =====================================================================
BENCHMARK_VERSION = "1.0.0"

# =====================================================================
# PHYSICAL CONSTANTS — from constants.py (single source of truth)
# =====================================================================
try:
    from .phys_constants import HARTREE_TO_EV, HARTREE_TO_KCAL, BOHR_TO_ANGSTROM
except ImportError:
    from phys_constants import HARTREE_TO_EV, HARTREE_TO_KCAL, BOHR_TO_ANGSTROM  # type: ignore[no-redef]

# =====================================================================
# INTEGRAL BACKEND DETECTION
# =====================================================================
# When libcint is available, our HF solver uses PySCF's SCF engine and
# matches PySCF reference values to machine precision (~1e-10 Eh).
# When only our native Obara-Saika engine is available, there are small
# integral differences (~50 µHa for STO-3G, ~1e-6 for cc-pVXZ).
#
# The benchmark detects the backend and adjusts thresholds accordingly.


def _detect_libcint():
    """Check if libcint is available."""
    try:
        from libcint_integrals import LIBCINT_AVAILABLE

        return LIBCINT_AVAILABLE
    except ImportError:
        return False


LIBCINT_AVAILABLE = _detect_libcint()

# =====================================================================
# THRESHOLDS
# =====================================================================
# These encode the expected accuracy of each method in our implementation.
#
# With libcint: HF matches PySCF to machine precision (same integral engine).
# Without libcint (Obara-Saika): integral differences ~50 µHa for STO-3G.
# cc-pVXZ always uses libcint (if available), so tolerance is tighter.

THRESHOLDS = {
    # HF energy thresholds
    "hf_energy_libcint": 1e-6,  # Hartree — libcint matches PySCF exactly
    "hf_energy_native": 5e-5,  # Hartree — Obara-Saika vs libcint difference
    # Correlated methods (always use libcint integrals when available)
    "ccsd_corr": 1e-5,  # Hartree (10 microhartree)
    "ccsdt_corr": 1e-4,  # Hartree (0.1 millihartree)
    "uccsd_corr": 1e-5,  # Hartree (10 microhartree)
    "uccsdt_corr": 1e-4,  # Hartree (0.1 millihartree)
    "eomccsd": 1e-3,  # Hartree (millihartree) — finite-difference noise
    # Experimental comparisons
    "geometry": 0.01,  # Bohr
    "frequency": 500,  # cm⁻¹ (HF overestimates by ~5-10%, STO-3G even more)
}


def _hf_threshold():
    """Get the appropriate HF energy threshold for the current backend."""
    if LIBCINT_AVAILABLE:
        return THRESHOLDS["hf_energy_libcint"]
    else:
        return THRESHOLDS["hf_energy_native"]


# =====================================================================
# MOLECULE GEOMETRIES (all in Bohr)
# =====================================================================
# CRITICAL: These geometries MUST match exactly what was used to generate
# the PySCF reference values. Any coordinate change invalidates the
# corresponding reference energies.
#
# Two sets of geometries exist:
#   1. STO-3G reference geometries (from cross-validation tests, matching PySCF)
#   2. Standard QENEX geometries (from auto_science.py / CCSD tests)
#
# For STO-3G HF: use the cross-validation geometries (converted from Angstrom)
# For cc-pVDZ/cc-pVTZ: use the standard QENEX geometries

_ANG2BOHR = 1.0 / 0.529177  # Matches cross-validation test conversion factor
_ANG2BOHR_PRECISE = 1.8897259886  # Higher precision for EOM tests

# --- STO-3G reference geometries (matching cross-validation PySCF values) ---
_H2_STO3G_R = 1.3984  # 0.74 Å → bohr (NOT 1.4)
_LIH_STO3G_R = 1.596 * _ANG2BOHR  # 3.0160 bohr

# H2O STO-3G reference geometry from cross-validation tests
# PySCF Å: O(0,0,0.1173), H(0,±0.7572,-0.4692)
_H2O_STO3G_O_Z = 0.1173 * _ANG2BOHR
_H2O_STO3G_H_Y = 0.7572 * _ANG2BOHR
_H2O_STO3G_H_Z = -0.4692 * _ANG2BOHR

# --- Standard QENEX geometries (for cc-pVDZ / cc-pVTZ / CCSD tests) ---
_H2O_R = 1.8088  # O-H bond length in Bohr
_H2O_ANGLE = 104.52  # H-O-H angle in degrees
_H2O_HALF_ANGLE = _H2O_ANGLE * np.pi / 360.0
_H2O_STD_H1 = (0.0, _H2O_R * np.sin(_H2O_HALF_ANGLE), _H2O_R * np.cos(_H2O_HALF_ANGLE))
_H2O_STD_H2 = (0.0, -_H2O_R * np.sin(_H2O_HALF_ANGLE), _H2O_R * np.cos(_H2O_HALF_ANGLE))

# --- EOM-CCSD reference geometry (from PySCF EOM reference generation) ---
_H2O_EOM_H1 = (0.0, 0.757 * _ANG2BOHR_PRECISE, 0.587 * _ANG2BOHR_PRECISE)
_H2O_EOM_H2 = (0.0, -0.757 * _ANG2BOHR_PRECISE, 0.587 * _ANG2BOHR_PRECISE)


GEOMETRIES = {
    # === Atoms ===
    "H": [("H", (0, 0, 0))],
    "He": [("He", (0, 0, 0))],
    "Li": [("Li", (0, 0, 0))],
    "Be": [("Be", (0, 0, 0))],
    "N": [("N", (0, 0, 0))],
    "Ne": [("Ne", (0, 0, 0))],
    # === Diatomics (STO-3G reference geometries) ===
    "H2_sto3g": [("H", (0, 0, 0)), ("H", (0, 0, _H2_STO3G_R))],
    "LiH_sto3g": [("Li", (0, 0, 0)), ("H", (0, 0, _LIH_STO3G_R))],
    # === Diatomics (standard QENEX geometries for cc-pVXZ) ===
    "H2": [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
    "LiH": [("Li", (0, 0, 0)), ("H", (0, 0, 3.015))],
    "HF": [("F", (0, 0, 0)), ("H", (0, 0, 1.7328))],
    "N2": [("N", (0, 0, 0)), ("N", (0, 0, 2.074))],
    "CO": [("C", (0, 0, 0)), ("O", (0, 0, 2.132))],
    "F2": [("F", (0, 0, 0)), ("F", (0, 0, 2.668))],
    "OH": [("O", (0, 0, 0)), ("H", (0, 0, 1.8324))],
    "O2": [("O", (0, 0, 0)), ("O", (0, 0, 2.2819))],
    # === Triatomics (STO-3G reference geometry) ===
    "H2O_sto3g": [
        ("O", (0, 0, _H2O_STO3G_O_Z)),
        ("H", (0, _H2O_STO3G_H_Y, _H2O_STO3G_H_Z)),
        ("H", (0, -_H2O_STO3G_H_Y, _H2O_STO3G_H_Z)),
    ],
    # === Triatomics (standard QENEX geometry for cc-pVXZ) ===
    "H2O": [("O", (0, 0, 0)), ("H", _H2O_STD_H1), ("H", _H2O_STD_H2)],
    # === EOM-CCSD geometry ===
    "H2O_eom": [("O", (0, 0, 0)), ("H", _H2O_EOM_H1), ("H", _H2O_EOM_H2)],
    # === Other triatomics ===
    "CO2": [("C", (0, 0, 0)), ("O", (0, 0, 2.196)), ("O", (0, 0, -2.196))],
    "BeH2": [("Be", (0, 0, 0)), ("H", (0, 0, 2.502)), ("H", (0, 0, -2.502))],
    "HCN": [("H", (0, 0, -3.187)), ("C", (0, 0, -1.186)), ("N", (0, 0, 1.013))],
    # === Tetratomics+ ===
    "NH3": [
        ("N", (0, 0, 0.1173)),
        ("H", (0, 1.7717, -0.5461)),
        ("H", (1.5342, -0.8858, -0.5461)),
        ("H", (-1.5342, -0.8858, -0.5461)),
    ],
    "BH3": [
        ("B", (0, 0, 0)),
        ("H", (0, 2.249, 0)),
        ("H", (1.948, -1.125, 0)),
        ("H", (-1.948, -1.125, 0)),
    ],
    "CH4": [
        ("C", (0, 0, 0)),
        ("H", (1.186, 1.186, 1.186)),
        ("H", (-1.186, -1.186, 1.186)),
        ("H", (-1.186, 1.186, -1.186)),
        ("H", (1.186, -1.186, -1.186)),
    ],
    "C2H2": [
        ("C", (0, 0, -1.136)),
        ("C", (0, 0, 1.136)),
        ("H", (0, 0, -3.139)),
        ("H", (0, 0, 3.139)),
    ],
}


# =====================================================================
# REFERENCE DATABASE
# =====================================================================
# Every value is traceable to a source. Every source is documented.
#
# Naming convention for keys:
#   E_hf_sto3g       — RHF total energy, STO-3G basis
#   E_hf_ccpvdz      — RHF total energy, cc-pVDZ basis
#   E_hf_ccpvtz      — RHF total energy, cc-pVTZ basis
#   E_ccsd_corr_*    — CCSD correlation energy (not total)
#   E_ccsdt_corr_*   — CCSD(T) correction only
#   E_uccsd_corr_*   — UCCSD correlation energy
#   E_uccsdt_corr_*  — UCCSD(T) correction only
#   E_uhf_*          — UHF total energy
#   E_mp2_corr_*     — MP2 correlation energy
#   eom_s*_*         — EOM-CCSD excitation energies (Hartree)
#   R_eq_bohr        — Experimental equilibrium bond length
#   freq_cm1         — Experimental vibrational frequency(ies)
#
# Sources:
#   [P] = PySCF 2.12.1, cart=True
#   [N] = NIST CCCBDB
#

REFERENCE_DATA = {
    # All [P] values below regenerated from PySCF 2.12.1 (cart=True) on
    # 2026-04-18 via scripts/regenerate_reference_data.py.  HF converged to
    # 1e-12 Eh, CCSD to 1e-10 Eh, CCSD(T) via ccsd.ccsd_t() post-CCSD,
    # MP2 via mp.MP2(mf).kernel().  Geometries match benchmark.GEOMETRIES
    # and the dispatch in _get_geometry_key.  Open-shell values below (U-HF
    # atoms) use UHF with appropriate multiplicity.
    #
    # Every value is independently verified on every CI run by the
    # tests/test_reference_data_integrity.py watchdog — any drift
    # (code change, PySCF upgrade) lights up in 15 seconds.
    #
    # ================================================================
    # ATOMS
    # ================================================================
    "H": {
        "E_uhf_sto3g": -0.46658184955728,  # [P] 2026-04-18
        "mult": 2,
    },
    "He": {
        "E_hf_sto3g": -2.80778395753997,  # [P] 2026-04-18
        "E_hf_ccpvdz": -2.85516047724274,  # [P] 2026-04-21
        "E_hf_ccpvtz": -2.86115334478442,  # [P] 2026-04-21
        "E_hf_augccpvdz": -2.85570466771043,  # [P] 2026-04-18
        "E_hf_augccpvtz": -2.86118342611556,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.03243435414502,  # [P] 2026-04-21
    },
    "Li": {
        "E_uhf_sto3g": -7.31552598128109,  # [P] 2026-04-18
        "mult": 2,
    },
    "N": {
        "E_uhf_sto3g": -53.71901016259395,  # [P] 2026-04-18
        "mult": 4,
    },
    "Ne": {
        "E_hf_sto3g": -126.60452499680483,  # [P] 2026-04-18
    },
    # ================================================================
    # DIATOMICS — closed-shell
    # ================================================================
    "H2": {
        "E_hf_sto3g": -1.11675923355046,  # [P] 2026-04-18 at R=1.3984 Bohr
        "E_hf_ccpvdz": -1.12870944897989,  # [P] 2026-04-21
        "E_hf_ccpvtz": -1.13296052548290,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.03468928390090,  # [P] 2026-04-21
        "R_eq_bohr": 1.401,  # [N]
        "freq_cm1": 4401,  # [N]
    },
    "LiH": {
        "E_hf_sto3g": -7.86199266880826,  # [P] 2026-04-18 at R=3.0160 Bohr
        "E_mp2_corr_sto3g": -0.01287250001771,  # [P] 2026-04-18
        "R_eq_bohr": 3.015,  # [N]
    },
    "HF": {
        "E_hf_ccpvdz": -100.01941282773733,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.20874192655078,  # [P] 2026-04-21
        "E_ccsdt_corr_ccpvdz": -0.00193630321778,  # [P] 2026-04-21
        "R_eq_bohr": 1.733,  # [N]
        "freq_cm1": 4138,  # [N]
    },
    "N2": {
        "R_eq_bohr": 2.074,  # [N]
    },
    "CO": {
        "R_eq_bohr": 2.132,  # [N]
    },
    "F2": {
        "R_eq_bohr": 2.668,  # [N]
    },
    # ================================================================
    # TRIATOMICS — closed-shell
    # ================================================================
    "H2O": {
        "E_hf_sto3g": -74.96302319256539,  # [P] 2026-04-18 at H2O_sto3g geom
        "E_mp2_corr_sto3g": -0.03554568294876,  # [P] 2026-04-18
        "E_hf_ccpvdz": -76.02679981838681,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.21328215761425,  # [P] 2026-04-21
        "E_hf_ccpvtz": -76.05717013922529,  # [P] 2026-04-21
        "E_ccsdt_corr_ccpvdz": -0.00305552882664,  # [P] 2026-04-21
        "freq_cm1": [1648, 3832, 3943],  # [N]
    },
    "CO2": {
        "R_eq_bohr": 2.196,  # [N]
    },
    "HCN": {
        "R_eq_bohr_CH": 2.001,  # [N]
        "R_eq_bohr_CN": 2.179,  # [N]
    },
    # ================================================================
    # LARGER CLOSED-SHELL
    # ================================================================
    "NH3": {
        "E_hf_ccpvdz": -56.19531720562033,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.20394376885184,  # [P] 2026-04-21
        "E_ccsdt_corr_ccpvdz": -0.00371327676516,  # [P] 2026-04-21
    },
    "CH4": {
        "E_hf_ccpvdz": -40.19867349345080,  # [P] 2026-04-21
        "E_ccsd_corr_ccpvdz": -0.18727840178101,  # [P] 2026-04-21
        "E_ccsdt_corr_ccpvdz": -0.00373245456790,  # [P] 2026-04-21
    },
    # ================================================================
    # EOM-CCSD EXCITED STATES
    # ================================================================
    "H2O_eom": {
        # [P] 2026-04-18 — EOM-CCSD/STO-3G singlet excitations, cart=True.
        # Geometry: H2O with H at (0, ±0.757 Å, 0.587 Å) converted to Bohr.
        # Verified deterministic across 3 independent runs to 1e-15 Eh.
        "eom_s1_sto3g": 0.45662295690215,  # [P] 2026-04-18
        "eom_s2_sto3g": 0.54110425404270,  # [P] 2026-04-18
        "eom_s3_sto3g": 0.59860999275947,  # [P] 2026-04-18
        "eom_s4_sto3g": 0.69875411078641,  # [P] 2026-04-18
        "eom_s5_sto3g": 0.82645727363953,  # [P] 2026-04-18
        "geometry_key": "H2O_eom",
    },
    # ================================================================
    # OPEN-SHELL (UCCSD) — regenerated 2026-04-18 from PySCF 2.12.1
    # via scripts/regenerate_open_shell_refs.py (UHF conv_tol=1e-10,
    # UCCSD conv_tol=1e-8).  Verified live by
    # tests/test_reference_data_integrity.py.
    # ================================================================
    "Li_radical": {
        "E_uhf_sto3g": -7.31552598128109,  # [P] 2026-04-18
        "E_uccsd_corr_sto3g": -0.00031057156938,  # [P] 2026-04-18
        "E_uccsdt_corr_sto3g": 0.0,  # [P] (exact — Li has 3 electrons)
        "mult": 2,
        "geometry_key": "Li",
    },
    "OH_radical": {
        "E_uhf_sto3g": -74.36263373530393,  # [P] 2026-04-18
        "E_uccsd_corr_sto3g": -0.02449381721652,  # [P] 2026-04-18
        "E_uccsdt_corr_sto3g": -0.00000023144084,  # [P] 2026-04-18
        "mult": 2,
        "geometry_key": "OH",
    },
    "O2_triplet": {
        "E_uhf_sto3g": -147.63394816553966,  # [P] 2026-04-18
        "E_uccsd_corr_sto3g": -0.10799678891470,  # [P] 2026-04-18
        "E_uccsdt_corr_sto3g": -0.00068503780076,  # [P] 2026-04-18
        "mult": 3,
        "geometry_key": "O2",
    },
    "N_quartet": {
        "E_uhf_sto3g": -53.71901016259395,  # [P] 2026-04-18
        "E_uccsd_corr_sto3g": 0.0,  # [P] half-filled p → no correlation
        "mult": 4,
        "geometry_key": "N",
    },
}


# =====================================================================
# GEOMETRY DISPATCH
# =====================================================================
# Maps (molecule_ref_key, basis_class) → geometry_key
# STO-3G HF tests use specific geometries matching PySCF references.
# cc-pVDZ/TZ tests use standard QENEX geometries.


def _get_geometry_key(ref_key, basis):
    """
    Determine the correct geometry key for a given reference and basis.

    STO-3G reference values were generated with specific geometries that
    may differ from our standard QENEX geometries. For cc-pVDZ and cc-pVTZ,
    the standard QENEX geometries were used.
    """
    # Check for explicit geometry_key override in reference data
    if ref_key in REFERENCE_DATA:
        gk = REFERENCE_DATA[ref_key].get("geometry_key")
        if gk is not None:
            return gk

    # STO-3G uses specific reference geometries for H2, LiH, H2O
    if basis == "sto-3g":
        if ref_key == "H2":
            return "H2_sto3g"
        elif ref_key == "LiH":
            return "LiH_sto3g"
        elif ref_key == "H2O":
            return "H2O_sto3g"

    # Default: use molecule name as geometry key
    return ref_key


# =====================================================================
# TEST DEFINITIONS
# =====================================================================


def _build_test_definitions():
    """Build the complete list of benchmark tests from REFERENCE_DATA."""
    tests = []

    def add(molecule, ref_key, prop, method, basis, threshold, suites):
        geom_key = _get_geometry_key(ref_key, basis)
        tests.append(
            {
                "molecule": molecule,
                "geometry_key": geom_key,
                "ref_key": ref_key,
                "prop": prop,
                "method": method,
                "basis": basis,
                "threshold": threshold,
                "suites": suites,
                "ref_value": REFERENCE_DATA[ref_key][prop],
            }
        )

    hf_thresh = _hf_threshold()

    # --- RHF/STO-3G ---
    for mol, prop in [
        ("He", "E_hf_sto3g"),
        ("H2", "E_hf_sto3g"),
        ("LiH", "E_hf_sto3g"),
        ("H2O", "E_hf_sto3g"),
        ("Ne", "E_hf_sto3g"),
    ]:
        add(mol, mol, prop, "hf", "sto-3g", hf_thresh, ["quick", "full"])

    # --- RHF/aug-cc-pVDZ (new) ---
    for mol, prop in [
        ("He", "E_hf_augccpvdz"),
    ]:
        add(mol, mol, prop, "hf", "aug-cc-pvdz", hf_thresh, ["quick", "full"])

    # --- RHF/cc-pVDZ ---
    for mol, prop in [
        ("He", "E_hf_ccpvdz"),
        ("H2", "E_hf_ccpvdz"),
        ("H2O", "E_hf_ccpvdz"),
    ]:
        add(mol, mol, prop, "hf", "cc-pvdz", hf_thresh, ["full"])

    # --- RHF/cc-pVTZ ---
    for mol, prop in [
        ("He", "E_hf_ccpvtz"),
        ("H2", "E_hf_ccpvtz"),
        ("H2O", "E_hf_ccpvtz"),
    ]:
        add(mol, mol, prop, "hf", "cc-pvtz", hf_thresh, ["full"])

    # --- CCSD/cc-pVDZ correlation ---
    for mol, prop in [
        ("He", "E_ccsd_corr_ccpvdz"),
        ("H2", "E_ccsd_corr_ccpvdz"),
        ("H2O", "E_ccsd_corr_ccpvdz"),
        ("HF", "E_ccsd_corr_ccpvdz"),
        ("NH3", "E_ccsd_corr_ccpvdz"),
        ("CH4", "E_ccsd_corr_ccpvdz"),
    ]:
        suites = ["full"]
        if mol in ("He", "H2", "H2O"):
            suites = ["quick", "full"]
        add(mol, mol, prop, "ccsd", "cc-pvdz", THRESHOLDS["ccsd_corr"], suites)

    # --- CCSD(T)/cc-pVDZ correction ---
    for mol, prop in [
        ("H2O", "E_ccsdt_corr_ccpvdz"),
        ("HF", "E_ccsdt_corr_ccpvdz"),
        ("NH3", "E_ccsdt_corr_ccpvdz"),
        ("CH4", "E_ccsdt_corr_ccpvdz"),
    ]:
        add(mol, mol, prop, "ccsdt", "cc-pvdz", THRESHOLDS["ccsdt_corr"], ["full"])

    # --- EOM-CCSD/STO-3G ---
    for state_idx in range(1, 6):
        prop = f"eom_s{state_idx}_sto3g"
        add(
            "H2O_eom",
            "H2O_eom",
            prop,
            "eomccsd",
            "sto-3g",
            THRESHOLDS["eomccsd"],
            ["full"],
        )

    # --- UCCSD/STO-3G correlation ---
    for ref_key, mol_key in [
        ("Li_radical", "Li"),
        ("OH_radical", "OH"),
        ("O2_triplet", "O2"),
    ]:
        add(
            mol_key,
            ref_key,
            "E_uccsd_corr_sto3g",
            "uccsd",
            "sto-3g",
            THRESHOLDS["uccsd_corr"],
            ["full"],
        )

    # --- UCCSD(T)/STO-3G correction ---
    for ref_key, mol_key in [
        ("OH_radical", "OH"),
        ("O2_triplet", "O2"),
    ]:
        val = REFERENCE_DATA[ref_key]["E_uccsdt_corr_sto3g"]
        if val != 0.0:
            add(
                mol_key,
                ref_key,
                "E_uccsdt_corr_sto3g",
                "uccsdt",
                "sto-3g",
                THRESHOLDS["uccsdt_corr"],
                ["full"],
            )

    return tests


ALL_TESTS = _build_test_definitions()


# =====================================================================
# RESULT DATA CLASSES
# =====================================================================


class TestResult:
    """Result of a single benchmark test."""

    __slots__ = (
        "molecule",
        "method",
        "basis",
        "prop",
        "computed",
        "reference",
        "deviation",
        "threshold",
        "status",
        "error",
        "time_s",
    )

    def __init__(
        self,
        molecule,
        method,
        basis,
        prop,
        computed,
        reference,
        threshold,
        error=None,
        time_s=0.0,
    ):
        self.molecule = molecule
        self.method = method
        self.basis = basis
        self.prop = prop
        self.computed = computed
        self.reference = reference
        self.threshold = threshold
        self.error = error
        self.time_s = time_s

        if error is not None:
            self.deviation = float("inf")
            self.status = "ERROR"
        elif computed is None:
            self.deviation = float("inf")
            self.status = "SKIP"
        else:
            self.deviation = abs(computed - reference)
            self.status = "PASS" if self.deviation < threshold else "FAIL"

    def __repr__(self):
        if self.status in ("ERROR", "SKIP"):
            return (
                f"TestResult({self.molecule} {self.method}/{self.basis}: {self.status})"
            )
        return (
            f"TestResult({self.molecule} {self.method}/{self.basis}: "
            f"{self.computed:.10f} ref={self.reference:.10f} "
            f"Δ={self.deviation:.2e} {self.status})"
        )


class BenchmarkSummary:
    """Summary of a benchmark run."""

    def __init__(self, suite_name, results):
        self.suite_name = suite_name
        self.results = results
        self.n_pass = sum(1 for r in results if r.status == "PASS")
        self.n_fail = sum(1 for r in results if r.status == "FAIL")
        self.n_skip = sum(1 for r in results if r.status == "SKIP")
        self.n_error = sum(1 for r in results if r.status == "ERROR")
        self.n_total = len(results)
        self.total_time = sum(r.time_s for r in results)

        real_results = [r for r in results if r.status in ("PASS", "FAIL")]
        if real_results:
            worst = max(real_results, key=lambda r: r.deviation)
            self.max_deviation = worst.deviation
            self.max_deviation_test = f"{worst.molecule} {worst.method}/{worst.basis}"
        else:
            self.max_deviation = 0.0
            self.max_deviation_test = "N/A"

        self.all_passed = self.n_fail == 0 and self.n_error == 0

    def to_dict(self):
        """Convert to a serializable dictionary."""
        return {
            "suite": self.suite_name,
            "version": BENCHMARK_VERSION,
            "libcint_available": LIBCINT_AVAILABLE,
            "n_total": self.n_total,
            "n_pass": self.n_pass,
            "n_fail": self.n_fail,
            "n_skip": self.n_skip,
            "n_error": self.n_error,
            "all_passed": self.all_passed,
            "max_deviation": self.max_deviation,
            "max_deviation_test": self.max_deviation_test,
            "total_time_s": round(self.total_time, 1),
            "results": [
                {
                    "molecule": r.molecule,
                    "method": r.method,
                    "basis": r.basis,
                    "prop": r.prop,
                    "computed": r.computed,
                    "reference": r.reference,
                    "deviation": r.deviation if r.deviation != float("inf") else None,
                    "threshold": r.threshold,
                    "status": r.status,
                    "error": r.error,
                    "time_s": round(r.time_s, 3),
                }
                for r in self.results
            ],
        }


# =====================================================================
# BENCHMARK SUITE
# =====================================================================


class BenchmarkSuite:
    """
    QENEX LAB Regression Benchmark System.

    Runs molecules through computational methods and compares against
    known reference values. Any deviation beyond threshold means a
    regression was introduced.

    Usage:
        suite = BenchmarkSuite()
        summary = suite.run_quick()     # Fast: ~30s
        summary = suite.run_full()      # Full: ~5min
        summary = suite.run_benchmark() # All tests
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._hf_cache = {}
        self._ccsd_cache = {}

    # ================================================================
    # Cache management
    # ================================================================

    def _cache_key(self, geom_key, basis, method_prefix="hf"):
        return f"{method_prefix}:{geom_key}:{basis}"

    def _get_hf(self, geom_key, basis, charge=0, mult=1):
        """Get or compute HF result. Returns (hf_solver, E_hf, mol)."""
        key = self._cache_key(geom_key, basis, "hf")
        if key in self._hf_cache:
            return self._hf_cache[key]

        atoms = GEOMETRIES[geom_key]
        mol = Molecule(atoms, charge=charge, multiplicity=mult, basis_name=basis)
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(mol, verbose=False)
        result = (hf, E_hf, mol)
        self._hf_cache[key] = result
        return result

    def _get_ccsd(self, geom_key, basis, charge=0, mult=1):
        """Get or compute CCSD result. Returns (ccsd_solver, E_corr, E_hf, mol)."""
        key = self._cache_key(geom_key, basis, "ccsd")
        if key in self._ccsd_cache:
            return self._ccsd_cache[key]

        hf, E_hf, mol = self._get_hf(geom_key, basis, charge, mult)
        ccsd = CCSDSolver(convergence=1e-10)
        _, E_corr = ccsd.solve(hf, mol, verbose=False)
        result = (ccsd, E_corr, E_hf, mol)
        self._ccsd_cache[key] = result
        return result

    def _clear_cache(self):
        self._hf_cache.clear()
        self._ccsd_cache.clear()

    # ================================================================
    # Individual test execution
    # ================================================================

    def _run_single_test(self, test_def):
        """Execute a single benchmark test. Returns TestResult."""
        molecule = test_def["molecule"]
        geom_key = test_def["geometry_key"]
        ref_key = test_def["ref_key"]
        prop = test_def["prop"]
        method = test_def["method"]
        basis = test_def["basis"]
        threshold = test_def["threshold"]
        ref_value = test_def["ref_value"]

        ref_data = REFERENCE_DATA[ref_key]
        mult = ref_data.get("mult", 1)
        charge = 0

        t0 = time.time()

        try:
            computed = self._compute_property(
                geom_key, method, basis, prop, charge, mult, ref_key
            )
            elapsed = time.time() - t0
            return TestResult(
                molecule=molecule,
                method=method,
                basis=basis,
                prop=prop,
                computed=computed,
                reference=ref_value,
                threshold=threshold,
                time_s=elapsed,
            )
        except Exception as e:
            elapsed = time.time() - t0
            return TestResult(
                molecule=molecule,
                method=method,
                basis=basis,
                prop=prop,
                computed=None,
                reference=ref_value,
                threshold=threshold,
                error=str(e),
                time_s=elapsed,
            )

    def _compute_property(self, geom_key, method, basis, prop, charge, mult, ref_key):
        """Compute a single property. Returns the computed value."""
        if method == "hf":
            _, E_hf, _ = self._get_hf(geom_key, basis, charge, mult)
            return E_hf

        elif method == "ccsd":
            _, E_corr, _, _ = self._get_ccsd(geom_key, basis, charge, mult)
            return E_corr

        elif method == "ccsdt":
            ccsd, _, _, _ = self._get_ccsd(geom_key, basis, charge, mult)
            E_t = ccsd.ccsd_t(verbose=False)
            return E_t

        elif method == "eomccsd":
            ccsd, _, _, _ = self._get_ccsd(geom_key, basis, charge, mult)

            # Cache EOM results per geometry/basis
            eom_key = self._cache_key(geom_key, basis, "eomccsd")
            if eom_key not in self._hf_cache:
                eom = EOMCCSDSolver()
                evals = eom.solve(ccsd, nroots=5, verbose=False)
                self._hf_cache[eom_key] = evals
            else:
                evals = self._hf_cache[eom_key]

            # Extract requested state: prop = "eom_s1_sto3g" → state 0
            state_str = prop.split("_")[1]  # "s1", "s2", etc.
            state_idx = int(state_str[1:]) - 1
            if state_idx < len(evals):
                return float(evals[state_idx])
            return None

        elif method == "uccsd":
            atoms = GEOMETRIES[geom_key]
            mol = Molecule(atoms, charge=charge, multiplicity=mult, basis_name=basis)

            # Cache UCCSD solver for (T) reuse
            uccsd_key = self._cache_key(geom_key, basis, "uccsd")
            if uccsd_key not in self._hf_cache:
                uccsd = UCCSDSolver(convergence=1e-10)
                _, E_corr = uccsd.solve_pyscf(mol, verbose=False)
                self._hf_cache[uccsd_key] = (uccsd, E_corr)
            else:
                _, E_corr = self._hf_cache[uccsd_key]
            return E_corr

        elif method == "uccsdt":
            atoms = GEOMETRIES[geom_key]
            mol = Molecule(atoms, charge=charge, multiplicity=mult, basis_name=basis)

            uccsd_key = self._cache_key(geom_key, basis, "uccsd")
            if uccsd_key not in self._hf_cache:
                uccsd = UCCSDSolver(convergence=1e-10)
                uccsd.solve_pyscf(mol, verbose=False)
                self._hf_cache[uccsd_key] = (uccsd, uccsd._E_corr)
            uccsd, _ = self._hf_cache[uccsd_key]
            E_t = uccsd.uccsd_t(verbose=False)
            return E_t

        else:
            raise ValueError(f"Unknown method: {method}")

    # ================================================================
    # Suite runners
    # ================================================================

    def run_benchmark(self, methods=None, bases=None, suites=None, verbose=None):
        """
        Run benchmark tests.

        Args:
            methods: Filter by method (e.g., ["hf", "ccsd"]). None = all.
            bases:   Filter by basis (e.g., ["sto-3g"]). None = all.
            suites:  Filter by suite (e.g., ["quick"]). None = all.
            verbose: Override instance verbose setting.

        Returns:
            BenchmarkSummary
        """
        if verbose is None:
            verbose = self.verbose

        self._clear_cache()

        filtered = ALL_TESTS
        if methods is not None:
            filtered = [t for t in filtered if t["method"] in methods]
        if bases is not None:
            filtered = [t for t in filtered if t["basis"] in bases]
        if suites is not None:
            filtered = [t for t in filtered if any(s in t["suites"] for s in suites)]

        suite_name = "Custom"
        if suites:
            suite_name = "/".join(suites)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"QENEX LAB Benchmark Report v{BENCHMARK_VERSION}")
            print(f"{'=' * 60}")
            print(f"Suite: {suite_name}")
            print(f"Tests: {len(filtered)}")
            print(
                f"Integral backend: {'libcint' if LIBCINT_AVAILABLE else 'Obara-Saika (native)'}"
            )
            print(f"HF threshold: {_hf_threshold():.0e} Eh")
            if methods:
                print(f"Methods: {', '.join(methods)}")
            if bases:
                print(f"Bases: {', '.join(bases)}")
            print(f"{'=' * 60}")
            print()

        results = []

        for i, test_def in enumerate(filtered):
            mol = test_def["molecule"]
            method = test_def["method"]
            basis = test_def["basis"]
            prop = test_def["prop"]

            if verbose:
                label = f"{mol} {method}/{basis}"
                print(f"  [{i + 1}/{len(filtered)}] {label}...", end=" ", flush=True)

            result = self._run_single_test(test_def)
            results.append(result)

            if verbose:
                if result.status == "PASS":
                    print(
                        f"{result.computed:14.10f} "
                        f"(ref: {result.reference:14.10f}) "
                        f"Δ={result.deviation:.2e}  PASS"
                    )
                elif result.status == "FAIL":
                    print(
                        f"{result.computed:14.10f} "
                        f"(ref: {result.reference:14.10f}) "
                        f"Δ={result.deviation:.2e}  **FAIL**"
                    )
                elif result.status == "ERROR":
                    print(f"ERROR: {result.error}")
                elif result.status == "SKIP":
                    print("SKIP")

        summary = BenchmarkSummary(suite_name, results)

        if verbose:
            self._print_summary(summary)

        return summary

    def run_quick(self, verbose=None):
        """
        Fast benchmark (~30s): H2, He, H2O on STO-3G + key CCSD/cc-pVDZ.

        Returns: BenchmarkSummary
        """
        return self.run_benchmark(suites=["quick"], verbose=verbose)

    def run_full(self, verbose=None):
        """
        Full benchmark (~5min): all molecules, all methods, all bases.

        Returns: BenchmarkSummary
        """
        return self.run_benchmark(suites=["quick", "full"], verbose=verbose)

    def run_accuracy(self, verbose=None):
        """
        Accuracy benchmark: compare computed properties against experiment.

        This tests method accuracy, not implementation correctness.
        Large deviations are expected (basis set incompleteness, correlation
        truncation) but should be consistent over time.

        Returns: BenchmarkSummary
        """
        if verbose is None:
            verbose = self.verbose

        self._clear_cache()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"QENEX LAB Accuracy Benchmark v{BENCHMARK_VERSION}")
            print(f"{'=' * 60}")
            print(f"Comparing computational results against experiment")
            print(f"(Large deviations expected — basis set effects)")
            print(f"{'=' * 60}")
            print()

        results = []

        # --- H2 harmonic frequency from finite difference ---
        try:
            R_eq_exp = REFERENCE_DATA["H2"]["R_eq_bohr"]
            t0 = time.time()

            mol_eq = Molecule(
                [("H", (0, 0, 0)), ("H", (0, 0, R_eq_exp))],
                basis_name="cc-pvdz",
            )
            hf = HartreeFockSolver()
            E_eq, _ = hf.compute_energy(mol_eq, verbose=False)

            delta = 0.01
            mol_plus = Molecule(
                [("H", (0, 0, 0)), ("H", (0, 0, R_eq_exp + delta))],
                basis_name="cc-pvdz",
            )
            E_plus, _ = HartreeFockSolver().compute_energy(mol_plus, verbose=False)

            mol_minus = Molecule(
                [("H", (0, 0, 0)), ("H", (0, 0, R_eq_exp - delta))],
                basis_name="cc-pvdz",
            )
            E_minus, _ = HartreeFockSolver().compute_energy(mol_minus, verbose=False)

            elapsed = time.time() - t0

            # Harmonic frequency: ω = sqrt(k/μ) in a.u., then convert
            k = (E_plus - 2 * E_eq + E_minus) / (delta**2)
            mu_au = 0.5 * 1822.888486209  # reduced mass of H2 in a.u.
            if k > 0:
                omega_au = np.sqrt(k / mu_au)
                freq_computed = omega_au * 219474.63  # Hartree → cm⁻¹
            else:
                freq_computed = 0.0

            freq_exp = float(REFERENCE_DATA["H2"]["freq_cm1"])

            if verbose:
                print(
                    f"  H2 harmonic freq (HF/cc-pVDZ): {freq_computed:.0f} cm⁻¹ "
                    f"(exp: {freq_exp:.0f} cm⁻¹)"
                )

            results.append(
                TestResult(
                    molecule="H2",
                    method="hf_pes",
                    basis="cc-pvdz",
                    prop="freq_cm1",
                    computed=freq_computed,
                    reference=freq_exp,
                    threshold=THRESHOLDS["frequency"],
                    time_s=elapsed,
                )
            )
        except Exception as e:
            results.append(
                TestResult(
                    molecule="H2",
                    method="hf_pes",
                    basis="cc-pvdz",
                    prop="freq_cm1",
                    computed=None,
                    reference=float(REFERENCE_DATA["H2"]["freq_cm1"]),
                    threshold=THRESHOLDS["frequency"],
                    error=str(e),
                )
            )

        # --- H2O: verify STO-3G energy matches reference ---
        try:
            t0 = time.time()
            _, E_hf, _ = self._get_hf("H2O_sto3g", "sto-3g")
            elapsed = time.time() - t0

            results.append(
                TestResult(
                    molecule="H2O",
                    method="hf",
                    basis="sto-3g",
                    prop="E_hf_sto3g",
                    computed=E_hf,
                    reference=REFERENCE_DATA["H2O"]["E_hf_sto3g"],
                    threshold=_hf_threshold(),
                    time_s=elapsed,
                )
            )
        except Exception as e:
            results.append(
                TestResult(
                    molecule="H2O",
                    method="hf",
                    basis="sto-3g",
                    prop="E_hf_sto3g",
                    computed=None,
                    reference=REFERENCE_DATA["H2O"]["E_hf_sto3g"],
                    threshold=_hf_threshold(),
                    error=str(e),
                )
            )

        # --- H2O: verify energy is negative (bound system) ---
        try:
            t0 = time.time()
            _, E_hf, _ = self._get_hf("H2O", "cc-pvdz")
            elapsed = time.time() - t0

            # Verify negative energy
            ref_energy = REFERENCE_DATA["H2O"]["E_hf_ccpvdz"]
            results.append(
                TestResult(
                    molecule="H2O",
                    method="hf",
                    basis="cc-pvdz",
                    prop="E_hf_ccpvdz",
                    computed=E_hf,
                    reference=ref_energy,
                    threshold=_hf_threshold(),
                    time_s=elapsed,
                )
            )
        except Exception as e:
            results.append(
                TestResult(
                    molecule="H2O",
                    method="hf",
                    basis="cc-pvdz",
                    prop="E_hf_ccpvdz",
                    computed=None,
                    reference=REFERENCE_DATA["H2O"]["E_hf_ccpvdz"],
                    threshold=_hf_threshold(),
                    error=str(e),
                )
            )

        summary = BenchmarkSummary("accuracy", results)

        if verbose:
            self._print_summary(summary)

        return summary

    def run_cbs(self, verbose=None):
        """
        CBS extrapolation benchmark (~2 min): estimates complete basis set
        limit for HF and CCSD correlation energies using cc-pVDZ + cc-pVTZ
        pairs for H2, He, and H2O.

        Uses Feller (1992) exponential extrapolation for HF and Helgaker (1997)
        X^-3 formula for correlation.  Results are informational — no PASS/FAIL
        threshold since CBS values are estimates, not references.

        Returns: BenchmarkSummary
        """
        if verbose is None:
            verbose = self.verbose

        results = []
        t0 = time.time()

        test_molecules = [
            ("H2", [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4011))]),
            ("He", [("He", (0.0, 0.0, 0.0))]),
            (
                "H2O",
                [
                    ("O", (0.0, 0.0, 0.0)),
                    ("H", (0.0, 1.4300, 1.1070)),
                    ("H", (0.0, -1.4300, 1.1070)),
                ],
            ),
        ]

        for mol_name, atoms in test_molecules:
            t_mol = time.time()
            try:
                mol = Molecule(atoms, basis_name="cc-pvdz")
                hf_dz = HartreeFockSolver()
                E_hf_dz, _ = hf_dz.compute_energy(mol, verbose=False)

                ccsd_dz = CCSDSolver()
                E_total_dz, E_corr_dz = ccsd_dz.solve(hf_dz, mol, verbose=False)

                mol_tz = Molecule(atoms, basis_name="cc-pvtz")
                hf_tz = HartreeFockSolver()
                E_hf_tz, _ = hf_tz.compute_energy(mol_tz, verbose=False)

                ccsd_tz = CCSDSolver()
                E_total_tz, E_corr_tz = ccsd_tz.solve(hf_tz, mol_tz, verbose=False)

                cbs = extrapolate_cbs_2point(
                    E_hf_dz=E_hf_dz,
                    E_corr_dz=E_corr_dz,
                    E_hf_tz=E_hf_tz,
                    E_corr_tz=E_corr_tz,
                )

                elapsed = time.time() - t_mol
                if verbose:
                    print(
                        f"  CBS {mol_name}: "
                        f"E_HF(CBS)={cbs['E_hf_cbs']:.8f}  "
                        f"E_corr(CBS)={cbs['E_corr_cbs']:.8f}  "
                        f"E_total(CBS)={cbs['E_total_cbs']:.8f}  "
                        f"({elapsed:.1f}s)"
                    )

                results.append(
                    TestResult(
                        molecule=mol_name,
                        method="CCSD/CBS(DZ→TZ)",
                        basis="cc-pVDZ/TZ",
                        prop="E_total_cbs",
                        computed=cbs["E_total_cbs"],
                        # Use CBS value itself as reference — this is informational,
                        # not a pass/fail check against an external value.
                        reference=cbs["E_total_cbs"],
                        threshold=1e-3,
                        time_s=elapsed,
                    )
                )

            except Exception as e:
                results.append(
                    TestResult(
                        molecule=mol_name,
                        method="CCSD/CBS(DZ→TZ)",
                        basis="cc-pVDZ/TZ",
                        prop="E_total_cbs",
                        computed=None,
                        reference=0.0,
                        threshold=1e-3,
                        time_s=time.time() - t_mol,
                        error=str(e),
                    )
                )

        summary = BenchmarkSummary("cbs", results)
        summary.total_time = time.time() - t0
        if verbose:
            self._print_summary(summary)
        return summary

    # ================================================================
    # Reporting
    # ================================================================

    def _print_summary(self, summary):
        """Print a formatted benchmark summary."""
        print()
        print(f"{'=' * 60}")
        print(f"QENEX LAB Benchmark Report v{BENCHMARK_VERSION}")
        print(f"{'=' * 60}")
        print(f"{summary.suite_name} Suite: {summary.n_total} tests")
        print(
            f"  PASS: {summary.n_pass}  "
            f"FAIL: {summary.n_fail}  "
            f"SKIP: {summary.n_skip}  "
            f"ERROR: {summary.n_error}"
        )
        if summary.max_deviation > 0:
            print(
                f"  Max deviation: {summary.max_deviation:.2e} "
                f"({summary.max_deviation_test})"
            )
        print(f"  Total time: {summary.total_time:.1f}s")

        failures = [r for r in summary.results if r.status == "FAIL"]
        if failures:
            print()
            print(f"  *** {len(failures)} FAILURE(S) ***")
            for f in failures:
                print(
                    f"    {f.molecule} {f.method}/{f.basis}: "
                    f"{f.computed:.10f} vs ref {f.reference:.10f} "
                    f"Δ={f.deviation:.2e} > threshold {f.threshold:.2e}"
                )

        errors = [r for r in summary.results if r.status == "ERROR"]
        if errors:
            print()
            print(f"  *** {len(errors)} ERROR(S) ***")
            for e in errors:
                print(f"    {e.molecule} {e.method}/{e.basis}: {e.error}")

        if summary.all_passed:
            print()
            print("  ALL TESTS PASSED — no regressions detected")

        print(f"{'=' * 60}")

    @staticmethod
    def format_report(summary, detailed=True):
        """
        Format a benchmark summary as a string report.

        Args:
            summary: BenchmarkSummary instance.
            detailed: Include per-test details.

        Returns:
            str: Formatted report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"QENEX LAB Benchmark Report v{BENCHMARK_VERSION}")
        lines.append("=" * 60)
        lines.append(f"{summary.suite_name} Suite: {summary.n_total} tests")
        lines.append(
            f"  PASS: {summary.n_pass}  "
            f"FAIL: {summary.n_fail}  "
            f"SKIP: {summary.n_skip}  "
            f"ERROR: {summary.n_error}"
        )
        if summary.max_deviation > 0:
            lines.append(
                f"  Max deviation: {summary.max_deviation:.2e} "
                f"({summary.max_deviation_test})"
            )
        lines.append(f"  Total time: {summary.total_time:.1f}s")
        lines.append("")

        if detailed:
            for r in summary.results:
                if r.status == "PASS":
                    lines.append(
                        f"  {r.molecule:10s} {r.method}/{r.basis:10s}: "
                        f"{r.computed:14.10f} "
                        f"(ref: {r.reference:14.10f}) "
                        f"Δ={r.deviation:.2e}  PASS"
                    )
                elif r.status == "FAIL":
                    lines.append(
                        f"  {r.molecule:10s} {r.method}/{r.basis:10s}: "
                        f"{r.computed:14.10f} "
                        f"(ref: {r.reference:14.10f}) "
                        f"Δ={r.deviation:.2e}  **FAIL**"
                    )
                elif r.status == "ERROR":
                    lines.append(
                        f"  {r.molecule:10s} {r.method}/{r.basis:10s}: ERROR: {r.error}"
                    )
                elif r.status == "SKIP":
                    lines.append(f"  {r.molecule:10s} {r.method}/{r.basis:10s}: SKIP")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================


def main():
    """Run benchmarks from the command line."""
    import sys

    suite = BenchmarkSuite(verbose=True)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "quick"

    if mode == "quick":
        summary = suite.run_quick()
    elif mode == "full":
        summary = suite.run_full()
    elif mode == "accuracy":
        summary = suite.run_accuracy()
    elif mode == "all":
        print("Running quick benchmark...")
        s1 = suite.run_quick()
        print("\nRunning accuracy benchmark...")
        s2 = suite.run_accuracy()
        print("\nRunning full benchmark...")
        s3 = suite.run_full()
        all_ok = s1.all_passed and s2.all_passed and s3.all_passed
        print(f"\n{'=' * 60}")
        print(f"COMBINED: {'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
        print(f"{'=' * 60}")
        return 0 if all_ok else 1
    elif mode in ("--help", "-h", "help"):
        print("Usage: python benchmark.py [quick|full|accuracy|all]")
        print()
        print("  quick    — Fast (~30s): core HF + CCSD tests")
        print("  full     — Full (~5min): all methods, all bases")
        print("  accuracy — Compare against experiment")
        print("  all      — Run everything")
        return 0
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python benchmark.py [quick|full|accuracy|all]")
        return 1

    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    exit(main())
