"""
6-31G* (6-31G(d)) Basis Set Parameters for Z=1-10
===================================================
Pople split-valence basis set with d-type polarization functions on
heavy atoms (Li-Ne). The most widely used basis set in organic chemistry.

Structure:
  H, He:   Inner 1s (3 primitives) + Outer 1s (1 primitive) = 2 s-functions
  Li-Ne:   Core 1s (6 primitives) + Valence inner 2sp (3 prims, shared exponents)
           + Valence outer 2sp (1 prim) + 1 d-polarization function (6 Cartesian)

Data extracted from standard published coefficients to ensure exact integral
agreement with PySCF for cross-validation.

Original references:
  - H, He:   Hehre, Ditchfield, Pople, JCP 56, 2257 (1972)
  - Li-Ne:   Hehre, Ditchfield, Pople, JCP 56, 2257 (1972)
  - d-pol:   Hariharan, Pople, Theor. Chim. Acta 28, 213 (1973)

Shell encoding matches basis_cc_pvdz.py convention:
  Each shell is a dict with:
    "angular_momentum": int (0=s, 1=p, 2=d)
    "exponents": list of float
    "coefficients": list of list of float
    "spherical": bool (True for d-shells: 6 Cartesian components used)

IMPORTANT: Uses segmented/PySCF contraction convention.
"""

import numpy as np

__all__ = ["BASIS_631GS", "get_basis_info_631gs", "validate_basis_631gs"]

# ============================================================
#  6-31G* Basis Set Data
# ============================================================
# Format: For each element, we store atomic number Z and a list of shells.
#
# For H/He (row 1): No polarization in 6-31G*, only split-valence s-functions.
#   Shell 1: 3-primitive contracted s (inner)
#   Shell 2: 1-primitive uncontracted s (outer)
#
# For Li-Ne (row 2): Core + split valence + d-polarization
#   Shell 1: 6-primitive contracted s (core 1s)
#   Shell 2: 3-primitive sp (valence inner 2s) — s-coefficients
#   Shell 3: 3-primitive sp (valence inner 2p) — p-coefficients (shared exponents)
#   Shell 4: 1-primitive sp (valence outer 2s) — s uncontracted
#   Shell 5: 1-primitive sp (valence outer 2p) — p uncontracted
#   Shell 6: 1-primitive d (polarization)
#
# The sp-shells share exponents but have different contraction coefficients
# for the s and p components. We store them as separate s and p shells.

BASIS_631GS = {
    # ================================================================
    # Hydrogen  (Z=1)
    # ================================================================
    "H": {
        "Z": 1,
        "shells": [
            # Inner 1s: 3 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [18.7311370, 2.8253937, 0.6401217],
                "coefficients": [[0.03349460, 0.23472695, 0.81375733]],
                "spherical": False,
            },
            # Outer 1s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.1612778],
                "coefficients": [[1.0]],
                "spherical": False,
            },
        ],
    },
    # ================================================================
    # Helium  (Z=2)
    # ================================================================
    "He": {
        "Z": 2,
        "shells": [
            # Inner 1s: 3 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [38.4216340, 5.7780300, 1.2417740],
                "coefficients": [[0.02376605, 0.15467889, 0.46963000]],
                "spherical": False,
            },
            # Outer 1s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.2979640],
                "coefficients": [[1.0]],
                "spherical": False,
            },
        ],
    },
    # ================================================================
    # Lithium  (Z=3)
    # ================================================================
    "Li": {
        "Z": 3,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    642.4189200,
                    96.7985150,
                    22.0911210,
                    6.2010703,
                    1.9351177,
                    0.6367358,
                ],
                "coefficients": [
                    [
                        0.0021426,
                        0.0162089,
                        0.0773156,
                        0.2457860,
                        0.4700942,
                        0.3454710,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [2.3249184, 0.6324306, 0.0790534],
                "coefficients": [[-0.0350917, -0.1912328, 1.0839878]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [2.3249184, 0.6324306, 0.0790534],
                "coefficients": [[0.0089415, 0.1410095, 0.9453637]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.0359620],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.0359620],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization
            {
                "angular_momentum": 2,
                "exponents": [0.2000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Beryllium  (Z=4)
    # ================================================================
    "Be": {
        "Z": 4,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    1264.5857000,
                    189.9368100,
                    43.1590890,
                    12.0986630,
                    3.8063232,
                    1.2728903,
                ],
                "coefficients": [
                    [
                        0.0019448,
                        0.0148351,
                        0.0720906,
                        0.2371542,
                        0.4691987,
                        0.3565202,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [3.1964631, 0.7478133, 0.2199663],
                "coefficients": [[-0.1126487, -0.2295064, 1.1869167]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [3.1964631, 0.7478133, 0.2199663],
                "coefficients": [[0.0559802, 0.2615506, 0.7939723]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.0823099],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.0823099],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization
            {
                "angular_momentum": 2,
                "exponents": [0.4000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Boron  (Z=5)
    # ================================================================
    "B": {
        "Z": 5,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    2068.8823000,
                    310.6495700,
                    70.6830330,
                    19.8610800,
                    6.2993048,
                    2.1270270,
                ],
                "coefficients": [
                    [
                        0.0018663,
                        0.0142515,
                        0.0695516,
                        0.2325729,
                        0.4670787,
                        0.3634314,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [4.7279710, 1.1903377, 0.3594117],
                "coefficients": [[-0.1303938, -0.1307889, 1.1309444]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [4.7279710, 1.1903377, 0.3594117],
                "coefficients": [[0.0745976, 0.3078467, 0.7434568]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.1267512],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.1267512],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent)
            {
                "angular_momentum": 2,
                "exponents": [0.6000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Carbon  (Z=6)
    # ================================================================
    "C": {
        "Z": 6,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    3047.5249000,
                    457.3695100,
                    103.9486900,
                    29.2101550,
                    9.2866630,
                    3.1639270,
                ],
                "coefficients": [
                    [
                        0.0018347,
                        0.0140373,
                        0.0688426,
                        0.2321844,
                        0.4679413,
                        0.3623120,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [7.8682724, 1.8812885, 0.5442493],
                "coefficients": [[-0.1193324, -0.1608542, 1.1434564]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [7.8682724, 1.8812885, 0.5442493],
                "coefficients": [[0.0689991, 0.3164240, 0.7443083]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.1687144],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.1687144],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent = 0.8)
            {
                "angular_momentum": 2,
                "exponents": [0.8000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Nitrogen  (Z=7)
    # ================================================================
    "N": {
        "Z": 7,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    4173.5110000,
                    627.4579000,
                    142.9021000,
                    40.2343300,
                    12.8202100,
                    4.3904370,
                ],
                "coefficients": [
                    [
                        0.0018348,
                        0.0139950,
                        0.0685870,
                        0.2322410,
                        0.4690700,
                        0.3604550,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [11.6263580, 2.7162800, 0.7722180],
                "coefficients": [[-0.1149612, -0.1593168, 1.1354180]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [11.6263580, 2.7162800, 0.7722180],
                "coefficients": [[0.0675797, 0.3239072, 0.7408951]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.2120313],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.2120313],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent = 0.8)
            {
                "angular_momentum": 2,
                "exponents": [0.8000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Oxygen  (Z=8)
    # ================================================================
    "O": {
        "Z": 8,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    5484.6717000,
                    825.2349500,
                    188.0469600,
                    52.9645000,
                    16.8975700,
                    5.7996353,
                ],
                "coefficients": [
                    [
                        0.0018311,
                        0.0139501,
                        0.0684451,
                        0.2327143,
                        0.4701930,
                        0.3585209,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [15.5396160, 3.5999336, 1.0137618],
                "coefficients": [[-0.1107775, -0.1480263, 1.1307670]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [15.5396160, 3.5999336, 1.0137618],
                "coefficients": [[0.0708743, 0.3397528, 0.7271586]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.2700058],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.2700058],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent = 0.8)
            {
                "angular_momentum": 2,
                "exponents": [0.8000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Fluorine  (Z=9)
    # ================================================================
    "F": {
        "Z": 9,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    7001.7130900,
                    1051.3660900,
                    239.2856900,
                    67.3974530,
                    21.5199573,
                    7.4031014,
                ],
                "coefficients": [
                    [
                        0.0018196,
                        0.0139160,
                        0.0684053,
                        0.2331860,
                        0.4712680,
                        0.3566186,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [20.8479528, 4.8083068, 1.3440700],
                "coefficients": [[-0.1085070, -0.1464080, 1.1286390]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [20.8479528, 4.8083068, 1.3440700],
                "coefficients": [[0.0716200, 0.3459451, 0.7224699]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.3581514],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.3581514],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent = 0.8)
            {
                "angular_momentum": 2,
                "exponents": [0.8000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    # ================================================================
    # Neon  (Z=10)
    # ================================================================
    "Ne": {
        "Z": 10,
        "shells": [
            # Core 1s: 6 primitives contracted
            {
                "angular_momentum": 0,
                "exponents": [
                    8425.8515300,
                    1268.5194000,
                    289.6214100,
                    81.8590040,
                    26.2515079,
                    9.0947205,
                ],
                "coefficients": [
                    [
                        0.0018843,
                        0.0143368,
                        0.0701096,
                        0.2373317,
                        0.4730071,
                        0.3484578,
                    ]
                ],
                "spherical": False,
            },
            # Valence inner 2s: 3 primitives (sp shell, s-coefficients)
            {
                "angular_momentum": 0,
                "exponents": [26.5321310, 6.1017550, 1.6962715],
                "coefficients": [[-0.1071830, -0.1461200, 1.1277740]],
                "spherical": False,
            },
            # Valence inner 2p: 3 primitives (sp shell, p-coefficients)
            {
                "angular_momentum": 1,
                "exponents": [26.5321310, 6.1017550, 1.6962715],
                "coefficients": [[0.0719095, 0.3495139, 0.7199398]],
                "spherical": False,
            },
            # Valence outer 2s: 1 primitive uncontracted
            {
                "angular_momentum": 0,
                "exponents": [0.4458187],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # Valence outer 2p: 1 primitive uncontracted
            {
                "angular_momentum": 1,
                "exponents": [0.4458187],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # d-polarization (Hariharan & Pople exponent = 0.8)
            {
                "angular_momentum": 2,
                "exponents": [0.8000000],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
}


# ============================================================
#  Utility Functions
# ============================================================

SHELL_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}


def get_basis_info_631gs(element):
    """
    Get basis set information for a given element in 6-31G*.

    Args:
        element: Chemical element symbol (e.g. 'C', 'O')

    Returns:
        dict with element, Z, n_basis_functions, and shells_summary.

    Raises:
        ValueError: If element not in 6-31G* basis set.
    """
    if element not in BASIS_631GS:
        raise ValueError(
            f"Element '{element}' not in 6-31G*. "
            f"Available: {sorted(BASIS_631GS.keys())}"
        )
    data = BASIS_631GS[element]
    n_basis = 0
    for shell in data["shells"]:
        L = shell["angular_momentum"]
        n_cont = len(shell["coefficients"])
        n_comp = {0: 1, 1: 3, 2: 6, 3: 10}.get(L, 2 * L + 1)
        n_basis += n_cont * n_comp
    return {
        "element": element,
        "Z": data["Z"],
        "n_basis_functions": n_basis,
        "shells_summary": "6-31G* (Pople split-valence + d-polarization)",
    }


def validate_basis_631gs(element):
    """
    Validate basis set data for a given element in 6-31G*.

    Checks:
      - Element exists in basis data
      - All exponents are positive
      - All coefficient lists are non-empty
      - Correct number of shells for the element type

    Args:
        element: Chemical element symbol.

    Returns:
        dict with element, Z, total_basis_functions, and checks list.
    """
    if element not in BASIS_631GS:
        return {"element": element, "checks": [f"FAIL: not in 6-31G*"]}

    checks = []
    data = BASIS_631GS[element]
    Z = data["Z"]

    # Check exponents are positive
    for i, shell in enumerate(data["shells"]):
        for exp in shell["exponents"]:
            if exp <= 0:
                checks.append(f"FAIL: Shell {i} has non-positive exponent {exp}")

    # Check coefficient lists are non-empty
    for i, shell in enumerate(data["shells"]):
        for j, coeff_vec in enumerate(shell["coefficients"]):
            if len(coeff_vec) == 0:
                checks.append(
                    f"FAIL: Shell {i}, contraction {j} has empty coefficients"
                )
            if len(coeff_vec) != len(shell["exponents"]):
                checks.append(
                    f"FAIL: Shell {i}, contraction {j}: "
                    f"{len(coeff_vec)} coefficients != "
                    f"{len(shell['exponents'])} exponents"
                )

    # Check expected shell count
    if Z <= 2:
        expected_shells = 2  # H, He: 2 s-shells
    else:
        expected_shells = 6  # Li-Ne: core_s + val_inner_s + val_inner_p + val_outer_s + val_outer_p + d
    actual_shells = len(data["shells"])
    if actual_shells != expected_shells:
        checks.append(
            f"WARN: Expected {expected_shells} shells for Z={Z}, found {actual_shells}"
        )

    if not checks:
        checks = ["ALL CHECKS PASSED"]

    n_bf = sum(
        len(s["coefficients"]) * {0: 1, 1: 3, 2: 6}.get(s["angular_momentum"], 1)
        for s in data["shells"]
    )

    return {
        "element": element,
        "Z": Z,
        "total_basis_functions": n_bf,
        "checks": checks,
    }


def validate_all_631gs():
    """Validate all elements in the 6-31G* basis set."""
    return all(
        "ALL CHECKS PASSED" in validate_basis_631gs(e)["checks"] for e in BASIS_631GS
    )
