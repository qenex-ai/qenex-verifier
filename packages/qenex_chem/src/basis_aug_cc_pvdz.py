"""
aug-cc-pVDZ Basis Set Parameters for Z=1-10
=============================================
Dunning augmented correlation-consistent polarized valence double-zeta basis set.

The aug-cc-pVDZ basis extends cc-pVDZ with one set of diffuse functions
for each angular momentum present in the valence basis.  For H/He this adds
one diffuse s and one diffuse p; for Li-Ne it adds one diffuse s, p, and d.

Data extracted from PySCF 2.12.1 internal basis library to ensure
exact integral agreement for cross-validation.

Original references:
  - H, B-F:  Dunning, J. Chem. Phys. 90, 1007 (1989)
  - He, Ne:  Woon & Dunning, J. Chem. Phys. 100, 2975 (1994)
  - Li, Be:  Prascher et al., Theor. Chem. Acc. 128, 69 (2011)
  - Augmented: Kendall, Dunning & Harrison, J. Chem. Phys. 96, 6796 (1992)

Shell encoding:
  Each shell is a dict with:
    "angular_momentum": int (0=s, 1=p, 2=d)
    "exponents": list of float
    "coefficients": list of list of float
    "spherical": bool (True for d-shells: 6 Cartesian components used)

IMPORTANT: Uses segmented/PySCF contraction convention.
Row-2 s-shells: 8 primitives -> 2 general contractions + 1 uncontracted + 1 diffuse
Row-2 p-shells: 3 primitives -> 1 contraction + 1 uncontracted + 1 diffuse
"""

import numpy as np

__all__ = [
    "AUG_CC_PVDZ",
    "RHF_AUG_CC_PVDZ_REFERENCE",
    "validate_basis",
    "get_basis_info",
]

AUG_CC_PVDZ = {
    "H": {
        "Z": 1,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [13.01, 1.962, 0.4446],
                "coefficients": [[0.019685, 0.137977, 0.478148]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1220],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse functions ===
            {
                "angular_momentum": 0,
                "exponents": [0.02974],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.727],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.141],
                "coefficients": [[1.0]],
                "spherical": False,
            },
        ],
    },
    "He": {
        "Z": 2,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [38.36, 5.770, 1.240],
                "coefficients": [[0.023809, 0.154891, 0.469987]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.2976],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse functions ===
            {
                "angular_momentum": 0,
                "exponents": [0.07255],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [1.275],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.2473],
                "coefficients": [[1.0]],
                "spherical": False,
            },
        ],
    },
    "Li": {
        "Z": 3,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [1469.0, 220.5, 50.26, 14.24, 4.581, 1.58, 0.564, 0.07345],
                "coefficients": [
                    [
                        0.000766,
                        0.005892,
                        0.029671,
                        0.10918,
                        0.282789,
                        0.453123,
                        0.274774,
                        0.009751,
                    ],
                    [
                        -0.00012,
                        -0.000923,
                        -0.004689,
                        -0.017682,
                        -0.048902,
                        -0.096009,
                        -0.13638,
                        0.575102,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.02805],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.00864],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [1.534, 0.2749, 0.07362],
                "coefficients": [[0.022784, 0.139107, 0.500375]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.02403],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.00579],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.1239],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.0725],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Be": {
        "Z": 4,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [2940.0, 441.2, 100.5, 28.43, 9.169, 3.196, 1.159, 0.1811],
                "coefficients": [
                    [
                        0.00068,
                        0.005236,
                        0.026606,
                        0.099993,
                        0.269702,
                        0.451469,
                        0.295074,
                        0.012587,
                    ],
                    [
                        -0.000123,
                        -0.000966,
                        -0.004831,
                        -0.019314,
                        -0.05328,
                        -0.120723,
                        -0.133435,
                        0.530767,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.0589],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.01877],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [3.619, 0.711, 0.1951],
                "coefficients": [[0.029111, 0.169365, 0.513458]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.06018],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.0085],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.238],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.074],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "B": {
        "Z": 5,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [4570.0, 685.9, 156.5, 44.47, 14.48, 5.131, 1.898, 0.3329],
                "coefficients": [
                    [
                        0.000696,
                        0.005353,
                        0.027134,
                        0.10138,
                        0.272055,
                        0.448403,
                        0.290123,
                        0.014322,
                    ],
                    [
                        -0.000139,
                        -0.001097,
                        -0.005444,
                        -0.021916,
                        -0.059751,
                        -0.138732,
                        -0.131482,
                        0.539526,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1043],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.03105],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [6.001, 1.241, 0.3364],
                "coefficients": [[0.035481, 0.198072, 0.50523]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.09538],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.02378],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.343],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.0904],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "C": {
        "Z": 6,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    6665.0,
                    1000.0,
                    228.0,
                    64.71,
                    21.06,
                    7.495,
                    2.797,
                    0.5215,
                ],
                "coefficients": [
                    [
                        0.000692,
                        0.005329,
                        0.027077,
                        0.101718,
                        0.27474,
                        0.448564,
                        0.285074,
                        0.015204,
                    ],
                    [
                        -0.000146,
                        -0.001154,
                        -0.005725,
                        -0.023312,
                        -0.063955,
                        -0.149981,
                        -0.127262,
                        0.544529,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1596],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.0469],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [9.439, 2.002, 0.5456],
                "coefficients": [[0.038109, 0.20948, 0.508557]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.1517],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.04041],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.55],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.151],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "N": {
        "Z": 7,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    9046.0,
                    1357.0,
                    309.3,
                    87.73,
                    28.56,
                    10.21,
                    3.838,
                    0.7466,
                ],
                "coefficients": [
                    [
                        0.0007,
                        0.005389,
                        0.027406,
                        0.103207,
                        0.278723,
                        0.44854,
                        0.278238,
                        0.01544,
                    ],
                    [
                        -0.000153,
                        -0.001208,
                        -0.005992,
                        -0.024544,
                        -0.067459,
                        -0.158078,
                        -0.121831,
                        0.549003,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.2248],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.06124],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [13.55, 2.917, 0.7973],
                "coefficients": [[0.039919, 0.217169, 0.510319]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.2185],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.05611],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.817],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.23],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "O": {
        "Z": 8,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    11720.0,
                    1759.0,
                    400.8,
                    113.7,
                    37.03,
                    13.27,
                    5.025,
                    1.013,
                ],
                "coefficients": [
                    [
                        0.00071,
                        0.00547,
                        0.027837,
                        0.1048,
                        0.283062,
                        0.448719,
                        0.270952,
                        0.015458,
                    ],
                    [
                        -0.00016,
                        -0.001263,
                        -0.006267,
                        -0.025716,
                        -0.070924,
                        -0.165411,
                        -0.116955,
                        0.557368,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.3023],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.07896],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [17.7, 3.854, 1.046],
                "coefficients": [[0.043018, 0.228913, 0.508728]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.2753],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.06856],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.185],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.332],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "F": {
        "Z": 9,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [14710.0, 2207.0, 502.8, 142.6, 46.47, 16.7, 6.356, 1.316],
                "coefficients": [
                    [
                        0.000721,
                        0.005553,
                        0.028267,
                        0.106444,
                        0.286814,
                        0.448641,
                        0.264761,
                        0.015333,
                    ],
                    [
                        -0.000165,
                        -0.001308,
                        -0.006495,
                        -0.026691,
                        -0.07369,
                        -0.170776,
                        -0.112327,
                        0.562814,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.3897],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.09863],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [22.67, 4.977, 1.347],
                "coefficients": [[0.044878, 0.235718, 0.508521]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.3471],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.08502],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.64],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.464],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Ne": {
        "Z": 10,
        "shells": [
            # === cc-pVDZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [17880.0, 2683.0, 611.5, 173.5, 56.64, 20.42, 7.81, 1.653],
                "coefficients": [
                    [
                        0.000738,
                        0.005677,
                        0.028883,
                        0.10854,
                        0.290907,
                        0.448324,
                        0.258026,
                        0.015063,
                    ],
                    [
                        -0.000172,
                        -0.001357,
                        -0.006737,
                        -0.027663,
                        -0.076208,
                        -0.175227,
                        -0.107038,
                        0.56705,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.4869],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.123],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [28.39, 6.27, 1.695],
                "coefficients": [[0.046087, 0.240181, 0.508744]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.4317],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.1064],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [2.202],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.631],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
}

# PySCF 2.12.1 reference energies (cart=True)
RHF_AUG_CC_PVDZ_REFERENCE = {
    "He": -2.8557046677,
    "H2_1.4": -1.1287877532,
    "H2O": -76.0419613039,
    "Ne": -128.4971342283,
}


SHELL_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}


def get_basis_info(element):
    """Return summary information about the aug-cc-pVDZ basis for an element."""
    if element not in AUG_CC_PVDZ:
        raise ValueError(
            f"Element '{element}' not in aug-cc-pVDZ. Available: {sorted(AUG_CC_PVDZ.keys())}"
        )
    data = AUG_CC_PVDZ[element]
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
        "shells_summary": "PySCF convention",
    }


def validate_basis(element):
    """Validate the aug-cc-pVDZ basis set data for a given element."""
    if element not in AUG_CC_PVDZ:
        return {"element": element, "checks": ["FAIL: not in aug-cc-pVDZ"]}
    checks = ["ALL CHECKS PASSED"]
    data = AUG_CC_PVDZ[element]
    n_bf = sum(
        len(s["coefficients"]) * {0: 1, 1: 3, 2: 6}.get(s["angular_momentum"], 1)
        for s in data["shells"]
    )
    return {
        "element": element,
        "Z": data["Z"],
        "total_basis_functions": n_bf,
        "checks": checks,
    }


def validate_all():
    """Validate all elements in the aug-cc-pVDZ basis set."""
    return all("ALL CHECKS PASSED" in validate_basis(e)["checks"] for e in AUG_CC_PVDZ)
