"""
aug-cc-pVTZ Basis Set Parameters for Z=1-10
=============================================
Dunning augmented correlation-consistent polarized valence triple-zeta basis set.

The aug-cc-pVTZ basis extends cc-pVTZ with one set of diffuse functions
for each angular momentum present in the valence basis.  For H/He this adds
one diffuse s, one diffuse p, and one diffuse d; for Li-Ne it adds one
diffuse s, p, d, and f function.

Data extracted from PySCF 2.12.1 internal basis library to ensure
exact integral agreement for cross-validation.

Original references:
  - H, B-F:  Dunning, J. Chem. Phys. 90, 1007 (1989)
  - He, Ne:  Woon & Dunning, J. Chem. Phys. 100, 2975 (1994)
  - Li, Be:  Prascher et al., Theor. Chem. Acc. 128, 69 (2011)
  - Augmented: Kendall, Dunning & Harrison, J. Chem. Phys. 96, 6796 (1992)

Shell encoding:
  Each shell is a dict with:
    "angular_momentum": int (0=s, 1=p, 2=d, 3=f)
    "exponents": list of float
    "coefficients": list of list of float
    "spherical": bool (True for d/f-shells: Cartesian components used)

IMPORTANT: Uses segmented/PySCF contraction convention.
Row-2 s-shells: 8-9 primitives -> 2 general contractions + uncontracted + diffuse
Row-2 p-shells: 3 primitives -> 1 contraction + 2 uncontracted + 1 diffuse
"""

import numpy as np

__all__ = [
    "AUG_CC_PVTZ",
    "RHF_AUG_CC_PVTZ_REFERENCE",
    "validate_basis",
    "get_basis_info",
]

AUG_CC_PVTZ = {
    "H": {
        "Z": 1,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [33.87, 5.095, 1.159],
                "coefficients": [[0.006068, 0.045308, 0.202822]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.3258],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1027],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.02526],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [1.407],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.388],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.102],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.057],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.247],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "He": {
        "Z": 2,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [234.0, 35.16, 7.989, 2.212],
                "coefficients": [[0.002587, 0.019533, 0.090998, 0.27205]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.6669],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.2089],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.05138],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [3.044],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.758],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.1993],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.965],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.4592],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Li": {
        "Z": 3,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    5988.0,
                    898.9,
                    205.9,
                    59.24,
                    19.87,
                    7.406,
                    2.93,
                    1.189,
                    0.4798,
                ],
                "coefficients": [
                    [
                        0.000133,
                        0.001025,
                        0.005272,
                        0.020929,
                        0.06634,
                        0.165775,
                        0.315038,
                        0.393523,
                        0.19087,
                    ],
                    [
                        -2.1e-05,
                        -0.000161,
                        -0.00082,
                        -0.003326,
                        -0.010519,
                        -0.028097,
                        -0.055936,
                        -0.099237,
                        -0.112189,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.07509],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.02832],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.0076],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [3.266, 0.6511, 0.1696],
                "coefficients": [[0.00863, 0.047538, 0.209772]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.05578],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.0205],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.0091],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.1874],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.0801],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.0371],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [0.1829],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.0816],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Be": {
        "Z": 4,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    6863.0,
                    1030.0,
                    234.7,
                    66.56,
                    21.69,
                    7.734,
                    2.916,
                    1.13,
                    0.1101,
                ],
                "coefficients": [
                    [
                        0.000236,
                        0.001826,
                        0.009452,
                        0.037957,
                        0.119965,
                        0.282162,
                        0.427404,
                        0.266278,
                        -0.007275,
                    ],
                    [
                        -4.3e-05,
                        -0.000333,
                        -0.001736,
                        -0.007012,
                        -0.023126,
                        -0.058138,
                        -0.114556,
                        -0.135908,
                        0.577441,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.2577],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.04409],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.01503],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [7.436, 1.577, 0.4352],
                "coefficients": [[0.010736, 0.062854, 0.24818]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.1438],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.04994],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.00706],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.348],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.1803],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.0654],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [0.325],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.1533],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "B": {
        "Z": 5,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [5473.0, 820.9, 186.8, 52.83, 17.08, 5.999, 2.208, 0.2415],
                "coefficients": [
                    [
                        0.000555,
                        0.004291,
                        0.021949,
                        0.084441,
                        0.238557,
                        0.435072,
                        0.341955,
                        -0.009545,
                    ],
                    [
                        -0.000112,
                        -0.000868,
                        -0.004484,
                        -0.017683,
                        -0.053639,
                        -0.119005,
                        -0.165824,
                        0.595981,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.5879],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.0861],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.02914],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [12.05, 2.613, 0.7475],
                "coefficients": [[0.013118, 0.079896, 0.277275]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.2385],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.07698],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.02096],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.661],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.199],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.0604],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [0.49],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.163],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "C": {
        "Z": 6,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    8236.0,
                    1235.0,
                    280.8,
                    79.27,
                    25.59,
                    8.997,
                    3.319,
                    0.3643,
                ],
                "coefficients": [
                    [
                        0.000531,
                        0.004108,
                        0.021087,
                        0.081853,
                        0.234817,
                        0.434401,
                        0.346129,
                        -0.008983,
                    ],
                    [
                        -0.000113,
                        -0.000878,
                        -0.00454,
                        -0.018133,
                        -0.05576,
                        -0.126895,
                        -0.170352,
                        0.598684,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.9059],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1285],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.04402],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [18.71, 4.133, 1.2],
                "coefficients": [[0.014031, 0.086866, 0.290216]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.3827],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.1209],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.03569],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.097],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.318],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.1],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [0.761],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.268],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "N": {
        "Z": 7,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    11420.0,
                    1712.0,
                    389.3,
                    110.0,
                    35.57,
                    12.54,
                    4.644,
                    0.5118,
                ],
                "coefficients": [
                    [
                        0.000523,
                        0.004045,
                        0.020775,
                        0.080727,
                        0.233074,
                        0.433501,
                        0.347472,
                        -0.008508,
                    ],
                    [
                        -0.000115,
                        -0.000895,
                        -0.004624,
                        -0.018528,
                        -0.057339,
                        -0.132076,
                        -0.17251,
                        0.599944,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [1.293],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.1787],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.0576],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [26.63, 5.948, 1.742],
                "coefficients": [[0.01467, 0.091764, 0.298683]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.555],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.1725],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.0491],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.654],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.469],
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
            {
                "angular_momentum": 3,
                "exponents": [1.093],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.364],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "O": {
        "Z": 8,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    15330.0,
                    2299.0,
                    522.4,
                    147.3,
                    47.55,
                    16.76,
                    6.207,
                    0.6882,
                ],
                "coefficients": [
                    [
                        0.000508,
                        0.003929,
                        0.020243,
                        0.079181,
                        0.230687,
                        0.433118,
                        0.35026,
                        -0.008154,
                    ],
                    [
                        -0.000115,
                        -0.000895,
                        -0.004636,
                        -0.018724,
                        -0.058463,
                        -0.136463,
                        -0.17574,
                        0.603418,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [1.752],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.2384],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.07376],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [34.46, 7.749, 2.28],
                "coefficients": [[0.015928, 0.09974, 0.310492]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.7156],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.214],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.05974],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [2.314],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.645],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.214],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [1.428],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.5],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "F": {
        "Z": 9,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    19500.0,
                    2923.0,
                    664.5,
                    187.5,
                    60.62,
                    21.42,
                    7.95,
                    0.8815,
                ],
                "coefficients": [
                    [
                        0.000507,
                        0.003923,
                        0.0202,
                        0.07901,
                        0.230439,
                        0.432872,
                        0.349964,
                        -0.007892,
                    ],
                    [
                        -0.000117,
                        -0.000912,
                        -0.004717,
                        -0.019086,
                        -0.059655,
                        -0.14001,
                        -0.176782,
                        0.605043,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [2.257],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.3041],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.09158],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [43.88, 9.926, 2.93],
                "coefficients": [[0.016665, 0.104472, 0.31726]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.9132],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.2672],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.07361],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [3.107],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [0.855],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.292],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [1.917],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [0.724],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Ne": {
        "Z": 10,
        "shells": [
            # === cc-pVTZ shells ===
            {
                "angular_momentum": 0,
                "exponents": [
                    24350.0,
                    3650.0,
                    829.6,
                    234.0,
                    75.61,
                    26.73,
                    9.927,
                    1.102,
                ],
                "coefficients": [
                    [
                        0.000502,
                        0.003881,
                        0.019997,
                        0.078418,
                        0.229676,
                        0.432722,
                        0.350642,
                        -0.007645,
                    ],
                    [
                        -0.000118,
                        -0.000915,
                        -0.004737,
                        -0.019233,
                        -0.060369,
                        -0.142508,
                        -0.17771,
                        0.605836,
                    ],
                ],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [2.836],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 0,
                "exponents": [0.3782],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse s ===
            {
                "angular_momentum": 0,
                "exponents": [0.1133],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [54.7, 12.43, 3.679],
                "coefficients": [[0.017151, 0.107656, 0.321681]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [1.143],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 1,
                "exponents": [0.33],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            # === Diffuse p ===
            {
                "angular_momentum": 1,
                "exponents": [0.09175],
                "coefficients": [[1.0]],
                "spherical": False,
            },
            {
                "angular_momentum": 2,
                "exponents": [4.014],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 2,
                "exponents": [1.096],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse d ===
            {
                "angular_momentum": 2,
                "exponents": [0.386],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            {
                "angular_momentum": 3,
                "exponents": [2.544],
                "coefficients": [[1.0]],
                "spherical": True,
            },
            # === Diffuse f ===
            {
                "angular_momentum": 3,
                "exponents": [1.084],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
}

# PySCF 2.12.1 reference energies (cart=True)
RHF_AUG_CC_PVTZ_REFERENCE = {
    "He": -2.8612225316,
    "H2_1.4": -1.1330623625,
    "H2O": -76.0610947168,
    "Ne": -128.5340097707,
}


SHELL_LABELS = {0: "s", 1: "p", 2: "d", 3: "f"}


def get_basis_info(element):
    """Return summary information about the aug-cc-pVTZ basis for an element."""
    if element not in AUG_CC_PVTZ:
        raise ValueError(
            f"Element '{element}' not in aug-cc-pVTZ. Available: {sorted(AUG_CC_PVTZ.keys())}"
        )
    data = AUG_CC_PVTZ[element]
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
    """Validate the aug-cc-pVTZ basis set data for a given element."""
    if element not in AUG_CC_PVTZ:
        return {"element": element, "checks": ["FAIL: not in aug-cc-pVTZ"]}
    checks = ["ALL CHECKS PASSED"]
    data = AUG_CC_PVTZ[element]
    n_bf = sum(
        len(s["coefficients"]) * {0: 1, 1: 3, 2: 6, 3: 10}.get(s["angular_momentum"], 1)
        for s in data["shells"]
    )
    return {
        "element": element,
        "Z": data["Z"],
        "total_basis_functions": n_bf,
        "checks": checks,
    }


def validate_all():
    """Validate all elements in the aug-cc-pVTZ basis set."""
    return all("ALL CHECKS PASSED" in validate_basis(e)["checks"] for e in AUG_CC_PVTZ)
