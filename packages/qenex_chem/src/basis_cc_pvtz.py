"""
cc-pVTZ Basis Set Parameters for H, He, C-Ne
==============================================
Data extracted from PySCF 2.12.1 internal basis library.
Uses PySCF's segmented contraction convention for exact integral match.

References:
  - H, C-F:  Dunning, J. Chem. Phys. 90, 1007 (1989)
  - He, Ne:  Woon & Dunning, J. Chem. Phys. 100, 2975 (1994)
"""

__all__ = ["CC_PVTZ", "RHF_CC_PVTZ_REFERENCE", "MP2_CC_PVTZ_REFERENCE"]

CC_PVTZ = {
    "H": {
        "Z": 1,
        "shells": [
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
            {
                "angular_momentum": 2,
                "exponents": [1.057],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "He": {
        "Z": 2,
        "shells": [
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
            {
                "angular_momentum": 2,
                "exponents": [1.965],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "C": {
        "Z": 6,
        "shells": [
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
            {
                "angular_momentum": 3,
                "exponents": [0.761],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "N": {
        "Z": 7,
        "shells": [
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
            {
                "angular_momentum": 3,
                "exponents": [1.093],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "O": {
        "Z": 8,
        "shells": [
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
            {
                "angular_momentum": 3,
                "exponents": [1.428],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "F": {
        "Z": 9,
        "shells": [
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
            {
                "angular_momentum": 3,
                "exponents": [1.917],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
    "Ne": {
        "Z": 10,
        "shells": [
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
            {
                "angular_momentum": 3,
                "exponents": [2.544],
                "coefficients": [[1.0]],
                "spherical": True,
            },
        ],
    },
}

RHF_CC_PVTZ_REFERENCE = {
    "He": -2.8611535740,
    "H2_1.4": -1.1329814896,
    "H2O": -76.0577239273,
}

MP2_CC_PVTZ_REFERENCE = {
    "He": {"E_corr": -0.0332872876, "E_tot": -2.8944408617},
    "H2_1.4": {"E_corr": -0.0317953220, "E_tot": -1.1647768116},
    "H2O": {"E_corr": -0.2789525747, "E_tot": -76.3366765020},
}
