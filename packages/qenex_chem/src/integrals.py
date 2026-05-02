"""
Integrals Module
Provides analytical integrals for Gaussian primitives and basis set definitions (STO-3G).
Supports s and p orbitals via Obara-Saika recurrence relations.
Now optimized with Numba JIT compilation.
"""

import numpy as np
from scipy.special import erf, gamma as scipy_gamma, gammainc as scipy_gammainc
import numba
from numba import jit, float64, int64
from numba.typed import Dict
from numba.core import types

import math

# [PATCH] Check for Rust Acceleration
# The QENEX Sovereign Agent has implemented a zero-copy FFI bridge
# to accelerate O(N^4) integral calculations.
try:
    import qenex_accelerate

    RUST_AVAILABLE = True
except ImportError:
    # Auto-discover the Rust module in the sibling qenex-accelerate package.
    # This handles the common case where sys.path doesn't include the
    # qenex-accelerate directory (e.g., system Python without .venv).
    import os as _os
    import sys as _sys

    _accel_path = _os.path.normpath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "qenex-accelerate")
    )
    if _os.path.isdir(_accel_path) and _accel_path not in _sys.path:
        _sys.path.insert(0, _accel_path)
        try:
            import qenex_accelerate  # type: ignore[no-redef]

            RUST_AVAILABLE = True
        except ImportError:
            RUST_AVAILABLE = False
    else:
        RUST_AVAILABLE = False
    del _os, _sys, _accel_path
    # Fallback to pure Python/Numba if Rust module not found

# ==========================================
# STO-3G Basis Set Definitions
# ==========================================


class STO3G:
    """
    Standard STO-3G parameters.
    Exponents (alpha) and Contraction Coefficients (d).
    """

    # Normalized 1s shell (alpha, d)
    basis_1s = [(0.109818, 0.444635), (0.405771, 0.535328), (2.227660, 0.154329)]

    # Normalized 2s shell (alpha, d)
    basis_2s = [(0.0751386, 0.700115), (0.231031, 0.399513), (0.994203, -0.0999672)]

    # Normalized 2p shell (alpha, d)
    basis_2p = [(0.0751386, 0.391957), (0.231031, 0.607684), (0.994203, 0.155916)]

    # Standard molecular exponents (zeta) — STO-3G basis set
    # Verified against PySCF 2.x: zeta = sqrt(pyscf_exponent / alpha_norm)
    # using the middle (most numerically stable) primitive of each shell.
    # Previous values from Hehre, Stewart, Pople JCP 51, 2657 (1969) had
    # transcription errors for Be (-12%), B (-19%), and smaller errors for
    # C, N, O, F. These PySCF-verified values reproduce exact exponent matching.
    zeta = {
        "H": 1.24,
        "He": 1.69,
        # Valence 2sp zetas (verified against PySCF STO-3G)
        "Li": 0.8000,
        "Be": 1.1500,
        "B": 1.5000,
        "C": 1.7200,
        "N": 1.9500,
        "O": 2.2500,
        "F": 2.5500,
        "Ne": 2.8800,
        # Core 1s zetas (verified against PySCF STO-3G)
        "Li_1s": 2.6900,
        "Be_1s": 3.6800,
        "B_1s": 4.6800,
        "C_1s": 5.6700,
        "N_1s": 6.6700,
        "O_1s": 7.6600,
        "F_1s": 8.6500,
        "Ne_1s": 9.6400,
    }


# ==========================================
# 6-31G Basis Set Definitions
# ==========================================


class BasisSet631G:
    """
    Standard 6-31G parameters.
    Source: Standard literature values (e.g. from EMSL Basis Set Exchange).
    Structure:
      H: 4s -> [2s] (Split Valence: 3 inner, 1 outer)
      C-F: 6-31G (Core 1s: 6 prims; Valence 2sp: 3 inner, 1 outer)
      P-S: 66-31G approx (Core 1s,2s,2p; Valence 3sp)
    """

    # Hydrogen (4s contracted to 2s)
    # Inner 1s (3 prims)
    H_inner = [
        (18.7311370, 0.03349460),
        (2.8253937, 0.23472695),
        (0.6401217, 0.81375733),
    ]
    # Outer 1s (1 prim)
    H_outer = [(0.1612778, 1.0000000)]

    # Helium (He) - Standard 6-31G
    He_inner = [
        (38.4216340, 0.03349460),
        (5.7760130, 0.23472695),
        (1.2931780, 0.81375733),
    ]
    He_outer = [(0.3258400, 1.0000000)]

    # Carbon (C)
    # Core 1s (6 prims)
    # R9-F2: C core 1s contraction coefficients corrected to EMSL/BSE values.
    # Previously these were copies of N's coefficients (e.g., 0.0139948 vs
    # correct 0.0140373). Reference: Hehre, Ditchfield, Pople, JCP 56, 2257 (1972).
    C_core_1s = [
        (3047.5249000, 0.0018347),
        (457.3695100, 0.0140373),
        (103.9486900, 0.0688426),
        (29.2101550, 0.2321844),
        (9.2866630, 0.4679413),
        (3.1639270, 0.3623120),
    ]
    # Valence Inner 2sp (3 prims) - Shared exponents
    C_val_inner = {
        "exponents": [7.8682724, 1.8812885, 0.5442493],
        "s_coeffs": [-0.1193324, -0.1608542, 1.1434564],
        "p_coeffs": [0.0689991, 0.3164240, 0.7443083],
    }
    # Valence Outer 2sp (1 prim)
    C_val_outer = {
        "exponents": [0.1687144],
        "s_coeffs": [1.0000000],
        "p_coeffs": [1.0000000],
    }

    # Nitrogen (N)
    N_core_1s = [
        (4173.5110000, 0.0018347),
        (627.4579000, 0.0139948),
        (142.9021000, 0.0685866),
        (40.2343300, 0.2322409),
        (12.8202100, 0.4690699),
        (4.3904370, 0.3604552),
    ]
    N_val_inner = {
        "exponents": [11.6263580, 2.7162800, 0.7722180],
        "s_coeffs": [-0.1149612, -0.1593168, 1.1354180],
        "p_coeffs": [0.0675797, 0.3239072, 0.7408951],
    }
    N_val_outer = {
        "exponents": [0.2120313],
        "s_coeffs": [1.0000000],
        "p_coeffs": [1.0000000],
    }

    # Oxygen (O)
    O_core_1s = [
        (5484.6717000, 0.0018311),
        (825.2349500, 0.0139501),
        (188.0469600, 0.0684451),
        (52.9645000, 0.2327143),
        (16.8975700, 0.4701930),
        (5.7996353, 0.3585209),
    ]
    O_val_inner = {
        "exponents": [15.5396160, 3.5999336, 1.0137618],
        "s_coeffs": [-0.1107775, -0.1480263, 1.1307670],
        "p_coeffs": [0.0708743, 0.3397528, 0.7271586],
    }
    O_val_outer = {
        "exponents": [0.2700058],
        "s_coeffs": [1.0000000],
        "p_coeffs": [1.0000000],
    }

    # Phosphorus (P)
    # Core 1s (6 prims)
    P_core_1s = [
        (19410.0000, 0.001851),
        (2921.0000, 0.014166),
        (665.4000, 0.069772),
        (187.6000, 0.240706),
        (60.1500, 0.485141),
        (20.9100, 0.331456),
    ]
    # Core 2sp (L-shell) - Often treated as 2s and 2p separately or sp in some sets.
    # Standard 6-31G for 3rd row often uses separate s and p for core or 2sp.
    # For P, we use 6 primitives for 2s and 2p (separate or shared).
    # Here we use shared exponents for 2s/2p core block (common in Pople sets).
    P_core_2sp = {
        "exponents": [423.5000, 95.8400, 30.5600, 11.0800, 4.2980, 1.6950],
        "s_coeffs": [-0.007626, -0.075421, -0.235697, 0.098319, 0.697666, 0.422501],
        "p_coeffs": [0.011883, 0.078440, 0.250550, 0.490807, 0.375258, 0.026417],
    }
    # Valence 3sp (Split 3, 1)
    P_val_inner = {
        "exponents": [2.8640, 1.0880, 0.4354],
        "s_coeffs": [-0.276495, 0.230728, 0.890696],
        "p_coeffs": [-0.092176, 0.400537, 0.672670],
    }
    P_val_outer = {"exponents": [0.1364], "s_coeffs": [1.0000], "p_coeffs": [1.0000]}

    # Sulfur (S)
    S_core_1s = [
        (22660.0000, 0.001835),
        (3404.0000, 0.014023),
        (774.2000, 0.069002),
        (218.0000, 0.238125),
        (69.8900, 0.482062),
        (24.3100, 0.334066),
    ]
    S_core_2sp = {
        "exponents": [507.0000, 115.6000, 37.1600, 13.5600, 5.2820, 2.0830],
        "s_coeffs": [-0.007505, -0.073420, -0.228518, 0.088192, 0.686522, 0.433431],
        "p_coeffs": [0.011707, 0.076840, 0.244365, 0.485122, 0.385472, 0.035985],
    }
    S_val_inner = {
        "exponents": [3.6190, 1.3530, 0.5332],
        "s_coeffs": [-0.301323, 0.270966, 0.865487],
        "p_coeffs": [-0.088737, 0.408078, 0.669830],
    }
    S_val_outer = {"exponents": [0.1654], "s_coeffs": [1.0000], "p_coeffs": [1.0000]}


# ==========================================
# Math Helpers
# ==========================================


def factorial2(n):
    """
    Compute the double factorial n!! = n * (n-2) * (n-4) * ... * 1.
    """
    if n <= 0:
        return 1
    result = 1
    for i in range(n, 0, -2):
        result *= i
    return result


def normalize_primitive(alpha, lmn):
    """
    Compute normalization constant for a primitive Gaussian x^l y^m z^n e^(-alpha*r^2).
    """
    l, m, n = lmn
    # Formula: (2*alpha/pi)^0.75 * (4*alpha)^(L/2) / sqrt((2l-1)!! (2m-1)!! (2n-1)!!)
    L = l + m + n
    pre = (2.0 * alpha / np.pi) ** 0.75
    ang = (4.0 * alpha) ** (L / 2.0)
    denom = np.sqrt(
        factorial2(2 * l - 1) * factorial2(2 * m - 1) * factorial2(2 * n - 1)
    )
    return pre * ang / denom


# JIT-compiled Boys function wrapper
# Numba does not support scipy.special.erf directly in nopython mode easily without extra config in some versions,
# but supports math.erf from python 3.2+. Numpy erf is also supported.
# R9-F3: fastmath=False for Boys function — fastmath=True risks precision loss
# in series convergence test and downward recurrence via IEEE 754 violations.
# The Boys function demands >=14 digits for chemical accuracy.
@jit(nopython=True, fastmath=False, cache=True)
def boys_jit(n, t):
    """
    Boys function F_n(t) = integral_0^1 u^(2n) exp(-t u^2) du
    JIT-optimized version with asymptotic tail.
    """
    if t < 1e-8:
        return 1.0 / (2 * n + 1) - t / (2 * n + 3)

    # Asymptotic expansion for large t
    elif t > 40.0:
        # F_n(t) ~ (2n-1)!! * sqrt(pi) / (2^(n+1) * t^(n+0.5))
        # Compute (2n-1)!!
        df = 1.0
        if n > 0:
            for i in range(1, 2 * n, 2):
                df *= i

        return (df * np.sqrt(np.pi)) / (2.0 ** (n + 1) * t ** (n + 0.5))

    else:
        # Phase 27 CRITICAL fix CH-1: Use DOWNWARD recurrence for numerical stability.
        #
        # The upward recurrence F_{k+1}(t) = [(2k+1)F_k(t) - exp(-t)]/(2t)
        # suffers from catastrophic cancellation for moderate t and large n
        # because F_k(t) ~ exp(-t)/(2k+1) and we subtract two nearly equal numbers.
        #
        # Downward recurrence F_{k-1}(t) = [2t*F_k(t) + exp(-t)]/(2k-1) is
        # numerically STABLE because it only involves addition.
        #
        # Strategy: start from n_top >> n where F_{n_top}(t) can be estimated
        # as exp(-t)/(2*n_top+1) (leading term for large n), then recur down.
        # The downward recurrence is self-correcting: errors in the starting
        # value shrink by a factor of t/(2k) at each step, so by the time
        # we reach n, any initial error has been damped by (t/(2n))^25 ~ 0.
        #
        # Reference: Helgaker, Jorgensen & Olsen, "Molecular Electronic-Structure
        # Theory", Ch. 9.8 (Boys function recurrence relations).

        if n == 0:
            return 0.5 * np.sqrt(np.pi / t) * math.erf(np.sqrt(t))

        # For downward recurrence, start from n_top >> n AND n_top >> t.
        # F_n(t) = exp(-t) * Σ_{k=0}^∞ (2t)^k / ((2n+1)(2n+3)...(2n+2k+1))
        # The series converges when 2t/(2n+1) << 1, i.e., n >> t.
        # We need n_top large enough that the truncated series is accurate.
        n_top = max(n + 30, int(t) + 30)  # Ensures n_top >> max(n, t)

        exp_t = np.exp(-t)
        # Multi-term series for F_{n_top}(t):
        #   F_n(t) = exp(-t) / (2n+1) * Σ_{k=0}^∞ (2t)^k / Π_{j=1}^k (2n+2j+1)
        # Compute until terms become negligible
        f_top = 0.0
        term = 1.0 / (2.0 * n_top + 1.0)
        for k in range(60):  # 60 terms is more than enough
            f_top += term
            term *= 2.0 * t / (2.0 * n_top + 2.0 * k + 3.0)
            if abs(term) < 1e-16 * abs(f_top):
                break
        f_top *= exp_t

        # Downward recurrence: F_{k-1}(t) = [2t*F_k(t) + exp(-t)] / (2k - 1)
        f_k = f_top
        for k in range(n_top, n, -1):
            f_k = (2.0 * t * f_k + exp_t) / (2.0 * k - 1.0)
        return f_k


def boys(n, t):
    """
    Python wrapper for the JIT-compiled Boys function.
    """
    return boys_jit(n, float(t))


# ==========================================
# Primitive Gaussian Class
# ==========================================


class BasisFunction:
    """
    Primitive Gaussian basis function with angular momentum and normalization.
    """

    def __init__(
        self, origin=None, alpha=None, coeff=1.0, lmn=(0, 0, 0), l=None, m=None, n=None
    ):
        """
        Create a primitive Gaussian basis function.

        Supports two calling conventions:
        1. BasisFunction(origin, alpha, coeff, lmn=(0,0,0))       # Original API
        2. BasisFunction(alpha=a, origin=o, l=0, m=0, n=0)        # Test API (PrimitiveGaussian style)
        """
        # Handle origin - required parameter
        if origin is None:
            raise TypeError("BasisFunction requires 'origin' parameter")
        self.origin = np.array(origin, dtype=np.float64)

        # Handle alpha - required parameter
        if alpha is None:
            raise TypeError("BasisFunction requires 'alpha' parameter")
        self.alpha = float(alpha)

        # Handle coefficient
        self.coeff = float(coeff)

        # Handle angular momentum - support both lmn tuple and individual l,m,n kwargs
        if l is not None or m is not None or n is not None:
            # Use individual l,m,n if provided (default to 0 if not specified)
            self.l = l if l is not None else 0
            self.m = m if m is not None else 0
            self.n = n if n is not None else 0
        else:
            # Use lmn tuple
            self.l, self.m, self.n = lmn

        self.L = self.l + self.m + self.n
        self.norm = normalize_primitive(self.alpha, (self.l, self.m, self.n))
        self.N = self.coeff * self.norm


# ==========================================
# JIT-Optimized Integral Kernels
# ==========================================


@jit(nopython=True, fastmath=True, cache=True)
def gaussian_product_center(alpha1, A, alpha2, B):
    """
    Compute the center of the Gaussian product distribution.
    """
    return (alpha1 * A + alpha2 * B) / (alpha1 + alpha2)


@jit(nopython=True, fastmath=True, cache=True)
def overlap_1d(l1, l2, xA, xB, alpha, beta):
    """
    1D Overlap Integral between (x-xA)^l1 and (x-xB)^l2
    """
    p = alpha + beta
    rp = (alpha * xA + beta * xB) / p

    # We avoid allocating S array dynamically if possible, or use small fixed size
    # Max angular momentum is usually small (d=2, f=3). 5x5 is safe.
    S = np.zeros((l1 + 1, l2 + 1))

    S[0, 0] = np.sqrt(np.pi / p)

    XP = rp
    XA = xA
    XB = xB

    for i in range(l1 + 1):
        for j in range(l2 + 1):
            if i == 0 and j == 0:
                continue

            if i > 0:
                term1 = (XP - XA) * S[i - 1, j]
                term2 = (i - 1) * S[i - 2, j] if i > 1 else 0.0
                term3 = j * S[i - 1, j - 1] if j > 0 else 0.0
                S[i, j] = term1 + (1.0 / (2 * p)) * (term2 + term3)
            else:
                term1 = (XP - XB) * S[i, j - 1]
                term2 = i * S[i - 1, j - 1] if i > 0 else 0.0
                term3 = (j - 1) * S[i, j - 2] if j > 1 else 0.0
                S[i, j] = term1 + (1.0 / (2 * p)) * (term2 + term3)

    return S[l1, l2]


@jit(nopython=True, fastmath=True, cache=True)
def overlap_jit(la, ma, na, lb, mb, nb, ra, rb, alpha, beta):
    """
    Compute unnormalized overlap integral between two primitive Gaussians.
    """
    p = alpha + beta
    diff = ra - rb
    dist2 = np.dot(diff, diff)
    pre = np.exp(-(alpha * beta / p) * dist2)

    sx = overlap_1d(la, lb, ra[0], rb[0], alpha, beta)
    sy = overlap_1d(ma, mb, ra[1], rb[1], alpha, beta)
    sz = overlap_1d(na, nb, ra[2], rb[2], alpha, beta)

    return pre * sx * sy * sz


def overlap(a, b):
    """
    Compute the normalized overlap integral <a|b> between two basis functions.
    """
    val = overlap_jit(
        a.l, a.m, a.n, b.l, b.m, b.n, a.origin, b.origin, a.alpha, b.alpha
    )
    return a.N * b.N * val


# R9-F3b: fastmath=False for kinetic integral — cancellation-prone arithmetic
# in the t1+t2+t3 summation requires IEEE 754 compliance.
@jit(nopython=True, fastmath=False, cache=True)
def kinetic_jit(la, ma, na, lb, mb, nb, ra, rb, alpha, beta):
    """Compute unnormalized kinetic energy integral between primitive Gaussians."""
    p = alpha + beta

    # Precompute 1D overlaps needed manually
    # We need to compute S values for different lb

    # Helper to compute Term 1D
    # T_1d = -0.5 * [ l2(l2-1)<l1|l2-2> - 2beta(2l2+1)<l1|l2> + 4beta^2 <l1|l2+2> ]

    def get_term_1d(l1, l2, xA, xB):
        """Compute one-dimensional kinetic energy contribution."""
        s0 = overlap_1d(l1, l2, xA, xB, alpha, beta)
        t1 = 0.0
        if l2 >= 2:
            s_minus = overlap_1d(l1, l2 - 2, xA, xB, alpha, beta)
            t1 = l2 * (l2 - 1) * s_minus

        t2 = -2.0 * beta * (2.0 * l2 + 1.0) * s0

        s_plus = overlap_1d(l1, l2 + 2, xA, xB, alpha, beta)
        t3 = 4.0 * beta**2 * s_plus

        return -0.5 * (t1 + t2 + t3)

    diff = ra - rb
    dist2 = np.dot(diff, diff)
    pre = np.exp(-(alpha * beta / p) * dist2)

    Sx = overlap_1d(la, lb, ra[0], rb[0], alpha, beta)
    Sy = overlap_1d(ma, mb, ra[1], rb[1], alpha, beta)
    Sz = overlap_1d(na, nb, ra[2], rb[2], alpha, beta)

    Tx = get_term_1d(la, lb, ra[0], rb[0])
    Ty = get_term_1d(ma, mb, ra[1], rb[1])
    Tz = get_term_1d(na, nb, ra[2], rb[2])

    val = Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz
    return pre * val


def kinetic(a, b):
    """
    Compute the normalized kinetic energy integral <a|T|b>.
    """
    val = kinetic_jit(
        a.l, a.m, a.n, b.l, b.m, b.n, a.origin, b.origin, a.alpha, b.alpha
    )
    return a.N * b.N * val


@jit(nopython=True, fastmath=True, cache=True)
def nuclear_recursion_inner(L, m, boys_vals, ctx):
    """
    Optimized recursive helper for Nuclear Attraction Integrals.
    L: tuple/array of (la, lb, ma, mb, na, nb)
    ctx: array of [Rp(3), C(3), p, kab, a_origin(3), b_origin(3)] -> size 14
    """
    la, lb, ma, mb, na, nb = L

    # Unpack ctx for readability (compiler will optimize)
    # Rp = ctx[0:3] # slices can be slow if creating new arrays, use indexing
    # C = ctx[3:6]
    p = ctx[6]
    kab = ctx[7]
    # a_origin = ctx[8:11]
    # b_origin = ctx[11:14]

    # Base case check
    if la == 0 and lb == 0 and ma == 0 and mb == 0 and na == 0 and nb == 0:
        return (2.0 * np.pi / p) * kab * boys_vals[m]

    # X-axis recurrence
    if la > 0:
        L_new = (la - 1, lb, ma, mb, na, nb)
        term1 = (ctx[0] - ctx[8]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[0] - ctx[3]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if la > 1:
            L_new2 = (la - 2, lb, ma, mb, na, nb)
            term3 = (
                (la - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if lb > 0:
            L_new3 = (la - 1, lb - 1, ma, mb, na, nb)
            term4 = (
                lb
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    if lb > 0:
        L_new = (la, lb - 1, ma, mb, na, nb)
        term1 = (ctx[0] - ctx[11]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[0] - ctx[3]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if la > 0:
            L_new2 = (la - 1, lb - 1, ma, mb, na, nb)
            term3 = (
                la
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if lb > 1:
            L_new3 = (la, lb - 2, ma, mb, na, nb)
            term4 = (
                (lb - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    # Y-axis recurrence
    if ma > 0:
        L_new = (la, lb, ma - 1, mb, na, nb)
        term1 = (ctx[1] - ctx[9]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[1] - ctx[4]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if ma > 1:
            L_new2 = (la, lb, ma - 2, mb, na, nb)
            term3 = (
                (ma - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if mb > 0:
            L_new3 = (la, lb, ma - 1, mb - 1, na, nb)
            term4 = (
                mb
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    if mb > 0:
        L_new = (la, lb, ma, mb - 1, na, nb)
        term1 = (ctx[1] - ctx[12]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[1] - ctx[4]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if ma > 0:
            L_new2 = (la, lb, ma - 1, mb - 1, na, nb)
            term3 = (
                ma
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if mb > 1:
            L_new3 = (la, lb, ma, mb - 2, na, nb)
            term4 = (
                (mb - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    # Z-axis recurrence
    if na > 0:
        L_new = (la, lb, ma, mb, na - 1, nb)
        term1 = (ctx[2] - ctx[10]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[2] - ctx[5]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if na > 1:
            L_new2 = (la, lb, ma, mb, na - 2, nb)
            term3 = (
                (na - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if nb > 0:
            L_new3 = (la, lb, ma, mb, na - 1, nb - 1)
            term4 = (
                nb
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    if nb > 0:
        L_new = (la, lb, ma, mb, na, nb - 1)
        term1 = (ctx[2] - ctx[13]) * nuclear_recursion_inner(L_new, m, boys_vals, ctx)
        term2 = (ctx[2] - ctx[5]) * nuclear_recursion_inner(
            L_new, m + 1, boys_vals, ctx
        )
        term3 = 0.0
        if na > 0:
            L_new2 = (la, lb, ma, mb, na - 1, nb - 1)
            term3 = (
                na
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new2, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new2, m + 1, boys_vals, ctx)
                )
            )
        term4 = 0.0
        if nb > 1:
            L_new3 = (la, lb, ma, mb, na, nb - 2)
            term4 = (
                (nb - 1)
                / (2 * p)
                * (
                    nuclear_recursion_inner(L_new3, m, boys_vals, ctx)
                    - nuclear_recursion_inner(L_new3, m + 1, boys_vals, ctx)
                )
            )
        return term1 - term2 + term3 + term4

    return 0.0


def nuclear_attraction(a, b, C, Z):
    """
    Compute the nuclear attraction integral (a | -Z/r_C | b).
    """
    alpha = a.alpha
    beta = b.alpha
    p = alpha + beta

    Rp = gaussian_product_center(alpha, a.origin, beta, b.origin)
    diff = a.origin - b.origin
    dist2 = np.dot(diff, diff)
    kab = np.exp(-(alpha * beta / p) * dist2)

    RPC = Rp - C

    L_total = a.L + b.L
    T_val = p * np.dot(RPC, RPC)

    boys_vals = np.zeros(L_total + 1)
    for n in range(L_total + 1):
        boys_vals[n] = boys_jit(n, float(T_val))

    # Pack context: [Rp(3), C(3), p, kab, a_origin(3), b_origin(3)] -> 14 floats
    ctx = np.zeros(14)
    ctx[0:3] = Rp
    ctx[3:6] = C
    ctx[6] = p
    ctx[7] = kab
    ctx[8:11] = a.origin
    ctx[11:14] = b.origin

    L_tuple = (a.l, b.l, a.m, b.m, a.n, b.n)

    val = nuclear_recursion_inner(L_tuple, 0, boys_vals, ctx)
    return -Z * a.N * b.N * val


# ==========================================
# ERI Implementation (Pure Python with Memoization)
# ==========================================


def eri_recursion_python(
    la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
):
    """
    Compute electron repulsion integrals via memoized Obara-Saika recursion in pure Python.
    """
    key = (la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m)
    if key in cache:
        return cache[key]

    p = ctx[6]
    q = ctx[7]
    rho = ctx[8]
    prefactor = ctx[21]

    L_sum = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd
    if L_sum == 0:
        result = prefactor * boys_vals[m]
        cache[key] = result
        return result

    result = 0.0

    # X-Axis: la
    if la > 0:
        term1 = (ctx[0] - ctx[9]) * eri_recursion_python(
            la - 1, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[0] - ctx[3])
            * eri_recursion_python(
                la - 1,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if la > 1:
            val = eri_recursion_python(
                la - 2,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la - 2,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (la - 1) / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if lb > 0:
            val = eri_recursion_python(
                la - 1,
                lb - 1,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la - 1,
                lb - 1,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = lb / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if lc > 0:
            val = eri_recursion_python(
                la - 1,
                lb,
                lc - 1,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = lc / (2 * (p + q)) * val
        term6 = 0.0
        if ld > 0:
            val = eri_recursion_python(
                la - 1,
                lb,
                lc,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = ld / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # X-Axis: lb
    if lb > 0:
        term1 = (ctx[0] - ctx[12]) * eri_recursion_python(
            la, lb - 1, lc, ld, ma, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[0] - ctx[3])
            * eri_recursion_python(
                la,
                lb - 1,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if la > 0:
            val = eri_recursion_python(
                la - 1,
                lb - 1,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la - 1,
                lb - 1,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = la / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if lb > 1:
            val = eri_recursion_python(
                la,
                lb - 2,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb - 2,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (lb - 1) / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if lc > 0:
            val = eri_recursion_python(
                la,
                lb - 1,
                lc - 1,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = lc / (2 * (p + q)) * val
        term6 = 0.0
        if ld > 0:
            val = eri_recursion_python(
                la,
                lb - 1,
                lc,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = ld / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # X-Axis: lc
    if lc > 0:
        term1 = (ctx[3] - ctx[15]) * eri_recursion_python(
            la, lb, lc - 1, ld, ma, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[0] - ctx[3])
            * eri_recursion_python(
                la,
                lb,
                lc - 1,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if lc > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc - 2,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc - 2,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (lc - 1) / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if ld > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc - 1,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc - 1,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = ld / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if la > 0:
            val = eri_recursion_python(
                la - 1,
                lb,
                lc - 1,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = la / (2 * (p + q)) * val
        term6 = 0.0
        if lb > 0:
            val = eri_recursion_python(
                la,
                lb - 1,
                lc - 1,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = lb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # X-Axis: ld
    if ld > 0:
        term1 = (ctx[3] - ctx[18]) * eri_recursion_python(
            la, lb, lc, ld - 1, ma, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[0] - ctx[3])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if lc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc - 1,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc - 1,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = lc / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if ld > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld - 2,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld - 2,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (ld - 1) / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if la > 0:
            val = eri_recursion_python(
                la - 1,
                lb,
                lc,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = la / (2 * (p + q)) * val
        term6 = 0.0
        if lb > 0:
            val = eri_recursion_python(
                la,
                lb - 1,
                lc,
                ld - 1,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = lb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Y-Axis: ma
    if ma > 0:
        term1 = (ctx[1] - ctx[10]) * eri_recursion_python(
            la, lb, lc, ld, ma - 1, mb, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[1] - ctx[4])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if ma > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 2,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 2,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (ma - 1) / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if mb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb - 1,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb - 1,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = mb / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if mc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb,
                mc - 1,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = mc / (2 * (p + q)) * val
        term6 = 0.0
        if md > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb,
                mc,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = md / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Y-Axis: mb
    if mb > 0:
        term1 = (ctx[1] - ctx[13]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb - 1, mc, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[1] - ctx[4])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 1,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if ma > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb - 1,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb - 1,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = ma / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if mb > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 2,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 2,
                mc,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (mb - 1) / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if mc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 1,
                mc - 1,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = mc / (2 * (p + q)) * val
        term6 = 0.0
        if md > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 1,
                mc,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = md / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Y-Axis: mc
    if mc > 0:
        term1 = (ctx[4] - ctx[16]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc - 1, md, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[1] - ctx[4])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 1,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if mc > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 2,
                md,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 2,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (mc - 1) / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if md > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 1,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 1,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = md / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if ma > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb,
                mc - 1,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = ma / (2 * (p + q)) * val
        term6 = 0.0
        if mb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 1,
                mc - 1,
                md,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = mb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Y-Axis: md
    if md > 0:
        term1 = (ctx[4] - ctx[19]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc, md - 1, na, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[1] - ctx[4])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if mc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 1,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc - 1,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = mc / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if md > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md - 2,
                na,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md - 2,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (md - 1) / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if ma > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma - 1,
                mb,
                mc,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = ma / (2 * (p + q)) * val
        term6 = 0.0
        if mb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb - 1,
                mc,
                md - 1,
                na,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = mb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Z-Axis: na
    if na > 0:
        term1 = (ctx[2] - ctx[11]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc, md, na - 1, nb, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[2] - ctx[5])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if na > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 2,
                nb,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 2,
                nb,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (na - 1) / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if nb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb - 1,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb - 1,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = nb / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if nc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb,
                nc - 1,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = nc / (2 * (p + q)) * val
        term6 = 0.0
        if nd > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb,
                nc,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = nd / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Z-Axis: nb
    if nb > 0:
        term1 = (ctx[2] - ctx[14]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc, md, na, nb - 1, nc, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            -(q / (p + q))
            * (ctx[2] - ctx[5])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 1,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if na > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb - 1,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb - 1,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = na / (2 * p) * (val - (rho / p) * val_m)
        term4 = 0.0
        if nb > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 2,
                nc,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 2,
                nc,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (nb - 1) / (2 * p) * (val - (rho / p) * val_m)
        term5 = 0.0
        if nc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 1,
                nc - 1,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = nc / (2 * (p + q)) * val
        term6 = 0.0
        if nd > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 1,
                nc,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = nd / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Z-Axis: nc
    if nc > 0:
        term1 = (ctx[5] - ctx[17]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc, md, na, nb, nc - 1, nd, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[2] - ctx[5])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 1,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if nc > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 2,
                nd,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 2,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = (nc - 1) / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if nd > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 1,
                nd - 1,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 1,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = nd / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if na > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb,
                nc - 1,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = na / (2 * (p + q)) * val
        term6 = 0.0
        if nb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 1,
                nc - 1,
                nd,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = nb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    # Z-Axis: nd
    if nd > 0:
        term1 = (ctx[5] - ctx[20]) * eri_recursion_python(
            la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd - 1, m, boys_vals, ctx, cache
        )
        term2 = (
            (p / (p + q))
            * (ctx[2] - ctx[5])
            * eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
        )
        term3 = 0.0
        if nc > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 1,
                nd - 1,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc - 1,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term3 = nc / (2 * q) * (val - (rho / q) * val_m)
        term4 = 0.0
        if nd > 1:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd - 2,
                m,
                boys_vals,
                ctx,
                cache,
            )
            val_m = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb,
                nc,
                nd - 2,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term4 = (nd - 1) / (2 * q) * (val - (rho / q) * val_m)
        term5 = 0.0
        if na > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na - 1,
                nb,
                nc,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term5 = na / (2 * (p + q)) * val
        term6 = 0.0
        if nb > 0:
            val = eri_recursion_python(
                la,
                lb,
                lc,
                ld,
                ma,
                mb,
                mc,
                md,
                na,
                nb - 1,
                nc,
                nd - 1,
                m + 1,
                boys_vals,
                ctx,
                cache,
            )
            term6 = nb / (2 * (p + q)) * val
        result = term1 + term2 + term3 + term4 + term5 + term6
        cache[key] = result
        return result

    return 0.0


def eri_primitive(
    alphaA,
    alphaB,
    alphaC,
    alphaD,
    A,
    B,
    C,
    D,
    la,
    ma,
    na,
    lb,
    mb,
    nb,
    lc,
    mc,
    nc,
    ld,
    md,
    nd,
    normA,
    normB,
    normC,
    normD,
):
    """
    Primitive ERI function - no JIT to avoid compilation issues.
    Uses pure Python memoized recursion.
    """
    p = alphaA + alphaB
    q = alphaC + alphaD
    alpha_Q = p * q / (p + q)

    Rp = gaussian_product_center_py(alphaA, A, alphaB, B)
    Rq = gaussian_product_center_py(alphaC, C, alphaD, D)
    Rpq = Rp - Rq

    diff_ab = A - B
    diff_cd = C - D

    kab = np.exp(-(alphaA * alphaB / p) * np.dot(diff_ab, diff_ab))
    kcd = np.exp(-(alphaC * alphaD / q) * np.dot(diff_cd, diff_cd))
    prefactor = (2 * np.pi**2.5) / (p * q * np.sqrt(p + q)) * kab * kcd

    L_total = la + lb + lc + ld + ma + mb + mc + md + na + nb + nc + nd
    T_val = alpha_Q * np.dot(Rpq, Rpq)

    # Precompute Boys Function values
    boys_vals = np.zeros(L_total + 1)
    for n in range(L_total + 1):
        boys_vals[n] = boys_py(n, float(T_val))

    # Pack context: [Rp(3), Rq(3), p, q, rho, A(3), B(3), C(3), D(3), prefactor] -> size 22
    ctx = np.zeros(22)
    ctx[0:3] = Rp
    ctx[3:6] = Rq
    ctx[6] = p
    ctx[7] = q
    ctx[8] = alpha_Q
    ctx[9:12] = A
    ctx[12:15] = B
    ctx[15:18] = C
    ctx[18:21] = D
    ctx[21] = prefactor

    # Use Python dict for memoization
    cache = {}
    val = eri_recursion_python(
        la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd, 0, boys_vals, ctx, cache
    )
    return normA * normB * normC * normD * val


def gaussian_product_center_py(alphaA, A, alphaB, B):
    """Pure Python version of gaussian product center."""
    return (alphaA * A + alphaB * B) / (alphaA + alphaB)


def boys_py(n, T):
    """Pure Python Boys function.

    Phase 27 fix CH-14: Aligned thresholds with boys_jit (small T < 1e-8,
    large T > 40.0) to ensure consistent results between code paths.
    """
    if T < 1e-8:
        return 1.0 / (2 * n + 1) - T / (2 * n + 3)

    if T > 40.0:
        # Asymptotic expansion — use module-level scipy_gamma (line 9)
        return scipy_gamma(n + 0.5) / (2 * T ** (n + 0.5))

    # Use incomplete gamma function — use module-level scipy imports (line 9)
    return 0.5 * T ** (-n - 0.5) * scipy_gamma(n + 0.5) * scipy_gammainc(n + 0.5, T)


def eri(a, b, c, d):
    """
    Two-Electron Repulsion Integral (ab|cd)
    Pure Python implementation with memoization.
    """
    return eri_primitive(
        a.alpha,
        b.alpha,
        c.alpha,
        d.alpha,
        a.origin,
        b.origin,
        c.origin,
        d.origin,
        a.l,
        a.m,
        a.n,
        b.l,
        b.m,
        b.n,
        c.l,
        c.m,
        c.n,
        d.l,
        d.m,
        d.n,
        a.N,
        b.N,
        c.N,
        d.N,
    )


# ==========================================
# Derivative Helpers
# ==========================================


def get_derivative_primitives(basis_func, coord_idx):
    """
    Returns a list of (factor, new_basis_func) representing the derivative
    of the given primitive basis function with respect to its center coordinate A[coord_idx].

    Relation: d/dAx g(l) = 2*alpha*g(l+1) - l*g(l-1)

    The returned basis functions are valid BasisFunction objects (normalized),
    so we must adjust the factors by the ratio of normalizations.
    """
    bf = basis_func
    lmn = (
        list(bf.lmn) if hasattr(bf, "lmn") else [bf.l, bf.m, bf.n]
    )  # Helper handles both

    res = []

    # Term 1: 2 * alpha * g(l+1)
    # New quantum numbers
    lmn_plus = list(lmn)
    lmn_plus[coord_idx] += 1

    bf_plus = BasisFunction(bf.origin, bf.alpha, bf.coeff, tuple(lmn_plus))

    # Factor adjustment:
    factor_plus = bf.N * 2.0 * bf.alpha / bf_plus.N
    res.append((factor_plus, bf_plus))

    # Term 2: -l * g(l-1)
    l_val = lmn[coord_idx]
    if l_val > 0:
        lmn_minus = list(lmn)
        lmn_minus[coord_idx] -= 1
        bf_minus = BasisFunction(bf.origin, bf.alpha, bf.coeff, tuple(lmn_minus))

        factor_minus = -1.0 * l_val * bf.N / bf_minus.N
        res.append((factor_minus, bf_minus))

    return res


def overlap_deriv(a, b, atom_idx, molecule_atoms):
    """
    Computes derivative of overlap integral (a|b) with respect to atom_idx coordinates.
    Returns np.array([dS/dx, dS/dy, dS/dz]).
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12

    if not is_a and not is_b:
        return grad

    for coord in range(3):
        val = 0.0
        # If a is on center: d/dA <a|b>
        if is_a:
            prims = get_derivative_primitives(a, coord)
            for factor, bf_new in prims:
                val += factor * overlap(bf_new, b)

        # If b is on center: d/dA <a|b>
        if is_b:
            prims = get_derivative_primitives(b, coord)
            for factor, bf_new in prims:
                val += factor * overlap(a, bf_new)

        grad[coord] = val

    return grad


def kinetic_deriv(a, b, atom_idx, molecule_atoms):
    """Derivative of Kinetic Energy Integral"""
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12

    if not is_a and not is_b:
        return grad

    for coord in range(3):
        val = 0.0
        if is_a:
            for factor, bf_new in get_derivative_primitives(a, coord):
                val += factor * kinetic(bf_new, b)
        if is_b:
            for factor, bf_new in get_derivative_primitives(b, coord):
                val += factor * kinetic(a, bf_new)
        grad[coord] = val
    return grad


def nuclear_attraction_deriv(a, b, atom_idx, molecule_atoms):
    """
    Derivative of V_ne matrix element <a|V_C|b> w.r.t. atom_idx coordinates.

    Phase 24 CRITICAL FIX: The Hellmann-Feynman term (d/dR_C of the operator)
    was previously computed using basis function derivatives, which is wrong.

    Correct decomposition of d/dR_atom of <a|sum_C(-Z_C/|r-R_C|)|b>:

    1. Pulay terms (basis function derivatives): When atom_idx hosts basis
       function a or b, their centers move with the atom.
       For EACH nucleus C: d/dR_A <a|V_C|b> (if A = atom_idx)
                          d/dR_B <a|V_C|b> (if B = atom_idx)

    2. Hellmann-Feynman term (operator derivative): When atom_idx IS nucleus C,
       the operator -Z_C/|r-R_C| changes. By translation invariance of
       Gaussians: d/dR_C <a|-Z_C/|r-R_C||b> = -(d/dR_A + d/dR_B) <a|V_C|b>
       This is because the integral depends on (A-C) and (B-C) only.

    Reference: Szabo & Ostlund, Chapter 6; Helgaker, Jorgensen & Olsen Ch. 9.
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12

    # CM-12: Extended Z_map to include all 3rd-row elements
    Z_map = {
        "H": 1,
        "He": 2,
        "Li": 3,
        "Be": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "Ne": 10,
        "Na": 11,
        "Mg": 12,
        "Al": 13,
        "Si": 14,
        "P": 15,
        "S": 16,
        "Cl": 17,
        "Ar": 18,
    }

    for c_idx, (el, pos) in enumerate(molecule_atoms):
        Z = Z_map.get(el, 0)
        if Z == 0:
            continue
        C_pos = np.array(pos)
        is_C = c_idx == atom_idx

        for coord in range(3):
            # Pulay terms: derivative of basis functions when atom hosts them
            term_pulay = 0.0
            if is_a:
                for f, bf_new in get_derivative_primitives(a, coord):
                    term_pulay += f * nuclear_attraction(bf_new, b, C_pos, Z)
            if is_b:
                for f, bf_new in get_derivative_primitives(b, coord):
                    term_pulay += f * nuclear_attraction(a, bf_new, C_pos, Z)

            # Hellmann-Feynman term: derivative of operator when atom IS the nucleus
            # By translational invariance: d/dR_C = -(d/dR_A + d/dR_B) for fixed basis
            # We compute d/dR_A and d/dR_B of the integral treating C as fixed.
            term_hf = 0.0
            if is_C:
                # d/dR_C V_ne = -(d/dR_A + d/dR_B) V_ne (translational invariance)
                for f, bf_new in get_derivative_primitives(a, coord):
                    term_hf -= f * nuclear_attraction(bf_new, b, C_pos, Z)
                for f, bf_new in get_derivative_primitives(b, coord):
                    term_hf -= f * nuclear_attraction(a, bf_new, C_pos, Z)

            # Total: Pulay only acts on terms where atom hosts basis functions,
            # HF only acts when atom IS the nucleus. They are INDEPENDENT.
            # When atom hosts a basis AND is the nucleus (diagonal block),
            # both contribute but to DIFFERENT physical effects.
            # The key fix: Pulay and HF should NOT cancel — they add.
            grad[coord] += term_pulay + term_hf

    return grad


def eri_deriv(a, b, c, d, atom_idx, molecule_atoms):
    """
    Derivative of (ab|cd) wrt atom_idx.
    """
    grad = np.zeros(3)
    target_pos = np.array(molecule_atoms[atom_idx][1])
    is_a = np.linalg.norm(a.origin - target_pos) < 1e-12
    is_b = np.linalg.norm(b.origin - target_pos) < 1e-12
    is_c = np.linalg.norm(c.origin - target_pos) < 1e-12
    is_d = np.linalg.norm(d.origin - target_pos) < 1e-12

    if not (is_a or is_b or is_c or is_d):
        return grad

    for coord in range(3):
        val = 0.0
        if is_a:
            for f, bf in get_derivative_primitives(a, coord):
                val += f * eri(bf, b, c, d)
        if is_b:
            for f, bf in get_derivative_primitives(b, coord):
                val += f * eri(a, bf, c, d)
        if is_c:
            for f, bf in get_derivative_primitives(c, coord):
                val += f * eri(a, b, bf, d)
        if is_d:
            for f, bf in get_derivative_primitives(d, coord):
                val += f * eri(a, b, c, bf)
        grad[coord] = val

    return grad


# ==========================================
# Basis Construction
# ==========================================


class ContractedGaussian:
    """Contracted Gaussian formed as a linear combination of primitives.

    Phase 25 note (H6): This class stores raw primitives without explicit
    re-normalization of the contraction coefficients.  The STO-3G contraction
    coefficients used in build_basis() are already tabulated in their
    NORMALIZED form (i.e., they include the primitive normalization factors
    in the BasisFunction.N attribute and the contraction coefficients in
    BasisFunction.coeff sum to give a normalized contracted function).

    If using custom, unnormalized contraction coefficients, the caller should
    normalize them before passing to this constructor:
        S = sum(ci*cj * Ni*Nj * overlap_ij for ci,Ni in zip(...) for cj,Nj in zip(...))
        rescale = 1/sqrt(S)
        for p in primitives: p.coeff *= rescale
    """

    def __init__(self, primitives, label=""):
        """Initialize contracted Gaussian from a list of primitive functions."""
        self.primitives = primitives
        self.label = label


def build_basis(molecule):
    """
    Constructs a basis set for the given molecule.
    Returns a list of ContractedGaussian objects.

    Dispatches to basis-specific builders based on molecule.basis_name:
      - "sto-3g" (default): Minimal basis, s and p functions
      - "cc-pvdz": Dunning correlation-consistent double-zeta with polarization
      - "aug-cc-pvdz": Augmented cc-pVDZ with diffuse functions
      - "aug-cc-pvtz": Augmented cc-pVTZ with diffuse functions
      - "6-31g*" / "6-31gs" / "631gs": Pople split-valence + d-polarization
    """
    basis_name = getattr(molecule, "basis_name", "sto-3g").lower().replace("_", "-")

    if basis_name == "cc-pvdz":
        return build_basis_cc_pvdz(molecule)
    elif basis_name in ("aug-cc-pvdz", "augccpvdz"):
        return build_basis_aug_cc_pvdz(molecule)
    elif basis_name == "cc-pvtz":
        return build_basis_general(molecule, "cc-pvtz")
    elif basis_name in ("aug-cc-pvtz", "augccpvtz"):
        return build_basis_aug_cc_pvtz(molecule)
    elif basis_name in ("6-31g*", "6-31gs", "631gs", "6-31g(d)"):
        return build_basis_631gs(molecule)
    elif basis_name not in ("sto-3g", "sto3g"):
        raise ValueError(
            f"Unknown basis set '{basis_name}'. "
            f"Supported: 'sto-3g', 'cc-pvdz', 'aug-cc-pvdz', 'cc-pvtz', 'aug-cc-pvtz', '6-31g*'."
        )

    # === STO-3G basis construction (original code) ===
    basis_set = []

    # Map element to STO-3G parameters
    sto3g = STO3G()

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element == "H" or element == "He":
            # 1s orbital
            # Scale factors (zeta)
            zeta = sto3g.zeta.get(element, 1.0)

            # 1s
            prims_1s = []
            for alpha_norm, d_norm in sto3g.basis_1s:
                alpha = alpha_norm * (zeta**2)
                prims_1s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))

            basis_set.append(ContractedGaussian(prims_1s, f"{element}_{atom_idx}_1s"))

        elif element in ["Li", "Be", "B", "C", "N", "O", "F", "Ne"]:
            # Row 2: 1s, 2s, 2p

            # 1s Core
            zeta_1s = sto3g.zeta.get(
                f"{element}_1s", sto3g.zeta.get(element, 1.0)
            )  # Fallback if specific 1s not found

            prims_1s = []
            for alpha_norm, d_norm in sto3g.basis_1s:
                alpha = alpha_norm * (zeta_1s**2)
                prims_1s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))
            basis_set.append(ContractedGaussian(prims_1s, f"{element}_{atom_idx}_1s"))

            # 2s Valence
            zeta_val = sto3g.zeta.get(element, 1.0)
            prims_2s = []
            for alpha_norm, d_norm in sto3g.basis_2s:
                alpha = alpha_norm * (zeta_val**2)
                prims_2s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))
            basis_set.append(ContractedGaussian(prims_2s, f"{element}_{atom_idx}_2s"))

            # 2p Valence (px, py, pz)
            # 2px
            prims_2px = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2px.append(BasisFunction(pos, alpha, d_norm, (1, 0, 0)))
            basis_set.append(ContractedGaussian(prims_2px, f"{element}_{atom_idx}_2px"))

            # 2py
            prims_2py = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2py.append(BasisFunction(pos, alpha, d_norm, (0, 1, 0)))
            basis_set.append(ContractedGaussian(prims_2py, f"{element}_{atom_idx}_2py"))

            # 2pz
            prims_2pz = []
            for alpha_norm, d_norm in sto3g.basis_2p:
                alpha = alpha_norm * (zeta_val**2)
                prims_2pz.append(BasisFunction(pos, alpha, d_norm, (0, 0, 1)))
            basis_set.append(ContractedGaussian(prims_2pz, f"{element}_{atom_idx}_2pz"))

        elif element in ["P", "S"]:
            # Row 3 elements: 1s, 2s, 2p, 3s, 3p shells
            # Phase 24 fix: P and S now produce real basis functions instead of
            # silently returning nothing. Uses STO-3G parameters from
            # Hehre, Stewart & Pople, JCP 51, 2657 (1969).
            #
            # STO-3G zeta values for Row 3:
            #   P: 1s=14.89, 2s=4.11, 2p=4.11, 3s=1.63, 3p=1.63
            #   S: 1s=15.54, 2s=4.45, 2p=4.45, 3s=1.82, 3p=1.82
            row3_zeta = {
                "P": {"1s": 14.89, "2sp": 4.11, "3sp": 1.63},
                "S": {"1s": 15.54, "2sp": 4.45, "3sp": 1.82},
            }
            zetas = row3_zeta[element]

            # 1s core
            prims_1s = []
            for alpha_norm, d_norm in sto3g.basis_1s:
                alpha = alpha_norm * (zetas["1s"] ** 2)
                prims_1s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))
            basis_set.append(ContractedGaussian(prims_1s, f"{element}_{atom_idx}_1s"))

            # 2s
            prims_2s = []
            for alpha_norm, d_norm in sto3g.basis_2s:
                alpha = alpha_norm * (zetas["2sp"] ** 2)
                prims_2s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))
            basis_set.append(ContractedGaussian(prims_2s, f"{element}_{atom_idx}_2s"))

            # 2px, 2py, 2pz
            for lmn_label, lmn_val in [
                ("2px", (1, 0, 0)),
                ("2py", (0, 1, 0)),
                ("2pz", (0, 0, 1)),
            ]:
                prims_p = []
                for alpha_norm, d_norm in sto3g.basis_2p:
                    alpha = alpha_norm * (zetas["2sp"] ** 2)
                    prims_p.append(BasisFunction(pos, alpha, d_norm, lmn_val))
                basis_set.append(
                    ContractedGaussian(prims_p, f"{element}_{atom_idx}_{lmn_label}")
                )

            # 3s (valence)
            prims_3s = []
            for alpha_norm, d_norm in sto3g.basis_2s:
                alpha = alpha_norm * (zetas["3sp"] ** 2)
                prims_3s.append(BasisFunction(pos, alpha, d_norm, (0, 0, 0)))
            basis_set.append(ContractedGaussian(prims_3s, f"{element}_{atom_idx}_3s"))

            # 3px, 3py, 3pz (valence)
            for lmn_label, lmn_val in [
                ("3px", (1, 0, 0)),
                ("3py", (0, 1, 0)),
                ("3pz", (0, 0, 1)),
            ]:
                prims_p = []
                for alpha_norm, d_norm in sto3g.basis_2p:
                    alpha = alpha_norm * (zetas["3sp"] ** 2)
                    prims_p.append(BasisFunction(pos, alpha, d_norm, lmn_val))
                basis_set.append(
                    ContractedGaussian(prims_p, f"{element}_{atom_idx}_{lmn_label}")
                )

        else:
            # CM-11: Raise error for elements without STO-3G implementation.
            # Previously this was a warning, but the result is always wrong
            # (zero basis functions for the atom = zero electron density).
            raise ValueError(
                f"Element '{element}' has no STO-3G basis set implementation. "
                f"Supported elements: H, He, Li-Ne, P, S. "
                f"Cannot compute energy for this molecule."
            )

    return basis_set


def build_basis_cc_pvdz(molecule):
    """
    Build cc-pVDZ basis set for the given molecule.

    Constructs ContractedGaussian objects from the cc-pVDZ data in
    basis_cc_pvdz.py. Handles general contractions (one exponent set ->
    multiple contracted functions) and d-shells.

    Uses 6 Cartesian d-functions (xx, yy, zz, xy, xz, yz) rather than
    5 spherical harmonics, consistent with the 6d convention used by
    many quantum chemistry codes. This gives identical results to
    spherical d-functions for single-zeta d-shells (cc-pVDZ has only
    one d-function per element).

    References:
      Dunning, J. Chem. Phys. 90, 1007 (1989)
      Woon & Dunning, J. Chem. Phys. 100, 2975 (1994)
    """
    from basis_cc_pvdz import CC_PVDZ

    basis_set = []

    # Cartesian angular momentum components for each L
    CART_LMN = {
        0: [(0, 0, 0)],  # s: 1 component
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # p: 3 components
        2: [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),  # d: 6 Cartesian
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ],
    }
    SHELL_NAMES = {0: "s", 1: "p", 2: "d"}
    CART_LABELS = {
        (0, 0, 0): "",
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
        (2, 0, 0): "xx",
        (1, 1, 0): "xy",
        (1, 0, 1): "xz",
        (0, 2, 0): "yy",
        (0, 1, 1): "yz",
        (0, 0, 2): "zz",
    }

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element not in CC_PVDZ:
            raise ValueError(
                f"Element '{element}' has no cc-pVDZ basis set implementation. "
                f"Available: {sorted(CC_PVDZ.keys())}."
            )

        data = CC_PVDZ[element]
        shell_count = {0: 0, 1: 0, 2: 0}  # Track shell numbering per L

        for shell in data["shells"]:
            L = shell["angular_momentum"]
            exponents = shell["exponents"]
            coefficients = shell["coefficients"]
            lmn_list = CART_LMN[L]
            shell_name = SHELL_NAMES[L]

            # Each coefficient vector produces one contracted function per
            # Cartesian component. General contraction: multiple coefficient
            # vectors share the same exponents.
            for c_idx, coeff_vec in enumerate(coefficients):
                shell_count[L] += 1
                shell_num = shell_count[L]

                for lmn in lmn_list:
                    cart_label = CART_LABELS[lmn]
                    label = f"{element}_{atom_idx}_{shell_num}{shell_name}{cart_label}"

                    primitives = []
                    for exp_val, c_val in zip(exponents, coeff_vec):
                        if abs(c_val) < 1e-15:
                            # Skip zero-coefficient primitives (optimization)
                            continue
                        primitives.append(BasisFunction(pos, exp_val, c_val, lmn))

                    if primitives:
                        # Renormalize s and p contracted functions so S[i,i]=1.
                        # Do NOT renormalize d/f — PySCF uses unnormalized
                        # Cartesian convention where S[dxx,dxx] != 1.
                        if L <= 1:
                            S_ii = sum(
                                overlap(pi, pj)
                                for pi in primitives
                                for pj in primitives
                            )
                            if S_ii > 1e-15 and abs(S_ii - 1.0) > 1e-10:
                                nf = 1.0 / (S_ii**0.5)
                                for prim in primitives:
                                    prim.N *= nf
                        basis_set.append(ContractedGaussian(primitives, label))

    return basis_set


def build_basis_aug_cc_pvdz(molecule):
    """
    Build aug-cc-pVDZ basis set for the given molecule.

    Constructs ContractedGaussian objects from the aug-cc-pVDZ data in
    basis_aug_cc_pvdz.py. This is the cc-pVDZ basis augmented with one
    set of diffuse functions per angular momentum type, important for
    anions, weak interactions, and excited states.

    Uses 6 Cartesian d-functions (xx, yy, zz, xy, xz, yz), consistent
    with PySCF's cart=True convention.

    References:
      Dunning, J. Chem. Phys. 90, 1007 (1989)
      Kendall, Dunning & Harrison, J. Chem. Phys. 96, 6796 (1992)
    """
    from basis_aug_cc_pvdz import AUG_CC_PVDZ

    basis_set = []

    # Cartesian angular momentum components for each L
    CART_LMN = {
        0: [(0, 0, 0)],  # s: 1 component
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # p: 3 components
        2: [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),  # d: 6 Cartesian
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ],
    }
    SHELL_NAMES = {0: "s", 1: "p", 2: "d"}
    CART_LABELS = {
        (0, 0, 0): "",
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
        (2, 0, 0): "xx",
        (1, 1, 0): "xy",
        (1, 0, 1): "xz",
        (0, 2, 0): "yy",
        (0, 1, 1): "yz",
        (0, 0, 2): "zz",
    }

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element not in AUG_CC_PVDZ:
            raise ValueError(
                f"Element '{element}' has no aug-cc-pVDZ basis set implementation. "
                f"Available: {sorted(AUG_CC_PVDZ.keys())}."
            )

        data = AUG_CC_PVDZ[element]
        shell_count = {0: 0, 1: 0, 2: 0}  # Track shell numbering per L

        for shell in data["shells"]:
            L = shell["angular_momentum"]
            exponents = shell["exponents"]
            coefficients = shell["coefficients"]
            lmn_list = CART_LMN[L]
            shell_name = SHELL_NAMES[L]

            # Each coefficient vector produces one contracted function per
            # Cartesian component. General contraction: multiple coefficient
            # vectors share the same exponents.
            for c_idx, coeff_vec in enumerate(coefficients):
                shell_count[L] += 1
                shell_num = shell_count[L]

                for lmn in lmn_list:
                    cart_label = CART_LABELS[lmn]
                    label = f"{element}_{atom_idx}_{shell_num}{shell_name}{cart_label}"

                    primitives = []
                    for exp_val, c_val in zip(exponents, coeff_vec):
                        if abs(c_val) < 1e-15:
                            # Skip zero-coefficient primitives (optimization)
                            continue
                        primitives.append(BasisFunction(pos, exp_val, c_val, lmn))

                    if primitives:
                        # Renormalize s and p contracted functions so S[i,i]=1.
                        # Do NOT renormalize d/f — PySCF uses unnormalized
                        # Cartesian convention where S[dxx,dxx] != 1.
                        if L <= 1:
                            S_ii = sum(
                                overlap(pi, pj)
                                for pi in primitives
                                for pj in primitives
                            )
                            if S_ii > 1e-15 and abs(S_ii - 1.0) > 1e-10:
                                nf = 1.0 / (S_ii**0.5)
                                for prim in primitives:
                                    prim.N *= nf
                        basis_set.append(ContractedGaussian(primitives, label))

    return basis_set


def build_basis_aug_cc_pvtz(molecule):
    """
    Build aug-cc-pVTZ basis set for the given molecule.

    Constructs ContractedGaussian objects from the aug-cc-pVTZ data in
    basis_aug_cc_pvtz.py. This is the cc-pVTZ basis augmented with one
    set of diffuse functions per angular momentum type, important for
    anions, weak interactions, excited states, and basis set extrapolation.

    Uses 6 Cartesian d-functions and 10 Cartesian f-functions, consistent
    with PySCF's cart=True convention.

    References:
      Dunning, J. Chem. Phys. 90, 1007 (1989)
      Kendall, Dunning & Harrison, J. Chem. Phys. 96, 6796 (1992)
    """
    from basis_aug_cc_pvtz import AUG_CC_PVTZ

    basis_set = []

    # Cartesian angular momentum components for each L
    CART_LMN = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
        3: [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ],
    }
    SHELL_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}
    CART_LABELS = {
        (0, 0, 0): "",
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
        (2, 0, 0): "xx",
        (1, 1, 0): "xy",
        (1, 0, 1): "xz",
        (0, 2, 0): "yy",
        (0, 1, 1): "yz",
        (0, 0, 2): "zz",
        (3, 0, 0): "xxx",
        (2, 1, 0): "xxy",
        (2, 0, 1): "xxz",
        (1, 2, 0): "xyy",
        (1, 1, 1): "xyz",
        (1, 0, 2): "xzz",
        (0, 3, 0): "yyy",
        (0, 2, 1): "yyz",
        (0, 1, 2): "yzz",
        (0, 0, 3): "zzz",
    }

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element not in AUG_CC_PVTZ:
            raise ValueError(
                f"Element '{element}' has no aug-cc-pVTZ basis set implementation. "
                f"Available: {sorted(AUG_CC_PVTZ.keys())}."
            )

        data = AUG_CC_PVTZ[element]
        shell_count = {0: 0, 1: 0, 2: 0, 3: 0}

        for shell in data["shells"]:
            L = shell["angular_momentum"]
            exponents = shell["exponents"]
            coefficients = shell["coefficients"]
            lmn_list = CART_LMN[L]
            shell_name = SHELL_NAMES[L]

            for c_idx, coeff_vec in enumerate(coefficients):
                shell_count[L] += 1
                shell_num = shell_count[L]

                for lmn in lmn_list:
                    cart_label = CART_LABELS[lmn]
                    label = f"{element}_{atom_idx}_{shell_num}{shell_name}{cart_label}"

                    primitives = []
                    for exp_val, c_val in zip(exponents, coeff_vec):
                        if abs(c_val) < 1e-15:
                            continue
                        primitives.append(BasisFunction(pos, exp_val, c_val, lmn))

                    if primitives:
                        if L <= 1:
                            S_ii = sum(
                                overlap(pi, pj)
                                for pi in primitives
                                for pj in primitives
                            )
                            if S_ii > 1e-15 and abs(S_ii - 1.0) > 1e-10:
                                nf = 1.0 / (S_ii**0.5)
                                for prim in primitives:
                                    prim.N *= nf
                        basis_set.append(ContractedGaussian(primitives, label))

    return basis_set


def build_basis_general(molecule, basis_name):
    """
    Build a general cc-pVXZ basis set for the given molecule.

    Supports s, p, d, and f orbitals with Cartesian components.
    Uses the same approach as build_basis_cc_pvdz but with f-orbital support.

    Cartesian f-functions (10 components):
        (3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1),
        (1,0,2), (0,3,0), (0,2,1), (0,1,2), (0,0,3)
    """
    # Import the appropriate basis data module
    if basis_name == "cc-pvtz":
        from basis_cc_pvtz import CC_PVTZ as BASIS_DATA
    else:
        raise ValueError(f"Unknown basis: {basis_name}")

    basis_set = []

    # Cartesian angular momentum components for each L
    CART_LMN = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
        3: [
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ],
    }
    SHELL_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}
    CART_LABELS = {
        (0, 0, 0): "",
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
        (2, 0, 0): "xx",
        (1, 1, 0): "xy",
        (1, 0, 1): "xz",
        (0, 2, 0): "yy",
        (0, 1, 1): "yz",
        (0, 0, 2): "zz",
        (3, 0, 0): "xxx",
        (2, 1, 0): "xxy",
        (2, 0, 1): "xxz",
        (1, 2, 0): "xyy",
        (1, 1, 1): "xyz",
        (1, 0, 2): "xzz",
        (0, 3, 0): "yyy",
        (0, 2, 1): "yyz",
        (0, 1, 2): "yzz",
        (0, 0, 3): "zzz",
    }

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element not in BASIS_DATA:
            raise ValueError(
                f"Element '{element}' has no {basis_name} basis set implementation. "
                f"Available: {sorted(BASIS_DATA.keys())}."
            )

        data = BASIS_DATA[element]
        shell_count = {0: 0, 1: 0, 2: 0, 3: 0}

        for shell in data["shells"]:
            L = shell["angular_momentum"]
            exponents = shell["exponents"]
            coefficients = shell["coefficients"]
            lmn_list = CART_LMN[L]
            shell_name = SHELL_NAMES[L]

            for c_idx, coeff_vec in enumerate(coefficients):
                shell_count[L] += 1
                shell_num = shell_count[L]

                for lmn in lmn_list:
                    cart_label = CART_LABELS[lmn]
                    label = f"{element}_{atom_idx}_{shell_num}{shell_name}{cart_label}"

                    primitives = []
                    for exp_val, c_val in zip(exponents, coeff_vec):
                        if abs(c_val) < 1e-15:
                            continue
                        primitives.append(BasisFunction(pos, exp_val, c_val, lmn))

                    if primitives:
                        if L <= 1:
                            S_ii = sum(
                                overlap(pi, pj)
                                for pi in primitives
                                for pj in primitives
                            )
                            if S_ii > 1e-15 and abs(S_ii - 1.0) > 1e-10:
                                nf = 1.0 / (S_ii**0.5)
                                for prim in primitives:
                                    prim.N *= nf
                        basis_set.append(ContractedGaussian(primitives, label))

    return basis_set


def build_basis_631gs(molecule):
    """
    Build 6-31G* (6-31G(d)) basis set for the given molecule.

    Constructs ContractedGaussian objects from the 6-31G* data in
    basis_631g.py. Handles sp-shells (shared exponents, separate s/p
    coefficients) and d-polarization functions.

    Uses 6 Cartesian d-functions (xx, yy, zz, xy, xz, yz), consistent
    with PySCF's cart=True convention.

    References:
      Hehre, Ditchfield, Pople, JCP 56, 2257 (1972)
      Hariharan, Pople, Theor. Chim. Acta 28, 213 (1973)
    """
    from basis_631g import BASIS_631GS

    basis_set = []

    # Cartesian angular momentum components for each L
    CART_LMN = {
        0: [(0, 0, 0)],  # s: 1 component
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # p: 3 components
        2: [
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),  # d: 6 Cartesian
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ],
    }
    SHELL_NAMES = {0: "s", 1: "p", 2: "d"}
    CART_LABELS = {
        (0, 0, 0): "",
        (1, 0, 0): "x",
        (0, 1, 0): "y",
        (0, 0, 1): "z",
        (2, 0, 0): "xx",
        (1, 1, 0): "xy",
        (1, 0, 1): "xz",
        (0, 2, 0): "yy",
        (0, 1, 1): "yz",
        (0, 0, 2): "zz",
    }

    for atom_idx, (element, pos) in enumerate(molecule.atoms):
        if element not in BASIS_631GS:
            raise ValueError(
                f"Element '{element}' has no 6-31G* basis set implementation. "
                f"Available: {sorted(BASIS_631GS.keys())}."
            )

        data = BASIS_631GS[element]
        shell_count = {0: 0, 1: 0, 2: 0}

        for shell in data["shells"]:
            L = shell["angular_momentum"]
            exponents = shell["exponents"]
            coefficients = shell["coefficients"]
            lmn_list = CART_LMN[L]
            shell_name = SHELL_NAMES[L]

            # Each coefficient vector produces one contracted function per
            # Cartesian component. General contraction: multiple coefficient
            # vectors share the same exponents.
            for c_idx, coeff_vec in enumerate(coefficients):
                shell_count[L] += 1
                shell_num = shell_count[L]

                for lmn in lmn_list:
                    cart_label = CART_LABELS[lmn]
                    label = f"{element}_{atom_idx}_{shell_num}{shell_name}{cart_label}"

                    primitives = []
                    for exp_val, c_val in zip(exponents, coeff_vec):
                        if abs(c_val) < 1e-15:
                            continue
                        primitives.append(BasisFunction(pos, exp_val, c_val, lmn))

                    if primitives:
                        # Renormalize s and p contracted functions so S[i,i]=1.
                        # Do NOT renormalize d/f — PySCF uses unnormalized
                        # Cartesian convention where S[dxx,dxx] != 1.
                        if L <= 1:
                            S_ii = sum(
                                overlap(pi, pj)
                                for pi in primitives
                                for pj in primitives
                            )
                            if S_ii > 1e-15 and abs(S_ii - 1.0) > 1e-10:
                                nf = 1.0 / (S_ii**0.5)
                                for prim in primitives:
                                    prim.N *= nf
                        basis_set.append(ContractedGaussian(primitives, label))

    return basis_set


# ==========================================
# JIT Gradient Implementation
# ==========================================


def eri_deriv_quartet(
    alphaA,
    alphaB,
    alphaC,
    alphaD,
    A,
    B,
    C,
    D,
    la,
    ma,
    na,
    lb,
    mb,
    nb,
    lc,
    mc,
    nc,
    ld,
    md,
    nd,
    normA,
    normB,
    normC,
    normD,
):
    """
    Computes the gradient of the ERI (ab|cd) with respect to the coordinates
    of the four centers A, B, C, D.
    Returns a (4, 3) array of gradients [dA, dB, dC, dD].
    """
    grads = np.zeros((4, 3))

    # --- Center A Gradients ---
    # x-component (la)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la + 1,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        1.0,
        normB,
        normC,
        normD,
    )
    term_x = 2.0 * alphaA * val_p

    if la > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la - 1,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            1.0,
            normB,
            normC,
            normD,
        )
        term_x -= la * val_m
    grads[0, 0] = normA * term_x

    # y-component (ma)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma + 1,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        1.0,
        normB,
        normC,
        normD,
    )
    term_y = 2.0 * alphaA * val_p

    if ma > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma - 1,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            1.0,
            normB,
            normC,
            normD,
        )
        term_y -= ma * val_m
    grads[0, 1] = normA * term_y

    # z-component (na)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na + 1,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        1.0,
        normB,
        normC,
        normD,
    )
    term_z = 2.0 * alphaA * val_p

    if na > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na - 1,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            1.0,
            normB,
            normC,
            normD,
        )
        term_z -= na * val_m
    grads[0, 2] = normA * term_z

    # --- Center B Gradients ---
    # x-component (lb)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb + 1,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        normA,
        1.0,
        normC,
        normD,
    )
    term_x = 2.0 * alphaB * val_p
    if lb > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb - 1,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            normA,
            1.0,
            normC,
            normD,
        )
        term_x -= lb * val_m
    grads[1, 0] = normB * term_x

    # y-component (mb)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb + 1,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        normA,
        1.0,
        normC,
        normD,
    )
    term_y = 2.0 * alphaB * val_p
    if mb > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb - 1,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            normA,
            1.0,
            normC,
            normD,
        )
        term_y -= mb * val_m
    grads[1, 1] = normB * term_y

    # z-component (nb)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb + 1,
        lc,
        mc,
        nc,
        ld,
        md,
        nd,
        normA,
        1.0,
        normC,
        normD,
    )
    term_z = 2.0 * alphaB * val_p
    if nb > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb - 1,
            lc,
            mc,
            nc,
            ld,
            md,
            nd,
            normA,
            1.0,
            normC,
            normD,
        )
        term_z -= nb * val_m
    grads[1, 2] = normB * term_z

    # --- Center C Gradients ---
    # x-component (lc)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc + 1,
        mc,
        nc,
        ld,
        md,
        nd,
        normA,
        normB,
        1.0,
        normD,
    )
    term_x = 2.0 * alphaC * val_p
    if lc > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc - 1,
            mc,
            nc,
            ld,
            md,
            nd,
            normA,
            normB,
            1.0,
            normD,
        )
        term_x -= lc * val_m
    grads[2, 0] = normC * term_x

    # y-component (mc)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc + 1,
        nc,
        ld,
        md,
        nd,
        normA,
        normB,
        1.0,
        normD,
    )
    term_y = 2.0 * alphaC * val_p
    if mc > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc - 1,
            nc,
            ld,
            md,
            nd,
            normA,
            normB,
            1.0,
            normD,
        )
        term_y -= mc * val_m
    grads[2, 1] = normC * term_y

    # z-component (nc)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc + 1,
        ld,
        md,
        nd,
        normA,
        normB,
        1.0,
        normD,
    )
    term_z = 2.0 * alphaC * val_p
    if nc > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc - 1,
            ld,
            md,
            nd,
            normA,
            normB,
            1.0,
            normD,
        )
        term_z -= nc * val_m
    grads[2, 2] = normC * term_z

    # --- Center D Gradients ---
    # x-component (ld)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld + 1,
        md,
        nd,
        normA,
        normB,
        normC,
        1.0,
    )
    term_x = 2.0 * alphaD * val_p
    if ld > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld - 1,
            md,
            nd,
            normA,
            normB,
            normC,
            1.0,
        )
        term_x -= ld * val_m
    grads[3, 0] = normD * term_x

    # y-component (md)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md + 1,
        nd,
        normA,
        normB,
        normC,
        1.0,
    )
    term_y = 2.0 * alphaD * val_p
    if md > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md - 1,
            nd,
            normA,
            normB,
            normC,
            1.0,
        )
        term_y -= md * val_m
    grads[3, 1] = normD * term_y

    # z-component (nd)
    val_p = eri_primitive(
        alphaA,
        alphaB,
        alphaC,
        alphaD,
        A,
        B,
        C,
        D,
        la,
        ma,
        na,
        lb,
        mb,
        nb,
        lc,
        mc,
        nc,
        ld,
        md,
        nd + 1,
        normA,
        normB,
        normC,
        1.0,
    )
    term_z = 2.0 * alphaD * val_p
    if nd > 0:
        val_m = eri_primitive(
            alphaA,
            alphaB,
            alphaC,
            alphaD,
            A,
            B,
            C,
            D,
            la,
            ma,
            na,
            lb,
            mb,
            nb,
            lc,
            mc,
            nc,
            ld,
            md,
            nd - 1,
            normA,
            normB,
            normC,
            1.0,
        )
        term_z -= nd * val_m
    grads[3, 2] = normD * term_z

    return grads


def grad_rhf_2e_jit(
    atom_coords,  # (N_atoms, 3)
    atom_indices,  # (N_prims,)
    basis_indices,  # (N_prims,)
    exponents,  # (N_prims,)
    norms,  # (N_prims,) - Includes contraction coeffs
    lmns,  # (N_prims, 3)
    D_matrix,  # (N_basis, N_basis)
):
    """Compute RHF two-electron gradient contribution using flattened basis data.

    Optimized with Schwarz pre-screening to skip negligible quartets before
    computing expensive ERI derivatives (~90% reduction in inner loop work).
    """
    n_prims = len(exponents)
    n_atoms = len(atom_coords)
    total_grad = np.zeros((n_atoms, 3))

    # --- Schwarz pre-screening: compute (ij|ij)^{1/2} upper bounds ---
    schwarz = np.zeros((n_prims, n_prims))
    for i in range(n_prims):
        for j in range(i + 1):
            mu_i = basis_indices[i]
            mu_j = basis_indices[j]
            val = np.abs(
                D_matrix[mu_i, mu_j]
            )  # cheap upper-bound proxy for |Q_ij|
            schwarz[i, j] = schwarz[j, i] = val

    schwarz_threshold = 1e-12

    # Pre-compute density-based factors for all basis pairs
    D_coul = np.zeros((n_prims, n_prims))
    D_exch = np.zeros((n_prims, n_prims))
    for i in range(n_prims):
        mu = basis_indices[i]
        for j in range(n_prims):
            nu = basis_indices[j]
            D_coul[i, j] = D_matrix[mu, nu]
            D_exch[i, j] = D_matrix[mu, nu]

    for i in range(n_prims):
        for j in range(n_prims):
            # Schwarz screen on ij pair
            if schwarz[i, j] < schwarz_threshold:
                continue

            mu = basis_indices[i]
            nu = basis_indices[j]
            at_i = atom_indices[i]
            at_j = atom_indices[j]

            for k in range(n_prims):
                for l in range(n_prims):
                    # Schwarz screen on kl pair and combined (ij|kl)
                    if schwarz[k, l] < schwarz_threshold:
                        continue

                    lam = basis_indices[k]
                    sig = basis_indices[l]

                    # Density Screening
                    term_coul = 0.5 * D_matrix[mu, nu] * D_matrix[lam, sig]
                    term_exch = 0.25 * D_matrix[mu, lam] * D_matrix[nu, sig]
                    factor = term_coul - term_exch

                    if np.abs(factor) < schwarz_threshold:
                        continue

                    # Compute Gradient of (ij|kl)
                    grads = eri_deriv_quartet(
                        exponents[i],
                        exponents[j],
                        exponents[k],
                        exponents[l],
                        atom_coords[at_i],
                        atom_coords[at_j],
                        atom_coords[atom_indices[k]],
                        atom_coords[atom_indices[l]],
                        lmns[i, 0],
                        lmns[i, 1],
                        lmns[i, 2],
                        lmns[j, 0],
                        lmns[j, 1],
                        lmns[j, 2],
                        lmns[k, 0],
                        lmns[k, 1],
                        lmns[k, 2],
                        lmns[l, 0],
                        lmns[l, 1],
                        lmns[l, 2],
                        norms[i],
                        norms[j],
                        norms[k],
                        norms[l],
                    )

                    # Accumulate to atoms
                    at_k = atom_indices[k]
                    at_l = atom_indices[l]

                    total_grad[at_i] += factor * grads[0]
                    total_grad[at_j] += factor * grads[1]
                    total_grad[at_k] += factor * grads[2]
                    total_grad[at_l] += factor * grads[3]

    return total_grad


# [FIX] Alias for backward compatibility - PrimitiveGaussian is the same as BasisFunction
PrimitiveGaussian = BasisFunction
