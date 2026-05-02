"""
PROMETHEUS Backend for QENEX Chemistry
=======================================

This module provides PROMETHEUS-accelerated matrix operations for the
Hartree-Fock solver. Falls back to NumPy when PROMETHEUS is unavailable.

Binding Priority:
    1. PyO3 bindings (qenex_accelerate) - lowest overhead
    2. ctypes bindings (prometheus module) - legacy support
    3. NumPy fallback

Key optimizations:
    - DGEMM for matrix orthogonalization (X.T @ H @ X)
    - DGEMM for density matrix construction
    - DGEMM for Fock matrix diagonalization prep

Performance:
    - PyO3 overhead: ~0.05ms per call
    - Competitive with OpenBLAS for N=200-300
    - Some sweet spots where PROMETHEUS wins

Usage:
    from prometheus_backend import use_prometheus, matmul, build_density_matrix

    # Check if PROMETHEUS is available
    if use_prometheus():
        C = matmul(X.T, matmul(H, X))  # Uses PROMETHEUS
    else:
        C = X.T @ H @ X  # Falls back to NumPy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Any, TYPE_CHECKING
import os

# Type aliases
FloatArray = NDArray[np.float64]

# Backend selection - use lazy initialization
_backend: Optional[str] = None  # 'pyo3', 'ctypes', or None
_prometheus_available: Optional[bool] = None  # None means not initialized yet
_initialized: bool = False

# Module references (set during initialization)
_pyo3_module: Any = None
_ctypes_dgemm_func: Any = None
_ctypes_dot_func: Any = None


# Canonical search paths for libprometheus_c.so, in priority order.
# Extend this list when new install locations become standard.
_PROMETHEUS_LIBRARY_CANDIDATES = (
    # 1. User-supplied override
    "PROMETHEUS_LIB_PATH",  # indirect: os.environ["PROMETHEUS_LIB_PATH"]
    # 2. User-supplied home (auto-constructs <home>/build/libprometheus_c.so)
    "PROMETHEUS_HOME",  # indirect: $PROMETHEUS_HOME/build/libprometheus_c.so
    # 3. Standard ldconfig-visible install locations
    "/usr/local/lib/libprometheus_c.so",
    "/usr/lib/libprometheus_c.so",
    "/usr/lib/x86_64-linux-gnu/libprometheus_c.so",
    # 4. Canonical developer build tree on a QENEX workstation.  Referenced
    #    only as a fallback if the lib was not installed system-wide.
    "/home/ubuntu/prometheus-unchained/build/libprometheus_c.so",
    # 5. A build tree co-located with the qenex-lab repo, e.g. for CI
    #    containers that check out prometheus-unchained as a sibling repo.
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "..",
        "prometheus-unchained",
        "build",
        "libprometheus_c.so",
    ),
)


def _find_prometheus_library() -> Optional[str]:
    """Search canonical locations for ``libprometheus_c.so``.

    Returns the absolute path to the library if found, or ``None`` if
    no candidate exists.  Propagates the discovered path into the
    ``PROMETHEUS_LIB_PATH`` environment variable so the Rust PyO3
    extension's ``load_prometheus_library()`` function picks it up.
    """
    # 1. Explicit environment override — always honour it first if set.
    explicit = os.environ.get("PROMETHEUS_LIB_PATH")
    if explicit and os.path.exists(explicit):
        return explicit

    # 2. $PROMETHEUS_HOME/build/libprometheus_c.so
    home = os.environ.get("PROMETHEUS_HOME")
    if home:
        candidate = os.path.join(home, "build", "libprometheus_c.so")
        if os.path.exists(candidate):
            os.environ["PROMETHEUS_LIB_PATH"] = candidate
            return candidate

    # 3. Concrete filesystem paths.
    for candidate in _PROMETHEUS_LIBRARY_CANDIDATES:
        if candidate in ("PROMETHEUS_LIB_PATH", "PROMETHEUS_HOME"):
            continue  # handled above
        if os.path.exists(candidate):
            os.environ.setdefault("PROMETHEUS_LIB_PATH", candidate)
            return candidate

    return None


def _init_backend() -> None:
    """Initialize the PROMETHEUS backend (called lazily on first use).

    Discovery order:

      1. ``qenex_accelerate`` PyO3 extension (optimal overhead).  This
         wraps the Rust + FFI binding to ``libprometheus_c.so``.
      2. Legacy ``prometheus`` ctypes shim (older layouts).
      3. Silent NumPy fallback (diagnostic message emitted).

    Before either wrapper is attempted, we search a canonical list of
    paths for ``libprometheus_c.so`` and inject the discovery into
    ``PROMETHEUS_LIB_PATH`` so the Rust side picks it up even when
    the library is not on the default ``ldconfig`` path.  This makes
    Prometheus auto-detect on a freshly-cloned workspace without any
    per-developer environment setup.
    """
    global \
        _backend, \
        _prometheus_available, \
        _pyo3_module, \
        _ctypes_dgemm_func, \
        _ctypes_dot_func, \
        _initialized

    if _initialized:
        return

    _initialized = True

    # Locate libprometheus_c.so BEFORE attempting PyO3 import — the Rust
    # side reads PROMETHEUS_LIB_PATH during module init via a Once guard,
    # so the env var must be set first to avoid a silent "not available"
    # status from the Rust extension.
    lib_path = _find_prometheus_library()

    # ------------------------------------------------------------------
    # Attempt 1: PyO3 (qenex_accelerate) — the modern path
    # ------------------------------------------------------------------
    try:
        import qenex_accelerate  # type: ignore[import-not-found]

        if qenex_accelerate.prometheus_is_available():  # type: ignore[attr-defined]
            _backend = "pyo3"
            _prometheus_available = True
            _pyo3_module = qenex_accelerate
            lib_info = f" via {lib_path}" if lib_path else ""
            print(f"[PROMETHEUS] PyO3 backend loaded (qenex_accelerate){lib_info}")
            return
        else:
            # Rust extension is installed but the C library wasn't found
            # at Rust init time.  This happens on fresh VMs where neither
            # PROMETHEUS_LIB_PATH nor /usr/local/lib is set up yet.
            info = qenex_accelerate.prometheus_info()  # type: ignore[attr-defined]
            print(
                "[PROMETHEUS] qenex_accelerate imported but backend reports "
                f"unavailable: {info}"
            )
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Attempt 2: legacy ctypes shim
    # ------------------------------------------------------------------
    try:
        import sys

        accelerate_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "qenex-accelerate",
        )
        if accelerate_path not in sys.path:
            sys.path.insert(0, accelerate_path)

        from prometheus import dgemm, dot, is_available  # type: ignore[import-not-found]

        if is_available():
            _backend = "ctypes"
            _prometheus_available = True
            _ctypes_dgemm_func = dgemm
            _ctypes_dot_func = dot
            print("[PROMETHEUS] ctypes backend loaded (legacy)")
            return
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # Attempt 3: explicit PROMETHEUS_LIB_PATH + legacy ctypes
    # ------------------------------------------------------------------
    if lib_path:
        try:
            import ctypes

            ctypes.CDLL(lib_path)
            from prometheus import dgemm, dot, is_available  # type: ignore[import-not-found]

            if is_available():
                _backend = "ctypes"
                _prometheus_available = True
                _ctypes_dgemm_func = dgemm
                _ctypes_dot_func = dot
                print(f"[PROMETHEUS] ctypes backend loaded from {lib_path}")
                return
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Final state: not available — give diagnostic context so the
    # (now rare) situation is easy to debug.
    # ------------------------------------------------------------------
    _prometheus_available = False
    if lib_path is None:
        print(
            "[PROMETHEUS] Backend not available, using NumPy fallback.\n"
            "           libprometheus_c.so not found on any canonical path.\n"
            "           Set PROMETHEUS_LIB_PATH or install to /usr/local/lib."
        )
    else:
        print(
            "[PROMETHEUS] Backend not available, using NumPy fallback.\n"
            f"           library discovered at {lib_path} but neither\n"
            "           qenex_accelerate nor the legacy ctypes shim could\n"
            "           bind to it.  Check Python environment."
        )


def use_prometheus() -> bool:
    """Check if PROMETHEUS acceleration is available."""
    _init_backend()  # Lazy initialization
    return _prometheus_available if _prometheus_available is not None else False


def get_backend() -> Optional[str]:
    """Get the current backend type ('pyo3', 'ctypes', or None)."""
    _init_backend()  # Lazy initialization
    return _backend


def matmul(
    A: FloatArray,
    B: FloatArray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[FloatArray] = None,
) -> FloatArray:
    """
    Matrix multiplication with optional PROMETHEUS acceleration.

    Computes: C = alpha * A @ B + beta * C

    Args:
        A: Left matrix (M x K)
        B: Right matrix (K x N)
        alpha: Scalar for A @ B (default 1.0)
        beta: Scalar for C (default 0.0)
        C: Optional output matrix

    Returns:
        Result matrix (M x N)

    Safety
    ------
    The underlying ``prometheus_dgemm`` C kernel passes raw pointers to
    a C-contiguous double array and silently produces wrong results if
    the caller hands it a non-contiguous view (e.g. ``A.T`` or an
    ndarray slice with non-unit strides).  We therefore materialise
    both operands via ``np.ascontiguousarray`` before every DGEMM call.
    For already-contiguous inputs this is a cheap no-op; for views it
    adds one small copy but guarantees correctness.
    """
    _init_backend()  # Ensure initialization
    if _backend == "pyo3" and _pyo3_module is not None:
        A_c = np.ascontiguousarray(A, dtype=np.float64)
        B_c = np.ascontiguousarray(B, dtype=np.float64)
        return _pyo3_module.prometheus_dgemm(A_c, B_c, alpha, beta)
    elif _backend == "ctypes" and _ctypes_dgemm_func is not None:
        A_c = np.ascontiguousarray(A, dtype=np.float64)
        B_c = np.ascontiguousarray(B, dtype=np.float64)
        return _ctypes_dgemm_func(A_c, B_c, alpha=alpha, beta=beta, C=C)
    else:
        if C is None:
            return alpha * (A @ B)
        else:
            return alpha * (A @ B) + beta * C


def triple_product(A: FloatArray, B: FloatArray, C: FloatArray) -> FloatArray:
    """
    Compute A @ B @ C efficiently.

    Uses two DGEMM calls when PROMETHEUS is available.
    Optimal for orthogonalization transforms like X.T @ H @ X.

    Args:
        A, B, C: Input matrices

    Returns:
        A @ B @ C
    """
    _init_backend()  # Ensure initialization
    if _backend == "pyo3" and _pyo3_module is not None:
        return _pyo3_module.prometheus_triple_product(A, B, C)
    elif _backend == "ctypes" and _ctypes_dgemm_func is not None:
        tmp = _ctypes_dgemm_func(A, B)
        return _ctypes_dgemm_func(tmp, C)
    else:
        return A @ B @ C


def build_density_matrix(C: FloatArray, n_occ: int) -> FloatArray:
    """
    Build RHF density matrix P from MO coefficients.

    P_μν = 2 * Σ_i^occ C_μi * C_νi = 2 * C_occ @ C_occ.T

    This is a key operation that benefits from PROMETHEUS DGEMM.

    Args:
        C: MO coefficient matrix (N x N)
        n_occ: Number of occupied orbitals

    Returns:
        Density matrix P (N x N)
    """
    _init_backend()  # Ensure initialization
    if _backend == "pyo3" and _pyo3_module is not None:
        return _pyo3_module.prometheus_build_density(C, n_occ)
    elif _backend == "ctypes" and _ctypes_dgemm_func is not None:
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        return _ctypes_dgemm_func(C_occ, C_occ.T, alpha=2.0)
    else:
        C_occ = np.ascontiguousarray(C[:, :n_occ])
        return 2.0 * C_occ @ C_occ.T


def build_density_matrix_uhf(
    C_alpha: FloatArray, C_beta: FloatArray, n_alpha: int, n_beta: int
) -> Tuple[FloatArray, FloatArray]:
    """
    Build UHF density matrices from MO coefficients.

    P_α = C_α_occ @ C_α_occ.T
    P_β = C_β_occ @ C_β_occ.T

    Args:
        C_alpha, C_beta: MO coefficient matrices
        n_alpha, n_beta: Number of occupied orbitals

    Returns:
        Tuple of (P_alpha, P_beta)
    """
    _init_backend()  # Ensure initialization

    # Try optimized Rust UHF function first
    if _backend == "pyo3" and _pyo3_module is not None:
        if hasattr(_pyo3_module, "prometheus_build_density_uhf"):
            return _pyo3_module.prometheus_build_density_uhf(
                C_alpha, C_beta, n_alpha, n_beta
            )
        else:
            # Fall back to two separate DGEMM calls
            C_a_occ = np.ascontiguousarray(C_alpha[:, :n_alpha])
            C_b_occ = np.ascontiguousarray(C_beta[:, :n_beta])
            C_a_occ_T = np.ascontiguousarray(C_a_occ.T)
            C_b_occ_T = np.ascontiguousarray(C_b_occ.T)
            P_alpha = _pyo3_module.prometheus_dgemm(C_a_occ, C_a_occ_T, 1.0, 0.0)
            P_beta = _pyo3_module.prometheus_dgemm(C_b_occ, C_b_occ_T, 1.0, 0.0)
            return P_alpha, P_beta
    elif _backend == "ctypes" and _ctypes_dgemm_func is not None:
        C_a_occ = np.ascontiguousarray(C_alpha[:, :n_alpha])
        C_b_occ = np.ascontiguousarray(C_beta[:, :n_beta])
        C_a_occ_T = np.ascontiguousarray(C_a_occ.T)
        C_b_occ_T = np.ascontiguousarray(C_b_occ.T)
        P_alpha = _ctypes_dgemm_func(C_a_occ, C_a_occ_T)
        P_beta = _ctypes_dgemm_func(C_b_occ, C_b_occ_T)
        return P_alpha, P_beta
    else:
        C_a_occ = np.ascontiguousarray(C_alpha[:, :n_alpha])
        C_b_occ = np.ascontiguousarray(C_beta[:, :n_beta])
        P_alpha = C_a_occ @ C_a_occ.T
        P_beta = C_b_occ @ C_b_occ.T
        return P_alpha, P_beta


def transform_fock(X: FloatArray, F: FloatArray) -> FloatArray:
    """
    Transform Fock matrix to orthogonal basis.

    F' = X.T @ F @ X

    This triple product is a prime candidate for PROMETHEUS acceleration.

    Args:
        X: Orthogonalization matrix (S^-1/2)
        F: Fock matrix in AO basis

    Returns:
        F' in orthogonal basis
    """
    _init_backend()  # Ensure initialization
    if _backend == "pyo3" and _pyo3_module is not None:
        return _pyo3_module.prometheus_transform_fock(X, F)
    else:
        return triple_product(X.T, F, X)


def back_transform_coefficients(X: FloatArray, C_prime: FloatArray) -> FloatArray:
    """
    Back-transform MO coefficients from orthogonal to AO basis.

    C = X @ C'

    Args:
        X: Orthogonalization matrix
        C_prime: Coefficients in orthogonal basis

    Returns:
        C in AO basis
    """
    return matmul(X, C_prime)


def compute_electronic_energy(
    P: FloatArray, H_core: FloatArray, F: FloatArray
) -> float:
    """
    Compute electronic energy from density and Fock matrices.

    E_elec = 0.5 * Tr[P(H + F)] = 0.5 * Σ_μν P_μν (H_μν + F_μν)

    This can be computed as a dot product of flattened matrices.

    Args:
        P: Density matrix
        H_core: Core Hamiltonian
        F: Fock matrix

    Returns:
        Electronic energy
    """
    _init_backend()  # Ensure initialization
    combined = H_core + F
    if _backend == "pyo3" and _pyo3_module is not None:
        result = 0.5 * _pyo3_module.prometheus_dot(P.ravel(), combined.ravel())
        return float(result)
    elif _backend == "ctypes" and _ctypes_dot_func is not None:
        result = 0.5 * _ctypes_dot_func(P.ravel(), combined.ravel())
        return float(result)
    else:
        return float(0.5 * np.sum(P * combined))


def compute_energy_weighted_density(
    C: FloatArray, eps: FloatArray, n_occ: int
) -> FloatArray:
    """
    Compute energy-weighted density matrix for gradient calculations.

    W_μν = 2 * Σ_i^occ ε_i * C_μi * C_νi

    Can be reformulated as W = 2 * C_occ @ diag(eps_occ) @ C_occ.T

    Args:
        C: MO coefficients
        eps: Orbital energies
        n_occ: Number of occupied orbitals

    Returns:
        Energy-weighted density matrix W
    """
    _init_backend()  # Ensure initialization
    C_occ = np.ascontiguousarray(C[:, :n_occ])
    eps_occ = eps[:n_occ]

    # W = 2 * C_occ @ diag(eps_occ) @ C_occ.T
    # = 2 * (C_occ * eps_occ) @ C_occ.T
    C_weighted = C_occ * eps_occ  # Broadcasting

    if _backend == "pyo3" and _pyo3_module is not None:
        return _pyo3_module.prometheus_dgemm(C_weighted, C_occ.T, 2.0, 0.0)
    elif _backend == "ctypes" and _ctypes_dgemm_func is not None:
        return _ctypes_dgemm_func(C_weighted, C_occ.T, alpha=2.0)
    else:
        return 2.0 * C_weighted @ C_occ.T


# Benchmark utility
def benchmark_prometheus(N: int = 256, iterations: int = 10) -> dict[str, Any]:
    """
    Benchmark PROMETHEUS vs NumPy for typical HF operations.

    Args:
        N: Matrix size
        iterations: Number of iterations

    Returns:
        Dictionary with timing results
    """
    import time

    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)

    # Warmup
    _ = A @ B
    if _prometheus_available:
        _ = matmul(A, B)

    # NumPy timing
    start = time.perf_counter()
    for _ in range(iterations):
        _ = A @ B
    numpy_time = (time.perf_counter() - start) / iterations

    # PROMETHEUS timing
    prometheus_time: Optional[float] = None
    if _prometheus_available:
        start = time.perf_counter()
        for _ in range(iterations):
            _ = matmul(A, B)
        prometheus_time = (time.perf_counter() - start) / iterations

    results: dict[str, Any] = {
        "N": N,
        "numpy_ms": numpy_time * 1000,
        "prometheus_ms": prometheus_time * 1000 if prometheus_time else None,
        "speedup": numpy_time / prometheus_time if prometheus_time else None,
        "prometheus_available": _prometheus_available,
        "backend": _backend,
    }

    return results


def matmul_f32(
    A: NDArray[np.float32],
    B: NDArray[np.float32],
    alpha: float = 1.0,
    beta: float = 0.0,
) -> NDArray[np.float32]:
    """
    Single-precision matrix multiplication (SGEMM).

    Computes: C = alpha * A @ B + beta * C

    Args:
        A: Left matrix (M x K), float32
        B: Right matrix (K x N), float32
        alpha: Scalar for A @ B (default 1.0)
        beta: Scalar for C (default 0.0, note: C array not accepted via PyO3)

    Returns:
        Result matrix (M x N), float32

    Note:
        The PyO3 binding currently only supports beta=0.0 (creates fresh output).
        For non-zero beta, falls back to NumPy.
    """
    _init_backend()  # Ensure initialization

    # PyO3 SGEMM only supports beta=0 (always creates fresh output)
    if _backend == "pyo3" and _pyo3_module is not None and beta == 0.0:
        if hasattr(_pyo3_module, "prometheus_sgemm"):
            return _pyo3_module.prometheus_sgemm(A, B, alpha, 0.0)

    # NumPy fallback (handles non-zero beta)
    if beta == 0.0:
        return (alpha * (A @ B)).astype(np.float32)
    else:
        M, K = A.shape
        N = B.shape[1]
        C = np.zeros((M, N), dtype=np.float32)
        return (alpha * (A @ B) + beta * C).astype(np.float32)


# Export symbols
__all__ = [
    "use_prometheus",
    "get_backend",
    "matmul",
    "matmul_f32",
    "triple_product",
    "build_density_matrix",
    "build_density_matrix_uhf",
    "transform_fock",
    "back_transform_coefficients",
    "compute_electronic_energy",
    "compute_energy_weighted_density",
    "benchmark_prometheus",
]
