"""
Equation-of-Motion CCSD (EOM-CCSD) Solver for Excited States
==============================================================
Computes vertical excitation energies via the EOM-CCSD method.

The similarity-transformed Hamiltonian H-bar acts on trial vectors R = (r1, r2)
via numerical finite-difference of the CCSD residual:

    sigma = [Omega(T + eps*R) - Omega(T)] / eps

where Omega is the FULL CCSD residual (increment-update convention with full
Fae/Fmi including diagonal), copied exactly from ccsd.py (Stanton et al. 1991).

At convergence of the ground-state CCSD, Omega(T) = 0, so:

    sigma = Omega(T + eps*R) / eps

Two diagonalization methods are available:

1. **Direct** (method='direct'): Build the full H-bar matrix column by column and
   diagonalize with numpy.linalg.eig. Exact but O(dim^2) memory and O(dim^3) time.
   Best for small systems (dim < 500).

2. **Davidson** (method='davidson'): Iterative subspace diagonalization. Only needs
   matrix-vector products (sigma vectors). O(nroots * dim) memory per iteration.
   Required for large systems where the full H-bar matrix won't fit in memory.

The method is auto-selected based on dimension: direct for dim <= 500, Davidson
for dim > 500. Can be overridden via the `method` parameter.

All integrals in PHYSICIST notation: MO[p,q,r,s] = <pq|rs>.
Uses P^(ab)_(ij) symmetrizer: f(ijab) + f(jiba).

References:
    Stanton & Bartlett, J. Chem. Phys. 98, 7029 (1993)
    Stanton, Bartlett, et al., J. Chem. Phys. 94, 4334 (1991)
    Crawford & Schaefer, Rev. Comp. Chem. 14, 33 (2000)
    Davidson, J. Comput. Phys. 17, 87 (1975)
    Liu, in "Numerical Algorithms in Chemistry", LBL-8158 (1978)
"""

import numpy as np

__all__ = ["EOMCCSDSolver"]

# Physical constants — imported from constants.py (single source of truth)
try:
    from .phys_constants import HARTREE_TO_EV, EV_TO_NM
except ImportError:
    from phys_constants import HARTREE_TO_EV, EV_TO_NM  # type: ignore[no-redef]

# Dimension threshold for auto method selection
_DAVIDSON_THRESHOLD = 500


def _ccsd_full_residual(t1, t2, MO, F, o, v):
    """
    Compute the FULL CCSD residual (Omega) for given t1, t2 amplitudes.

    This is an EXACT copy of the CCSD residual from ccsd.py lines 96-241.
    The only difference: this is a standalone function (no self), and we
    return the raw residuals (r1, r2) WITHOUT dividing by denominators.

    At the converged T amplitudes, r1 ~ 0 and r2 ~ 0.
    At T + eps*R, the residual encodes the action of H-bar on R.

    Args:
        t1: T1 amplitudes, shape (nocc, nvir)
        t2: T2 amplitudes, shape (nocc, nocc, nvir, nvir)
        MO: MO integrals in physicist notation, shape (N, N, N, N)
        F:  Fock matrix, shape (N, N) — diagonal for canonical orbitals
        o:  slice for occupied orbitals
        v:  slice for virtual orbitals

    Returns:
        (r1, r2): T1 and T2 residuals
    """
    # === Intermediates (Stanton Eqs. 3-10) ===
    ttau = t2 + 0.5 * np.einsum("ia,jb->ijab", t1, t1)
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1)

    # Fae (Eq. 3) — FULL, including diagonal
    Fae = F[v, v].copy()
    Fae -= 0.5 * np.einsum("me,ma->ae", F[o, v], t1)
    Fae += 2 * np.einsum("mf,mafe->ae", t1, MO[o, v, v, v])
    Fae -= np.einsum("mf,maef->ae", t1, MO[o, v, v, v])
    Fae -= 2 * np.einsum("mnaf,mnef->ae", ttau, MO[o, o, v, v])
    Fae += np.einsum("mnaf,mnfe->ae", ttau, MO[o, o, v, v])

    # Fmi (Eq. 4) — FULL, including diagonal
    Fmi = F[o, o].copy()
    Fmi += 0.5 * np.einsum("ie,me->mi", t1, F[o, v])
    Fmi += 2 * np.einsum("ne,mnie->mi", t1, MO[o, o, o, v])
    Fmi -= np.einsum("ne,mnei->mi", t1, MO[o, o, v, o])
    Fmi += 2 * np.einsum("inef,mnef->mi", ttau, MO[o, o, v, v])
    Fmi -= np.einsum("inef,mnfe->mi", ttau, MO[o, o, v, v])

    # Fme (Eq. 5)
    Fme = F[o, v].copy()
    Fme += 2 * np.einsum("nf,mnef->me", t1, MO[o, o, v, v])
    Fme -= np.einsum("nf,mnfe->me", t1, MO[o, o, v, v])

    # Wmnij (Eq. 6)
    Wmnij = MO[o, o, o, o].copy()
    Wmnij += np.einsum("je,mnie->mnij", t1, MO[o, o, o, v])
    Wmnij += np.einsum("ie,mnej->mnij", t1, MO[o, o, v, o])
    Wmnij += np.einsum("ijef,mnef->mnij", tau, MO[o, o, v, v])

    # Wmbej (Eq. 8)
    Wmbej = MO[o, v, v, o].copy()
    Wmbej += np.einsum("jf,mbef->mbej", t1, MO[o, v, v, v])
    Wmbej -= np.einsum("nb,mnej->mbej", t1, MO[o, o, v, o])
    tw = 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1)
    Wmbej -= np.einsum("jnfb,mnef->mbej", tw, MO[o, o, v, v])
    Wmbej += np.einsum("njfb,mnef->mbej", t2, MO[o, o, v, v])
    Wmbej -= 0.5 * np.einsum("njfb,mnfe->mbej", t2, MO[o, o, v, v])

    # Wmbje (spin-factored exchange)
    Wmbje = -MO[o, v, o, v].copy()
    Wmbje -= np.einsum("jf,mbfe->mbje", t1, MO[o, v, v, v])
    Wmbje += np.einsum("nb,mnje->mbje", t1, MO[o, o, o, v])
    tw = 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1)
    Wmbje += np.einsum("jnfb,mnfe->mbje", tw, MO[o, o, v, v])

    # Zmbij
    Zmbij = np.einsum("mbef,ijef->mbij", MO[o, v, v, v], tau)

    # === T1 residual (Eq. 1) ===
    r1 = F[o, v].copy()
    r1 += np.einsum("ie,ae->ia", t1, Fae)
    r1 -= np.einsum("ma,mi->ia", t1, Fmi)
    r1 += 2 * np.einsum("imae,me->ia", t2, Fme)
    r1 -= np.einsum("imea,me->ia", t2, Fme)
    r1 += 2 * np.einsum("nf,nafi->ia", t1, MO[o, v, v, o])
    r1 -= np.einsum("nf,naif->ia", t1, MO[o, v, o, v])
    r1 += 2 * np.einsum("mief,maef->ia", t2, MO[o, v, v, v])
    r1 -= np.einsum("mife,maef->ia", t2, MO[o, v, v, v])
    r1 -= 2 * np.einsum("mnae,nmei->ia", t2, MO[o, o, v, o])
    r1 += np.einsum("mnae,nmie->ia", t2, MO[o, o, o, v])

    # === T2 residual (Eq. 2) ===
    r2 = MO[o, o, v, v].copy()

    # P^(ab)_(ij) {t2*Fae}
    tmp = np.einsum("ijae,be->ijab", t2, Fae)
    r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P^(ab)_(ij) {-0.5*t2*t1*Fme}
    tmp = np.einsum("mb,me->be", t1, Fme)
    f = 0.5 * np.einsum("ijae,be->ijab", t2, tmp)
    r2 -= f + f.swapaxes(0, 1).swapaxes(2, 3)

    # P^(ab)_(ij) {-t2*Fmi}
    tmp = np.einsum("imab,mj->ijab", t2, Fmi)
    r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P^(ab)_(ij) {0.5*t2*t1*Fme (ij)}
    tmp = np.einsum("je,me->jm", t1, Fme)
    f = 0.5 * np.einsum("imab,jm->ijab", t2, tmp)
    r2 -= f + f.swapaxes(0, 1).swapaxes(2, 3)

    # tau*Wmnij
    r2 += np.einsum("mnab,mnij->ijab", tau, Wmnij)

    # tau*<vvvv>
    r2 += np.einsum("ijef,abef->ijab", tau, MO[v, v, v, v])

    # -P^(ab)_(ij) {t1*Zmbij}
    tmp = np.einsum("ma,mbij->ijab", t1, Zmbij)
    r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Wmbej term 1
    tmp = np.einsum("imae,mbej->ijab", t2, Wmbej) - np.einsum(
        "imea,mbej->ijab", t2, Wmbej
    )
    r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Wmbej term 2
    tmp = np.einsum("imae,mbej->ijab", t2, Wmbej) + np.einsum(
        "imae,mbje->ijab", t2, Wmbje
    )
    r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # Wmbej term 3
    tmp = np.einsum("mjae,mbie->ijab", t2, Wmbje)
    r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P^(ab)_(ij) {t1*t1*<ovvo>}
    tt = np.einsum("ie,ma->imea", t1, t1)
    tmp = np.einsum("imea,mbej->ijab", tt, MO[o, v, v, o])
    r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P^(ab)_(ij) {t1*t1*<ovov>}
    tt = np.einsum("ie,mb->imeb", t1, t1)
    tmp = np.einsum("imeb,maje->ijab", tt, MO[o, v, o, v])
    r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # P^(ab)_(ij) {t1*<vvvo>}
    tmp = np.einsum("ie,abej->ijab", t1, MO[v, v, v, o])
    r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    # -P^(ab)_(ij) {t1*<ovoo>}
    tmp = np.einsum("ma,mbij->ijab", t1, MO[o, v, o, o])
    r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

    return r1, r2


def _numerical_sigma(r1, r2, t1_conv, t2_conv, MO, F, o, v, eps_fd=1e-5):
    """
    Compute sigma = H-bar * R via finite difference of the CCSD residual.

        sigma = [Omega(T + eps*R) - Omega(T)] / eps

    At convergence, Omega(T) ~ 0 (to machine precision), so we include it
    for numerical accuracy (the difference cancels systematic errors).

    Args:
        r1: Trial R1 vector, shape (nocc, nvir)
        r2: Trial R2 vector, shape (nocc, nocc, nvir, nvir)
        t1_conv: Converged T1 amplitudes
        t2_conv: Converged T2 amplitudes
        MO: MO integrals (physicist notation)
        F: Fock matrix
        o: Occupied slice
        v: Virtual slice
        eps_fd: Finite difference step size

    Returns:
        (sigma1, sigma2): H-bar action on R
    """
    # Omega at converged T (should be ~0)
    omega1_0, omega2_0 = _ccsd_full_residual(t1_conv, t2_conv, MO, F, o, v)

    # Omega at displaced T + eps*R
    t1_pert = t1_conv + eps_fd * r1
    t2_pert = t2_conv + eps_fd * r2
    omega1_p, omega2_p = _ccsd_full_residual(t1_pert, t2_pert, MO, F, o, v)

    # Finite difference
    sigma1 = (omega1_p - omega1_0) / eps_fd
    sigma2 = (omega2_p - omega2_0) / eps_fd

    return sigma1, sigma2


def _pack_vector(r1, r2, dim_s):
    """Pack (r1, r2) tensors into a flat vector."""
    vec = np.zeros(dim_s + r2.size)
    vec[:dim_s] = r1.ravel()
    vec[dim_s:] = r2.ravel()
    return vec


def _unpack_vector(vec, nocc, nvir, dim_s):
    """Unpack a flat vector into (r1, r2) tensors."""
    r1 = vec[:dim_s].reshape(nocc, nvir)
    r2 = vec[dim_s:].reshape(nocc, nocc, nvir, nvir)
    return r1, r2


class EOMCCSDSolver:
    """
    Equation-of-Motion CCSD solver for excited states.

    Computes excitation energies via the similarity-transformed Hamiltonian H-bar
    in the space of singles and doubles excitations.

    Two methods are available:

    - **direct**: Build the full H-bar matrix and diagonalize. Best for small
      systems (dim < 500). O(dim^3) time, O(dim^2) memory.

    - **davidson**: Iterative Davidson diagonalization. Only needs matrix-vector
      products. Best for large systems. O(nroots * dim * n_iter) time.

    The method is auto-selected based on dimension unless explicitly specified.

    Usage:
        ccsd = CCSDSolver()
        E_total, E_corr = ccsd.solve(hf, mol)
        eom = EOMCCSDSolver()
        energies = eom.solve(ccsd, nroots=5)

        # Force Davidson for testing:
        eom_dav = EOMCCSDSolver(method='davidson')
        energies = eom_dav.solve(ccsd, nroots=5)
    """

    def __init__(self, eps_fd=1e-5, method="auto"):
        """
        Args:
            eps_fd: Finite-difference step size for H-bar computation.
                    1e-5 gives a good balance of accuracy vs noise.
            method: Diagonalization method. One of:
                    - 'auto': Select based on dimension (direct if dim <= 500, else davidson)
                    - 'direct': Full matrix construction + diag (np.linalg.eig)
                    - 'davidson': Iterative Davidson algorithm
        """
        if method not in ("auto", "direct", "davidson"):
            raise ValueError(
                f"Unknown method '{method}'. Use 'auto', 'direct', or 'davidson'."
            )
        self.eps_fd = eps_fd
        self.method = method

    def auto_method(self, dim_total):
        """
        Select the best diagonalization method based on problem dimension.

        Args:
            dim_total: Total dimension of the H-bar matrix (dim_s + dim_d)

        Returns:
            str: 'direct' or 'davidson'
        """
        if dim_total <= _DAVIDSON_THRESHOLD:
            return "direct"
        else:
            return "davidson"

    def solve(self, ccsd_solver, nroots=5, verbose=True):
        """
        Compute EOM-CCSD excitation energies.

        Args:
            ccsd_solver: Converged CCSDSolver instance (must have called .solve())
            nroots: Number of excitation energies to return
            verbose: Print results

        Returns:
            excitation_energies: Array of excitation energies in Hartree
        """
        if not hasattr(ccsd_solver, "_t1"):
            raise RuntimeError("CCSDSolver must be converged first (call .solve())")

        t1 = ccsd_solver._t1
        t2 = ccsd_solver._t2
        MO = ccsd_solver._MO
        eps = ccsd_solver._eps
        nocc = ccsd_solver._nocc
        nvir = ccsd_solver._nvir
        N = nocc + nvir
        o = slice(0, nocc)
        v = slice(nocc, N)
        F = np.diag(eps)

        dim_s = nocc * nvir  # singles dimension
        dim_d = nocc * nocc * nvir * nvir  # doubles dimension
        dim_total = dim_s + dim_d

        if verbose:
            print(f"\nEOM-CCSD: nocc={nocc}, nvir={nvir}")
            print(f"  Singles dimension: {dim_s}")
            print(f"  Doubles dimension: {dim_d}")
            print(f"  Total H-bar dimension: {dim_total}")
            print(f"  Finite-difference eps: {self.eps_fd:.1e}")

        if dim_total == 0:
            if verbose:
                print("  No excitations possible (dim=0).")
            return np.array([])

        # Select method
        if self.method == "auto":
            method = self.auto_method(dim_total)
        else:
            method = self.method

        if verbose:
            print(f"  Method: {method}")

        # Store problem data for sigma computation
        self._t1 = t1
        self._t2 = t2
        self._MO = MO
        self._F = F
        self._o = o
        self._v = v
        self._nocc = nocc
        self._nvir = nvir
        self._dim_s = dim_s
        self._dim_d = dim_d
        self._eps = eps

        if method == "direct":
            return self._solve_direct(nroots, verbose)
        else:
            return self._solve_davidson(nroots, verbose)

    def _compute_sigma_vec(self, vec):
        """
        Compute sigma = H-bar * R for a packed vector.

        Args:
            vec: Packed trial vector of length dim_total

        Returns:
            sigma_vec: Packed sigma vector of length dim_total
        """
        nocc = self._nocc
        nvir = self._nvir
        dim_s = self._dim_s

        r1, r2 = _unpack_vector(vec, nocc, nvir, dim_s)
        sigma1, sigma2 = _numerical_sigma(
            r1,
            r2,
            self._t1,
            self._t2,
            self._MO,
            self._F,
            self._o,
            self._v,
            eps_fd=self.eps_fd,
        )
        return _pack_vector(sigma1, sigma2, dim_s)

    def _solve_direct(self, nroots, verbose):
        """
        Solve EOM-CCSD by building the full H-bar matrix and diagonalizing.

        This is the original algorithm — exact but O(dim^3).
        """
        t1 = self._t1
        t2 = self._t2
        MO = self._MO
        F = self._F
        o = self._o
        v = self._v
        nocc = self._nocc
        nvir = self._nvir
        dim_s = self._dim_s
        dim_d = self._dim_d
        dim_total = dim_s + dim_d

        # Build full H-bar matrix column by column
        if verbose:
            print(f"  Building H-bar matrix ({dim_total} x {dim_total})...")

        Hbar = np.zeros((dim_total, dim_total))

        for col in range(dim_total):
            if verbose and dim_total > 20 and col % max(1, dim_total // 10) == 0:
                print(f"    Column {col}/{dim_total} ({100 * col / dim_total:.0f}%)")

            # Build unit vector in (r1, r2) space
            r1_trial = np.zeros((nocc, nvir))
            r2_trial = np.zeros((nocc, nocc, nvir, nvir))

            if col < dim_s:
                # Singles block: map linear index -> (i, a)
                i_idx = col // nvir
                a_idx = col % nvir
                r1_trial[i_idx, a_idx] = 1.0
            else:
                # Doubles block: map linear index -> (i, j, a, b)
                d_idx = col - dim_s
                i_idx = d_idx // (nocc * nvir * nvir)
                rem = d_idx % (nocc * nvir * nvir)
                j_idx = rem // (nvir * nvir)
                rem2 = rem % (nvir * nvir)
                a_idx = rem2 // nvir
                b_idx = rem2 % nvir
                r2_trial[i_idx, j_idx, a_idx, b_idx] = 1.0

            # Apply H-bar
            sigma1, sigma2 = _numerical_sigma(
                r1_trial, r2_trial, t1, t2, MO, F, o, v, eps_fd=self.eps_fd
            )

            # Pack sigma into column
            Hbar[:dim_s, col] = sigma1.ravel()
            Hbar[dim_s:, col] = sigma2.ravel()

        if verbose:
            print("  Diagonalizing H-bar matrix...")

        # H-bar is NOT symmetric in general (similarity transform).
        # Use general eigenvalue decomposition.
        eigenvalues, eigenvectors = np.linalg.eig(Hbar)

        # Take real parts (imaginary parts should be small for physical states)
        eigenvalues_real = np.real(eigenvalues)

        # Sort by real part
        sort_idx = np.argsort(eigenvalues_real)
        eigenvalues_sorted = eigenvalues_real[sort_idx]
        eigenvectors_sorted = eigenvectors[:, sort_idx]

        # Filter: positive excitation energies above threshold
        threshold = 0.01  # Hartree (~0.27 eV)
        mask = eigenvalues_sorted > threshold
        excitation_energies = eigenvalues_sorted[mask]
        excitation_vectors = eigenvectors_sorted[:, mask]

        # Keep only requested number of roots
        if len(excitation_energies) > nroots:
            excitation_energies = excitation_energies[:nroots]
            excitation_vectors = excitation_vectors[:, :nroots]

        # Print and store results
        self._print_results(excitation_energies, excitation_vectors, dim_s, verbose)

        self._excitation_energies = excitation_energies
        self._excitation_vectors = excitation_vectors
        self._eigenvalues_all = eigenvalues_sorted
        self._Hbar = Hbar

        return excitation_energies

    def _solve_davidson(
        self, nroots, verbose, max_iter=100, convergence=1e-5, max_subspace_factor=10
    ):
        """
        Solve EOM-CCSD using the Davidson iterative diagonalization algorithm.

        This method finds the lowest positive eigenvalues of the (non-symmetric)
        H-bar matrix without building it explicitly. Only matrix-vector products
        (sigma vectors) are needed.

        The algorithm:
        1. Generate initial guess vectors from CIS-like orbital energy gaps
        2. Iteratively expand the subspace with preconditioned residuals
        3. Restart when the subspace grows too large
        4. Converge when eigenvalue changes are below threshold

        For non-symmetric H-bar, we use the non-Hermitian Davidson algorithm
        which works with right eigenvectors only (sufficient for energies).

        Args:
            nroots: Number of eigenvalues to find
            verbose: Print progress
            max_iter: Maximum Davidson iterations
            convergence: Eigenvalue convergence threshold (Hartree)
            max_subspace_factor: Maximum subspace size = nroots * this factor

        Returns:
            excitation_energies: Array of excitation energies in Hartree
        """
        nocc = self._nocc
        nvir = self._nvir
        dim_s = self._dim_s
        dim_d = self._dim_d
        dim_total = dim_s + dim_d
        eps = self._eps

        if verbose:
            print(f"  Davidson iterative diagonalization")
            print(f"    nroots={nroots}, max_iter={max_iter}, tol={convergence:.1e}")

        max_subspace = max(nroots * max_subspace_factor, nroots + 20)
        # Don't let subspace exceed the dimension
        max_subspace = min(max_subspace, dim_total)

        # ================================================================
        # Step 1: Build diagonal approximation of H-bar for preconditioning
        # ================================================================
        # For canonical orbitals, the diagonal of H-bar is approximately:
        #   H_ii^{singles} ≈ eps_a - eps_i  (orbital energy gap)
        #   H_ii^{doubles} ≈ eps_a + eps_b - eps_i - eps_j
        # This is exact for the Fock part and approximate for the correlation part.
        diag_H = np.zeros(dim_total)

        eo = eps[:nocc]
        ev = eps[nocc : nocc + nvir]

        # Singles diagonal: eps_a - eps_i
        for i in range(nocc):
            for a in range(nvir):
                idx = i * nvir + a
                diag_H[idx] = ev[a] - eo[i]

        # Doubles diagonal: eps_a + eps_b - eps_i - eps_j
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        idx = (
                            dim_s
                            + i * (nocc * nvir * nvir)
                            + j * (nvir * nvir)
                            + a * nvir
                            + b
                        )
                        diag_H[idx] = ev[a] + ev[b] - eo[i] - eo[j]

        # ================================================================
        # Step 2: Generate initial guess vectors (CIS-like)
        # ================================================================
        # Use unit vectors in the singles space, sorted by orbital energy gap.
        # This gives the best initial approximation to the lowest excitations.
        gaps = []
        for i in range(nocc):
            for a in range(nvir):
                gaps.append((ev[a] - eo[i], i * nvir + a))
        gaps.sort(key=lambda x: x[0])

        n_guess = min(nroots, dim_s, dim_total)
        if n_guess == 0:
            if verbose:
                print("  No guess vectors possible.")
            return np.array([])

        # Initial basis vectors
        basis = []
        for k in range(n_guess):
            b = np.zeros(dim_total)
            b[gaps[k][1]] = 1.0
            basis.append(b)

        # If we need more guesses than singles, add doubles guesses
        if n_guess < nroots and dim_d > 0:
            dgaps = []
            for i in range(nocc):
                for j in range(nocc):
                    for a in range(nvir):
                        for b in range(nvir):
                            idx = (
                                dim_s
                                + i * (nocc * nvir * nvir)
                                + j * (nvir * nvir)
                                + a * nvir
                                + b
                            )
                            dgaps.append((ev[a] + ev[b] - eo[i] - eo[j], idx))
            dgaps.sort(key=lambda x: x[0])
            for k in range(min(nroots - n_guess, len(dgaps))):
                b = np.zeros(dim_total)
                b[dgaps[k][1]] = 1.0
                basis.append(b)
            n_guess = len(basis)

        # ================================================================
        # Step 3: Davidson iteration
        # ================================================================
        # Precompute sigma vectors for initial basis
        sigmas = []
        for b in basis:
            sigmas.append(self._compute_sigma_vec(b))

        old_eigenvalues = np.zeros(nroots) + 1e10
        converged = False

        for iteration in range(max_iter):
            n_basis = len(basis)

            # Build subspace Hamiltonian: H_sub[i,j] = basis[i] . sigma[j]
            # For non-symmetric H-bar, H_sub is also non-symmetric
            H_sub = np.zeros((n_basis, n_basis))
            for i in range(n_basis):
                for j in range(n_basis):
                    H_sub[i, j] = np.dot(basis[i], sigmas[j])

            # Diagonalize subspace Hamiltonian (non-symmetric)
            sub_eigenvalues, sub_eigenvectors = np.linalg.eig(H_sub)

            # Take real parts and sort
            sub_eigenvalues_real = np.real(sub_eigenvalues)
            sort_idx = np.argsort(sub_eigenvalues_real)
            sub_eigenvalues_real = sub_eigenvalues_real[sort_idx]
            sub_eigenvectors = np.real(sub_eigenvectors[:, sort_idx])

            # Select the lowest positive eigenvalues
            pos_mask = sub_eigenvalues_real > 0.005
            pos_evals = sub_eigenvalues_real[pos_mask]
            pos_evecs = sub_eigenvectors[:, pos_mask]

            if len(pos_evals) == 0:
                # If no positive eigenvalues, take smallest eigenvalues
                pos_evals = sub_eigenvalues_real[:nroots]
                pos_evecs = sub_eigenvectors[:, :nroots]

            n_found = min(nroots, len(pos_evals))
            current_eigenvalues = pos_evals[:n_found]
            current_ritz_coeffs = pos_evecs[:, :n_found]

            # Check convergence
            if n_found >= nroots:
                max_change = np.max(
                    np.abs(current_eigenvalues[:nroots] - old_eigenvalues[:nroots])
                )
            else:
                max_change = 1.0

            if verbose and (
                iteration < 3 or iteration % 5 == 0 or max_change < convergence
            ):
                print(
                    f"    Iter {iteration:3d}: {n_found} roots, "
                    f"ω₁={current_eigenvalues[0]:.8f} Eh, "
                    f"max_dω={max_change:.2e}, "
                    f"subspace={n_basis}"
                )

            if max_change < convergence and n_found >= nroots:
                converged = True
                if verbose:
                    print(f"    Converged at iteration {iteration}")
                break

            # Update old eigenvalues
            old_eigenvalues = np.zeros(nroots) + 1e10
            old_eigenvalues[:n_found] = current_eigenvalues[:n_found]

            # ================================================================
            # Compute residuals and expand basis
            # ================================================================
            new_vectors = []
            for k in range(n_found):
                omega = current_eigenvalues[k]
                coeffs = current_ritz_coeffs[:, k]

                # Ritz vector in full space: R = Σ_i c_i * basis[i]
                ritz_vec = np.zeros(dim_total)
                for i in range(n_basis):
                    ritz_vec += coeffs[i] * basis[i]

                # Sigma for Ritz vector: σ_R = Σ_i c_i * sigma[i]
                sigma_ritz = np.zeros(dim_total)
                for i in range(n_basis):
                    sigma_ritz += coeffs[i] * sigmas[i]

                # Residual: r = σ_R - ω * R
                residual = sigma_ritz - omega * ritz_vec

                res_norm = np.linalg.norm(residual)
                if res_norm < convergence * 0.1:
                    continue  # This root is already converged

                # Precondition: q_i = r_i / (diag_H_i - ω)
                # With safe division to avoid singularities
                precond = np.zeros(dim_total)
                for i in range(dim_total):
                    denom = diag_H[i] - omega
                    if abs(denom) > 1e-6:
                        precond[i] = residual[i] / denom
                    else:
                        precond[i] = residual[i] / (1e-6 if denom >= 0 else -1e-6)

                # Normalize
                norm = np.linalg.norm(precond)
                if norm > 1e-14:
                    precond /= norm
                    new_vectors.append(precond)

            # Orthogonalize new vectors against existing basis
            added = 0
            for new_vec in new_vectors:
                if n_basis + added >= max_subspace:
                    break

                # Modified Gram-Schmidt orthogonalization
                q = new_vec.copy()
                for b in basis:
                    q -= np.dot(q, b) * b
                # Second pass for numerical stability
                for b in basis:
                    q -= np.dot(q, b) * b

                norm = np.linalg.norm(q)
                if norm > 1e-8:
                    q /= norm
                    basis.append(q)
                    sigmas.append(self._compute_sigma_vec(q))
                    added += 1

            if added == 0:
                # No new vectors could be added — we've converged or stagnated
                if verbose:
                    print(
                        f"    No new vectors at iteration {iteration} (subspace exhausted)"
                    )
                break

            # ================================================================
            # Restart if subspace too large
            # ================================================================
            if len(basis) >= max_subspace:
                if verbose:
                    print(f"    Restarting: subspace reached {len(basis)}")

                # Collapse to the nroots best Ritz vectors
                n_keep = min(nroots + 2, n_found + 2, len(basis))
                new_basis = []
                new_sigmas = []

                for k in range(n_keep):
                    if k >= current_ritz_coeffs.shape[1]:
                        break
                    coeffs = current_ritz_coeffs[:, k]
                    ritz_vec = np.zeros(dim_total)
                    sigma_vec = np.zeros(dim_total)
                    for i in range(len(coeffs)):
                        ritz_vec += coeffs[i] * basis[i]
                        sigma_vec += coeffs[i] * sigmas[i]

                    # Re-orthogonalize
                    for prev in new_basis:
                        ritz_vec -= np.dot(ritz_vec, prev) * prev
                    norm = np.linalg.norm(ritz_vec)
                    if norm > 1e-10:
                        ritz_vec /= norm
                        # Recompute sigma for the collapsed vector (more accurate)
                        sigma_vec = self._compute_sigma_vec(ritz_vec)
                        new_basis.append(ritz_vec)
                        new_sigmas.append(sigma_vec)

                basis = new_basis
                sigmas = new_sigmas

        if not converged and verbose:
            print(f"    WARNING: Davidson did not converge in {max_iter} iterations")
            print(f"    Last max eigenvalue change: {max_change:.2e}")

        # Extract final results
        excitation_energies = current_eigenvalues[:nroots]

        # Build excitation vectors in full space
        excitation_vectors = np.zeros((dim_total, n_found))
        for k in range(min(nroots, n_found)):
            coeffs = current_ritz_coeffs[:, k]
            for i in range(len(coeffs)):
                excitation_vectors[:, k] += coeffs[i] * basis[i]

        # Print results
        self._print_results(excitation_energies, excitation_vectors, dim_s, verbose)

        self._excitation_energies = excitation_energies
        self._excitation_vectors = excitation_vectors

        return excitation_energies

    def _print_results(self, excitation_energies, excitation_vectors, dim_s, verbose):
        """Print formatted EOM-CCSD results table."""
        if not verbose:
            return

        dim_d = self._dim_d

        print(f"\n{'=' * 65}")
        print(f"EOM-CCSD Excitation Energies")
        print(f"{'=' * 65}")
        print(
            f"  {'State':>5s}  {'E_exc (Eh)':>14s}  {'E_exc (eV)':>12s}  "
            f"{'lambda (nm)':>12s}  {'%Singles':>8s}"
        )
        print(f"  {'-' * 60}")

        for k in range(len(excitation_energies)):
            e_h = excitation_energies[k]
            e_ev = e_h * HARTREE_TO_EV
            if e_ev > 0.0:
                lam_nm = EV_TO_NM / e_ev
            else:
                lam_nm = float("inf")

            # Analyze singles vs doubles character
            if k < excitation_vectors.shape[1]:
                vec = excitation_vectors[:, k]
                norm_s = np.sum(np.abs(vec[:dim_s]) ** 2)
                norm_d = np.sum(np.abs(vec[dim_s:]) ** 2)
                norm_total = norm_s + norm_d
                pct_singles = 100.0 * norm_s / norm_total if norm_total > 0 else 0.0
            else:
                pct_singles = 0.0

            print(
                f"  {k + 1:5d}  {e_h:14.6f}  {e_ev:12.4f}  "
                f"{lam_nm:12.2f}  {pct_singles:7.1f}%"
            )

        print(f"{'=' * 65}")

    def dominant_transitions(self, state_idx=0, n_transitions=5):
        """
        Analyze the dominant single-excitation contributions to a given state.

        Args:
            state_idx: Index of the excited state (0-based)
            n_transitions: Number of dominant transitions to return

        Returns:
            List of (i, a, coefficient) tuples for the dominant transitions.
        """
        if not hasattr(self, "_excitation_vectors"):
            raise RuntimeError("Call solve() first")

        if state_idx >= self._excitation_vectors.shape[1]:
            raise ValueError(
                f"State index {state_idx} out of range "
                f"(only {self._excitation_vectors.shape[1]} states)"
            )

        vec = self._excitation_vectors[:, state_idx]
        r1_vec = vec[: self._dim_s]

        # Determine nocc and nvir from dimensions
        dim_s = self._dim_s
        dim_d = self._dim_d

        # Find nocc, nvir such that nocc*nvir = dim_s
        nocc_found = 0
        nvir_found = 0
        for n in range(1, dim_s + 1):
            if dim_s % n == 0:
                nv = dim_s // n
                if n * n * nv * nv == dim_d:
                    nocc_found = n
                    nvir_found = nv
                    break

        if nocc_found == 0:
            raise RuntimeError("Could not determine nocc/nvir from dimensions")

        r1_mat = r1_vec.reshape(nocc_found, nvir_found)

        # Get indices of largest absolute values
        flat_indices = np.argsort(np.abs(r1_mat.ravel()))[::-1][:n_transitions]
        transitions = []
        for idx in flat_indices:
            i = idx // nvir_found
            a = idx % nvir_found
            coeff = r1_mat[i, a]
            if abs(coeff) > 1e-10:
                transitions.append((i, a, coeff))

        return transitions
