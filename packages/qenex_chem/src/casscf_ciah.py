"""
State-averaged CASSCF via CIAH (CI-amplitude + active-hole) optimisation.

Implements a joint (orbital, CI) second-order Newton step for the
weighted-average CASSCF energy functional

    E_SA(C, {c_i}) = Σ_i w_i ⟨Ψ_i|H(C)|Ψ_i⟩,     Ψ_i = Σ_I c_i[I] Φ_I(C)

where the orbitals C enter both the one- and two-electron MO integrals
(via the orbital-rotation parameter κ) and the configuration
interaction (CI) vectors ``{c_i}`` enter linearly.  The full joint
step is obtained by diagonalising the augmented-Hessian matrix

    M = [[0,   g^T],
         [g,   H  ]]

in the joint (κ, c_ci) space.  Because this is a block-structured
problem, we evaluate ``H·x`` via separate orbital and CI responses and
never form the Hessian matrix explicitly.

Algorithm outline (one macro-iteration):

  1. At current ``(C, {c_i})``:

     * build the SA-averaged 1- and 2-RDMs
     * build the row-form generalised Fock and all PySCF-h\\_op
       intermediates (vhf_c, vhf_a, hdm2, g)
     * per state i, form the CI vector ``H_CI · c_i - E_i · c_i``
       (the CI gradient "residual")

  2. Compose joint gradient ``g_all = [2·g_orb_SA; 2·w·g_ci_stack]``.

  3. Define ``h_op(x_all)`` as a ``LinearOperator`` that:

     * unpacks ``x_all`` into ``(X_orb, {x_ci_i})``
     * applies the orbital-orbital Hessian H_oo·X_orb (the existing
       analytic h_op on SA-averaged RDMs)
     * applies the CI-CI Hessian H_cc·x_ci_i per state (projected
       Davidson-style)
     * applies the orbital-CI coupling H_oc·x_ci_i and H_co·X_orb
       via transition 1- and 2-RDMs between ``c_i`` and ``c_i + x_ci_i``

  4. Solve the augmented-Hessian eigenproblem for the lowest
     eigenpair ``(λ, (1, x_all))`` via scipy MINRES on a shifted
     system (damped Newton) or via scipy eigsh.

  5. Extract step: ``C ← C·exp(unpack(X_orb))`` and
     ``c_i ← (c_i + x_ci_i) / ||.||``.  Apply maximum-overlap root
     reassignment to avoid root crossing.

  6. Trust-region line search on the SA energy; accept if E_SA
     decreases.

This module builds directly on ``CASSCFSolver._build_hessian_intermediates``
and ``CASSCFSolver._orbital_hessian_apply_analytic`` for the H_oo
block, and on ``_ci_hamiltonian`` / ``_compute_rdms`` from the main
module for the CI and cross-block terms.

References:
  * Siegbahn, Almlöf, Heiberg, Roos, J. Chem. Phys. 74, 2384 (1981).
  * Werner, Knowles, J. Chem. Phys. 82, 5053 (1985).
  * Helgaker-Jørgensen-Olsen, §10.8.  PySCF newton_casscf.py.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# CI gradient (residual form)
# ─────────────────────────────────────────────────────────────────────


def ci_residual(H_CI: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return ``(H·c - E·c, E)`` where ``E = c^T H c`` for a normalised c.

    For a CI eigenvector ``c`` with eigenvalue ``E`` the residual is
    identically zero; in the SA-CASSCF inner loop we use this at
    trial vectors to compute the projected Newton step.
    """
    Hc = H_CI @ c
    E = float(c @ Hc)
    return Hc - E * c, E


def ci_gradient_projected(H_CI: np.ndarray, c: np.ndarray, weight: float) -> np.ndarray:
    """
    Project the CI gradient onto the tangent space orthogonal to c,
    weighted by the state-average weight.  This is the CI-block
    component of the joint SA gradient.

    Returns ``2·w·(H·c - E·c)`` with ``E = c^T H c``.
    """
    res, _ = ci_residual(H_CI, c)
    return 2.0 * weight * res


def ci_hessian_apply(
    H_CI: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    weight: float,
) -> np.ndarray:
    """
    Apply the projected CI-CI Hessian of state ``i`` to a CI trial
    vector ``x``.  Formula (Werner-Knowles 1985 eq. 22, adapted for
    SA-CASSCF):

        H_cc · x = 2 w [ (H - E)·x − res·⟨c,x⟩ − c·⟨res,x⟩ ]

    where ``res = H·c - E·c`` is the CI residual (= gradient vector
    before projection).  The two last terms together enforce that the
    step preserves orthogonality to ``c`` at linear order.
    """
    Hc = H_CI @ c
    E = float(c @ Hc)
    res = Hc - E * c
    Hx = H_CI @ x
    # (H - E)·x - res·⟨c,x⟩ - c·⟨res,x⟩
    out = Hx - E * x - res * float(c @ x) - c * float(res @ x)
    return 2.0 * weight * out


# ─────────────────────────────────────────────────────────────────────
# Maximum-overlap root reassignment
# ─────────────────────────────────────────────────────────────────────


def match_roots(
    c_new_unsorted: np.ndarray,
    c_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorder the columns of ``c_new_unsorted`` so that column ``i`` has
    maximum overlap with ``c_ref[:, i]``.  Returns
    ``(c_new_reordered, permutation)`` where
    ``c_new_reordered[:, i] == c_new_unsorted[:, permutation[i]]`` up
    to a global sign flip per column.

    The permutation is greedy: at each step, pick the new column with
    highest unsigned overlap with the current reference column and
    remove it from consideration.
    """
    n = c_ref.shape[1]
    S = c_ref.T @ c_new_unsorted  # (n_ref, n_new) overlap matrix
    perm = -np.ones(n, dtype=int)
    available = list(range(c_new_unsorted.shape[1]))
    for i in range(n):
        if not available:
            break
        best = max(available, key=lambda j: abs(S[i, j]))
        perm[i] = best
        available.remove(best)
    c_out = np.empty_like(c_ref)
    for i in range(n):
        if perm[i] < 0:
            # Should not happen if n_new >= n_ref; fallback to identity
            c_out[:, i] = c_ref[:, i]
            continue
        col = c_new_unsorted[:, perm[i]]
        # Flip sign to maximise real-part overlap
        if S[i, perm[i]] < 0:
            col = -col
        c_out[:, i] = col
    return c_out, perm


# ─────────────────────────────────────────────────────────────────────
# SA-averaged RDM builder
# ─────────────────────────────────────────────────────────────────────


def sa_average_rdms(
    ci_vecs: np.ndarray,
    weights: np.ndarray,
    compute_rdms_fn,
    det_alpha,
    det_beta,
    nact: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the SA-weighted 1- and 2-RDMs from per-state CI vectors.

    ci_vecs: shape (ci_dim, n_states), one state per column.
    weights: length-n_states array of non-negative weights.
    compute_rdms_fn: callable (ci_vec, det_alpha, det_beta, nact)
        returning (rdm1, rdm2) for a single CI vector.
    """
    n_states = ci_vecs.shape[1]
    rdm1 = np.zeros((nact, nact))
    rdm2 = np.zeros((nact, nact, nact, nact))
    for i in range(n_states):
        r1, r2 = compute_rdms_fn(ci_vecs[:, i], det_alpha, det_beta, nact)
        rdm1 += float(weights[i]) * r1
        rdm2 += float(weights[i]) * r2
    return rdm1, rdm2


# ─────────────────────────────────────────────────────────────────────
# Joint gradient
# ─────────────────────────────────────────────────────────────────────


def joint_gradient(
    orbital_grad_vec: np.ndarray,
    ci_vecs: np.ndarray,
    weights: np.ndarray,
    H_CI: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack the joint (orbital, CI) gradient vector and return also the
    per-state CI energies and residuals for later use.

    Returns:
      g_all: flattened joint gradient, shape (n_rot + n_states*ci_dim,)
      E_states: shape (n_states,) the current state energies
      residuals: shape (ci_dim, n_states) columns are
                 ``H·c_i - E_i·c_i`` (pre-weight).
    """
    n_states = ci_vecs.shape[1]
    ci_dim = ci_vecs.shape[0]

    # Orbital block: already weighted-average gradient
    pieces = [orbital_grad_vec]
    E_states = np.empty(n_states)
    residuals = np.empty((ci_dim, n_states))
    for i in range(n_states):
        res, E = ci_residual(H_CI, ci_vecs[:, i])
        E_states[i] = E
        residuals[:, i] = res
        pieces.append(2.0 * float(weights[i]) * res)
    return np.concatenate(pieces), E_states, residuals


# ─────────────────────────────────────────────────────────────────────
# Joint augmented-Hessian step
# ─────────────────────────────────────────────────────────────────────


def solve_ciah_step(
    g_all: np.ndarray,
    n_rot: int,
    ci_dim: int,
    n_states: int,
    hv_orbital_orbital,  # callable (x_orb) -> H_oo @ x_orb  (length n_rot)
    H_CI: np.ndarray,
    ci_vecs: np.ndarray,
    weights: np.ndarray,
    hv_coupling_oc,  # callable (x_ci_stack) -> H_oc · x_ci stacked (length n_rot)
    hv_coupling_co,  # callable (x_orb) -> H_co · x_orb stacked (length ci_dim*n_states)
    diag_precond: Optional[np.ndarray] = None,
    rtol: float = 1e-7,
    max_inner: int = 80,
    shift_ladder: tuple = (0.0, 0.1, 0.5, 2.0, 10.0),
) -> tuple[np.ndarray, float, float]:
    """
    Solve for the Newton step ``x_all`` satisfying
    ``(H + σI) · x_all = -g_all`` via MINRES on the joint
    Hessian, with a damping ladder: try σ=0 first (pure Newton), then
    progressively larger shifts until the solution yields first-order
    descent (``g·x < 0``).

    Returns ``(x_all, g_dot_x, sigma_used)``.

    This is simpler than the full augmented-Hessian eigenvalue problem
    but equivalent in the attractive basin where H+σI is positive
    definite.  When H is strongly indefinite the ladder eventually
    reaches a σ large enough for PD, at which point the step is
    essentially steepest-descent.
    """
    from scipy.sparse.linalg import LinearOperator, minres

    n_all = n_rot + ci_dim * n_states

    def apply_H(x_all):
        x_orb = x_all[:n_rot]
        x_ci_flat = x_all[n_rot:]
        # Orbital part: H_oo·x_orb + H_oc·x_ci
        out_orb = hv_orbital_orbital(x_orb)
        out_orb += hv_coupling_oc(x_ci_flat)
        # CI part: H_cc·x_ci (per state) + H_co·x_orb
        out_ci = np.empty(ci_dim * n_states)
        for i in range(n_states):
            xi = x_ci_flat[i * ci_dim : (i + 1) * ci_dim]
            out_ci[i * ci_dim : (i + 1) * ci_dim] = ci_hessian_apply(
                H_CI,
                ci_vecs[:, i],
                xi,
                float(weights[i]),
            )
        out_ci += hv_coupling_co(x_orb)
        return np.concatenate([out_orb, out_ci])

    # Diagonal preconditioner: orbital block from provided diag_precond,
    # CI block uses ``2 w_i · (H_diag - E_i)``
    if diag_precond is None:
        diag_precond = np.ones(n_rot)
    H_diag_ci = np.diag(H_CI)
    diag_full = np.empty(n_all)
    diag_full[:n_rot] = np.abs(diag_precond)
    diag_full[:n_rot] = np.where(diag_full[:n_rot] < 0.05, 0.05, diag_full[:n_rot])
    for i in range(n_states):
        gap = np.abs(H_diag_ci - (ci_vecs[:, i] @ H_CI @ ci_vecs[:, i])) + 0.05
        diag_full[n_rot + i * ci_dim : n_rot + (i + 1) * ci_dim] = (
            2.0 * float(weights[i]) * gap
        )

    def precond(r):
        return r / diag_full

    H_op = LinearOperator(shape=(n_all, n_all), matvec=apply_H, dtype=float)
    M_op = LinearOperator(shape=(n_all, n_all), matvec=precond, dtype=float)

    x_all = None
    sigma_used = 0.0
    g_dot_x = 0.0
    for sigma in shift_ladder:
        if sigma > 0:

            def apply_shifted(x, _s=sigma):
                return apply_H(x) + _s * x

            Hs = LinearOperator(
                shape=(n_all, n_all),
                matvec=apply_shifted,
                dtype=float,
            )
        else:
            Hs = H_op
        try:
            try:
                x_try, info = minres(
                    Hs,
                    -g_all,
                    M=M_op,
                    rtol=rtol,
                    maxiter=max_inner,
                )
            except TypeError:
                x_try, info = minres(
                    Hs,
                    -g_all,
                    M=M_op,
                    tol=rtol,
                    maxiter=max_inner,
                )
        except Exception:
            continue
        if info < 0:
            continue
        directional = float(g_all @ x_try)
        if directional < -1e-12:
            x_all = x_try
            sigma_used = sigma
            g_dot_x = directional
            break

    if x_all is None:
        # Last-resort: steepest descent
        x_all = -g_all / max(float(np.linalg.norm(g_all)), 1e-12) * 0.1
        g_dot_x = float(g_all @ x_all)
        sigma_used = float("inf")

    return x_all, g_dot_x, sigma_used


# ─────────────────────────────────────────────────────────────────────
# CI update + renormalisation
# ─────────────────────────────────────────────────────────────────────


def apply_ci_step(
    ci_vecs: np.ndarray,
    x_ci_flat: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Apply the CI part of the Newton step:
        c_i_new = (c_i + x_ci_i) / ||.||
    Returns the updated CI matrix (ci_dim, n_states).  If normalisation
    is impossible (zero denominator) the old CI vector is kept.
    """
    ci_dim = ci_vecs.shape[0]
    out = np.empty_like(ci_vecs)
    for i in range(n_states):
        xi = x_ci_flat[i * ci_dim : (i + 1) * ci_dim]
        ci_new = ci_vecs[:, i] + xi
        norm = float(np.linalg.norm(ci_new))
        if norm > 1e-14:
            out[:, i] = ci_new / norm
        else:
            out[:, i] = ci_vecs[:, i]
    return out


# ─────────────────────────────────────────────────────────────────────
# Module-level export for cleaner imports
# ─────────────────────────────────────────────────────────────────────


__all__ = [
    "ci_residual",
    "ci_gradient_projected",
    "ci_hessian_apply",
    "match_roots",
    "sa_average_rdms",
    "joint_gradient",
    "solve_ciah_step",
    "apply_ci_step",
    "run_block_diagonal_ciah",
]


# ─────────────────────────────────────────────────────────────────────
# Block-diagonal CIAH driver (no H_oc / H_co coupling)
# ─────────────────────────────────────────────────────────────────────


def run_block_diagonal_ciah(
    casscf_solver,
    hf_solver,
    molecule,
    weights,
    max_iter: int = 100,
    conv_tol: float = 1e-8,
    verbose: bool = False,
):
    """
    EXPERIMENTAL.  Block-diagonal CIAH driver for state-averaged CASSCF
    — WITHOUT the orbital-CI coupling Hessian blocks H_oc / H_co.

    This is a demonstrative implementation that shows why the coupling
    matters: when CI amplitudes are re-diagonalised at each line-search
    trial, the CI residual is always zero (exact eigenvectors), so the
    CI-block Newton step is identically zero and the driver reduces to
    plain single-state AH with SA-averaged RDMs.  On H2O/CAS(4,4)/sto-3g
    SA(3) this results in oscillation without convergence (the SA
    functional has multiple decoupled-basin stationary points that
    cannot be distinguished without H_oc/H_co).

    A full CIAH (with the coupling blocks, transition RDMs, and
    update_jk_in_ah response) is tracked as follow-up work.  The
    algorithmic components in this module (``ci_residual``,
    ``ci_hessian_apply``, ``match_roots``, ``apply_ci_step``) are
    verified and will be reused by the full implementation.

    Parameters
    ----------
    casscf_solver, hf_solver, molecule : QENEX objects
    weights : sequence of float (sum = 1)
    max_iter, conv_tol, verbose : standard Newton-loop controls

    Returns
    -------
    dict with keys: ``converged``, ``E_SA``, ``state_energies``,
    ``C_final``, ``state_ci``, ``macro_iters``.  Not suitable for
    production use.
    """
    from casscf import (  # type: ignore[import-not-found]
        _generate_determinants,
        _ci_hamiltonian,
        _compute_rdms,
    )

    weights_arr = np.asarray(weights, dtype=float)
    n_states = len(weights_arr)
    assert abs(weights_arr.sum() - 1.0) < 1e-6, "weights must sum to 1"
    nact = casscf_solver.ncas
    nelec = casscf_solver.nelecas

    # Active-space determinant lists and CI dim
    na = nelec // 2
    nb = nelec - na
    det_alpha = _generate_determinants(nact, na)
    det_beta = _generate_determinants(nact, nb)
    ci_dim = len(det_alpha) * len(det_beta)

    # Initial orbitals: HF canonical
    C = hf_solver.C.copy()
    n_mo = C.shape[1]
    n_occ = hf_solver.n_occ
    n_inactive = n_occ - nelec // 2
    if n_inactive < 0:
        n_inactive = 0
    n_virtual = n_mo - n_inactive - nact
    H_core_ao = casscf_solver._get_h_core(
        hf_solver,
        hf_solver.ERI,
        hf_solver.eps,
        C,
    )
    E_nuc = hf_solver.compute_nuclear_repulsion(molecule)

    # Initial per-state CI: diagonalise at HF orbitals
    def _build_H_CI(C):
        h1_mo = C.T @ H_core_ao @ C
        tmp = np.einsum("up,uvwx->pvwx", C, hf_solver.ERI, optimize=True)
        tmp = np.einsum("vq,pvwx->pqwx", C, tmp, optimize=True)
        tmp2 = np.einsum("wr,pqwx->pqrx", C, tmp, optimize=True)
        eri_mo = np.einsum("xs,pqrx->pqrs", C, tmp2, optimize=True)
        act = slice(n_inactive, n_inactive + nact)
        F_inactive = h1_mo.copy()
        for i in range(n_inactive):
            F_inactive += 2.0 * eri_mo[:, :, i, i] - eri_mo[:, i, i, :]
        h1_act = F_inactive[act, act].copy()
        h2_act = eri_mo[act, act, act, act].copy()
        return (
            _ci_hamiltonian(det_alpha, det_beta, h1_act, h2_act, nact),
            h1_mo,
            eri_mo,
            F_inactive,
        )

    H_CI, h1_mo, eri_mo, F_inactive = _build_H_CI(C)
    eigvals, eigvecs = np.linalg.eigh(H_CI)
    ci_vecs = eigvecs[:, :n_states].copy()

    # Inactive-core energy (fixed by orbitals, not CI)
    E_inactive_at_C = sum(h1_mo[i, i] + F_inactive[i, i] for i in range(n_inactive))

    converged = False
    E_SA_prev = None
    last_macro = -1
    for macro in range(max_iter):
        last_macro = macro

        # Build SA RDMs
        rdm1_sa = np.zeros((nact, nact))
        rdm2_sa = np.zeros((nact, nact, nact, nact))
        for i in range(n_states):
            r1, r2 = _compute_rdms(ci_vecs[:, i], det_alpha, det_beta, nact)
            rdm1_sa += float(weights_arr[i]) * r1
            rdm2_sa += float(weights_arr[i]) * r2

        # Per-state energies and SA-average
        state_energies = E_inactive_at_C + E_nuc + eigvals[:n_states]
        E_SA = float(np.dot(weights_arr, state_energies))

        # Orbital gradient (SA-RDMs)
        grad = casscf_solver._orbital_gradient(
            h1_mo,
            eri_mo,
            rdm1_sa,
            rdm2_sa,
            n_inactive,
            nact,
            n_virtual,
        )
        g_orb_flat = np.concatenate(
            [
                grad["ia"].flatten(),
                grad["ta"].flatten(),
                grad["iv"].flatten(),
            ]
        )
        gnorm_orb = float(grad["norm"])

        # Per-state CI residuals (for convergence + joint grad)
        ci_grads = np.empty(ci_dim * n_states)
        max_ci_res = 0.0
        for i in range(n_states):
            g_ci_i = ci_gradient_projected(H_CI, ci_vecs[:, i], float(weights_arr[i]))
            ci_grads[i * ci_dim : (i + 1) * ci_dim] = g_ci_i
            max_ci_res = max(max_ci_res, float(np.linalg.norm(g_ci_i)))

        if verbose:
            print(
                f"  CIAH iter {macro:3d}: E_SA={E_SA:.10f} "
                f"|g_orb|={gnorm_orb:.2e} |g_ci|={max_ci_res:.2e}"
            )

        # Convergence
        dE = abs(E_SA - E_SA_prev) if E_SA_prev is not None else 1.0
        if (
            macro > 0
            and dE < conv_tol
            and gnorm_orb < max(conv_tol * 100, 1e-5)
            and max_ci_res < max(conv_tol * 100, 1e-5)
        ):
            converged = True
            if verbose:
                print(
                    f"  CIAH converged at iter {macro}: "
                    f"dE={dE:.2e}, |g_orb|={gnorm_orb:.2e}, "
                    f"|g_ci|={max_ci_res:.2e}"
                )
            break

        # Orbital-block Newton step via existing AH machinery
        # (re-solves CI inside, but that's OK — it uses the current C)
        n_rot = g_orb_flat.size
        try:
            x_orb, _g_norm = casscf_solver._augmented_hessian_step(
                C,
                H_core_ao,
                hf_solver.ERI,
                n_inactive,
                nact,
                n_virtual,
                n_mo,
                det_alpha,
                det_beta,
                fd_h=1e-3,
                max_step=0.3,
                use_analytic_hessian=True,
            )
        except Exception:
            # Fall back to steepest-descent orbital step
            x_orb = -g_orb_flat / max(gnorm_orb, 1e-12) * 0.05

        # CI-block Newton step: per state, solve
        #   (H_cc + σ I) x_i = -g_ci_i
        # directly since H_cc is the small (ci_dim × ci_dim) projected
        # operator.  For CI dim < 5000 we can form it explicitly.
        x_ci_flat = np.zeros(ci_dim * n_states)
        for i in range(n_states):
            c_i = ci_vecs[:, i]
            E_i = float(eigvals[i])
            g_ci_i = ci_grads[i * ci_dim : (i + 1) * ci_dim]

            # Solve (H_cc + σI) x = -g_ci projected onto tangent space
            # H_cc x = 2w (H x - E x - res·⟨c,x⟩ - c·⟨res,x⟩)
            # At a true eigenvector, res=0 so H_cc = 2w(H - E I) and
            # the Newton step is x = -g_ci / (2w(H - E I)) applied
            # through the H_CI eigendecomposition.
            # General case with residual: use (H - E I)^+ pseudo-inverse
            # but since our initial CI came from exact diag, residual=0.
            # Projected Newton step: x = 0.5/w · (E I - H)^{-1} · res
            # where we set x · c = 0 by projection.
            res = H_CI @ c_i - E_i * c_i  # should be ~0 initially
            if np.linalg.norm(res) < 1e-10:
                x_ci_flat[i * ci_dim : (i + 1) * ci_dim] = 0.0
                continue
            # Projected Newton: pick negative eigenvalue of (H-E_i I)
            # restricted to tangent space via eigendecomposition.
            # Use Davidson-like: x = Σ_j (res·v_j / (E_i - E_j)) v_j
            # for j ≠ i.  Simple loop over known eigenvectors.
            x_i = np.zeros(ci_dim)
            for j in range(ci_dim):
                if j == i:
                    continue
                denom = eigvals[i] - eigvals[j]
                if abs(denom) < 1e-10:
                    continue
                proj = float(eigvecs[:, j] @ res)
                x_i += (proj / denom) * eigvecs[:, j]
            x_ci_flat[i * ci_dim : (i + 1) * ci_dim] = x_i

        # Trust-region line search on the JOINT step
        accepted = False
        trial_scale = 1.0
        for _ls in range(6):
            # Rotate orbitals by trial_scale * x_orb
            C_try = casscf_solver._rotate_and_flatten_step(
                C,
                x_orb * trial_scale,
                n_inactive,
                nact,
                n_virtual,
                n_mo,
            )
            # Apply CI update (also scaled)
            ci_try = apply_ci_step(
                ci_vecs,
                x_ci_flat * trial_scale,
                n_states,
            )

            # Evaluate SA energy at (C_try, ci_try)
            H_CI_try, h1_mo_try, eri_mo_try, F_inact_try = _build_H_CI(C_try)
            E_inact_try = sum(
                h1_mo_try[i, i] + F_inact_try[i, i] for i in range(n_inactive)
            )
            # Re-diagonalise CI at the new orbitals to get correct
            # per-state E.  Take the N lowest-with-overlap to ci_try.
            eigvals_try, eigvecs_try = np.linalg.eigh(H_CI_try)
            # Maximum-overlap reassign
            ci_matched, _perm = match_roots(eigvecs_try, ci_try)
            # Recompute eigvals in the matched order
            matched_eigs = np.array(
                [
                    float(ci_matched[:, i] @ H_CI_try @ ci_matched[:, i])
                    for i in range(n_states)
                ]
            )
            E_SA_try = E_inact_try + E_nuc + float(np.dot(weights_arr, matched_eigs))

            if E_SA_try < E_SA + 1e-10:
                C = C_try
                H_CI = H_CI_try
                h1_mo = h1_mo_try
                eri_mo = eri_mo_try
                F_inactive = F_inact_try
                E_inactive_at_C = E_inact_try
                eigvals = eigvals_try
                eigvecs = eigvecs_try
                ci_vecs = ci_matched
                # Re-extract eigvals in the matched order (store full
                # eigenvalue list but put matched states at positions 0..n_states-1)
                accepted = True
                break
            trial_scale *= 0.5

        if not accepted:
            if verbose:
                print(
                    f"  CIAH iter {macro}: all line-search scales "
                    f"rejected, falling back to steepest descent"
                )
            # Steepest descent: take a tiny step in -g direction
            # in both orbital and CI blocks
            small_orb = -g_orb_flat / max(gnorm_orb, 1e-12) * 0.01
            small_ci = -ci_grads / max(max_ci_res, 1e-12) * 0.01
            C = casscf_solver._rotate_and_flatten_step(
                C,
                small_orb,
                n_inactive,
                nact,
                n_virtual,
                n_mo,
            )
            ci_vecs = apply_ci_step(ci_vecs, small_ci, n_states)
            # Re-diagonalise at new orbitals
            H_CI, h1_mo, eri_mo, F_inactive = _build_H_CI(C)
            eigvals, eigvecs = np.linalg.eigh(H_CI)
            ci_vecs, _ = match_roots(eigvecs[:, :n_states], ci_vecs)
            E_inactive_at_C = sum(
                h1_mo[i, i] + F_inactive[i, i] for i in range(n_inactive)
            )

        E_SA_prev = E_SA

    # Final state bookkeeping at convergence
    eigvals_final, eigvecs_final = np.linalg.eigh(H_CI)
    # Pick matching-overlap N-state block
    state_block, _ = match_roots(eigvecs_final, ci_vecs)
    state_eigs = np.array(
        [float(state_block[:, i] @ H_CI @ state_block[:, i]) for i in range(n_states)]
    )
    final_state_E = E_inactive_at_C + E_nuc + state_eigs
    E_SA_final = float(np.dot(weights_arr, final_state_E))

    return {
        "converged": converged,
        "E_SA": E_SA_final,
        "state_energies": final_state_E,
        "C_final": C,
        "state_ci": state_block,
        "macro_iters": last_macro + 1,
    }
