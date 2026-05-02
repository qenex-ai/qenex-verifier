"""
Coupled Cluster Singles and Doubles (CCSD) Solver
===================================================
Gold standard of single-reference quantum chemistry.

Native spatial-orbital closed-shell CCSD implementation using
spin-factored equations from Stanton et al. (1991).

All integrals in PHYSICIST notation: MO[p,q,r,s] = <pq|rs> = (pr|qs).
Uses P^(ab)_(ij) symmetrizer: f(ijab) + f(jiba).

Equations follow Psi4NumPy / Crawford Group convention:
    https://github.com/CrawfordGroup/ProgrammingProjects

Performance upgrades:
    - Frozen core approximation: exclude core MOs from correlation space
    - Density fitting (RI) interface: see density_fitting.py

References:
    Stanton, Bartlett, et al., J. Chem. Phys. 94, 4334 (1991)
    Crawford & Schaefer, Rev. Comp. Chem. 14, 33 (2000)
    Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989) — (T)
"""

import numpy as np

__all__ = ["CCSDSolver"]

# ============================================================
# Frozen Core Utilities
# ============================================================

# Atomic number lookup (H through Ar)
_Z_MAP = {
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

# Number of core orbitals per element (spatial orbitals, not spin-orbitals):
#   Row 1 (H, He):    0 core orbitals
#   Row 2 (Li–Ne):    1 core orbital  (1s)
#   Row 3 (Na–Ar):    5 core orbitals (1s, 2s, 2px, 2py, 2pz)
_CORE_ORBITALS = {
    "H": 0,
    "He": 0,
    "Li": 1,
    "Be": 1,
    "B": 1,
    "C": 1,
    "N": 1,
    "O": 1,
    "F": 1,
    "Ne": 1,
    "Na": 5,
    "Mg": 5,
    "Al": 5,
    "Si": 5,
    "P": 5,
    "S": 5,
    "Cl": 5,
    "Ar": 5,
}


def count_frozen_core(atoms):
    """
    Count total number of frozen core orbitals for a molecular system.

    Args:
        atoms: list of (element_symbol, (x, y, z)) tuples

    Returns:
        n_frozen: number of spatial MOs to freeze (lowest-energy occupied)
    """
    n_frozen = 0
    for element, _ in atoms:
        n_frozen += _CORE_ORBITALS.get(element, 0)
    return n_frozen


class CCSDSolver:
    """Closed-shell RHF-CCSD solver — fully native, zero external dependencies."""

    def __init__(self, max_iter=100, convergence=1e-10, diis_size=8, frozen_core=False):
        """
        Initialize CCSD solver.

        Args:
            max_iter: Maximum number of CCSD iterations
            convergence: Energy convergence threshold (Hartree)
            diis_size: Number of DIIS vectors to store
            frozen_core: If True, freeze core orbitals (1s for 2nd row,
                        1s2s2p for 3rd row). Gives 2-5x speedup with
                        negligible accuracy loss (<0.1 mHartree).
                        Default False for backward compatibility.
        """
        self.max_iter = max_iter
        self.convergence = convergence
        self.diis_size = diis_size
        self.frozen_core = frozen_core

    def solve(self, hf_solver, molecule, verbose=True, frozen_core=None):
        """
        Solve CCSD on top of converged RHF.

        Args:
            hf_solver: Converged HartreeFockSolver instance
            molecule: Molecule object
            verbose: Print convergence info
            frozen_core: Override instance-level frozen_core setting.
                        If None, uses self.frozen_core.

        Returns: (E_total, E_correlation)
        """
        # Determine frozen core setting: method arg overrides instance default
        use_frozen = frozen_core if frozen_core is not None else self.frozen_core

        C = hf_solver.C
        eps = hf_solver.eps
        ERI_ao = hf_solver.ERI
        nocc_total = hf_solver.n_occ
        E_hf, _ = hf_solver.compute_energy(molecule, verbose=False)
        N = C.shape[0]

        # --- Frozen core handling ---
        if use_frozen:
            n_frozen = count_frozen_core(molecule.atoms)
            # Safety: can't freeze more than we have occupied
            if n_frozen >= nocc_total:
                if verbose:
                    print(
                        f"CCSD: WARNING: n_frozen={n_frozen} >= nocc={nocc_total}, "
                        f"disabling frozen core"
                    )
                n_frozen = 0
        else:
            n_frozen = 0

        self._n_frozen = n_frozen

        # Active occupied = total occupied - frozen core
        nocc = nocc_total - n_frozen
        nvir = N - nocc_total  # virtual space is unchanged

        if verbose:
            if n_frozen > 0:
                print(
                    f"\nCCSD: N={N}, total_occ={nocc_total}, frozen_core={n_frozen}, "
                    f"active_occ={nocc}, vir={nvir}"
                )
            else:
                print(f"\nCCSD: N={N}, occ={nocc}, vir={nvir}")
        if nvir == 0:
            if verbose:
                print(
                    f"  CCSD: No virtual orbitals (N={N}, occ={nocc}). "
                    f"Correlation energy is exactly 0.0."
                )
            self.E_corr = 0.0
            self.t1 = np.zeros((0, 0))
            self.t2 = np.zeros((0, 0, 0, 0))
            self._t1 = self.t1
            self._t2 = self.t2
            self.t1_diagnostic = 0.0
            self.converged = True
            self._E_hf = E_hf
            self._E_corr = 0.0
            self._nocc = nocc
            self._nvir = 0
            self._n_frozen = n_frozen
            self._eps = eps[n_frozen:] if n_frozen > 0 else eps
            self._MO = np.zeros((0, 0, 0, 0))
            self._g_chem = np.zeros((0, 0, 0, 0))
            self._MO_full = np.zeros((0, 0, 0, 0))
            self._g_chem_full = np.zeros((0, 0, 0, 0))
            self._eps_full = eps
            self._nocc_total = nocc_total
            return E_hf, 0.0

        if nvir < 2:
            if verbose:
                print(
                    f"  CCSD: Only {nvir} virtual orbital(s). "
                    f"Correlation energy will be minimal."
                )

        # Full MO transformation: g[p,q,r,s] = (pq|rs) chemist
        if verbose:
            print("CCSD: MO integral transformation...")
        t = np.einsum("up,uvwx->pvwx", C, ERI_ao, optimize=True)
        t = np.einsum("vq,pvwx->pqwx", C, t, optimize=True)
        t2_tmp = np.einsum("wr,pqwx->pqrx", C, t, optimize=True)
        del t
        g_full = np.einsum("xs,pqrx->pqrs", C, t2_tmp, optimize=True)
        del t2_tmp

        # Physicist notation: MO[p,q,r,s] = <pq|rs> = (pr|qs) = g[p,r,q,s]
        MO_full = g_full.swapaxes(1, 2)

        # --- Slice to active space if frozen core ---
        # Active indices: frozen..nocc_total (occupied), nocc_total..N (virtual)
        # We exclude the first n_frozen MOs from the correlation space.
        if n_frozen > 0:
            # Active orbital range: n_frozen..N
            n_active = N - n_frozen
            # Slice MO integrals to active space
            MO = MO_full[n_frozen:, n_frozen:, n_frozen:, n_frozen:]
            g = g_full[n_frozen:, n_frozen:, n_frozen:, n_frozen:]
            # Active Fock diagonal
            eps_active = eps[n_frozen:]
            # Slices within the active space
            o = slice(0, nocc)
            v = slice(nocc, n_active)
            F = np.diag(eps_active)
        else:
            MO = MO_full
            g = g_full
            eps_active = eps
            n_active = N
            o = slice(0, nocc)
            v = slice(nocc, N)
            F = np.diag(eps)

        eo = eps_active[:nocc]
        ev = eps_active[nocc:]
        Dia = eo[:, None] - ev[None, :]
        Dijab = (
            eo[:, None, None, None]
            + eo[None, :, None, None]
            - ev[None, None, :, None]
            - ev[None, None, None, :]
        )

        # Initialize: T1=0, T2=MP2
        t1 = np.zeros((nocc, nvir))
        t2 = MO[o, o, v, v] / Dijab

        if verbose:
            print("CCSD: Iterating...")

        # DIIS
        d_t1, d_t2, d_e1, d_e2 = [], [], [], []
        E_old = 0.0

        for it in range(self.max_iter):
            # === Intermediates (Stanton Eqs. 3-10) ===
            # S4: compute t1⊗t1 outer product once, derive both ttau and tau
            _t1t1 = np.einsum("ia,jb->ijab", t1, t1)
            ttau = t2 + 0.5 * _t1t1
            tau = t2 + _t1t1
            del _t1t1

            # S5: precompute 2*MO[oovv] - MO[oovv].swapaxes(2,3) once per iteration
            # Halves the number of oovv einsums for Fae, Fmi, Fme
            _MO_oovv_comb = 2 * MO[o, o, v, v] - MO[o, o, v, v].swapaxes(2, 3)

            # Fae (Eq. 3)
            Fae = F[v, v].copy()
            Fae -= 0.5 * np.einsum("me,ma->ae", F[o, v], t1)
            Fae += 2 * np.einsum("mf,mafe->ae", t1, MO[o, v, v, v])
            Fae -= np.einsum("mf,maef->ae", t1, MO[o, v, v, v])
            Fae -= np.einsum("mnaf,mnef->ae", ttau, _MO_oovv_comb)

            # Fmi (Eq. 4)
            Fmi = F[o, o].copy()
            Fmi += 0.5 * np.einsum("ie,me->mi", t1, F[o, v])
            Fmi += 2 * np.einsum("ne,mnie->mi", t1, MO[o, o, o, v])
            Fmi -= np.einsum("ne,mnei->mi", t1, MO[o, o, v, o])
            Fmi += np.einsum("inef,mnef->mi", ttau, _MO_oovv_comb)

            # Fme (Eq. 5)
            Fme = F[o, v].copy()
            Fme += np.einsum("nf,mnef->me", t1, _MO_oovv_comb)

            # Wmnij (Eq. 6) — also uses _MO_oovv_comb for the tau term
            Wmnij = MO[o, o, o, o].copy()
            Wmnij += np.einsum("je,mnie->mnij", t1, MO[o, o, o, v])
            Wmnij += np.einsum("ie,mnej->mnij", t1, MO[o, o, v, o])
            Wmnij += np.einsum("ijef,mnef->mnij", tau, MO[o, o, v, v])

            # Wmbej (Eq. 8)
            # S3: compute tw once (was computed identically at lines 297 and 306)
            Wmbej = MO[o, v, v, o].copy()
            Wmbej += np.einsum("jf,mbef->mbej", t1, MO[o, v, v, v])
            Wmbej -= np.einsum("nb,mnej->mbej", t1, MO[o, o, v, o])
            tw = 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1)  # computed ONCE
            Wmbej -= np.einsum("jnfb,mnef->mbej", tw, MO[o, o, v, v])
            Wmbej += np.einsum("njfb,mnef->mbej", t2, MO[o, o, v, v])
            Wmbej -= 0.5 * np.einsum("njfb,mnfe->mbej", t2, MO[o, o, v, v])

            # Wmbje (spin-factored exchange)
            Wmbje = -MO[o, v, o, v].copy()
            Wmbje -= np.einsum("jf,mbfe->mbje", t1, MO[o, v, v, v])
            Wmbje += np.einsum("nb,mnje->mbje", t1, MO[o, o, o, v])
            Wmbje += np.einsum("jnfb,mnfe->mbje", tw, MO[o, o, v, v])  # reuse tw

            # Zmbij — S13: BLAS DGEMM via reshape (O²V³ contraction)
            _nocc_l = tau.shape[0]
            _nvir_l = tau.shape[2]
            Zmbij = (
                MO[o, v, v, v].reshape(_nocc_l * _nvir_l, _nvir_l * _nvir_l)
                @ tau.reshape(_nocc_l * _nocc_l, _nvir_l * _nvir_l).T
            ).reshape(_nocc_l, _nvir_l, _nocc_l, _nocc_l)

            # === Energy ===
            E_corr = (
                2 * np.einsum("ia,ia", F[o, v], t1)
                + 2 * np.einsum("ijab,ijab", tau, MO[o, o, v, v])
                - np.einsum("ijab,ijba", tau, MO[o, o, v, v])
            )

            dE = abs(E_corr - E_old)
            if verbose and (it < 5 or it % 10 == 0 or dE < self.convergence):
                print(f"  Iter {it:3d}: E_corr = {E_corr:16.10f}  dE = {dE:.2e}")
            if dE < self.convergence and it > 0:
                if verbose:
                    print(f"  Converged at iteration {it}")
                break
            E_old = E_corr

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
            # Psi4NumPy convention: increment with full Fae/Fmi in residual
            t1_new = t1 + r1 / Dia

            # === T2 residual (Eq. 2) ===
            # Uses Psi4NumPy convention: full Fae/Fmi with increment update.
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

            # tau*<vvvv> — S7: BLAS DGEMM via reshape (O²V⁴ dominant term)
            r2 += (
                tau.reshape(_nocc_l * _nocc_l, _nvir_l * _nvir_l)
                @ MO[v, v, v, v].reshape(_nvir_l * _nvir_l, _nvir_l * _nvir_l).T
            ).reshape(_nocc_l, _nocc_l, _nvir_l, _nvir_l)

            # -P^(ab)_(ij) {t1*Zmbij}
            tmp = np.einsum("ma,mbij->ijab", t1, Zmbij)
            r2 -= tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

            # Wmbej terms — S6: cache the O³V³ contraction computed twice
            _X_mbej = np.einsum("imae,mbej->ijab", t2, Wmbej)  # computed ONCE

            # Wmbej term 1
            tmp = _X_mbej - np.einsum("imea,mbej->ijab", t2, Wmbej)
            r2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

            # Wmbej term 2 (reuse _X_mbej)
            tmp = _X_mbej + np.einsum("imae,mbje->ijab", t2, Wmbje)
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

            # Increment (not replace): t2_new = t2 + r2/D
            t2_new = t2 + r2 / Dijab

            # === DIIS ===
            d_t1.append(t1_new.copy())
            d_t2.append(t2_new.copy())
            d_e1.append(t1_new - t1)
            d_e2.append(t2_new - t2)
            if len(d_t1) > self.diis_size:
                d_t1.pop(0)
                d_t2.pop(0)
                d_e1.pop(0)
                d_e2.pop(0)
            if len(d_t1) >= 2:
                nd = len(d_t1)
                B = np.zeros((nd + 1, nd + 1))
                B[-1, :] = -1
                B[:, -1] = -1
                B[-1, -1] = 0
                # DIIS B-matrix via BLAS DGEMM (replaces nd² Python np.sum calls)
                _e1 = np.stack([e.ravel() for e in d_e1])  # (nd, o*v)
                _e2 = np.stack([e.ravel() for e in d_e2])  # (nd, o²v²)
                _E = np.concatenate([_e1, _e2], axis=1)  # (nd, o*v + o²v²)
                B[:nd, :nd] = _E @ _E.T
                rhs = np.zeros(nd + 1)
                rhs[-1] = -1
                try:
                    if np.linalg.cond(B) < 1e12:
                        c = np.linalg.solve(B, rhs)
                        # DIIS extrapolation via tensordot (replaces Python sum() generator)
                        T1_stack = np.stack(d_t1)  # (nd, nocc, nvir)
                        T2_stack = np.stack(d_t2)  # (nd, nocc, nocc, nvir, nvir)
                        t1_new = np.tensordot(c[:nd], T1_stack, axes=[[0], [0]])
                        t2_new = np.tensordot(c[:nd], T2_stack, axes=[[0], [0]])
                except np.linalg.LinAlgError:
                    pass

            t1 = t1_new
            t2 = t2_new

        else:
            if verbose:
                print(f"  WARNING: not converged in {self.max_iter} iterations")

        # T1 diagnostic (Lee & Taylor, JCP 94, 5463, 1989)
        t1_norm = np.linalg.norm(t1)
        n_occ_active = t1.shape[0]
        self.t1_diagnostic = t1_norm / np.sqrt(2 * n_occ_active)
        if verbose:
            print(f"  T1 diagnostic: {self.t1_diagnostic:.6f}", end="")
            if self.t1_diagnostic > 0.02:
                print(" (WARNING: T1 > 0.02, multi-reference character detected)")
            else:
                print(" (single-reference OK)")

        E_total = E_hf + E_corr
        self._t1 = t1
        self._t2 = t2
        self.t1 = t1  # public alias for T1 diagnostic
        self.t2 = t2  # public alias
        self._MO = MO
        self._g_chem = g  # (pq|rs) chemist for (T)
        self._eps = eps_active
        self._nocc = nocc
        self._nvir = nvir
        self._E_hf = E_hf
        self._E_corr = E_corr
        # Store full-space quantities for potential downstream use
        self._MO_full = MO_full
        self._g_chem_full = g_full
        self._eps_full = eps
        self._nocc_total = nocc_total

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"CCSD Results")
            if n_frozen > 0:
                print(f"  Frozen core orbitals: {n_frozen}")
            print(f"{'=' * 50}")
            print(f"  E(HF)        = {E_hf:16.10f} Eh")
            print(f"  E(CCSD corr) = {E_corr:16.10f} Eh")
            print(f"  E(CCSD tot)  = {E_total:16.10f} Eh")
            print(f"{'=' * 50}")

        return E_total, E_corr

    def ccsd_t(self, verbose=True):
        """
        Compute perturbative triples (T) correction.

        Native implementation following PySCF's ccsd_t_slow.py exactly:
        - Loop over unique abc triples (a >= b >= c)
        - 6 permutations of abc for W and V
        - r3 spin-adaptation function
        - 36 contractions per abc triple
        - Symmetry factor in denominator

        Frozen core: when active, the (T) uses only active-space MO integrals
        and orbital energies, consistent with the frozen-core CCSD amplitudes.

        Reference: JCP 94, 442 (1991); PySCF ccsd_t_slow.py

        Returns: E(T) in Hartree
        """
        if not hasattr(self, "_t1"):
            raise RuntimeError("Call solve() before ccsd_t()")

        t1 = self._t1
        t2 = self._t2
        MO = self._MO
        eps = self._eps
        nocc = self._nocc
        nvir = self._nvir

        if self.t2.size == 0 or nvir == 0:
            if verbose:
                print("  (T): No virtual orbitals, E(T) = 0.0")
            return 0.0

        if nocc < 2:
            if verbose:
                print("(T): Need >= 2 occupied orbitals. E(T) = 0.")
            return 0.0

        if verbose:
            n_frozen = self._n_frozen
            if n_frozen > 0:
                print(
                    f"(T): Computing perturbative triples "
                    f"(frozen_core={n_frozen}, active_occ={nocc}, nvir={nvir})..."
                )
            else:
                print(
                    f"(T): Computing perturbative triples (nocc={nocc}, nvir={nvir})..."
                )

        eo = eps[:nocc]
        ev = eps[nocc : nocc + nvir]
        t1T = t1.T  # (nvir, nocc)
        t2T = t2.transpose(2, 3, 0, 1)  # (nvir, nvir, nocc, nocc)

        # PySCF stores eris in CHEMIST notation (pq|rs).
        # Our g_chem[p,q,r,s] = (pq|rs) and MO[p,q,r,s] = <pq|rs> = g_chem[p,r,q,s].
        # PySCF's ccsd_t_slow uses specific transpositions to convert eris → physicist.
        # We match exactly:
        #   eris_vvov[a,b,i,c] = eris.ovvv.T[b,a,i,c].conj() → (va|ic) in some convention
        # Actually, from PySCF source:
        #   eris_vvov = eris.get_ovvv().conj().transpose(1,3,0,2)
        #   eris_vooo = eris.ovoo.conj().transpose(1,0,2,3)
        #   eris_vvoo = eris.ovov.conj().transpose(1,3,0,2)
        # Where eris.ovvv[i,a,b,c] = (ia|bc), eris.ovoo[i,a,j,k] = (ia|jk)
        # eris.ovov[i,a,j,b] = (ia|jb)
        #
        # So: eris_vvov[a,b,i,c] = (ia|bc).T(1,3,0,2) = eris.ovvv[i,a,b,c] → [a,c,i,b]
        # Wait, .conj().transpose(1,3,0,2) on ovvv[i,a,b,c] gives [a,c,i,b]
        # That means eris_vvov[a,c,i,b] = (ia|bc)*

        # Use our chemist g directly (already sliced to active space)
        n_active = nocc + nvir
        o = slice(0, nocc)
        v = slice(nocc, n_active)
        g_chem = self._g_chem  # (pq|rs) chemist, active space

        # eris_vvov = ovvv.conj().T(1,3,0,2): ovvv[i,a,b,c]=(ia|bc) → [a,c,i,b]
        eris_vvov = g_chem[o, v, v, v].transpose(1, 3, 0, 2)  # (nvir,nvir,nocc,nvir)
        # eris_vooo = ovoo.conj().T(1,0,2,3): ovoo[i,a,j,k]=(ia|jk) → [a,i,j,k]
        eris_vooo = g_chem[o, v, o, o].transpose(1, 0, 2, 3)  # (nvir,nocc,nocc,nocc)
        # eris_vvoo = ovov.conj().T(1,3,0,2): ovov[i,a,j,b]=(ia|jb) → [a,b,i,j]
        eris_vvoo = g_chem[o, v, o, v].transpose(1, 3, 0, 2)  # (nvir,nvir,nocc,nocc)
        # f_vo = fock[nocc:,:nocc] — zero for canonical
        fvo = np.zeros((nvir, nocc))

        eijk = eo[:, None, None] + eo[None, :, None] + eo[None, None, :]

        def get_w(a, b, c):
            """W(ijk, abc) = Σ_f eris_vvov[a,b,i,f]*t2T[c,f,k,j] - Σ_m eris_vooo[a,i,j,m]*t2T[b,c,m,k]
            Matches PySCF ccsd_t_slow.py: einsum('if,fkj->ijk') and einsum('ijm,mk->ijk').
            Note: t2T[c] index order is (f,k,j) not (f,j,k) — j/k swap is critical."""
            # PySCF: np.einsum('if,fkj->ijk', eris_vvov[a,b], t2T[c])
            # = sum_f eris_vvov[a,b,i,f] * t2T[c,f,k,j]
            w = np.einsum('if,fkj->ijk', eris_vvov[a, b], t2T[c])
            # PySCF: np.einsum('ijm,mk->ijk', eris_vooo[a], t2T[b,c])
            w -= np.einsum('ijm,mk->ijk', eris_vooo[a], t2T[b, c])
            return w

        def get_v(a, b, c):
            """V(ijk, abc) = <ab|ij> * t1T[c] + t2T[a,b] * fvo[c]"""
            v = np.einsum("ij,k->ijk", eris_vvoo[a, b], t1T[c])
            v += np.einsum("ij,k->ijk", t2T[a, b], fvo[c])
            return v

        def r3(w):
            """Spin-adaptation factor for closed-shell (T)."""
            return (
                4 * w
                + w.transpose(1, 2, 0)
                + w.transpose(2, 0, 1)
                - 2 * w.transpose(2, 1, 0)
                - 2 * w.transpose(0, 2, 1)
                - 2 * w.transpose(1, 0, 2)
            )

        def _contract_36(
            wabc, wacb, wbac, wbca, wcab, wcba, zabc, zacb, zbac, zbca, zcab, zcba
        ):
            """Compute all 36 contractions for one (a,b,c) triple.

            Replaces 36 np.einsum('XYZ,ijk',...) calls with np.sum(w.T * z).
            Transpose mapping: ijk→(0,1,2), ikj→(0,2,1), jik→(1,0,2),
                               jki→(2,0,1), kij→(1,2,0), kji→(2,1,0).
            """
            e = 0.0
            # zabc block (6 terms)
            e += np.sum(wabc * zabc)
            e += np.sum(wacb.transpose(0, 2, 1) * zabc)
            e += np.sum(wbac.transpose(1, 0, 2) * zabc)
            e += np.sum(wbca.transpose(2, 0, 1) * zabc)
            e += np.sum(wcab.transpose(1, 2, 0) * zabc)
            e += np.sum(wcba.transpose(2, 1, 0) * zabc)
            # zacb block
            e += np.sum(wacb * zacb)
            e += np.sum(wabc.transpose(0, 2, 1) * zacb)
            e += np.sum(wcab.transpose(1, 0, 2) * zacb)
            e += np.sum(wcba.transpose(2, 0, 1) * zacb)
            e += np.sum(wbac.transpose(1, 2, 0) * zacb)
            e += np.sum(wbca.transpose(2, 1, 0) * zacb)
            # zbac block
            e += np.sum(wbac * zbac)
            e += np.sum(wbca.transpose(0, 2, 1) * zbac)
            e += np.sum(wabc.transpose(1, 0, 2) * zbac)
            e += np.sum(wacb.transpose(2, 0, 1) * zbac)
            e += np.sum(wcba.transpose(1, 2, 0) * zbac)
            e += np.sum(wcab.transpose(2, 1, 0) * zbac)
            # zbca block
            e += np.sum(wbca * zbca)
            e += np.sum(wbac.transpose(0, 2, 1) * zbca)
            e += np.sum(wcba.transpose(1, 0, 2) * zbca)
            e += np.sum(wcab.transpose(2, 0, 1) * zbca)
            e += np.sum(wabc.transpose(1, 2, 0) * zbca)
            e += np.sum(wacb.transpose(2, 1, 0) * zbca)
            # zcab block
            e += np.sum(wcab * zcab)
            e += np.sum(wcba.transpose(0, 2, 1) * zcab)
            e += np.sum(wacb.transpose(1, 0, 2) * zcab)
            e += np.sum(wabc.transpose(2, 0, 1) * zcab)
            e += np.sum(wbca.transpose(1, 2, 0) * zcab)
            e += np.sum(wbac.transpose(2, 1, 0) * zcab)
            # zcba block
            e += np.sum(wcba * zcba)
            e += np.sum(wcab.transpose(0, 2, 1) * zcba)
            e += np.sum(wbca.transpose(1, 0, 2) * zcba)
            e += np.sum(wbac.transpose(2, 0, 1) * zcba)
            e += np.sum(wacb.transpose(1, 2, 0) * zcba)
            e += np.sum(wabc.transpose(2, 1, 0) * zcba)
            return e

        et = 0.0
        for a in range(nvir):
            for b in range(a + 1):
                # Batch over c: process all c in [0, b+1) at once
                # Precompute W and V intermediates for all c values
                c_range = range(b + 1)
                n_c = b + 1

                # Precompute get_w and get_v for all 6 permutations of (a,b,c)
                # for each c value in the batch. Store as lists for vectorized access.
                for c in c_range:
                    d3 = eijk - ev[a] - ev[b] - ev[c]
                    if a == c:  # a == b == c
                        d3 *= 6
                    elif a == b or b == c:
                        d3 *= 2

                    wabc = get_w(a, b, c)
                    wacb = get_w(a, c, b)
                    wbac = get_w(b, a, c)
                    wbca = get_w(b, c, a)
                    wcab = get_w(c, a, b)
                    wcba = get_w(c, b, a)
                    vabc = get_v(a, b, c)
                    vacb = get_v(a, c, b)
                    vbac = get_v(b, a, c)
                    vbca = get_v(b, c, a)
                    vcab = get_v(c, a, b)
                    vcba = get_v(c, b, a)

                    zabc = r3(wabc + 0.5 * vabc) / d3
                    zacb = r3(wacb + 0.5 * vacb) / d3
                    zbac = r3(wbac + 0.5 * vbac) / d3
                    zbca = r3(wbca + 0.5 * vbca) / d3
                    zcab = r3(wcab + 0.5 * vcab) / d3
                    zcba = r3(wcba + 0.5 * vcba) / d3

                    # All 36 contractions via np.sum instead of np.einsum
                    et += _contract_36(
                        wabc,
                        wacb,
                        wbac,
                        wbca,
                        wcab,
                        wcba,
                        zabc,
                        zacb,
                        zbac,
                        zbca,
                        zcab,
                        zcba,
                    )

        et *= 2  # Final factor of 2

        if verbose:
            print(f"  E(T)       = {et:16.10f} Eh")
            print(f"  E(CCSD)    = {self._E_corr:16.10f} Eh")
            print(f"  E(CCSD(T)) = {self._E_corr + et:16.10f} Eh")
            print(f"  E_total    = {self._E_hf + self._E_corr + et:16.10f} Eh")

        return et
