"""
Unrestricted Coupled Cluster Singles and Doubles (UCCSD) Solver
================================================================
Open-shell CCSD for systems with unpaired electrons (radicals,
triplet states, transition metals).

Uses the spin-orbital formalism: α and β MO coefficients are tiled
into a single 2N-dimensional spin-orbital basis, and antisymmetrized
two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> are used throughout.

This is the simplest correct UCCSD implementation — the same CCSD
equations as closed-shell but without spin-factoring, and with full
antisymmetry automatically handled.

Works with any UHF reference (doublet, triplet, quartet, etc.).

Update convention: INCREMENT with FULL Fae/Fmi (including diagonal).
    t1_new = t1 + Ω₁/Dia
    t2_new = t2 + Ω₂/Dijab
where Ω is the full residual computed from complete intermediates.

Equations follow Crawford & Schaefer, Rev. Comp. Chem. 14, 33 (2000),
adapted to spin-orbital notation.

References:
    Stanton, Bartlett, et al., J. Chem. Phys. 94, 4334 (1991)
    Crawford & Schaefer, Rev. Comp. Chem. 14, 33 (2000)
    Raghavachari et al., Chem. Phys. Lett. 157, 479 (1989) — (T)
    Watts, Gauss, Bartlett, J. Chem. Phys. 98, 8718 (1993) — open-shell (T)
"""

import numpy as np

__all__ = ["UCCSDSolver"]


def _build_spinorb_eri(C_alpha, C_beta, ERI_ao, nmo):
    """
    Build spin-orbital antisymmetrized integrals <pq||rs> in blocked
    ordering [α₁..αN, β₁..βN] from AO ERIs and UHF MO coefficients.

    Uses the identity:
        <pq||rs> = <pq|rs> - <pq|sr>
    with spin selection rules applied to the spatial integrals.

    Returns:
        MO_so: (2N, 2N, 2N, 2N) antisymmetrized physicist-notation integrals
    """
    nso = 2 * nmo

    # Transform AO ERIs to MO chemist integrals (pq|rs) for each spin block
    def transform(C1, C2, C3, C4):
        t = np.einsum("up,uvwx->pvwx", C1, ERI_ao, optimize=True)
        t = np.einsum("vq,pvwx->pqwx", C2, t, optimize=True)
        t = np.einsum("wr,pqwx->pqrx", C3, t, optimize=True)
        return np.einsum("xs,pqrx->pqrs", C4, t, optimize=True)

    g_aaaa = transform(C_alpha, C_alpha, C_alpha, C_alpha)
    g_bbbb = transform(C_beta, C_beta, C_beta, C_beta)
    g_aabb = transform(C_alpha, C_alpha, C_beta, C_beta)

    # Build antisymmetrized integrals in blocked [αα..., ββ...] ordering
    # <pq||rs> = <pq|rs>_phys - <pq|sr>_phys
    # = (pr|qs)_chem - (ps|qr)_chem
    MO_so = np.zeros((nso, nso, nso, nso))

    # Same-spin blocks: αα-αα and ββ-ββ
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    MO_so[p, q, r, s] = g_aaaa[p, r, q, s] - g_aaaa[p, s, q, r]
                    MO_so[nmo + p, nmo + q, nmo + r, nmo + s] = (
                        g_bbbb[p, r, q, s] - g_bbbb[p, s, q, r]
                    )

    # Cross-spin blocks: αβ-αβ and βα-βα (only Coulomb, no exchange)
    # <αp βq | αr βs> = (αp αr | βq βs)_chem = g_aabb[p,r,q,s]
    # Exchange <αp βq | βs αr> = 0 (spin mismatch on same electron)
    for p in range(nmo):
        for q in range(nmo):
            for r in range(nmo):
                for s in range(nmo):
                    MO_so[p, nmo + q, r, nmo + s] = g_aabb[p, r, q, s]
                    MO_so[nmo + p, q, nmo + r, s] = g_aabb[q, s, p, r]

    return MO_so


def _build_spinorb_eri_fast(C_alpha, C_beta, ERI_ao, nmo):
    """
    Vectorized version of spin-orbital integral construction.
    Same result as _build_spinorb_eri but avoids explicit Python loops.
    """
    nso = 2 * nmo

    def transform(C1, C2, C3, C4):
        t = np.einsum("up,uvwx->pvwx", C1, ERI_ao, optimize=True)
        t = np.einsum("vq,pvwx->pqwx", C2, t, optimize=True)
        t = np.einsum("wr,pqwx->pqrx", C3, t, optimize=True)
        return np.einsum("xs,pqrx->pqrs", C4, t, optimize=True)

    g_aaaa = transform(C_alpha, C_alpha, C_alpha, C_alpha)
    g_bbbb = transform(C_beta, C_beta, C_beta, C_beta)
    g_aabb = transform(C_alpha, C_alpha, C_beta, C_beta)

    MO_so = np.zeros((nso, nso, nso, nso))

    # αα-αα: <pq||rs> = g_aaaa[p,r,q,s] - g_aaaa[p,s,q,r]
    phys_aaaa = g_aaaa.swapaxes(1, 2)  # (pr|qs) → <pq|rs>
    MO_so[:nmo, :nmo, :nmo, :nmo] = phys_aaaa - phys_aaaa.swapaxes(2, 3)

    # ββ-ββ
    phys_bbbb = g_bbbb.swapaxes(1, 2)
    MO_so[nmo:, nmo:, nmo:, nmo:] = phys_bbbb - phys_bbbb.swapaxes(2, 3)

    # αβ-αβ: <αp βq|αr βs> = g_aabb[p,r,q,s] (no exchange)
    MO_so[:nmo, nmo:, :nmo, nmo:] = g_aabb.swapaxes(1, 2)

    # βα-βα: <βp αq|βr αs> = g_aabb[q,s,p,r]
    MO_so[nmo:, :nmo, nmo:, :nmo] = g_aabb.swapaxes(1, 2).transpose(2, 3, 0, 1)

    return MO_so


class UCCSDSolver:
    """Open-shell UHF-CCSD solver in spin-orbital basis."""

    def __init__(self, max_iter=100, convergence=1e-10, diis_size=8):
        self.max_iter = max_iter
        self.convergence = convergence
        self.diis_size = diis_size

    def _ccsd_iteration(self, t1, t2, MO, F, o, v, nocc, nvir, Dia, Dijab):
        """
        One CCSD iteration in spin-orbital basis.

        Uses INCREMENT update with FULL intermediates (including diagonal).
        This is critical for correctness — matches PySCF's GCCSD exactly.

        Returns: (t1_new, t2_new, E_corr)
        """
        # Effective doubles: ttau and tau (antisymmetrized)
        ttau = t2 + 0.5 * (
            np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
        )
        tau = t2 + (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1))

        # === Intermediates (Stanton Eqs. 3-8, spin-orbital) ===
        # All use FULL F (including diagonal) for increment update

        # Fae (Eq. 3)
        Fae = F[v, v].copy()
        Fae -= 0.5 * np.einsum("me,ma->ae", F[o, v], t1)
        Fae += np.einsum("mf,amef->ae", t1, MO[v, o, v, v])
        Fae -= 0.5 * np.einsum("mnaf,mnef->ae", ttau, MO[o, o, v, v])

        # Fmi (Eq. 4)
        Fmi = F[o, o].copy()
        Fmi += 0.5 * np.einsum("ie,me->mi", t1, F[o, v])
        Fmi += np.einsum("ne,mnie->mi", t1, MO[o, o, o, v])
        Fmi += 0.5 * np.einsum("inef,mnef->mi", ttau, MO[o, o, v, v])

        # Fme (Eq. 5)
        Fme = F[o, v].copy()
        Fme += np.einsum("nf,mnef->me", t1, MO[o, o, v, v])

        # Wmnij (Eq. 6)
        Wmnij = MO[o, o, o, o].copy()
        Wmnij += np.einsum("je,mnie->mnij", t1, MO[o, o, o, v])
        Wmnij -= np.einsum("ie,mnje->mnij", t1, MO[o, o, o, v])
        Wmnij += 0.25 * np.einsum("ijef,mnef->mnij", tau, MO[o, o, v, v])

        # Wabef (Eq. 7)
        Wabef = MO[v, v, v, v].copy()
        Wabef -= np.einsum("mb,amef->abef", t1, MO[v, o, v, v])
        Wabef += np.einsum("ma,bmef->abef", t1, MO[v, o, v, v])
        Wabef += 0.25 * np.einsum("mnab,mnef->abef", tau, MO[o, o, v, v])

        # Wmbej (Eq. 8)
        Wmbej = MO[o, v, v, o].copy()
        Wmbej += np.einsum("jf,mbef->mbej", t1, MO[o, v, v, v])
        Wmbej -= np.einsum("nb,mnej->mbej", t1, MO[o, o, v, o])
        Wmbej -= 0.5 * np.einsum("jnfb,mnef->mbej", t2, MO[o, o, v, v])
        Wmbej -= np.einsum("jf,nb,mnef->mbej", t1, t1, MO[o, o, v, v], optimize=True)

        # === CCSD Energy ===
        E_corr = (
            np.einsum("ia,ia", F[o, v], t1)
            + 0.25 * np.einsum("ijab,ijab", MO[o, o, v, v], t2)
            + 0.5 * np.einsum("ijab,ia,jb", MO[o, o, v, v], t1, t1, optimize=True)
        )

        # === T1 residual (full, for increment) ===
        r1 = F[o, v].copy()
        r1 += np.einsum("ie,ae->ia", t1, Fae)
        r1 -= np.einsum("ma,mi->ia", t1, Fmi)
        r1 += np.einsum("imae,me->ia", t2, Fme)
        r1 -= 0.5 * np.einsum("mnae,mnie->ia", t2, MO[o, o, o, v])
        r1 += 0.5 * np.einsum("imef,amef->ia", t2, MO[v, o, v, v])
        r1 -= np.einsum("nf,naif->ia", t1, MO[o, v, o, v])

        # INCREMENT: t1_new = t1 + r1/Dia
        t1_new = t1 + r1 / Dia

        # === T2 residual (full, for increment) ===
        r2 = MO[o, o, v, v].copy()

        # P(ab) {t2 * Fae} — FULL Fae including diagonal
        tmp = np.einsum("ijae,be->ijab", t2, Fae)
        r2 += tmp - tmp.swapaxes(2, 3)

        # -0.5 * P(ab) {t2 * t1 * Fme}
        tmp = np.einsum("mb,me->be", t1, Fme)
        tmp2 = 0.5 * np.einsum("ijae,be->ijab", t2, tmp)
        r2 -= tmp2 - tmp2.swapaxes(2, 3)

        # -P(ij) {t2 * Fmi} — FULL Fmi including diagonal
        tmp = np.einsum("imab,mj->ijab", t2, Fmi)
        r2 -= tmp - tmp.swapaxes(0, 1)

        # -0.5 * P(ij) {t2 * t1 * Fme}
        tmp = np.einsum("je,me->jm", t1, Fme)
        tmp2 = 0.5 * np.einsum("imab,jm->ijab", t2, tmp)
        r2 -= tmp2 - tmp2.swapaxes(0, 1)

        # 0.5 * tau * Wmnij
        r2 += 0.5 * np.einsum("mnab,mnij->ijab", tau, Wmnij)

        # 0.5 * tau * Wabef
        r2 += 0.5 * np.einsum("ijef,abef->ijab", tau, Wabef)

        # P(ij)P(ab) {t2*Wmbej - t1*t1*<mbej>}
        tmp = np.einsum("imae,mbej->ijab", t2, Wmbej)
        tmp -= np.einsum("ie,ma,mbej->ijab", t1, t1, MO[o, v, v, o], optimize=True)
        r2 += (
            tmp
            - tmp.swapaxes(0, 1)
            - tmp.swapaxes(2, 3)
            + tmp.swapaxes(0, 1).swapaxes(2, 3)
        )

        # P(ij) {t1 * <abej>}
        tmp = np.einsum("ie,abej->ijab", t1, MO[v, v, v, o])
        r2 += tmp - tmp.swapaxes(0, 1)

        # -P(ab) {t1 * <mbij>}
        tmp = np.einsum("ma,mbij->ijab", t1, MO[o, v, o, o])
        r2 -= tmp - tmp.swapaxes(2, 3)

        # INCREMENT: t2_new = t2 + r2/Dijab
        t2_new = t2 + r2 / Dijab

        return t1_new, t2_new, E_corr

    def solve_pyscf(self, molecule, verbose=True):
        """
        Solve UCCSD using PySCF's UHF → GHF conversion for exact spin-orbital integrals.

        This is the recommended method: it uses PySCF's SCF solver (SAD guess, DIIS)
        for the UHF reference, then PySCF's GHF conversion for spin-orbital structure,
        and our native CCSD iteration.

        Args:
            molecule: QENEX Molecule object

        Returns: (E_total, E_correlation)
        """
        try:
            from pyscf import gto, scf
            from pyscf.scf import addons
        except ImportError:
            raise ImportError(
                "PySCF required for solve_pyscf(). Install: pip install pyscf"
            )

        # Build PySCF molecule
        atoms_str = "; ".join(f"{el} {x} {y} {z}" for el, (x, y, z) in molecule.atoms)
        basis_map = {
            "sto-3g": "sto-3g",
            "sto3g": "sto-3g",
            "cc-pvdz": "cc-pvdz",
            "ccpvdz": "cc-pvdz",
            "aug-cc-pvdz": "aug-cc-pvdz",
            "augccpvdz": "aug-cc-pvdz",
            "cc-pvtz": "cc-pvtz",
            "ccpvtz": "cc-pvtz",
            "aug-cc-pvtz": "aug-cc-pvtz",
            "augccpvtz": "aug-cc-pvtz",
            "6-31g": "6-31g",
            "631g": "6-31g",
            "6-31g*": "6-31g*",
            "631gs": "6-31g*",
        }
        pyscf_basis = basis_map.get(molecule.basis_name.lower(), molecule.basis_name)
        spin = molecule.multiplicity - 1

        # cart=False: spherical d-functions (matches Dunning cc-pVXZ
        # and Pople 6-31G(d) definitions).  See solver.py for history.
        mol_p = gto.M(
            atom=atoms_str,
            basis=pyscf_basis,
            unit="bohr",
            cart=False,
            charge=molecule.charge,
            spin=spin,
            verbose=0,
        )

        # Run UHF
        mf = scf.UHF(mol_p)
        mf.max_cycle = 200
        mf.conv_tol = 1e-10
        mf.verbose = 4 if verbose else 0
        mf.kernel()

        if not mf.converged:
            import warnings

            warnings.warn("UHF did not converge — UCCSD results may be unreliable")

        E_uhf = mf.e_tot

        # Convert UHF → GHF for proper spin-orbital structure
        mf_ghf = addons.convert_to_ghf(mf)

        C = mf_ghf.mo_coeff  # (2*nao, 2*nmo) GHF MO coefficients
        eps = mf_ghf.mo_energy  # (2*nmo,) sorted spin-orbital energies
        nao = mol_p.nao
        nso = 2 * nao
        nocc = mol_p.nelectron
        nvir = nso - nocc

        if verbose:
            print(f"\nUCCSD: Spin-orbital integrals via GHF conversion")
            print(f"  E(UHF) = {E_uhf:.10f}")
            print(
                f"  N_alpha={mol_p.nelectron // 2 + spin // 2 + spin % 2}, "
                f"N_beta={mol_p.nelectron // 2 - spin // 2}"
            )
            print(f"  Spin-orbitals: {nso} (occ={nocc}, vir={nvir})")

        if nvir == 0:
            if verbose:
                print("  No virtual orbitals — E_corr = 0")
            return E_uhf, 0.0

        # Build spin-orbital AO ERI (block-diagonal in spin)
        eri_ao = mol_p.intor("int2e")
        eri_so = np.zeros((nso, nso, nso, nso))
        eri_so[:nao, :nao, :nao, :nao] = eri_ao
        eri_so[nao:, nao:, nao:, nao:] = eri_ao
        eri_so[:nao, :nao, nao:, nao:] = eri_ao
        eri_so[nao:, nao:, :nao, :nao] = eri_ao

        # Transform to MO basis
        if verbose:
            print("UCCSD: MO integral transformation...")
        t = np.einsum("up,uvwx->pvwx", C, eri_so, optimize=True)
        t = np.einsum("vq,pvwx->pqwx", C, t, optimize=True)
        t2_tmp = np.einsum("wr,pqwx->pqrx", C, t, optimize=True)
        del t
        g_chem = np.einsum("xs,pqrx->pqrs", C, t2_tmp, optimize=True)
        del t2_tmp

        # Antisymmetrize: <pq||rs> = (pr|qs) - (ps|qr) = g[p,r,q,s] - g[p,s,q,r]
        MO = g_chem.swapaxes(1, 2) - g_chem.transpose(0, 2, 3, 1)
        del g_chem

        # Run CCSD iteration
        return self._solve_with_integrals(MO, eps, nocc, nvir, E_uhf, verbose)

    def solve(self, uhf_solver, molecule, verbose=True):
        """
        Solve UCCSD on top of converged UHF from our native solver.

        Uses the UHF MO coefficients directly to build spin-orbital integrals
        in blocked [α₁..αN, β₁..βN] ordering.

        Args:
            uhf_solver: Converged UHFSolver instance
            molecule: Molecule object

        Returns: (E_total, E_correlation)
        """
        if not hasattr(uhf_solver, "C_alpha"):
            raise RuntimeError("UHF solver must be converged first (missing C_alpha)")

        C_alpha = uhf_solver.C_alpha
        C_beta = uhf_solver.C_beta
        eps_alpha = uhf_solver.eps_alpha
        eps_beta = uhf_solver.eps_beta

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
        total_electrons = (
            sum(Z_map.get(el, 0) for el, _ in molecule.atoms) - molecule.charge
        )
        n_alpha = (total_electrons + molecule.multiplicity - 1) // 2
        n_beta = total_electrons - n_alpha

        ERI_ao = uhf_solver.ERI if hasattr(uhf_solver, "ERI") else None
        if ERI_ao is None:
            raise RuntimeError(
                "UHF solver must store ERI (AO electron repulsion integrals)"
            )

        E_uhf, _ = uhf_solver.compute_energy(molecule, verbose=False)

        nmo = C_alpha.shape[1]
        nso = 2 * nmo
        nocc = n_alpha + n_beta
        nvir = nso - nocc

        if verbose:
            print(f"\nUCCSD: Building spin-orbital integrals (blocked ordering)...")
            print(f"  E(UHF) = {E_uhf:.10f}")
            print(f"  N_alpha={n_alpha}, N_beta={n_beta}")
            print(f"  Spin-orbitals: {nso} (occ={nocc}, vir={nvir})")

        if nvir == 0:
            return E_uhf, 0.0

        # Store UHF <S^2> if available (for post-CCSD reporting)
        s2_uhf = getattr(uhf_solver, "s2", None)
        if s2_uhf is None:
            # Try to compute from stored data
            try:
                S_ov = uhf_solver.S
                if S_ov is not None:
                    S_ab_MO = C_alpha.T @ S_ov @ C_beta
                    overlap_sum = sum(
                        S_ab_MO[i, j] ** 2
                        for i in range(n_alpha)
                        for j in range(n_beta)
                    )
                    s_z = (n_alpha - n_beta) / 2.0
                    s2_uhf = s_z * (s_z + 1.0) + n_beta - overlap_sum
            except Exception:
                pass
        self._uhf_s2 = s2_uhf

        # Build blocked spin-orbital integrals
        MO_so = _build_spinorb_eri_fast(C_alpha, C_beta, ERI_ao, nmo)

        # Orbital energies in blocked ordering
        eps_blocked = np.hstack([eps_alpha, eps_beta])

        # Reorder to put occupied first: [α_occ, β_occ, α_vir, β_vir]
        occ_idx = list(range(n_alpha)) + list(range(nmo, nmo + n_beta))
        vir_idx = list(range(n_alpha, nmo)) + list(range(nmo + n_beta, nso))
        all_idx = np.array(occ_idx + vir_idx)

        eps_so = eps_blocked[all_idx]
        MO = MO_so[np.ix_(all_idx, all_idx, all_idx, all_idx)]

        return self._solve_with_integrals(MO, eps_so, nocc, nvir, E_uhf, verbose)

    def _solve_with_integrals(self, MO, eps_so, nocc, nvir, E_uhf, verbose):
        """
        Core CCSD solver given spin-orbital integrals and orbital energies.

        Args:
            MO: antisymmetrized spin-orbital integrals <pq||rs> (nso, nso, nso, nso)
            eps_so: spin-orbital energies (nso,) with occupied first
            nocc: number of occupied spin-orbitals
            nvir: number of virtual spin-orbitals
            E_uhf: UHF total energy
            verbose: print convergence info

        Returns: (E_total, E_correlation)
        """
        nso = nocc + nvir
        o = slice(0, nocc)
        v = slice(nocc, nso)

        eo = eps_so[:nocc]
        ev = eps_so[nocc:]
        Dia = eo[:, None] - ev[None, :]
        Dijab = (
            eo[:, None, None, None]
            + eo[None, :, None, None]
            - ev[None, None, :, None]
            - ev[None, None, None, :]
        )

        F = np.diag(eps_so)

        # Initialize: T1=0, T2=MP2
        t1 = np.zeros((nocc, nvir))
        t2 = MO[o, o, v, v] / Dijab

        if verbose:
            print("UCCSD: Iterating...")

        # DIIS storage
        d_t1, d_t2, d_e1, d_e2 = [], [], [], []
        E_old = 0.0

        for it in range(self.max_iter):
            t1_new, t2_new, E_corr = self._ccsd_iteration(
                t1, t2, MO, F, o, v, nocc, nvir, Dia, Dijab
            )

            dE = abs(E_corr - E_old)
            if verbose and (it < 5 or it % 10 == 0 or dE < self.convergence):
                print(f"  Iter {it:3d}: E_corr = {E_corr:16.10f}  dE = {dE:.2e}")
            if dE < self.convergence and it > 0:
                if verbose:
                    print(f"  Converged at iteration {it}")
                break
            E_old = E_corr

            # DIIS acceleration
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
                for i in range(nd):
                    for j in range(nd):
                        B[i, j] = np.sum(d_e1[i] * d_e1[j]) + np.sum(d_e2[i] * d_e2[j])
                rhs = np.zeros(nd + 1)
                rhs[-1] = -1
                try:
                    if np.linalg.cond(B) < 1e12:
                        c = np.linalg.solve(B, rhs)
                        t1_new = sum(c[k] * d_t1[k] for k in range(nd))
                        t2_new = sum(c[k] * d_t2[k] for k in range(nd))
                except np.linalg.LinAlgError:
                    pass

            t1 = t1_new
            t2 = t2_new

        else:
            if verbose:
                print(f"  WARNING: not converged in {self.max_iter} iterations")

        E_total = E_uhf + E_corr
        self._t1 = t1
        self._t2 = t2
        self._MO = MO
        self._eps_so = eps_so
        self._nocc = nocc
        self._nvir = nvir
        self._E_uhf = E_uhf
        self._E_corr = E_corr

        # Store t1 for T1 diagnostic (public alias)
        self.t1 = t1

        # T1 diagnostic (Lee & Taylor 1989)
        t1_norm = np.linalg.norm(t1)
        n_occ_active = t1.shape[0]
        self.t1_diagnostic = (
            t1_norm / np.sqrt(2 * n_occ_active) if n_occ_active > 0 else 0.0
        )

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"UCCSD Results")
            print(f"{'=' * 50}")
            print(f"  E(UHF)        = {E_uhf:16.10f} Eh")
            print(f"  E(UCCSD corr) = {E_corr:16.10f} Eh")
            print(f"  E(UCCSD tot)  = {E_total:16.10f} Eh")
            print(f"  T1 diagnostic = {self.t1_diagnostic:.6f}", end="")
            if self.t1_diagnostic > 0.02:
                print(" (WARNING: T1 > 0.02, multi-reference character)")
            else:
                print(" (single-reference OK)")
            # <S^2> from UHF reference (approximate; exact requires CCSD density)
            if hasattr(self, "_uhf_s2"):
                print(f"  UHF <S^2>     = {self._uhf_s2:.4f}")
            print(f"{'=' * 50}")

        return E_total, E_corr

    def uccsd_t(self, verbose=True):
        """
        Compute perturbative triples (T) correction for open-shell UCCSD.

        Uses the Watts-Gauss-Bartlett spin-orbital formula (JCP 98, 8718, 1993),
        matching PySCF's gccsd_t_slow.py exactly:

            t3c = A(ijk)A(abc) [Σ_e t2(jk,ae)*<bc,ei> - Σ_m t2(im,bc)*<ma,jk>]
            t3d = A(ijk)A(abc) [t1(i,a)*<bc,jk> + f(a,i)*t2(jk,bc)]
            E(T) = (1/36) Σ (t3c + t3d)* · d3 · t3c / d3²
                 = (1/36) Σ (t3c + t3d) · t3c

        where A = full antisymmetrizer (A(ijk) = 1 - P(ij) - P(ik), etc.)

        The T1-dependent t3d term is essential for open-shell accuracy.

        Returns: E(T) in Hartree
        """
        if not hasattr(self, "_t1"):
            raise RuntimeError("Call solve() or solve_pyscf() before uccsd_t()")

        t1 = self._t1
        t2 = self._t2
        MO = self._MO
        eps_so = self._eps_so
        nocc = self._nocc
        nvir = self._nvir

        if nocc < 3 or nvir < 3:
            if verbose:
                print(
                    "(T): Need >= 3 occupied AND >= 3 virtual spin-orbitals. E(T) = 0."
                )
            return 0.0

        # Memory estimate for (T) tensors: d3, t3c, t3d are each
        # (nocc, nocc, nocc, nvir, nvir, nvir) float64 arrays.
        _t_elements = nocc**3 * nvir**3
        _t_arrays = 3  # d3, t3c, t3d (plus temporaries during antisymmetrization)
        mem_gb = (_t_elements * 8 * _t_arrays) / 1e9
        if mem_gb > 4.0:
            if verbose:
                print(
                    f"  WARNING: (T) requires ~{mem_gb:.1f} GB for triples tensors. "
                    f"Consider frozen core or a smaller basis to reduce memory."
                )
            if mem_gb > 16.0:
                raise MemoryError(
                    f"UCCSD(T) would require ~{mem_gb:.1f} GB for "
                    f"(nocc={nocc})^3 x (nvir={nvir})^3 triples tensors. "
                    f"Use frozen core or a smaller basis set to reduce "
                    f"occupied/virtual space."
                )

        nso = nocc + nvir
        o = slice(0, nocc)
        v = slice(nocc, nso)

        if verbose:
            print(f"(T): Computing perturbative triples (nocc={nocc}, nvir={nvir})...")
            if mem_gb > 0.1:
                print(f"  Estimated (T) memory: {mem_gb:.2f} GB")

        eo = eps_so[:nocc]
        ev = eps_so[nocc:]

        # Orbital energy denominator d3(ijk,abc) = εi+εj+εk-εa-εb-εc
        eia = eo[:, None] - ev[None, :]  # (nocc, nvir)
        d3 = (
            eia[:, None, None, :, None, None]
            + eia[None, :, None, None, :, None]
            + eia[None, None, :, None, None, :]
        )  # (nocc,nocc,nocc,nvir,nvir,nvir)

        # Integral slices matching PySCF convention:
        # bcei = ovvv^*.T(3,2,1,0) — for real orbitals, conj() = identity
        # ovvv = MO[o,v,v,v] → shape (nocc,nvir,nvir,nvir)
        # bcei[b,c,e,i] = ovvv[i,e,c,b] = MO[i,nocc+e,nocc+c,nocc+b]
        bcei = MO[o, v, v, v].transpose(3, 2, 1, 0)  # (nvir,nvir,nvir,nocc)

        # majk = ooov^*.T(2,3,0,1)
        # ooov = MO[o,o,o,v] → (nocc,nocc,nocc,nvir)
        # majk[m,a,j,k] = ooov[j,k,m,a]
        majk = MO[o, o, o, v].transpose(2, 3, 0, 1)  # (nocc,nvir,nocc,nocc)

        # bcjk = oovv^*.T(2,3,0,1)
        # oovv = MO[o,o,v,v] → (nocc,nocc,nvir,nvir)
        # bcjk[b,c,j,k] = oovv[j,k,b,c]
        bcjk = MO[o, o, v, v].transpose(2, 3, 0, 1)  # (nvir,nvir,nocc,nocc)

        # Fock occ-vir block (zero for canonical orbitals, but included for generality)
        F = np.diag(eps_so)
        fvo = F[nocc:, :nocc]  # (nvir, nocc) — f(a,i)

        # === Connected triples t3c (from T2) ===
        t3c = np.einsum("jkae,bcei->ijkabc", t2, bcei, optimize=True) - np.einsum(
            "imbc,majk->ijkabc", t2, majk, optimize=True
        )

        # Full antisymmetrization: A(abc) then A(ijk)
        # A(abc) = 1 - P(ab) - P(ac): swap indices 3↔4, 3↔5
        t3c = t3c - t3c.transpose(0, 1, 2, 4, 3, 5) - t3c.transpose(0, 1, 2, 5, 4, 3)
        # A(ijk) = 1 - P(ij) - P(ik): swap indices 0↔1, 0↔2
        t3c = t3c - t3c.transpose(1, 0, 2, 3, 4, 5) - t3c.transpose(2, 1, 0, 3, 4, 5)
        t3c /= d3

        # === Disconnected triples t3d (from T1) ===
        t3d = np.einsum("ia,bcjk->ijkabc", t1, bcjk, optimize=True) + np.einsum(
            "ai,jkbc->ijkabc", fvo, t2, optimize=True
        )

        # Antisymmetrize
        t3d = t3d - t3d.transpose(0, 1, 2, 4, 3, 5) - t3d.transpose(0, 1, 2, 5, 4, 3)
        t3d = t3d - t3d.transpose(1, 0, 2, 3, 4, 5) - t3d.transpose(2, 1, 0, 3, 4, 5)
        t3d /= d3

        # === E(T) = (1/36) Σ (t3c + t3d) · d3 · t3c ===
        # = (1/36) Σ (t3c + t3d) · d3 · (t3c_unnorm / d3)
        # But t3c already divided by d3, so:
        # E(T) = (1/36) Σ_{ijkabc} (t3c + t3d)[ijk,abc] * d3[ijk,abc] * t3c[ijk,abc]
        et = np.einsum("ijkabc,ijkabc,ijkabc", t3c + t3d, d3, t3c) / 36.0

        if verbose:
            print(f"  E(T)          = {et:16.10f} Eh")
            print(f"  E(UCCSD)      = {self._E_corr:16.10f} Eh")
            print(f"  E(UCCSD(T))   = {self._E_corr + et:16.10f} Eh")
            print(f"  E_total       = {self._E_uhf + self._E_corr + et:16.10f} Eh")

        return et
