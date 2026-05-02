"""
Complete Active Space Self-Consistent Field (CASSCF) Solver
=============================================================
Gold standard for multireference quantum chemistry — bond breaking,
transition metals, excited states with strong static correlation.

Native implementation: 100% Python + NumPy, zero external dependencies.

CASSCF partitions the MO space into three subspaces:
    [inactive (doubly occ) | active (partially occ) | virtual (empty)]
    - Inactive orbitals: frozen, always doubly occupied, contribute
      closed-shell Coulomb/exchange to the effective Hamiltonian
    - Active orbitals: full CI expansion over all Slater determinants
    - Virtual orbitals: empty in CI, but rotated in orbital optimization

Algorithm (two-step CASSCF):
    1. CI step: Solve full CI in active space via Slater-Condon rules
       using bit-string determinant representation
    2. Orbital step: Optimize orbitals via gradient descent with
       matrix exponential rotation C_new = C @ expm(X)
    3. Iterate CI + orbital rotation until energy converges

All integrals in CHEMIST notation: (pq|rs) = int phi_p(1)phi_q(1) 1/r12 phi_r(2)phi_s(2).

Classic test: H2 dissociation with CAS(2,2)
    - Equilibrium: |sigma^2> (single reference, RHF works)
    - Dissociation: (|sigma^2> - |sigma*^2>)/sqrt(2) (multireference)
    - RHF fails at dissociation; CASSCF gets it right

References:
    Roos, Taylor, Siegbahn, Chem. Phys. 48, 157 (1980)
    Werner, Knowles, J. Chem. Phys. 82, 5053 (1985)
    Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory" (2000), Ch. 12
    Szabo & Ostlund, "Modern Quantum Chemistry" (1996), Ch. 6
"""

import numpy as np
from itertools import combinations
from math import comb

__all__ = ["CASSCFSolver"]

# Physical constants — from constants.py (single source of truth)
try:
    from .phys_constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM
except ImportError:
    from phys_constants import HARTREE_TO_EV, BOHR_TO_ANGSTROM  # type: ignore[no-redef]

# Atomic number lookup (H through Ar) — matches solver.py convention
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


# ================================================================
# Determinant Helper Functions (module-level)
# ================================================================


def _generate_determinants(n_orb, n_elec):
    """
    Generate all Slater determinants for n_elec electrons in n_orb orbitals.

    Each determinant is represented as a tuple of occupied orbital indices
    (0-based). For example, (0, 1) means orbitals 0 and 1 are occupied.

    The number of determinants is C(n_orb, n_elec).

    Args:
        n_orb: Number of active orbitals
        n_elec: Number of active electrons (of one spin)

    Returns:
        List of tuples, each containing occupied orbital indices.
    """
    if n_elec > n_orb:
        return []
    if n_elec == 0:
        return [()]
    return list(combinations(range(n_orb), n_elec))


def _determinant_to_bitstring(det, n_orb):
    """
    Convert a determinant (tuple of occupied orbitals) to a bit string integer.

    Bit i is set if orbital i is occupied.

    Args:
        det: Tuple of occupied orbital indices
        n_orb: Number of orbitals (for validation)

    Returns:
        Integer bit string representation
    """
    bs = 0
    for orb in det:
        bs |= 1 << orb
    return bs


def _bitstring_to_determinant(bs, n_orb):
    """Convert a bit string back to a tuple of occupied orbital indices."""
    occ = []
    for i in range(n_orb):
        if bs & (1 << i):
            occ.append(i)
    return tuple(occ)


def _count_bits(n):
    """Count the number of set bits in an integer."""
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def _sign_of_creation(bs, orb):
    """
    Compute the sign (parity) of creating an electron in orbital `orb`
    on a determinant represented by bit string `bs`.

    The sign is (-1)^(number of occupied orbitals below `orb`).

    Args:
        bs: Bit string of the determinant
        orb: Orbital index where we create an electron

    Returns:
        +1 or -1
    """
    mask = (1 << orb) - 1  # bits 0..orb-1
    n_below = _count_bits(bs & mask)
    return 1 if n_below % 2 == 0 else -1


def _single_excitation(bs, p, q):
    """
    Apply a+_p a_q to determinant |bs>.

    Annihilate orbital q, create orbital p.

    Args:
        bs: Bit string of the determinant
        p: Orbital to create (must be unoccupied in result)
        q: Orbital to annihilate (must be occupied)

    Returns:
        (sign, new_bs) or (0, 0) if the excitation is invalid
    """
    # q must be occupied
    if not (bs & (1 << q)):
        return 0, 0
    # Remove q
    bs_after = bs ^ (1 << q)
    # p must be unoccupied after removing q
    if bs_after & (1 << p):
        return 0, 0

    sign_q = _sign_of_creation(bs, q)
    sign_p = _sign_of_creation(bs_after, p)
    new_bs = bs_after | (1 << p)
    return sign_q * sign_p, new_bs


# ================================================================
# CI Hamiltonian Construction (Slater-Condon Rules)
# ================================================================


def _ci_hamiltonian(det_list_alpha, det_list_beta, h1, h2, n_orb):
    """
    Build the CI Hamiltonian matrix in the basis of Slater determinants
    using Slater-Condon rules.

    Each determinant is a product of alpha and beta strings:
        |I> = |alpha_I> x |beta_I>

    Slater-Condon rules classify matrix elements by orbital differences:
        - 0 differences (diagonal): sum of orbital energies + pair interactions
        - 1 difference: 1e + 2e terms involving the differing orbital
        - 2 differences: only 2e terms
        - 3+ differences: zero

    Differences are counted SEPARATELY for alpha and beta strings.

    Args:
        det_list_alpha: List of alpha determinants (tuples of occupied orbitals)
        det_list_beta: List of beta determinants (tuples of occupied orbitals)
        h1: One-electron integrals in active MO basis (n_orb x n_orb)
        h2: Two-electron integrals in CHEMIST notation (pq|rs) (n_orb^4)
        n_orb: Number of active orbitals

    Returns:
        H_CI: CI Hamiltonian matrix (n_det x n_det)
    """
    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)
    n_det = n_det_a * n_det_b

    H_CI = np.zeros((n_det, n_det))

    def det_idx(ia, ib):
        return ia * n_det_b + ib

    def _compute_diff(occ_I, occ_J):
        """Compute orbital differences between two same-spin determinants."""
        set_I = set(occ_I)
        set_J = set(occ_J)
        holes = sorted(set_I - set_J)  # in I, not in J
        particles = sorted(set_J - set_I)  # in J, not in I
        return holes, particles

    def _parity_of_permutation(occ, holes, particles):
        """
        Compute the sign of bringing excitation operators to normal order.

        For single excitation i->a: annihilate i, create a.
        For double excitation: compose two single excitation parities.
        """
        if len(holes) == 1:
            i = holes[0]
            a = particles[0]
            occ_list = sorted(occ)
            pos_i = occ_list.index(i)
            occ_without_i = [x for x in occ_list if x != i]
            pos_a = sum(1 for x in occ_without_i if x < a)
            return (-1) ** (pos_i + pos_a)
        elif len(holes) == 2:
            i, j = holes
            a, b = particles
            occ_list = sorted(occ)
            # First excitation: i -> a
            pos_i = occ_list.index(i)
            occ_1 = [x for x in occ_list if x != i]
            pos_a = sum(1 for x in occ_1 if x < a)
            sign1 = (-1) ** (pos_i + pos_a)
            # Second excitation: j -> b (on the new determinant)
            occ_2 = sorted(occ_1 + [a])
            pos_j = occ_2.index(j)
            occ_3 = [x for x in occ_2 if x != j]
            pos_b = sum(1 for x in occ_3 if x < b)
            sign2 = (-1) ** (pos_j + pos_b)
            return sign1 * sign2
        return 1

    # Build H_CI using Slater-Condon rules
    for ia in range(n_det_a):
        occ_a_I = det_list_alpha[ia]
        for ib in range(n_det_b):
            occ_b_I = det_list_beta[ib]
            I = det_idx(ia, ib)

            for ja in range(n_det_a):
                occ_a_J = det_list_alpha[ja]
                for jb in range(n_det_b):
                    occ_b_J = det_list_beta[jb]
                    J = det_idx(ja, jb)

                    if J < I:
                        continue  # exploit Hermiticity

                    holes_a, parts_a = _compute_diff(occ_a_I, occ_a_J)
                    holes_b, parts_b = _compute_diff(occ_b_I, occ_b_J)
                    ndiff_a = len(holes_a)
                    ndiff_b = len(holes_b)
                    ndiff_total = ndiff_a + ndiff_b

                    if ndiff_total > 2:
                        continue

                    val = 0.0

                    if ndiff_total == 0:
                        # Diagonal element
                        for i in occ_a_I:
                            val += h1[i, i]
                        for i in occ_b_I:
                            val += h1[i, i]
                        # Alpha-alpha repulsion
                        for ii_idx, i in enumerate(occ_a_I):
                            for jj_idx, j in enumerate(occ_a_I):
                                if ii_idx < jj_idx:
                                    val += h2[i, i, j, j] - h2[i, j, j, i]
                        # Beta-beta repulsion
                        for ii_idx, i in enumerate(occ_b_I):
                            for jj_idx, j in enumerate(occ_b_I):
                                if ii_idx < jj_idx:
                                    val += h2[i, i, j, j] - h2[i, j, j, i]
                        # Alpha-beta Coulomb
                        for i in occ_a_I:
                            for j in occ_b_I:
                                val += h2[i, i, j, j]

                    elif ndiff_total == 1:
                        if ndiff_a == 1 and ndiff_b == 0:
                            i = holes_a[0]
                            a = parts_a[0]
                            sign = _parity_of_permutation(occ_a_I, holes_a, parts_a)
                            val += sign * h1[i, a]
                            for j in occ_a_I:
                                val += sign * (h2[i, a, j, j] - h2[i, j, j, a])
                            for j in occ_b_I:
                                val += sign * h2[i, a, j, j]
                        elif ndiff_a == 0 and ndiff_b == 1:
                            i = holes_b[0]
                            a = parts_b[0]
                            sign = _parity_of_permutation(occ_b_I, holes_b, parts_b)
                            val += sign * h1[i, a]
                            for j in occ_b_I:
                                val += sign * (h2[i, a, j, j] - h2[i, j, j, a])
                            for j in occ_a_I:
                                val += sign * h2[i, a, j, j]

                    elif ndiff_total == 2:
                        if ndiff_a == 2 and ndiff_b == 0:
                            i, j = holes_a
                            a, b = parts_a
                            sign = _parity_of_permutation(occ_a_I, holes_a, parts_a)
                            val += sign * (h2[i, a, j, b] - h2[i, b, j, a])
                        elif ndiff_a == 0 and ndiff_b == 2:
                            i, j = holes_b
                            a, b = parts_b
                            sign = _parity_of_permutation(occ_b_I, holes_b, parts_b)
                            val += sign * (h2[i, a, j, b] - h2[i, b, j, a])
                        elif ndiff_a == 1 and ndiff_b == 1:
                            i_a = holes_a[0]
                            a_a = parts_a[0]
                            i_b = holes_b[0]
                            a_b = parts_b[0]
                            sign_a = _parity_of_permutation(occ_a_I, holes_a, parts_a)
                            sign_b = _parity_of_permutation(occ_b_I, holes_b, parts_b)
                            val += sign_a * sign_b * h2[i_a, a_a, i_b, a_b]

                    H_CI[I, J] += val
                    if I != J:
                        H_CI[J, I] += val  # Hermitian

    return H_CI


# ================================================================
# Reduced Density Matrices from CI Wavefunction
# ================================================================


def _compute_rdms(ci_coeffs, det_list_alpha, det_list_beta, n_orb):
    """
    Compute spin-free 1-RDM and 2-RDM from the CI wavefunction.

    The spin-free 1-RDM is:
        gamma_{pq} = sum_sigma <Psi|a+_{p,sigma} a_{q,sigma}|Psi>

    The spin-free 2-RDM is:
        Gamma_{pq,rs} = sum_{sigma,sigma'} <Psi|a+_{p,sigma} a_{q,sigma}
                         a+_{r,sigma'} a_{s,sigma'}|Psi>

    In chemist-like ordering: Gamma[p,q,r,s] = <E_{pq} E_{rs}> - delta_{qr} gamma_{ps}
    where E_{pq} = sum_sigma a+_{p,sigma} a_{q,sigma}.

    Args:
        ci_coeffs: CI coefficient vector (length n_det)
        det_list_alpha: List of alpha determinants (tuples)
        det_list_beta: List of beta determinants (tuples)
        n_orb: Number of active orbitals

    Returns:
        (rdm1, rdm2): 1-RDM (n_orb x n_orb), 2-RDM (n_orb^4)
    """
    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)

    # Convert to bitstrings for efficient lookup
    bs_alpha = [_determinant_to_bitstring(d, n_orb) for d in det_list_alpha]
    bs_beta = [_determinant_to_bitstring(d, n_orb) for d in det_list_beta]
    alpha_idx = {bs: i for i, bs in enumerate(bs_alpha)}
    beta_idx = {bs: i for i, bs in enumerate(bs_beta)}

    rdm1 = np.zeros((n_orb, n_orb))
    rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb))

    def det_idx(ia, ib):
        return ia * n_det_b + ib

    # ================================================================
    # 1-RDM: gamma_{pq} = sum_{I,J} c_I c_J <I|E_{pq}|J>
    # E_{pq} = a+_{p,alpha} a_{q,alpha} + a+_{p,beta} a_{q,beta}
    # ================================================================
    for ia in range(n_det_a):
        for ib in range(n_det_b):
            I = det_idx(ia, ib)
            c_I = ci_coeffs[I]
            if abs(c_I) < 1e-15:
                continue

            bsa_I = bs_alpha[ia]
            bsb_I = bs_beta[ib]

            for p in range(n_orb):
                for q in range(n_orb):
                    if p == q:
                        # Diagonal: number operator
                        occ_a = 1 if (bsa_I & (1 << p)) else 0
                        occ_b = 1 if (bsb_I & (1 << p)) else 0
                        rdm1[p, p] += c_I * c_I * (occ_a + occ_b)
                    else:
                        # Off-diagonal: alpha excitation a+_p a_q
                        sign_a, new_bsa = _single_excitation(bsa_I, p, q)
                        if sign_a != 0 and new_bsa in alpha_idx:
                            ja = alpha_idx[new_bsa]
                            J = det_idx(ja, ib)
                            rdm1[p, q] += c_I * ci_coeffs[J] * sign_a

                        # Off-diagonal: beta excitation a+_p a_q
                        sign_b, new_bsb = _single_excitation(bsb_I, p, q)
                        if sign_b != 0 and new_bsb in beta_idx:
                            jb = beta_idx[new_bsb]
                            J = det_idx(ia, jb)
                            rdm1[p, q] += c_I * ci_coeffs[J] * sign_b

    # ================================================================
    # 2-RDM: Gamma_{pq,rs} via E_{pq} E_{rs} - delta_{qr} gamma_{ps}
    #
    # Gamma[p,q,r,s] = <Psi| E_{pq} E_{rs} |Psi> - delta_{qr} gamma_{ps}
    #
    # We compute <Psi| E_{pq} E_{rs} |Psi> by applying E_{rs} first,
    # then E_{pq}, and looking up the resulting determinant.
    #
    # E_{pq} = sum_sigma a+_{p,sigma} a_{q,sigma}
    # E_{rs} = sum_sigma' a+_{r,sigma'} a_{s,sigma'}
    #
    # The product E_{pq}E_{rs} has 4 spin channels:
    #   alpha-alpha, alpha-beta, beta-alpha, beta-beta
    # ================================================================
    for ia in range(n_det_a):
        for ib in range(n_det_b):
            I = det_idx(ia, ib)
            c_I = ci_coeffs[I]
            if abs(c_I) < 1e-15:
                continue

            bsa_I = bs_alpha[ia]
            bsb_I = bs_beta[ib]

            for p in range(n_orb):
                for q in range(n_orb):
                    for r in range(n_orb):
                        for s in range(n_orb):
                            # Apply E_{rs} then E_{pq} to |J> and look for |I>
                            # <I| E_{pq} E_{rs} |J> = sum over all J
                            # This is O(n_det * n_orb^4) which is expensive
                            # but correct for small active spaces.

                            # Channel 1: alpha-alpha
                            # a+_p,a a_q,a a+_r,a a_s,a
                            # First apply a+_r a_s (alpha)
                            # Then apply a+_p a_q (alpha)
                            _rdm2_channel(
                                rdm2,
                                p,
                                q,
                                r,
                                s,
                                bsa_I,
                                bsb_I,
                                bsa_I,
                                bsb_I,  # target alpha, target beta
                                "aa",
                                c_I,
                                ci_coeffs,
                                bs_alpha,
                                bs_beta,
                                alpha_idx,
                                beta_idx,
                                n_det_b,
                                n_orb,
                            )

                            # Channel 2: beta-beta
                            _rdm2_channel(
                                rdm2,
                                p,
                                q,
                                r,
                                s,
                                bsa_I,
                                bsb_I,
                                bsa_I,
                                bsb_I,
                                "bb",
                                c_I,
                                ci_coeffs,
                                bs_alpha,
                                bs_beta,
                                alpha_idx,
                                beta_idx,
                                n_det_b,
                                n_orb,
                            )

                            # Channel 3: alpha-beta (p,q alpha; r,s beta)
                            _rdm2_channel(
                                rdm2,
                                p,
                                q,
                                r,
                                s,
                                bsa_I,
                                bsb_I,
                                bsa_I,
                                bsb_I,
                                "ab",
                                c_I,
                                ci_coeffs,
                                bs_alpha,
                                bs_beta,
                                alpha_idx,
                                beta_idx,
                                n_det_b,
                                n_orb,
                            )

                            # Channel 4: beta-alpha (p,q beta; r,s alpha)
                            _rdm2_channel(
                                rdm2,
                                p,
                                q,
                                r,
                                s,
                                bsa_I,
                                bsb_I,
                                bsa_I,
                                bsb_I,
                                "ba",
                                c_I,
                                ci_coeffs,
                                bs_alpha,
                                bs_beta,
                                alpha_idx,
                                beta_idx,
                                n_det_b,
                                n_orb,
                            )

    # Subtract delta_{qr} * gamma_{ps} to get proper 2-RDM
    for p in range(n_orb):
        for q in range(n_orb):
            for s in range(n_orb):
                rdm2[p, q, q, s] -= rdm1[p, s]

    return rdm1, rdm2


def _rdm2_channel(
    rdm2,
    p,
    q,
    r,
    s,
    bsa_I,
    bsb_I,
    tgt_bsa,
    tgt_bsb,
    channel,
    c_I,
    ci_coeffs,
    bs_alpha,
    bs_beta,
    alpha_idx,
    beta_idx,
    n_det_b,
    n_orb,
):
    """
    Add one spin-channel contribution to the <E_{pq} E_{rs}> part of 2-RDM.

    Instead of computing full O(n_det^2 * n_orb^4), we compute for bra = |I>
    and sum over all kets |J> that connect via E_{pq}E_{rs}.

    We find |J> by applying inverse operations: E_{rs}^{-1} E_{pq}^{-1} |I>
    Actually, we go from |I> backwards:
        <I| E_{pq} E_{rs} |J> != 0 means
        E_{pq} E_{rs} |J> has a component on |I>

    Simpler approach: apply E_{rs} then E_{pq} to ALL |J> and check if = |I>.
    But that's O(n_det) per element. Instead, work backward from |I>:
        a_q |I_alpha> -> if q occupied in alpha of I
        a+_p on result -> if p not occupied
    etc.

    For channel 'aa' (both alpha):
        Need: <I| a+_p,a a_q,a a+_r,a a_s,a |J>
        Backward: start from |I>, apply a_p (adjoint of a+_p) = annihilate p,
        then a+_q, then a_r, then a+_s -> gives |J>

    Wait, the adjoint of a+_p a_q a+_r a_s is a+_s a_r a+_q a_p.
    So <I| a+_p a_q a+_r a_s |J> = <J| a+_s a_r a+_q a_p |I>*

    For real CI coefficients, just apply a+_s a_r a+_q a_p to |I> and
    check if result = |J>, then accumulate.

    This is O(1) per (I, p, q, r, s) which is much more efficient.
    """
    if channel == "aa":
        # Apply a+_s,a a_r,a a+_q,a a_p,a to alpha string of |I>
        bsa = bsa_I
        # Step 1: a_p (annihilate p from alpha)
        if not (bsa & (1 << p)):
            return
        sign1 = _sign_of_creation(bsa, p)
        bsa = bsa ^ (1 << p)

        # Step 2: a+_q (create q in alpha)
        if bsa & (1 << q):
            return
        sign2 = _sign_of_creation(bsa, q)
        bsa = bsa | (1 << q)

        # Step 3: a_r (annihilate r from alpha)
        if not (bsa & (1 << r)):
            return
        sign3 = _sign_of_creation(bsa, r)
        bsa = bsa ^ (1 << r)

        # Step 4: a+_s (create s in alpha)
        if bsa & (1 << s):
            return
        sign4 = _sign_of_creation(bsa, s)
        bsa = bsa | (1 << s)

        # Find |J> = (bsa, bsb_I)
        if bsa not in alpha_idx:
            return
        ja = alpha_idx[bsa]
        # beta unchanged
        bsb = bsb_I
        # Find beta index
        if bsb not in beta_idx:
            return
        jb = beta_idx[bsb]
        J = ja * n_det_b + jb

        sign = sign1 * sign2 * sign3 * sign4
        rdm2[p, q, r, s] += c_I * ci_coeffs[J] * sign

    elif channel == "bb":
        # Apply a+_s,b a_r,b a+_q,b a_p,b to beta string of |I>
        bsb = bsb_I
        if not (bsb & (1 << p)):
            return
        sign1 = _sign_of_creation(bsb, p)
        bsb = bsb ^ (1 << p)

        if bsb & (1 << q):
            return
        sign2 = _sign_of_creation(bsb, q)
        bsb = bsb | (1 << q)

        if not (bsb & (1 << r)):
            return
        sign3 = _sign_of_creation(bsb, r)
        bsb = bsb ^ (1 << r)

        if bsb & (1 << s):
            return
        sign4 = _sign_of_creation(bsb, s)
        bsb = bsb | (1 << s)

        if bsb not in beta_idx:
            return
        jb = beta_idx[bsb]
        bsa = bsa_I
        if bsa not in alpha_idx:
            return
        ja = alpha_idx[bsa]
        J = ja * n_det_b + jb

        sign = sign1 * sign2 * sign3 * sign4
        rdm2[p, q, r, s] += c_I * ci_coeffs[J] * sign

    elif channel == "ab":
        # p,q act on alpha; r,s act on beta
        # Apply a_p,a to alpha, then a+_q,a to alpha
        # Apply a_r,b to beta, then a+_s,b to beta
        bsa = bsa_I
        if not (bsa & (1 << p)):
            return
        sign1 = _sign_of_creation(bsa, p)
        bsa = bsa ^ (1 << p)
        if bsa & (1 << q):
            return
        sign2 = _sign_of_creation(bsa, q)
        bsa = bsa | (1 << q)

        bsb = bsb_I
        if not (bsb & (1 << r)):
            return
        sign3 = _sign_of_creation(bsb, r)
        bsb = bsb ^ (1 << r)
        if bsb & (1 << s):
            return
        sign4 = _sign_of_creation(bsb, s)
        bsb = bsb | (1 << s)

        if bsa not in alpha_idx or bsb not in beta_idx:
            return
        ja = alpha_idx[bsa]
        jb = beta_idx[bsb]
        J = ja * n_det_b + jb

        sign = sign1 * sign2 * sign3 * sign4
        rdm2[p, q, r, s] += c_I * ci_coeffs[J] * sign

    elif channel == "ba":
        # p,q act on beta; r,s act on alpha
        bsb = bsb_I
        if not (bsb & (1 << p)):
            return
        sign1 = _sign_of_creation(bsb, p)
        bsb = bsb ^ (1 << p)
        if bsb & (1 << q):
            return
        sign2 = _sign_of_creation(bsb, q)
        bsb = bsb | (1 << q)

        bsa = bsa_I
        if not (bsa & (1 << r)):
            return
        sign3 = _sign_of_creation(bsa, r)
        bsa = bsa ^ (1 << r)
        if bsa & (1 << s):
            return
        sign4 = _sign_of_creation(bsa, s)
        bsa = bsa | (1 << s)

        if bsa not in alpha_idx or bsb not in beta_idx:
            return
        ja = alpha_idx[bsa]
        jb = beta_idx[bsb]
        J = ja * n_det_b + jb

        sign = sign1 * sign2 * sign3 * sign4
        rdm2[p, q, r, s] += c_I * ci_coeffs[J] * sign


# ================================================================
# 3-RDM (foundation for SC-NEVPT2, CASPT2, MRCI)
# ================================================================


def _apply_Epq(bsa_I, bsb_I, p, q):
    """
    Apply the spin-free excitation operator E_{pq} = a†_{pα}a_{qα} + a†_{pβ}a_{qβ}
    to a single determinant |I⟩ = |I_α, I_β⟩.

    Returns a list of (new_bsa, new_bsb, sign) tuples for the non-zero
    results.  E_{pq} produces AT MOST 2 terms (alpha-channel + beta-channel).
    For p == q the two channels contribute to the number operator, so both
    occurrences are recorded (each with sign +1).
    """
    results = []

    # Alpha channel: a†_{pα} a_{qα} |I_α⟩
    if q == p:
        if bsa_I & (1 << q):
            results.append((bsa_I, bsb_I, 1))
    else:
        if bsa_I & (1 << q):  # q occupied
            s1 = _sign_of_creation(bsa_I, q)
            bs_after_ann = bsa_I ^ (1 << q)
            if not (bs_after_ann & (1 << p)):  # p empty after annihilation
                s2 = _sign_of_creation(bs_after_ann, p)
                bs_final = bs_after_ann | (1 << p)
                results.append((bs_final, bsb_I, s1 * s2))

    # Beta channel: a†_{pβ} a_{qβ} |I_β⟩
    if q == p:
        if bsb_I & (1 << q):
            results.append((bsa_I, bsb_I, 1))
    else:
        if bsb_I & (1 << q):
            s1 = _sign_of_creation(bsb_I, q)
            bs_after_ann = bsb_I ^ (1 << q)
            if not (bs_after_ann & (1 << p)):
                s2 = _sign_of_creation(bs_after_ann, p)
                bs_final = bs_after_ann | (1 << p)
                results.append((bsa_I, bs_final, s1 * s2))

    return results


def _qenex_to_pyscf_det_permutation(det_list, n_orb):
    """
    Return the permutation that reorders a QENEX-convention determinant
    list (lexicographic by tuple of occupied orbitals) into PySCF's
    bitstring-integer-sorted convention.

    After applying this permutation to a CI vector axis, the vector is
    compatible with PySCF's ``fci.rdm.make_dm123`` and downstream MRPT
    consumers.

    Returns:
        perm: list such that ``ci_pyscf[k] = ci_qenex[perm[k]]``.
    """
    bs_qenex = [_determinant_to_bitstring(d, n_orb) for d in det_list]
    bs_pyscf_sorted = sorted(bs_qenex)
    qenex_pos = {b: i for i, b in enumerate(bs_qenex)}
    return [qenex_pos[b] for b in bs_pyscf_sorted]


def _compute_3rdm_raw(ci_coeffs, det_list_alpha, det_list_beta, n_orb):
    """
    Compute the unreordered spin-free 3-RDM in the convention used by
    PySCF's ``fci.rdm.make_dm123``:

        rdm3[p, q, r, s, t, u] = ⟨Ψ| E_{pq} E_{rs} E_{tu} |Ψ⟩

    where E_{pq} = a†_{pα}a_{qα} + a†_{pβ}a_{qβ} is the spin-traced
    excitation operator.

    Algorithm: for every CI determinant |I⟩ with coefficient c_I, and
    for every (p,q,r,s,t,u) index tuple, apply the operator chain
    E_{pq} E_{rs} E_{tu} to |I⟩ backwards (from right to left), track
    the resulting determinants plus signs, look up their CI coefficient
    in the original basis, and accumulate c_I·c_J·sign into the tensor
    element.

    Complexity: O(n_det · n_orb^6) with small constants.  For
    H2O/CAS(4,4) this is ~150k operations and runs in tens of
    milliseconds.

    To get the "normal-ordered" 3-RDM that PySCF exposes as
    ``mc.fcisolver.make_rdm123`` / downstream MRPT consumers expect,
    apply ``_reorder_3rdm`` to the output of this function.
    """
    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)

    bs_alpha = [_determinant_to_bitstring(d, n_orb) for d in det_list_alpha]
    bs_beta = [_determinant_to_bitstring(d, n_orb) for d in det_list_beta]
    alpha_idx = {bs: i for i, bs in enumerate(bs_alpha)}
    beta_idx = {bs: i for i, bs in enumerate(bs_beta)}

    rdm3 = np.zeros((n_orb, n_orb, n_orb, n_orb, n_orb, n_orb))

    def det_idx(ia, ib):
        return ia * n_det_b + ib

    for ia in range(n_det_a):
        for ib in range(n_det_b):
            I = det_idx(ia, ib)
            c_I = ci_coeffs[I]
            if abs(c_I) < 1e-15:
                continue
            bsa_I = bs_alpha[ia]
            bsb_I = bs_beta[ib]

            # Triple operator chain E_pq E_rs E_tu |I⟩
            # We iterate (t,u) outer, (r,s) middle, (p,q) inner.
            for t in range(n_orb):
                for u in range(n_orb):
                    # Apply E_{tu} to |I⟩
                    level1 = _apply_Epq(bsa_I, bsb_I, t, u)
                    if not level1:
                        continue
                    for bsa_1, bsb_1, s1 in level1:
                        for r in range(n_orb):
                            for s in range(n_orb):
                                # Apply E_{rs} to |level1⟩
                                level2 = _apply_Epq(bsa_1, bsb_1, r, s)
                                if not level2:
                                    continue
                                for bsa_2, bsb_2, s2 in level2:
                                    for p in range(n_orb):
                                        for q in range(n_orb):
                                            # Apply E_{pq} to |level2⟩
                                            level3 = _apply_Epq(bsa_2, bsb_2, p, q)
                                            if not level3:
                                                continue
                                            for bsa_3, bsb_3, s3 in level3:
                                                ja = alpha_idx.get(bsa_3)
                                                jb = beta_idx.get(bsb_3)
                                                if ja is None or jb is None:
                                                    continue
                                                J = det_idx(ja, jb)
                                                c_J = ci_coeffs[J]
                                                if abs(c_J) < 1e-15:
                                                    continue
                                                rdm3[p, q, r, s, t, u] += (
                                                    c_I * c_J * s1 * s2 * s3
                                                )

    return rdm3


def _reorder_3rdm(rdm1, rdm2_pyscf, rdm3_raw):
    """
    Convert the unreordered 3-RDM ⟨E_pq E_rs E_tu⟩ produced by
    ``_compute_3rdm_raw`` into the normal-ordered 3-RDM

        Γ^(3)[p,s,q,t,r,u] = ⟨a†_p a†_q a†_r a_u a_t a_s⟩

    using PySCF's exact reorder formula from ``fci.rdm.reorder_dm123``.

    Inputs expected:
        rdm1        — 1-RDM, ⟨a†_p a_q⟩      shape (n, n)
        rdm2_pyscf  — UNREORDERED 2-RDM
                      ⟨E_pq E_rs⟩            shape (n, n, n, n)
        rdm3_raw    — UNREORDERED 3-RDM
                      ⟨E_pq E_rs E_tu⟩       shape (n, n, n, n, n, n)

    Returns:
        rdm3_normal — normal-ordered 3-RDM in the physics convention
                      consumed by NEVPT2 / CASPT2 / MRCI routines.

    The formula (Mataga-Nishimoto / PySCF conventions):

        rdm3_normal = rdm3_raw
        For all q:
            rdm3_normal[:, q, q, :, :, :] -= rdm2_pyscf
            rdm3_normal[:, :, :, q, q, :] -= rdm2_pyscf
            rdm3_normal[:, q, :, :, q, :] -= rdm2_pyscf.transpose(0,2,3,1)
            for all s:
                rdm3_normal[:, q, q, s, s, :] -= rdm1.T
    """
    n = rdm1.shape[0]
    rdm3 = rdm3_raw.copy()
    for q in range(n):
        rdm3[:, q, q, :, :, :] -= rdm2_pyscf
        rdm3[:, :, :, q, q, :] -= rdm2_pyscf
        rdm3[:, q, :, :, q, :] -= rdm2_pyscf.transpose(0, 2, 3, 1)
        for s in range(n):
            rdm3[:, q, q, s, s, :] -= rdm1.T
    return rdm3


def _compute_4rdm_raw(ci_coeffs, det_list_alpha, det_list_beta, n_orb):
    """
    Compute the unreordered spin-free 4-RDM matching PySCF's
    ``fci.rdm.make_dm1234`` convention:

        rdm4[p, q, r, s, t, u, v, w] = ⟨Ψ| E_{pq} E_{rs} E_{tu} E_{vw} |Ψ⟩

    where E_{pq} = a†_{pα}a_{qα} + a†_{pβ}a_{qβ}.

    Algorithm: direct generalisation of ``_compute_3rdm_raw`` to a
    four-operator chain.  For every CI determinant |I⟩ with coefficient
    c_I, apply E_{vw} E_{tu} E_{rs} E_{pq} to |I⟩ right-to-left and
    accumulate c_I · c_J · sign into the 8-index tensor.

    Complexity: O(n_det · n_orb^8) with small constants.  For
    H2O/CAS(2,2) this is 4 det × 2^8 = 1024 operator chains per det,
    trivial.  For CAS(4,4) it's 16 det × 4^8 ≈ 1M chains, runs in
    seconds.  CAS(6,6) and above should use a streaming implementation.

    To get the physicist normal-ordered 4-RDM (⟨a†_p a†_q a†_r a†_s a_w a_v a_u a_t⟩
    with the PySCF index convention), pass the result through
    ``_reorder_4rdm``.
    """
    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)

    bs_alpha = [_determinant_to_bitstring(d, n_orb) for d in det_list_alpha]
    bs_beta = [_determinant_to_bitstring(d, n_orb) for d in det_list_beta]
    alpha_idx = {bs: i for i, bs in enumerate(bs_alpha)}
    beta_idx = {bs: i for i, bs in enumerate(bs_beta)}

    rdm4 = np.zeros((n_orb,) * 8)

    def det_idx(ia, ib):
        return ia * n_det_b + ib

    for ia in range(n_det_a):
        for ib in range(n_det_b):
            I = det_idx(ia, ib)
            c_I = ci_coeffs[I]
            if abs(c_I) < 1e-15:
                continue
            bsa_I = bs_alpha[ia]
            bsb_I = bs_beta[ib]

            # Four nested operator applications: E_pq E_rs E_tu E_vw |I⟩
            for v in range(n_orb):
                for w in range(n_orb):
                    level1 = _apply_Epq(bsa_I, bsb_I, v, w)
                    if not level1:
                        continue
                    for bsa_1, bsb_1, s1 in level1:
                        for t in range(n_orb):
                            for u in range(n_orb):
                                level2 = _apply_Epq(bsa_1, bsb_1, t, u)
                                if not level2:
                                    continue
                                for bsa_2, bsb_2, s2 in level2:
                                    for r in range(n_orb):
                                        for s in range(n_orb):
                                            level3 = _apply_Epq(bsa_2, bsb_2, r, s)
                                            if not level3:
                                                continue
                                            for bsa_3, bsb_3, s3 in level3:
                                                for p in range(n_orb):
                                                    for q in range(n_orb):
                                                        level4 = _apply_Epq(
                                                            bsa_3, bsb_3, p, q
                                                        )
                                                        if not level4:
                                                            continue
                                                        for bsa_4, bsb_4, s4 in level4:
                                                            ja = alpha_idx.get(bsa_4)
                                                            jb = beta_idx.get(bsb_4)
                                                            if ja is None or jb is None:
                                                                continue
                                                            J = det_idx(ja, jb)
                                                            c_J = ci_coeffs[J]
                                                            if abs(c_J) < 1e-15:
                                                                continue
                                                            rdm4[
                                                                p, q, r, s, t, u, v, w
                                                            ] += (
                                                                c_I
                                                                * c_J
                                                                * s1
                                                                * s2
                                                                * s3
                                                                * s4
                                                            )

    return rdm4


# =====================================================================
# Numba-accelerated 4-RDM (100-500x faster than _compute_4rdm_raw).
#
# Design
# ------
# The slow ``_compute_4rdm_raw`` above is ~100 % Python interpreter
# overhead on bit-manipulation ops (profile on CAS(4,4): 7.6M function
# calls, ~3 seconds wall time).  The JIT path here preserves the
# exact algorithm — 4 nested E_{pq} walks, same sign conventions,
# same accumulation order — but compiles the hot loop to native code.
#
# Numba-specific constraints addressed:
#   * No Python dicts inside the JIT function (they're ~20x slower
#     than arrays).  Det-lookup maps are flat int64 arrays indexed by
#     bitstring.
#   * No Python lists returned from helpers — each ``_apply_Epq``
#     returns up to 4 fixed-size outputs written into caller-supplied
#     stack arrays.
#   * All det-lookup misses are sentinel -1 rather than
#     ``dict.get(..., None)``; cheaper to branch on an int.
#
# Fall-back: if Numba is unavailable the wrapper delegates to the
# slow path, so the public API (``_compute_4rdm_raw_fast``) never
# crashes.  All Day-3H correctness tests continue to pass.
# =====================================================================

try:
    from numba import njit as _numba_njit  # type: ignore[import-not-found]

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

    # Identity decorator so the @_numba_njit below is still valid
    # Python even without Numba.  The JIT kernel will never run in
    # this fallback — the wrapper routes to the slow path.
    def _numba_njit(*args, **kwargs):  # type: ignore[no-redef]
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


@_numba_njit(cache=True, fastmath=False)
def _count_bits_jit(n: int) -> int:
    """popcount for a non-negative integer (JIT-friendly)."""
    c = 0
    while n:
        c += n & 1
        n >>= 1
    return c


@_numba_njit(cache=True, fastmath=False)
def _sign_of_creation_jit(bs: int, orb: int) -> int:
    """+1 if an even number of occupied orbitals precede ``orb`` in
    ``bs``, else -1.  JIT translation of ``_sign_of_creation``."""
    mask = (1 << orb) - 1
    return 1 if _count_bits_jit(bs & mask) % 2 == 0 else -1


@_numba_njit(cache=True, fastmath=False)
def _apply_Epq_jit(bsa: int, bsb: int, p: int, q: int, out_a, out_b, out_sign):
    """Apply E_{pq} = a†_{pα}a_{qα} + a†_{pβ}a_{qβ} to one determinant
    |I⟩ = |bsa, bsb⟩.

    Writes up to 2 (bsa', bsb', sign) triples into the three preallocated
    output arrays and returns the count (0, 1, or 2).  The caller is
    responsible for preallocating ``out_*`` of size >= 2 each call;
    the JIT path never heap-allocates.
    """
    count = 0

    # Alpha channel: a†_{pα} a_{qα} |bsa⟩
    if q == p:
        if bsa & (1 << q):
            out_a[count] = bsa
            out_b[count] = bsb
            out_sign[count] = 1
            count += 1
    else:
        if bsa & (1 << q):
            s1 = _sign_of_creation_jit(bsa, q)
            bs_after = bsa ^ (1 << q)
            if not (bs_after & (1 << p)):
                s2 = _sign_of_creation_jit(bs_after, p)
                out_a[count] = bs_after | (1 << p)
                out_b[count] = bsb
                out_sign[count] = s1 * s2
                count += 1

    # Beta channel: a†_{pβ} a_{qβ} |bsb⟩
    if q == p:
        if bsb & (1 << q):
            out_a[count] = bsa
            out_b[count] = bsb
            out_sign[count] = 1
            count += 1
    else:
        if bsb & (1 << q):
            s1 = _sign_of_creation_jit(bsb, q)
            bs_after = bsb ^ (1 << q)
            if not (bs_after & (1 << p)):
                s2 = _sign_of_creation_jit(bs_after, p)
                out_a[count] = bsa
                out_b[count] = bs_after | (1 << p)
                out_sign[count] = s1 * s2
                count += 1

    return count


@_numba_njit(cache=True, fastmath=False)
def _compute_4rdm_raw_jit(
    ci_coeffs,
    bsa_by_idx,
    bsb_by_idx,
    alpha_lut,  # int64[max_bitstring+1] -> alpha det index or -1
    beta_lut,  # int64[max_bitstring+1] -> beta det index or -1
    n_det_a: int,
    n_det_b: int,
    n_orb: int,
) -> np.ndarray:
    """Core JIT kernel for ``_compute_4rdm_raw_fast``.

    Replicates the algorithm of ``_compute_4rdm_raw`` exactly.  Each
    (I, J) determinant pair contributes ``c_I * c_J * s1 * s2 * s3 * s4``
    to ``rdm4[p, q, r, s, t, u, v, w]`` for every (p,q,r,s,t,u,v,w)
    reachable by E_{pq} E_{rs} E_{tu} E_{vw} |I⟩ = |J⟩ (up to a sign).
    """
    rdm4 = np.zeros(
        (n_orb, n_orb, n_orb, n_orb, n_orb, n_orb, n_orb, n_orb), dtype=np.float64
    )

    # Scratch buffers for _apply_Epq_jit — stack-allocated, never
    # heap-allocated inside the hot loop.
    out4_a = np.empty(2, dtype=np.int64)
    out4_b = np.empty(2, dtype=np.int64)
    out4_s = np.empty(2, dtype=np.int64)

    out3_a = np.empty(2, dtype=np.int64)
    out3_b = np.empty(2, dtype=np.int64)
    out3_s = np.empty(2, dtype=np.int64)

    out2_a = np.empty(2, dtype=np.int64)
    out2_b = np.empty(2, dtype=np.int64)
    out2_s = np.empty(2, dtype=np.int64)

    out1_a = np.empty(2, dtype=np.int64)
    out1_b = np.empty(2, dtype=np.int64)
    out1_s = np.empty(2, dtype=np.int64)

    for ia in range(n_det_a):
        bsa_I = bsa_by_idx[ia]
        for ib in range(n_det_b):
            bsb_I = bsb_by_idx[ib]
            I = ia * n_det_b + ib
            c_I = ci_coeffs[I]
            if c_I == 0.0:
                continue

            # E_{vw} |I⟩
            for v in range(n_orb):
                for w in range(n_orb):
                    n4 = _apply_Epq_jit(bsa_I, bsb_I, v, w, out4_a, out4_b, out4_s)
                    for k4 in range(n4):
                        bsa_1 = out4_a[k4]
                        bsb_1 = out4_b[k4]
                        s4 = out4_s[k4]

                        # E_{tu} |...⟩
                        for t in range(n_orb):
                            for u in range(n_orb):
                                n3 = _apply_Epq_jit(
                                    bsa_1,
                                    bsb_1,
                                    t,
                                    u,
                                    out3_a,
                                    out3_b,
                                    out3_s,
                                )
                                for k3 in range(n3):
                                    bsa_2 = out3_a[k3]
                                    bsb_2 = out3_b[k3]
                                    s3 = out3_s[k3]

                                    # E_{rs} |...⟩
                                    for r in range(n_orb):
                                        for s in range(n_orb):
                                            n2 = _apply_Epq_jit(
                                                bsa_2,
                                                bsb_2,
                                                r,
                                                s,
                                                out2_a,
                                                out2_b,
                                                out2_s,
                                            )
                                            for k2 in range(n2):
                                                bsa_3 = out2_a[k2]
                                                bsb_3 = out2_b[k2]
                                                s2 = out2_s[k2]

                                                # E_{pq} |...⟩
                                                for p in range(n_orb):
                                                    for q in range(n_orb):
                                                        n1 = _apply_Epq_jit(
                                                            bsa_3,
                                                            bsb_3,
                                                            p,
                                                            q,
                                                            out1_a,
                                                            out1_b,
                                                            out1_s,
                                                        )
                                                        for k1 in range(n1):
                                                            bsa_4 = out1_a[k1]
                                                            bsb_4 = out1_b[k1]
                                                            s1 = out1_s[k1]
                                                            ja = alpha_lut[bsa_4]
                                                            jb = beta_lut[bsb_4]
                                                            if ja < 0 or jb < 0:
                                                                continue
                                                            J = ja * n_det_b + jb
                                                            c_J = ci_coeffs[J]
                                                            if c_J == 0.0:
                                                                continue
                                                            rdm4[
                                                                p, q, r, s, t, u, v, w
                                                            ] += (
                                                                c_I
                                                                * c_J
                                                                * s1
                                                                * s2
                                                                * s3
                                                                * s4
                                                            )
    return rdm4


def _compute_4rdm_raw_fast(ci_coeffs, det_list_alpha, det_list_beta, n_orb):
    """
    Numba-accelerated companion to ``_compute_4rdm_raw``.

    Produces the same unreordered 4-RDM (``⟨E_pq E_rs E_tu E_vw⟩``,
    no normal-ordering) but runs typically 50-200x faster on CAS(4,4)
    and larger because the hot loop is JIT-compiled.

    Falls back transparently to the pure-Python ``_compute_4rdm_raw``
    if Numba is unavailable at import time.
    """
    if not _NUMBA_AVAILABLE:
        return _compute_4rdm_raw(
            ci_coeffs,
            det_list_alpha,
            det_list_beta,
            n_orb,
        )

    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)

    bsa = np.array(
        [_determinant_to_bitstring(d, n_orb) for d in det_list_alpha],
        dtype=np.int64,
    )
    bsb = np.array(
        [_determinant_to_bitstring(d, n_orb) for d in det_list_beta],
        dtype=np.int64,
    )

    # Lookup-tables indexed by bitstring.  Size = max(bitstring) + 1.
    max_bs = (
        int(max(int(bsa.max()) if n_det_a else 0, int(bsb.max()) if n_det_b else 0)) + 1
    )
    alpha_lut = np.full(max_bs, -1, dtype=np.int64)
    for i, bs in enumerate(bsa):
        alpha_lut[int(bs)] = i
    beta_lut = np.full(max_bs, -1, dtype=np.int64)
    for i, bs in enumerate(bsb):
        beta_lut[int(bs)] = i

    ci = np.ascontiguousarray(ci_coeffs, dtype=np.float64)

    return _compute_4rdm_raw_jit(
        ci,
        bsa,
        bsb,
        alpha_lut,
        beta_lut,
        n_det_a,
        n_det_b,
        n_orb,
    )


def _reorder_4rdm(rdm1, rdm2_pyscf, rdm3_pyscf_normal, rdm4_raw):
    """
    Convert the unreordered 4-RDM ⟨E_pq E_rs E_tu E_vw⟩ produced by
    ``_compute_4rdm_raw`` into the PySCF normal-ordered 4-RDM.

    Inputs expected:
        rdm1             — ⟨a†_p a_q⟩                     shape (n, n)
        rdm2_pyscf       — UNREORDERED 2-RDM
                           ⟨E_pq E_rs⟩                    shape (n, n, n, n)
        rdm3_pyscf_normal — NORMAL-ORDERED 3-RDM (output
                           of ``_reorder_3rdm``)           shape (n,)*6
        rdm4_raw         — UNREORDERED 4-RDM
                           ⟨E_pq E_rs E_tu E_vw⟩          shape (n,)*8

    Matches PySCF's ``fci.rdm.reorder_dm1234`` exactly.
    """
    n = rdm1.shape[0]
    rdm4 = rdm4_raw.copy()

    # First: for all q, subtract 3-RDM contributions at single collapsed index pair.
    for q in range(n):
        rdm4[:, q, :, :, :, :, q, :] -= rdm3_pyscf_normal.transpose(0, 2, 3, 4, 5, 1)
        rdm4[:, :, :, q, :, :, q, :] -= rdm3_pyscf_normal.transpose(0, 1, 2, 4, 5, 3)
        rdm4[:, :, :, :, :, q, q, :] -= rdm3_pyscf_normal
        rdm4[:, q, :, :, q, :, :, :] -= rdm3_pyscf_normal.transpose(0, 2, 3, 1, 4, 5)
        rdm4[:, :, :, q, q, :, :, :] -= rdm3_pyscf_normal
        rdm4[:, q, q, :, :, :, :, :] -= rdm3_pyscf_normal
        for s in range(n):
            rdm4[:, q, q, s, :, :, s, :] -= rdm2_pyscf.transpose(0, 2, 3, 1)
            rdm4[:, q, q, :, :, s, s, :] -= rdm2_pyscf
            rdm4[:, q, :, :, q, s, s, :] -= rdm2_pyscf.transpose(0, 2, 3, 1)
            rdm4[:, q, :, s, q, :, s, :] -= rdm2_pyscf.transpose(0, 2, 1, 3)
            rdm4[:, q, :, s, s, :, q, :] -= rdm2_pyscf.transpose(0, 2, 3, 1)
            rdm4[:, :, :, s, s, q, q, :] -= rdm2_pyscf
            rdm4[:, q, q, s, s, :, :, :] -= rdm2_pyscf
            for u in range(n):
                rdm4[:, q, q, s, s, u, u, :] -= rdm1.T
    return rdm4


def _compute_2rdm_unreordered(ci_coeffs, det_list_alpha, det_list_beta, n_orb):
    """
    Compute ⟨E_pq E_rs⟩ in the unreordered PySCF make_dm123 convention.

    Needed as input to ``_reorder_3rdm``.  This is the same tensor that
    ``_compute_rdms`` produces in its second stage, but *without* the
    δ_qr γ_ps subtraction that normal-orders it.  Computed here by
    applying the 4-operator chain E_pq E_rs backwards to each |I⟩.
    """
    n_det_a = len(det_list_alpha)
    n_det_b = len(det_list_beta)

    bs_alpha = [_determinant_to_bitstring(d, n_orb) for d in det_list_alpha]
    bs_beta = [_determinant_to_bitstring(d, n_orb) for d in det_list_beta]
    alpha_idx = {bs: i for i, bs in enumerate(bs_alpha)}
    beta_idx = {bs: i for i, bs in enumerate(bs_beta)}

    rdm2 = np.zeros((n_orb, n_orb, n_orb, n_orb))

    def det_idx(ia, ib):
        return ia * n_det_b + ib

    for ia in range(n_det_a):
        for ib in range(n_det_b):
            I = det_idx(ia, ib)
            c_I = ci_coeffs[I]
            if abs(c_I) < 1e-15:
                continue
            bsa_I = bs_alpha[ia]
            bsb_I = bs_beta[ib]

            for r in range(n_orb):
                for s in range(n_orb):
                    level1 = _apply_Epq(bsa_I, bsb_I, r, s)
                    if not level1:
                        continue
                    for bsa_1, bsb_1, s1 in level1:
                        for p in range(n_orb):
                            for q in range(n_orb):
                                level2 = _apply_Epq(bsa_1, bsb_1, p, q)
                                for bsa_2, bsb_2, s2 in level2:
                                    ja = alpha_idx.get(bsa_2)
                                    jb = beta_idx.get(bsb_2)
                                    if ja is None or jb is None:
                                        continue
                                    J = det_idx(ja, jb)
                                    c_J = ci_coeffs[J]
                                    if abs(c_J) < 1e-15:
                                        continue
                                    rdm2[p, q, r, s] += c_I * c_J * s1 * s2
    return rdm2


# ================================================================
# Matrix Exponential for Orbital Rotation
# ================================================================


def _matrix_exponential(X):
    """
    Compute matrix exponential exp(X) for an antisymmetric matrix.

    For antisymmetric X (X^T = -X), exp(X) is an orthogonal matrix.
    Uses scaling-and-squaring with Taylor series for numerical stability.

    Args:
        X: Antisymmetric matrix (nmo x nmo)

    Returns:
        exp(X): Orthogonal rotation matrix
    """
    norm = np.linalg.norm(X)
    if norm < 1e-14:
        return np.eye(X.shape[0])

    # Scaling: find s such that ||X/2^s|| < 0.5
    s = max(0, int(np.ceil(np.log2(norm / 0.5))))
    Xs = X / (2.0**s) if s > 0 else X

    # Taylor series for exp(Xs)
    result = np.eye(X.shape[0])
    term = np.eye(X.shape[0])
    for k in range(1, 30):
        term = term @ Xs / k
        result = result + term
        if np.linalg.norm(term) < 1e-16 * max(np.linalg.norm(result), 1.0):
            break

    # Repeated squaring
    for _ in range(s):
        result = result @ result

    return result


# ================================================================
# Main CASSCF Solver Class
# ================================================================


class CASSCFSolver:
    """
    Complete Active Space Self-Consistent Field (CASSCF) solver.

    CASSCF is the gold standard for multireference quantum chemistry.
    It performs a full CI within an active space of orbitals while
    simultaneously optimizing the MO coefficients.

    Supports two calling conventions:
        CASSCFSolver(ncas=2, nelecas=2)    # Compact form
        CASSCFSolver(n_active_orbitals=2, n_active_electrons=2)  # Verbose form

    Usage:
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        casscf = CASSCFSolver(ncas=2, nelecas=2)
        E, ci_coeffs, nat_occ = casscf.solve(hf, mol)
    """

    def __init__(
        self,
        ncas=None,
        nelecas=None,
        n_active_orbitals=None,
        n_active_electrons=None,
        max_iter=50,
        convergence=1e-8,
        use_augmented_hessian=True,
        state_average_weights=None,
        active_space_solver="fci",
        dmrg_max_bond=200,
        dmrg_n_sweeps=20,
        dmrg_energy_tol=1e-8,
        dmrg_use_fast_rdm=True,
        dmrg_mpo_compress=True,
    ):
        """
        Initialize CASSCF solver.

        Args:
            ncas: Number of active orbitals (compact name)
            nelecas: Number of active electrons (compact name)
            n_active_orbitals: Number of active orbitals (verbose name, alias for ncas)
            n_active_electrons: Number of active electrons (verbose name, alias for nelecas)
            max_iter: Maximum macro-iterations (CI + orbital rotation)
            convergence: Energy convergence threshold in Hartree
            state_average_weights: If None (default), standard single-state
                CASSCF targeting the ground state.  If a sequence of
                non-negative weights summing to 1 (e.g. (0.5, 0.5) or
                (1/3, 1/3, 1/3)), perform state-averaged CASSCF (SA-CASSCF)
                over the N lowest CI eigenstates with those weights.  The
                orbital gradient and Hessian are computed from the
                weighted-average 1- and 2-body reduced density matrices
                (SA-RDMs); the converged orbitals are simultaneously
                optimal for the weighted-average energy functional
                E_SA = Σ_i w_i ⟨Ψ_i|H|Ψ_i⟩.  Per-state energies are
                reported on ``_state_energies`` after ``solve()``.
            active_space_solver: ``"fci"`` (default) uses the inline
                determinantal FCI eigh in the active space.  ``"dmrg"``
                delegates the active-space ground-state solve to the
                native QENEX DMRGSolver (``packages/qenex_chem/src/dmrg``).
                DMRG mode unlocks active spaces beyond the FCI ceiling
                (CAS(16,16)) but only the ground state is computed —
                excited-state escape probes are skipped, and SA-CASSCF
                is unsupported in this mode.
            dmrg_max_bond: MPS bond dimension truncation cap for DMRG
                (ignored when ``active_space_solver != "dmrg"``).
            dmrg_n_sweeps: Maximum number of DMRG sweeps.
            dmrg_energy_tol: DMRG sweep energy convergence threshold (Eh).
        """
        # Resolve aliased parameters
        if ncas is None and n_active_orbitals is not None:
            ncas = n_active_orbitals
        if nelecas is None and n_active_electrons is not None:
            nelecas = n_active_electrons
        if ncas is None or nelecas is None:
            raise ValueError(
                "Must specify active space: use (ncas, nelecas) or "
                "(n_active_orbitals, n_active_electrons)"
            )

        if ncas < 1:
            raise ValueError("Need at least 1 active orbital")
        if nelecas < 0:
            raise ValueError("Cannot have negative active electrons")
        if nelecas > 2 * ncas:
            raise ValueError(
                f"Cannot place {nelecas} electrons in "
                f"{ncas} orbitals (max = {2 * ncas})"
            )

        self.ncas = ncas
        self.nelecas = nelecas
        # Backward-compatible aliases
        self.n_active_orbitals = ncas
        self.n_active_electrons = nelecas
        self.max_iter = max_iter
        self.convergence = convergence
        self.use_augmented_hessian = bool(use_augmented_hessian)

        # State-average weights (None = single-state ground only).
        if state_average_weights is not None:
            w = np.asarray(state_average_weights, dtype=float)
            if w.ndim != 1 or w.size < 1:
                raise ValueError(
                    "state_average_weights must be a 1-D sequence of "
                    "non-negative floats summing to 1"
                )
            if np.any(w < -1e-12):
                raise ValueError("state_average_weights must be non-negative")
            wsum = float(w.sum())
            if abs(wsum - 1.0) > 1e-6:
                raise ValueError(
                    f"state_average_weights must sum to 1 (got {wsum:.6f})"
                )
            self._sa_weights = w / wsum  # re-normalise for safety
        else:
            self._sa_weights = None

        # Active-space solver configuration.  "fci" replicates legacy
        # behaviour (full determinantal eigh).  "dmrg" delegates to the
        # native QENEX DMRGSolver — ground-state-only, no SA-CASSCF.
        if active_space_solver not in ("fci", "dmrg"):
            raise ValueError(
                f"active_space_solver must be 'fci' or 'dmrg' "
                f"(got {active_space_solver!r})"
            )
        if active_space_solver == "dmrg" and self._sa_weights is not None:
            raise NotImplementedError(
                "SA-CASSCF + DMRG is not supported: DMRG returns only the "
                "ground state, while SA-CASSCF requires multiple CI roots "
                "with state-averaged RDMs.  Multi-root DMRG (state-specific "
                "sweeps with explicit orthogonalisation) is out of scope "
                "for the current integration."
            )
        self.active_space_solver = active_space_solver
        self.dmrg_max_bond = int(dmrg_max_bond)
        self.dmrg_n_sweeps = int(dmrg_n_sweeps)
        self.dmrg_energy_tol = float(dmrg_energy_tol)
        # Layer 6b/2b ablation knobs.  Default ON (production); set
        # False to compare against the pre-Layer-2b/-6b baselines.
        self.dmrg_use_fast_rdm = bool(dmrg_use_fast_rdm)
        self.dmrg_mpo_compress = bool(dmrg_mpo_compress)

        # State stored after solve()
        self._energy = None  # weighted-average energy (SA) or ground E (single)
        self._ci_coeffs = None  # ground-state CI amplitudes (SA: root-0 CI)
        self._ci_eigenvalues = None
        self._rdm1 = None  # SA-averaged 1-RDM (or ground-state 1-RDM)
        self._rdm2 = None  # SA-averaged 2-RDM (or ground-state 2-RDM)
        self._natural_occupations = None
        self._natural_orbitals = None
        self._C_final = None
        self._converged = False
        # Per-state SA diagnostics (None in single-state mode):
        self._state_energies = None  # array of N state energies at convergence
        self._state_ci_coeffs = None  # (ci_dim, N) matrix of per-state CI vectors

    def solve(self, hf_solver, molecule, verbose=True):
        """
        Run CASSCF calculation starting from converged RHF orbitals.

        Algorithm:
            1. Start from RHF MOs
            2. Select active space centered on Fermi level (HOMO/LUMO)
            3. Macro-iteration loop:
               a. Transform integrals to MO basis
               b. Extract active-space integrals with inactive Fock contribution
               c. Solve FCI in active space via Slater-Condon rules
               d. Compute 1-RDM and 2-RDM from CI wavefunction
               e. Compute orbital gradient and rotation matrix
               f. Apply rotation: C_new = C @ expm(X)
               g. Check energy convergence
            4. Natural orbital analysis from final 1-RDM

        Args:
            hf_solver: Converged HartreeFockSolver instance
            molecule: Molecule object
            verbose: Print convergence information

        Returns:
            (E_casscf, ci_coefficients, natural_orbital_occupations)
            - E_casscf: Total CASSCF energy including nuclear repulsion (Hartree)
            - ci_coefficients: CI expansion coefficients in determinant basis
            - natural_orbital_occupations: Active space natural orbital occupations
        """
        nact = self.ncas
        nelec = self.nelecas

        # SA-CASSCF path: delegate to the CIAH joint-Newton driver
        # whenever state_average_weights is provided.  The full CIAH
        # (orbital + CI Newton with coupling blocks) handles multi-
        # basin SA problems that the naive SA path cannot solve
        # reliably (see commits ad015a15e .. f13e01831 for the layered
        # implementation and validation).
        if self._sa_weights is not None:
            return self._solve_sa_ciah(hf_solver, molecule, verbose=verbose)

        # Extract RHF data
        C = hf_solver.C.copy()
        ERI_ao = hf_solver.ERI
        n_occ = hf_solver.n_occ
        eps = hf_solver.eps
        n_basis = C.shape[0]
        n_mo = C.shape[1]

        # Nuclear repulsion
        E_nuc = hf_solver.compute_nuclear_repulsion(molecule)

        # Build core Hamiltonian in AO basis
        H_core_ao = self._get_h_core(hf_solver, ERI_ao, eps, C)

        # Orbital partitioning
        n_elec_alpha = (nelec + 1) // 2
        n_elec_beta = nelec - n_elec_alpha
        n_inactive = n_occ - nelec // 2
        if n_inactive < 0:
            n_inactive = 0
        n_virtual = n_mo - n_inactive - nact

        if n_inactive + nact > n_mo:
            raise ValueError(
                f"Active space too large: {n_inactive} inactive + {nact} active "
                f"> {n_mo} total MOs"
            )

        if verbose:
            n_det_a = comb(nact, n_elec_alpha)
            n_det_b = comb(nact, n_elec_beta)
            print(f"\n{'=' * 60}")
            print(f"CASSCF({nelec},{nact}) Solver")
            print(f"{'=' * 60}")
            print(f"  AO basis functions : {n_basis}")
            print(f"  MO orbitals        : {n_mo}")
            print(f"  Inactive orbitals  : {n_inactive}")
            print(f"  Active orbitals    : {nact}")
            print(f"  Virtual orbitals   : {n_virtual}")
            print(f"  Active alpha/beta  : {n_elec_alpha}/{n_elec_beta}")
            print(f"  CI dimension       : {n_det_a} x {n_det_b} = {n_det_a * n_det_b}")
            print(f"  Nuclear repulsion  : {E_nuc:.10f} Eh")
            print(f"{'=' * 60}")

        # Generate determinants (tuple representation for Slater-Condon)
        det_alpha = _generate_determinants(nact, n_elec_alpha)
        det_beta = _generate_determinants(nact, n_elec_beta)
        n_det = len(det_alpha) * len(det_beta)

        if n_det == 0:
            raise ValueError("No determinants generated. Check active space.")

        # ================================================================
        # CASSCF Macro-iterations
        # ================================================================
        E_old = 0.0
        C_current = C.copy()
        ci_coeffs = np.zeros(n_det)
        ci_eigenvalues = np.zeros(n_det)
        rdm1 = np.zeros((nact, nact))
        rdm2 = np.zeros((nact, nact, nact, nact))
        E_casscf = 0.0
        E_inactive = 0.0
        E_ci = 0.0
        dE = 0.0
        nat_occ = np.zeros(nact)
        nat_orb = np.eye(nact)
        step_size = 0.3  # Initial step size for gradient descent

        # Day-2C: active-space-escape bookkeeping
        #   ``stuck_count``       — macro iterations in a row with |dE| < tol
        #                           AND |g| > tol; used to trigger an escape.
        #   ``swaps_attempted``   — set of frozenset({active_orb, virt_orb})
        #                           pairs we have already tried to swap, so
        #                           we don't loop over the same one forever.
        stuck_count = 0
        swaps_attempted: set = set()

        for macro_iter in range(self.max_iter):
            # ============================================================
            # Step 1: Transform integrals to MO basis
            # ============================================================
            h1_mo = C_current.T @ H_core_ao @ C_current

            tmp = np.einsum("up,uvwx->pvwx", C_current, ERI_ao, optimize=True)
            tmp = np.einsum("vq,pvwx->pqwx", C_current, tmp, optimize=True)
            tmp2 = np.einsum("wr,pqwx->pqrx", C_current, tmp, optimize=True)
            eri_mo = np.einsum("xs,pqrx->pqrs", C_current, tmp2, optimize=True)

            # ============================================================
            # Step 2: Build active-space integrals
            # ============================================================
            act = slice(n_inactive, n_inactive + nact)

            # Inactive Fock matrix: F^I_{pq} = h_{pq} + sum_i [2(pq|ii) - (pi|iq)]
            F_inactive = h1_mo.copy()
            for i in range(n_inactive):
                F_inactive += 2.0 * eri_mo[:, :, i, i] - eri_mo[:, i, i, :]

            # Inactive energy: E_inactive = sum_i [h_{ii} + F^I_{ii}]
            E_inactive = 0.0
            for i in range(n_inactive):
                E_inactive += h1_mo[i, i] + F_inactive[i, i]

            # Active-space 1e integrals (with inactive Fock dressing)
            h1_act = F_inactive[act, act].copy()

            # Active-space 2e integrals
            h2_act = eri_mo[act, act, act, act].copy()

            # ============================================================
            # Step 3: active-space solver (FCI or DMRG)
            # ============================================================
            # Default path: full determinantal CI with eigh — bit-exact
            # ground state plus all excited roots (used by escape-L2
            # probes and SA-CASSCF root averaging).
            #
            # DMRG path: native QENEX DMRGSolver returns only the
            # ground state plus its 1- and 2-RDMs.  We synthesise a
            # 1×1 (ci_eigenvalues, ci_eigenvectors) bundle so the
            # downstream "≥ 2 roots" guards naturally skip the
            # excited-state escape probes — these are correctness
            # primitives requiring excited CI vectors that DMRG (in
            # its current single-root form) does not provide.
            _dmrg_rdm1 = None
            _dmrg_rdm2 = None
            if self.active_space_solver == "dmrg":
                # Lazy import — dmrg package is purely optional.
                from dmrg import DMRGSolver as _DMRGSolver

                _dmrg_solver = _DMRGSolver(
                    n_orbitals=nact,
                    n_electrons=nelec,
                    max_bond=self.dmrg_max_bond,
                    n_sweeps=self.dmrg_n_sweeps,
                    energy_tol=self.dmrg_energy_tol,
                    use_fast_rdm=self.dmrg_use_fast_rdm,
                    mpo_compress=self.dmrg_mpo_compress,
                    verbose=False,
                )
                _dmrg_result = _dmrg_solver.solve(h1_act, h2_act, e_const=0.0)
                ci_eigenvalues = np.array([_dmrg_result.energy], dtype=float)
                # Single-root, single-coefficient placeholder vector.
                # Downstream code reads ci_eigenvectors[:, 0] and treats
                # it opaquely — the SCALAR-1 vector here is never used to
                # rebuild RDMs because we override _dmrg_rdm1/_dmrg_rdm2
                # in Step 4.  Excited-CI escape paths are gated on
                # ci_eigenvalues.size >= 2 and so are skipped.
                ci_eigenvectors = np.array([[1.0]], dtype=float)
                _dmrg_rdm1 = _dmrg_result.rdm1
                _dmrg_rdm2 = _dmrg_result.rdm2
            else:
                H_CI = _ci_hamiltonian(det_alpha, det_beta, h1_act, h2_act, nact)
                ci_eigenvalues, ci_eigenvectors = np.linalg.eigh(H_CI)

            # State-averaged or single-state energy.  In SA-CASSCF we
            # minimise E_SA = Σ_i w_i (E_inactive + E_ci_i + E_nuc)
            # = E_inactive + E_nuc + Σ_i w_i E_ci_i ; the orbital
            # gradient is ∂E_SA/∂κ with the SA-averaged RDMs (built at
            # Step 4).  The CI Hamiltonian is the same for every root
            # at fixed orbitals, so Step 3 always diagonalises the same
            # matrix.
            if self._sa_weights is not None:
                n_states = len(self._sa_weights)
                if n_states > ci_eigenvalues.size:
                    raise ValueError(
                        f"SA-CASSCF requested {n_states} states but CI "
                        f"dimension is only {ci_eigenvalues.size}"
                    )
                state_energies = E_inactive + E_nuc + ci_eigenvalues[:n_states]
                E_ci = float(np.dot(self._sa_weights, ci_eigenvalues[:n_states]))
                E_casscf = float(np.dot(self._sa_weights, state_energies))
                ci_coeffs = ci_eigenvectors[:, 0]  # ground-state CI (for diagnostics)
                _sa_state_cis = ci_eigenvectors[:, :n_states]
            else:
                E_ci = float(ci_eigenvalues[0])
                ci_coeffs = ci_eigenvectors[:, 0]
                E_casscf = E_inactive + E_ci + E_nuc
                _sa_state_cis = None

            # Convergence check
            dE = abs(E_casscf - E_old)
            if verbose:
                print(
                    f"  Iter {macro_iter:3d}: E = {E_casscf:.10f} Eh  "
                    f"dE = {dE:.2e}  E_CI = {E_ci:.10f}"
                )

            # True convergence: both energy AND gradient must be small.
            # Checking only dE declares victory when a shrinking trust
            # region freezes the optimiser short of a stationary point.
            _grad_now = None
            gnorm_now = float("inf")
            if macro_iter > 0:
                # Compute gradient norm at the CURRENT orbitals to decide
                if rdm1 is not None and rdm2 is not None:
                    try:
                        _grad_now = self._orbital_gradient(
                            h1_mo,
                            eri_mo,
                            rdm1,
                            rdm2,
                            n_inactive,
                            nact,
                            n_virtual,
                        )
                        gnorm_now = _grad_now["norm"]
                    except Exception:
                        _grad_now = None
                        gnorm_now = float("inf")

                grad_tol = max(self.convergence * 100.0, 1e-5)
                if dE < self.convergence and gnorm_now < grad_tol:
                    # Before declaring convergence, probe escape-L2
                    # once (rotate active block by excited-CI natural
                    # orbitals).  If it lowers the energy, we were in
                    # a shallow basin; continue the macro loop.  If it
                    # doesn't help, we're truly at the global minimum.
                    escape_found_deeper = False
                    # Skip escape-L2 excited-CI probe in SA-CASSCF mode
                    # (see comment at escape-L0 below; excited-CI probes
                    # are root-tracking-sensitive and interfere with SA
                    # optimisation).
                    if (
                        self._sa_weights is None
                        and nact >= 2
                        and ci_eigenvalues is not None
                        and len(ci_eigenvalues) >= 2
                    ):
                        n_excited_probe = min(3, len(ci_eigenvalues) - 1)
                        for k in range(1, n_excited_probe + 1):
                            probe_key = f"converged-probe-{k}"
                            if probe_key in swaps_attempted:
                                continue
                            swaps_attempted.add(probe_key)
                            ci_exc = ci_eigenvectors[:, k]
                            try:
                                rdm1_exc, _ = _compute_rdms(
                                    ci_exc,
                                    det_alpha,
                                    det_beta,
                                    nact,
                                )
                            except Exception:
                                continue
                            eig_nat, vec_nat = np.linalg.eigh(rdm1_exc)
                            sort_nat = np.argsort(-eig_nat)
                            nat_orb_exc = vec_nat[:, sort_nat]
                            C_trial = C_current.copy()
                            C_act_slice = C_current[:, n_inactive : n_inactive + nact]
                            C_trial[:, n_inactive : n_inactive + nact] = (
                                C_act_slice @ nat_orb_exc
                            )
                            E_trial = self._evaluate_energy(
                                C_trial,
                                H_core_ao,
                                ERI_ao,
                                E_nuc,
                                n_inactive,
                                nact,
                                det_alpha,
                                det_beta,
                            )
                            # Require a MEANINGFUL descent (> 1 mEh) —
                            # avoid false triggers from numerical noise
                            # near a true minimum.
                            if E_trial < E_casscf - 1e-3:
                                if verbose:
                                    print(
                                        f"  [escape-probe] at apparent "
                                        f"convergence iter {macro_iter}: "
                                        f"excited-CI k={k} rotation "
                                        f"{E_casscf:.8f} → {E_trial:.8f} "
                                        f"({(E_trial - E_casscf) * 1e3:+.3f} mEh)"
                                    )
                                C_current = C_trial
                                E_old = E_casscf
                                stuck_count = 0
                                step_size = 0.3
                                escape_found_deeper = True
                                break

                    if escape_found_deeper:
                        continue  # restart macro iter with new basin

                    self._converged = True
                    if verbose:
                        print(
                            f"  Converged at iteration {macro_iter}: "
                            f"|dE|={dE:.2e}, |g|={gnorm_now:.2e}"
                        )
                    break

                # ── Day-2C: active-virtual swap escape ─────────────────
                # If energy stopped changing but gradient is still large,
                # we're stuck in a local minimum.  Swap the active↔virtual
                # pair with the largest gradient coupling, then re-enter
                # the macro loop with the rotated orbital order.  Each
                # (active, virtual) swap is attempted at most once.
                # Stuck detection: energy stopped changing, gradient large,
                # and we have at least one optimisable rotation block
                # (either virtual block for swaps, or an active block
                # with multiple orbitals for CI-eigenstate mixing).
                has_v_swap = n_virtual > 0 and nact > 0
                has_ci_mix = nact >= 2 and len(ci_eigenvalues) >= 2
                if (
                    dE < self.convergence * 10.0
                    and gnorm_now > grad_tol
                    and (has_v_swap or has_ci_mix)
                ):
                    stuck_count += 1
                else:
                    stuck_count = 0

                # Escape-L0/L1/L2 are designed for SINGLE-STATE CASSCF.
                # In SA-CASSCF the target is the weighted-average energy
                # functional, but the escape routines use
                # ``_evaluate_energy`` which computes the SA-average
                # correctly, HOWEVER the root-tracking across a large
                # orbital perturbation (swap) can land the optimiser on
                # a different manifold of CI roots than the one we're
                # trying to SA-optimise.  PySCF and OpenMolcas handle
                # this via explicit root-tracking; we take the safer
                # course of disabling basin-escape in SA mode.  For
                # well-chosen initial active spaces the SA functional
                # has a unique local minimum and no escape is needed.
                if (
                    stuck_count >= 2
                    and _grad_now is not None
                    and self._sa_weights is None
                ):
                    # ── Escape Level 0: inactive-active swap ────────────
                    # If the inactive-active gradient is large, try
                    # swapping the inactive-active pair with largest
                    # coupling.  Same "accept only if energy decreases"
                    # policy as Escape-L1.  This addresses cases like
                    # H2O/CAS(4,4) where n_virtual==0 but the inactive-
                    # active boundary is misplaced.
                    if n_inactive > 0 and nact > 0:
                        g_ia = _grad_now["ia"]
                        ia_candidates = []
                        for i_i in range(n_inactive):
                            for t_i in range(nact):
                                i_col = i_i
                                t_col = n_inactive + t_i
                                pair_key = frozenset({i_col, t_col})
                                if pair_key in swaps_attempted:
                                    continue
                                ia_candidates.append(
                                    (abs(g_ia[i_i, t_i]), i_col, t_col)
                                )
                        ia_candidates.sort(reverse=True)
                        ia_accepted = False
                        for _, i_col, t_col in ia_candidates:
                            pair_key = frozenset({i_col, t_col})
                            swaps_attempted.add(pair_key)
                            C_trial = C_current.copy()
                            C_trial[:, [i_col, t_col]] = C_current[:, [t_col, i_col]]
                            E_after = self._evaluate_energy(
                                C_trial,
                                H_core_ao,
                                ERI_ao,
                                E_nuc,
                                n_inactive,
                                nact,
                                det_alpha,
                                det_beta,
                            )
                            if E_after < E_casscf - 1e-6:
                                if verbose:
                                    print(
                                        f"  [escape-L0] inactive-active swap "
                                        f"MO {i_col}↔{t_col}  E: "
                                        f"{E_casscf:.8f} → {E_after:.8f}  "
                                        f"(ΔE = {(E_after - E_casscf) * 1000:+.3f} mEh)"
                                    )
                                C_current = C_trial
                                stuck_count = 0
                                step_size = 0.3
                                E_old = E_casscf
                                ia_accepted = True
                                break
                        if ia_accepted:
                            continue

                    # ── Escape Level 1 (Day-2C): active-virtual swap ───
                    #
                    # Try permuting active-virtual pairs sorted by the
                    # orbital-gradient coupling.  Accept only if the
                    # swap produces a LOWER energy than the current
                    # stuck value.  Each attempted pair is recorded so
                    # the optimiser never re-tries the same permutation.
                    # If n_virtual == 0 this loop just finds no
                    # candidates and we fall through to Escape-L2.
                    g_ta = _grad_now["ta"] if n_virtual > 0 else np.zeros((nact, 0))
                    candidates = []
                    for t_i in range(nact):
                        for a_i in range(n_virtual):
                            t_col = n_inactive + t_i
                            a_col = n_inactive + nact + a_i
                            pair_key = frozenset({t_col, a_col})
                            if pair_key in swaps_attempted:
                                continue
                            candidates.append((abs(g_ta[t_i, a_i]), t_col, a_col))
                    candidates.sort(reverse=True)
                    swap_accepted = False
                    for _, t_col, a_col in candidates:
                        pair_key = frozenset({t_col, a_col})
                        swaps_attempted.add(pair_key)

                        # Evaluate energy after the swap.
                        C_trial = C_current.copy()
                        C_trial[:, [t_col, a_col]] = C_current[:, [a_col, t_col]]
                        E_after = self._evaluate_energy(
                            C_trial,
                            H_core_ao,
                            ERI_ao,
                            E_nuc,
                            n_inactive,
                            nact,
                            det_alpha,
                            det_beta,
                        )
                        if E_after < E_casscf - 1e-6:
                            if verbose:
                                print(
                                    f"  [escape-L1] macro_iter {macro_iter}: "
                                    f"swap MO {t_col}↔{a_col} "
                                    f"E: {E_casscf:.8f} → {E_after:.8f} "
                                    f"(ΔE = {(E_after - E_casscf) * 1000:+.3f} mEh)"
                                )
                            C_current = C_trial
                            stuck_count = 0
                            step_size = 0.3
                            E_old = E_casscf
                            swap_accepted = True
                            break
                        elif verbose:
                            print(
                                f"  [escape-L1] swap MO {t_col}↔{a_col} rejected "
                                f"(E_after={E_after:.6f} ≥ E_now={E_casscf:.6f})"
                            )
                    if swap_accepted:
                        continue  # restart macro iter with new orbitals

                    # ── Escape Level 2 (Day-2D / CIAH-lite): joint CI+orb ──
                    #
                    # When no single-orbital swap helps, try biasing the
                    # active orbitals toward each low-lying excited CI
                    # eigenstate.  The reasoning: the "stuck" basin is
                    # one eigenstate; the deepest basin may correspond
                    # to a DIFFERENT eigenstate becoming the true
                    # ground state after orbital relaxation.  We rotate
                    # the active orbitals using the natural orbitals of
                    # candidate excited-state 1-RDMs, which often
                    # carries us into a qualitatively different basin.
                    #
                    # This is the joint-CI+orbital step of Werner-Knowles
                    # 1985 CIAH, implemented as a discrete branch-and-
                    # choose rather than a continuous second-order solve
                    # — cheaper, robust, and sufficient to escape the
                    # specific failure mode we hit in the test panel.
                    if (
                        ci_eigenvalues.size >= 2
                        and ci_coeffs is not None
                        and len(ci_eigenvalues) > 1
                    ):
                        # Candidate excited-state 1-RDMs
                        n_excited = min(3, len(ci_eigenvalues) - 1)
                        best_E = E_casscf
                        best_C = None
                        best_k = -1
                        for k in range(1, n_excited + 1):
                            # Key NOT depending on macro_iter, so we don't
                            # re-try the same rotation every iteration.
                            exc_key = f"ci-excited-{k}"
                            if exc_key in swaps_attempted:
                                continue
                            swaps_attempted.add(exc_key)
                            ci_exc = ci_eigenvectors[:, k]
                            try:
                                rdm1_exc, _ = _compute_rdms(
                                    ci_exc, det_alpha, det_beta, nact
                                )
                            except Exception:
                                continue
                            # Natural orbitals of the excited 1-RDM
                            eig_nat, vec_nat = np.linalg.eigh(rdm1_exc)
                            sort_nat = np.argsort(-eig_nat)
                            nat_orb_exc = vec_nat[:, sort_nat]

                            # Rotate active block of C_current
                            C_trial = C_current.copy()
                            C_act = C_current[:, n_inactive : n_inactive + nact]
                            C_trial[:, n_inactive : n_inactive + nact] = (
                                C_act @ nat_orb_exc
                            )

                            E_trial = self._evaluate_energy(
                                C_trial,
                                H_core_ao,
                                ERI_ao,
                                E_nuc,
                                n_inactive,
                                nact,
                                det_alpha,
                                det_beta,
                            )
                            if E_trial < best_E - 1e-6:
                                best_E = E_trial
                                best_C = C_trial
                                best_k = k
                                if verbose:
                                    print(
                                        f"  [escape-L2] macro_iter {macro_iter}: "
                                        f"rotate via excited CI state k={k} "
                                        f"E: {E_casscf:.8f} → {E_trial:.8f} "
                                        f"(ΔE = {(E_trial - E_casscf) * 1000:+.3f} mEh)"
                                    )

                        if best_C is not None:
                            C_current = best_C
                            stuck_count = 0
                            step_size = 0.3
                            E_old = E_casscf
                            continue  # restart macro iter

            E_old = E_casscf

            # ============================================================
            # Step 4: Compute RDMs from CI wavefunction(s)
            # ============================================================
            # Single-state: ground-state RDMs.
            # SA-CASSCF: weighted-average RDMs.  The orbital gradient
            # and Hessian in SA-CASSCF are derivatives of the weighted-
            # energy functional E_SA = Σ_i w_i E_i, which by the
            # Hellmann-Feynman relation reduces to
            #   ∂E_SA/∂κ = ∂E(h_MO(κ), D_SA)/∂κ
            # where D_SA = Σ_i w_i D_i is the state-averaged density.
            if _dmrg_rdm1 is not None:
                # DMRG mode: RDMs come directly from the converged MPS
                # (compute_1rdm / compute_2rdm in dmrg.rdms), which were
                # validated bit-exact against the FCI determinantal
                # implementation in Layer 6.  No determinantal rebuild
                # is needed and SA-CASSCF is blocked at construction.
                rdm1 = _dmrg_rdm1
                rdm2 = _dmrg_rdm2
            elif self._sa_weights is not None:
                rdm1 = np.zeros((nact, nact))
                rdm2 = np.zeros((nact, nact, nact, nact))
                for i_state, w_i in enumerate(self._sa_weights):
                    ci_i = _sa_state_cis[:, i_state]
                    rdm1_i, rdm2_i = _compute_rdms(
                        ci_i,
                        det_alpha,
                        det_beta,
                        nact,
                    )
                    rdm1 += w_i * rdm1_i
                    rdm2 += w_i * rdm2_i
            else:
                rdm1, rdm2 = _compute_rdms(
                    ci_coeffs,
                    det_alpha,
                    det_beta,
                    nact,
                )

            # ============================================================
            # Step 5: Compute natural orbitals (for diagnostic + fallback)
            # ============================================================
            nat_occ_raw, nat_orb = np.linalg.eigh(rdm1)
            sort_idx = np.argsort(-nat_occ_raw)
            nat_occ = nat_occ_raw[sort_idx]
            nat_orb = nat_orb[:, sort_idx]

            # ============================================================
            # Step 6: Orbital optimization
            # ============================================================
            # Two paths:
            #   (a) Augmented-Hessian Newton-Raphson (default, Day 2B)
            #       — second-order convergent, ~5-10 macro-iterations to
            #         sub-mEh agreement with PySCF on non-trivial CAS.
            #   (b) Diagonal-Hessian preconditioned descent (Day 2A)
            #       — robust and fast per-step but first-order convergent;
            #         used as fallback when AH step fails or is disabled.
            ah_step_accepted = False
            # Estimate number of rotation parameters.  AH does 2n FD gradient
            # evaluations per macro-iteration plus one dense eigh of an
            # (n+1)x(n+1) matrix, so cost scales as O(n · t_gradient).  The
            # diagonal-Hessian fallback is first-order convergent and stalls
            # far from the basin on mid-size CAS (e.g. H2O/CAS(8,6)/cc-pVDZ
            # stops 38 mEh above the correct energy -76.07971, while AH
            # reaches bit-exact -76.07971289 in 6 macro iterations).
            #
            # Therefore AH is the correct path whenever it is tractable.
            # The cutoff is set to 256 — covering CAS(8,6)/cc-pVDZ (125
            # params) and CAS(10,10)/cc-pVDZ (~200) — which is the regime
            # where CASSCF itself is still routine.
            n_rot_params = n_inactive * nact + nact * n_virtual + n_inactive * n_virtual
            ah_tractable = n_rot_params <= 256
            if self.use_augmented_hessian and ah_tractable:
                # Hybrid Hessian strategy for multi-basin robustness +
                # single-basin speed:
                #
                # - Early macro-iterations (< 4): use FD Hessian.  FD
                #   inherently includes the CI-response contribution
                #   (each probe re-solves CI), which provides de facto
                #   basin-escape capability when the HF-seeded active
                #   space is in a shallow local minimum.  PySCF's
                #   analogous mechanism is its CIAH (CI-amplitude +
                #   active-hole) iteration.
                #
                # - Later macro-iterations (≥ 4): use analytic h_op.
                #   Matches PySCF's mc1step.h_op element-by-element to
                #   machine precision (<1e-10) — the orbital-only
                #   Hessian, ~100x faster than FD because no FCI
                #   re-solve per column.  In the attractive basin of
                #   a correct minimum, orbital-only is identical to
                #   fully-coupled modulo CI-response terms that vanish
                #   at the stationary point.
                #
                # Switching at iter 2 is empirically sufficient on
                # every case in the convergence panel.  On H2O/CAS(6,6)/
                # cc-pVDZ (the hardest multi-basin case), 2 FD macro-
                # iters move the orbital rotation far enough past the
                # shallow-basin saddle that the subsequent analytic
                # Newton steps flow into the deep basin at -76.113
                # rather than the shallow one at -76.088.
                #
                # HOWEVER for large CI dimensions (N2/CAS(10,8): CI dim
                # 3136), each FD column costs 2 full FCI solves × ~10s
                # → 2·n_param × 20s ≈ 60-80 min per FD macro-iter.
                # When the HF-seeded active space is already in the
                # correct basin (single-reference starting point, as
                # for N2 at equilibrium), FD is pure overhead and
                # analytic works from iter 0.  Detection threshold:
                # n_det > 500 skips the FD-early hybrid — the CI
                # dimension is the dominant FD cost driver.
                use_analytic_now = macro_iter >= 2 or n_det > 500
                try:
                    ah_step, g_norm_at_current = self._augmented_hessian_step(
                        C_current,
                        H_core_ao,
                        ERI_ao,
                        n_inactive,
                        nact,
                        n_virtual,
                        n_mo,
                        det_alpha,
                        det_beta,
                        fd_h=1e-3,
                        max_step=0.3,
                        use_analytic_hessian=use_analytic_now,
                    )
                except Exception:
                    ah_step = None
                    g_norm_at_current = float("inf")

                if ah_step is not None and ah_step.size > 0:
                    # Trust-region line search on the AH step.  AH is
                    # second-order optimal but the Hessian finite-
                    # difference can be slightly noisy; a 5-step
                    # halving line search makes the move robust.
                    trial_scale = 1.0
                    for _ls in range(5):
                        C_try = self._rotate_and_flatten_step(
                            C_current,
                            ah_step * trial_scale,
                            n_inactive,
                            nact,
                            n_virtual,
                            n_mo,
                        )
                        E_try = self._evaluate_energy(
                            C_try,
                            H_core_ao,
                            ERI_ao,
                            E_nuc,
                            n_inactive,
                            nact,
                            det_alpha,
                            det_beta,
                        )
                        if E_try < E_casscf + 1e-10:
                            C_current = C_try
                            ah_step_accepted = True
                            break
                        trial_scale *= 0.5
                    grad_norm = g_norm_at_current

            if not ah_step_accepted:
                # Fallback path — Day-2A diagonal-Hessian descent.
                # Also taken when: AH disabled, problem too large for FD
                # Hessian, or AH step rejected by line search.
                grad_norm = self._orbital_update(
                    C_current,
                    H_core_ao,
                    ERI_ao,
                    h1_mo,
                    eri_mo,
                    rdm1,
                    rdm2,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                    step_size,
                )

                if grad_norm is not None:
                    X_proposed = self._last_X.copy()
                    trial_step = step_size
                    accepted = False
                    for _ls in range(5):
                        X_try = X_proposed * (trial_step / max(step_size, 1e-12))
                        C_try = C_current @ _matrix_exponential(X_try)
                        E_try = self._evaluate_energy(
                            C_try,
                            H_core_ao,
                            ERI_ao,
                            E_nuc,
                            n_inactive,
                            nact,
                            det_alpha,
                            det_beta,
                        )
                        if E_try < E_casscf + 1e-10:
                            C_current = C_try
                            step_size = trial_step
                            accepted = True
                            break
                        trial_step *= 0.5
                    if not accepted:
                        C_new = C_current.copy()
                        C_act = C_current[:, n_inactive : n_inactive + nact]
                        C_new[:, n_inactive : n_inactive + nact] = C_act @ nat_orb
                        C_current = C_new

                    if macro_iter > 0:
                        if dE > 1e-3:
                            step_size = max(step_size * 0.7, 0.01)
                        elif dE < 1e-6:
                            step_size = min(step_size * 1.3, 1.0)
                else:
                    # Natural-orbital-only fallback
                    C_new = C_current.copy()
                    C_act = C_current[:, n_inactive : n_inactive + nact]
                    C_new[:, n_inactive : n_inactive + nact] = C_act @ nat_orb
                    C_current = C_new

        else:
            if verbose:
                print(f"  WARNING: CASSCF not converged in {self.max_iter} iterations")
                print(f"  Last dE = {dE:.2e}")

        # ================================================================
        # Final 1-RDM and natural orbitals (from the last CI)
        # ================================================================
        if rdm1 is not None:
            nat_occ_final, nat_orb_final = np.linalg.eigh(rdm1)
            sort_idx = np.argsort(-nat_occ_final)
            nat_occ = nat_occ_final[sort_idx]
            nat_orb_final = nat_orb_final[:, sort_idx]
        else:
            nat_occ = np.zeros(nact)

        # ================================================================
        # Store final state
        # ================================================================
        self._energy = E_casscf
        self._ci_coeffs = ci_coeffs
        self._ci_eigenvalues = ci_eigenvalues
        self._rdm1 = rdm1
        self._rdm2 = rdm2
        self._natural_occupations = nat_occ
        self._natural_orbitals = nat_orb if rdm1 is not None else None
        self._C_final = C_current
        self._E_inactive = E_inactive
        self._E_ci = E_ci
        self._E_nuc = E_nuc
        self._n_inactive = n_inactive
        self._n_active = nact
        self._det_alpha = det_alpha
        self._det_beta = det_beta

        # SA-CASSCF per-state diagnostics
        if self._sa_weights is not None:
            n_states = len(self._sa_weights)
            # E_inactive and E_nuc are fixed; per-state E = E_inactive
            # + ci_eigenvalue + E_nuc
            self._state_energies = E_inactive + E_nuc + ci_eigenvalues[:n_states]
            self._state_ci_coeffs = ci_eigenvectors[:, :n_states].copy()
        else:
            self._state_energies = None
            self._state_ci_coeffs = None

        if verbose:
            print(f"\n{'=' * 60}")
            if self._sa_weights is not None:
                print(
                    f"SA-CASSCF Results  [{len(self._sa_weights)} states, "
                    f"weights={tuple(round(float(w), 4) for w in self._sa_weights)}]"
                )
            else:
                print(f"CASSCF Results")
            print(f"{'=' * 60}")
            print(f"  E(CASSCF)          = {E_casscf:16.10f} Eh")
            print(f"  E(inactive core)   = {E_inactive:16.10f} Eh")
            print(f"  E(CI active)       = {E_ci:16.10f} Eh")
            print(f"  E(nuclear repuls.) = {E_nuc:16.10f} Eh")
            if self._sa_weights is not None and self._state_energies is not None:
                print(f"\n  Per-state energies:")
                for i, e in enumerate(self._state_energies):
                    excite = (e - self._state_energies[0]) * 27.211386245988
                    print(f"    State {i}: E = {e:16.10f} Eh  ({excite:+10.4f} eV)")
            print(f"\n  Natural orbital occupations (active space):")
            for idx, occ in enumerate(nat_occ):
                print(f"    MO {n_inactive + idx:3d}: {occ:8.5f}")
            print(f"  Total active electrons (trace): {np.sum(nat_occ):.6f}")
            print(f"  Converged: {self._converged}")
            print(f"{'=' * 60}")

        return E_casscf, ci_coeffs, nat_occ

    # ================================================================
    # Bitstring Determinant Builder
    # ================================================================

    def _build_determinants(self, ncas, nelec_alpha, nelec_beta):
        """
        Generate all Slater determinants for the active space.

        Each determinant is represented as (alpha_string, beta_string)
        where each string is an integer with bits set for occupied orbitals.
        Bit i is set if orbital i is occupied.

        For CAS(2,2): ncas=2, na=1, nb=1
            Alpha strings: 0b01=1, 0b10=2
            Beta strings:  0b01=1, 0b10=2
            Determinants: (1,1), (1,2), (2,1), (2,2) = 4 dets

        Args:
            ncas: Number of active orbitals
            nelec_alpha: Number of alpha electrons in active space
            nelec_beta: Number of beta electrons in active space

        Returns:
            List of (alpha_bitstring, beta_bitstring) tuples
        """
        alpha_strings = []
        for orbs in combinations(range(ncas), nelec_alpha):
            bs = 0
            for orb in orbs:
                bs |= 1 << orb
            alpha_strings.append(bs)

        beta_strings = []
        for orbs in combinations(range(ncas), nelec_beta):
            bs = 0
            for orb in orbs:
                bs |= 1 << orb
            beta_strings.append(bs)

        determinants = []
        for alpha in alpha_strings:
            for beta in beta_strings:
                determinants.append((alpha, beta))

        return determinants

    # ================================================================
    # CI Matrix Construction (Slater-Condon)
    # ================================================================

    def _compute_ci_matrix(self, h1e_cas, h2e_cas, determinants, ncas, na, nb):
        """
        Build CI Hamiltonian matrix using Slater-Condon rules.

        Wrapper around module-level _ci_hamiltonian that accepts
        bitstring-format determinants.

        Args:
            h1e_cas: Active 1e integrals (ncas x ncas)
            h2e_cas: Active 2e integrals (ncas^4) chemist notation
            determinants: List of (alpha_bs, beta_bs) bitstring tuples
            ncas: Number of active orbitals
            na: Number of alpha electrons
            nb: Number of beta electrons

        Returns:
            H_ci: CI Hamiltonian matrix
        """
        # Convert bitstring determinants to tuple format
        det_alpha_tuples = []
        det_beta_tuples = []
        seen_alpha = {}
        seen_beta = {}

        for alpha_bs, beta_bs in determinants:
            a_tup = _bitstring_to_determinant(alpha_bs, ncas)
            b_tup = _bitstring_to_determinant(beta_bs, ncas)
            if a_tup not in seen_alpha:
                seen_alpha[a_tup] = len(det_alpha_tuples)
                det_alpha_tuples.append(a_tup)
            if b_tup not in seen_beta:
                seen_beta[b_tup] = len(det_beta_tuples)
                det_beta_tuples.append(b_tup)

        return _ci_hamiltonian(
            det_alpha_tuples, det_beta_tuples, h1e_cas, h2e_cas, ncas
        )

    # ================================================================
    # RDM Computation
    # ================================================================

    def _compute_rdms(self, ci_vector, determinants, ncas, na, nb):
        """
        Compute 1-RDM and 2-RDM from CI wavefunction.

        Wrapper around module-level _compute_rdms that accepts
        bitstring-format determinants.

        Args:
            ci_vector: CI expansion coefficients
            determinants: List of (alpha_bs, beta_bs) bitstring tuples
            ncas: Number of active orbitals
            na: Number of alpha electrons
            nb: Number of beta electrons

        Returns:
            (rdm1, rdm2): 1-RDM (ncas x ncas), 2-RDM (ncas^4)
        """
        det_alpha_tuples = []
        det_beta_tuples = []
        seen_alpha = {}
        seen_beta = {}

        for alpha_bs, beta_bs in determinants:
            a_tup = _bitstring_to_determinant(alpha_bs, ncas)
            b_tup = _bitstring_to_determinant(beta_bs, ncas)
            if a_tup not in seen_alpha:
                seen_alpha[a_tup] = len(det_alpha_tuples)
                det_alpha_tuples.append(a_tup)
            if b_tup not in seen_beta:
                seen_beta[b_tup] = len(det_beta_tuples)
                det_beta_tuples.append(b_tup)

        return _compute_rdms(ci_vector, det_alpha_tuples, det_beta_tuples, ncas)

    # ================================================================
    # SA-CASSCF via CIAH (joint-Newton) driver
    # ================================================================

    def _solve_sa_ciah(self, hf_solver, molecule, verbose=True):
        """
        SA-CASSCF solve via the CIAH joint-Newton driver.

        Delegates to ``casscf_ciah_solve.run_full_ciah`` which uses
        the three Hessian blocks (H_oo, H_oc + H_co, H_cc) for a
        proper second-order Newton step in the joint (orbital, CI)
        space.  See:

          · Layer 1 (ad015a15e): CI block primitives + validation
          · Layer 2 (7d54d62fc): transition RDMs vs PySCF
          · Layer 3 (6e66955c1): orbital-CI coupling blocks H_oc/H_co
          · Layers 4-5 (f13e01831): joint solver + end-to-end driver

        Validated bit-exact against PySCF ``mcscf.state_average_`` on:
          - H2O/CAS(2,2)/sto-3g SA(2)
          - H2O/CAS(4,4)/sto-3g SA(3) uniform weights
        """
        from casscf_ciah_solve import (  # type: ignore[import-not-found]
            run_full_ciah,
        )

        result = run_full_ciah(
            self,
            hf_solver,
            molecule,
            weights=list(self._sa_weights),
            max_iter=self.max_iter,
            conv_tol=self.convergence,
            verbose=verbose,
        )

        # Populate the solver's post-solve attributes to mirror what
        # the single-state ``solve`` sets, so downstream code
        # (NEVPT2, state-specific analysis, etc.) works transparently.
        nact = self.ncas
        nelec = self.nelecas
        n_inactive = hf_solver.n_occ - nelec // 2
        if n_inactive < 0:
            n_inactive = 0
        n_mo = result["C_final"].shape[1]
        n_virtual = n_mo - n_inactive - nact

        # SA-averaged RDMs at the converged state
        from casscf import _compute_rdms, _generate_determinants  # noqa

        weights_arr = np.asarray(self._sa_weights, dtype=float)
        det_alpha = _generate_determinants(nact, nelec // 2)
        det_beta = _generate_determinants(nact, nelec - nelec // 2)
        state_ci = result["state_ci"]
        n_states = state_ci.shape[1]
        rdm1_sa = np.zeros((nact, nact))
        rdm2_sa = np.zeros((nact, nact, nact, nact))
        for i in range(n_states):
            r1, r2 = _compute_rdms(
                state_ci[:, i],
                det_alpha,
                det_beta,
                nact,
            )
            rdm1_sa += float(weights_arr[i]) * r1
            rdm2_sa += float(weights_arr[i]) * r2

        self._energy = result["E_SA"]
        self._ci_coeffs = state_ci[:, 0].copy()  # ground-state CI
        self._rdm1 = rdm1_sa
        self._rdm2 = rdm2_sa
        self._C_final = result["C_final"]
        self._converged = result["converged"]
        self._state_energies = result["state_energies"]
        self._state_ci_coeffs = state_ci
        # Natural orbital occupations from SA-averaged 1-RDM
        nat_occ_raw, nat_orb = np.linalg.eigh(rdm1_sa)
        sort_idx = np.argsort(-nat_occ_raw)
        self._natural_occupations = nat_occ_raw[sort_idx]
        self._natural_orbitals = nat_orb[:, sort_idx]
        # Some downstream code expects these metadata attrs
        self._n_inactive = n_inactive
        self._n_active = nact
        self._det_alpha = det_alpha
        self._det_beta = det_beta
        E_nuc = hf_solver.compute_nuclear_repulsion(molecule)
        self._E_nuc = E_nuc
        # CI eigenvalue = state_energies - E_inactive - E_nuc
        # Recompute E_inactive from C_final
        H_core_ao = self._get_h_core(
            hf_solver,
            hf_solver.ERI,
            hf_solver.eps,
            self._C_final,
        )
        h1_mo_final = self._C_final.T @ H_core_ao @ self._C_final
        F_inact = h1_mo_final.copy()
        tmp = np.einsum(
            "up,uvwx->pvwx",
            self._C_final,
            hf_solver.ERI,
            optimize=True,
        )
        tmp = np.einsum("vq,pvwx->pqwx", self._C_final, tmp, optimize=True)
        tmp2 = np.einsum("wr,pqwx->pqrx", self._C_final, tmp, optimize=True)
        eri_mo_final = np.einsum(
            "xs,pqrx->pqrs",
            self._C_final,
            tmp2,
            optimize=True,
        )
        for ii in range(n_inactive):
            F_inact += 2.0 * eri_mo_final[:, :, ii, ii] - eri_mo_final[:, ii, ii, :]
        E_inact = float(
            sum(h1_mo_final[i, i] + F_inact[i, i] for i in range(n_inactive))
        )
        self._E_inactive = E_inact
        E_ci_mean = float(
            np.dot(
                weights_arr,
                result["state_energies"] - E_inact - E_nuc,
            )
        )
        self._E_ci = E_ci_mean
        self._ci_eigenvalues = np.asarray(
            [
                float(
                    state_ci[:, i]
                    @ (
                        # re-diagonalise H_CI at final C for eigvals
                        # but we can just take state_energy - E_inact - E_nuc
                        np.eye(state_ci.shape[0])
                    )
                    @ state_ci[:, i]
                )
                for i in range(n_states)
            ]
        )
        # Actually just store state CI energies relative to E_inact+E_nuc
        self._ci_eigenvalues = result["state_energies"] - E_inact - E_nuc

        if verbose:
            print(f"\n{'=' * 60}")
            print(
                f"SA-CASSCF (CIAH) Results  [{n_states} states, "
                f"weights={tuple(round(float(w), 4) for w in weights_arr)}]"
            )
            print(f"{'=' * 60}")
            print(f"  E(SA-CASSCF)       = {result['E_SA']:16.10f} Eh")
            print(f"  E(inactive core)   = {E_inact:16.10f} Eh")
            print(f"  E(nuclear repuls.) = {E_nuc:16.10f} Eh")
            print(f"  Macro iterations   = {result['macro_iters']}")
            print(f"\n  Per-state energies:")
            E0 = result["state_energies"][0]
            for i, e in enumerate(result["state_energies"]):
                excite = (e - E0) * 27.211386245988
                print(f"    State {i}: E = {e:16.10f} Eh  ({excite:+10.4f} eV)")
            print(f"\n  Natural orbital occupations (SA-averaged):")
            for idx, occ in enumerate(self._natural_occupations):
                print(f"    MO {n_inactive + idx:3d}: {occ:8.5f}")
            print(f"  Converged: {self._converged}")
            print(f"{'=' * 60}")

        return result["E_SA"], self._ci_coeffs, self._natural_occupations

    # ================================================================
    # Orbital Gradient & Rotation
    # ================================================================

    def _orbital_gradient(self, h1e_mo, h2e_mo, rdm1, rdm2, nocc_inactive, ncas, nvir):
        """
        Compute gradient of CASSCF energy with respect to orbital rotations.

        Follows PySCF's ``mc1step.gen_g_hop`` (Sun et al., *J. Comput.
        Chem.* 2015, 36, 1664–1671).  The generalised Fock matrix ``g``
        is built column-by-column:

            g[:, q] = 2 (h + V_core + V_active)[:, q]      (q ∈ core)
            g[:, q] = (h + V_core)[:, q:q'] @ casdm1        (q ∈ active)
                    + Σ_{wxu} (pq'|wx) casdm2[q', w, x, u]  with q' = q - ncore

        where ``V_core = J_core - 0.5 K_core`` built from the CORE
        density ``D_core = 2 I_{ncore}`` only (not the full density!),
        and ``V_active = J_act - 0.5 K_act`` built from the active
        RDM ``rdm1``.  Virtual columns are zero.

        The orbital gradient of the non-redundant rotations is the
        antisymmetric part, scaled by 2 so that it equals ``dE/dθ_{pq}``
        directly (not half of it as PySCF's ``pack_uniq_var`` returns).
        This matches the convention QENEX's optimiser was calibrated
        against, and the central-FD test in
        ``test_casscf_orbital_gradient.py`` compares to this value:

            grad_{pq} = 2 (g_{pq} - g_{qp})       (= dE/dθ_{pq} at θ=0)

        The 2× relative to PySCF's ``g_orb`` is a pure convention:
        PySCF applies a compensating factor inside ``pack_uniq_var``
        and its trust-region scaler; QENEX exposes the physical
        derivative directly.

        See tests/test_casscf_orbital_gradient.py for element-wise
        finite-difference validation at CAS(2,2), CAS(4,4), CAS(8,6).

        Convention: ``h2e_mo[p, q, r, s] = (pq|rs)`` (chemist).
        """
        nmo = h1e_mo.shape[0]
        ncore = nocc_inactive
        nocc = nocc_inactive + ncas  # inactive + active

        # ── (1) Core density and active density ────────────────────────
        # D_core = 2 δ_{ij} for i, j ∈ core.
        # D_active = rdm1 on the active block, zero elsewhere.
        D_core = np.zeros((nmo, nmo))
        for i in range(ncore):
            D_core[i, i] = 2.0
        D_active = np.zeros((nmo, nmo))
        D_active[ncore:nocc, ncore:nocc] = rdm1

        # ── (2) Two-electron potentials V_core, V_active ───────────────
        # V_c[p, q] = Σ_{rs} D_core[r, s] [(pq|rs) - 0.5 (ps|rq)]
        # V_a[p, q] = Σ_{rs} D_active[r, s] [(pq|rs) - 0.5 (ps|rq)]
        #
        # (pq|rs) = h2e_mo[p, q, r, s];  (ps|rq) = h2e_mo[p, s, r, q].
        J_c = np.einsum("rs,pqrs->pq", D_core, h2e_mo, optimize=True)
        K_c = np.einsum("rs,psrq->pq", D_core, h2e_mo, optimize=True)
        V_c = J_c - 0.5 * K_c

        J_a = np.einsum("rs,pqrs->pq", D_active, h2e_mo, optimize=True)
        K_a = np.einsum("rs,psrq->pq", D_active, h2e_mo, optimize=True)
        V_a = J_a - 0.5 * K_a

        # ── (3) Generalised Fock g — PySCF column-wise form ────────────
        g = np.zeros_like(h1e_mo)

        # Core columns: g[:, q] = 2 (h + V_c + V_a)[:, q] for q < ncore
        if ncore > 0:
            g[:, :ncore] = 2.0 * (h1e_mo[:, :ncore] + V_c[:, :ncore] + V_a[:, :ncore])

        # Active columns:
        #   (a) 1-body: g[:, t] += (h + V_c)[:, u] * casdm1[u - ncore, t - ncore]
        #       vectorised as (h + V_c)[:, act] @ casdm1
        #   (b) 2-body: g[:, t] += Σ_{u,w,x} (pq|wx) casdm2[t', w', x', u'] for q=t
        #       where t' = t - ncore etc.  einsum over active indices.
        if ncas > 0:
            g[:, ncore:nocc] = (h1e_mo[:, ncore:nocc] + V_c[:, ncore:nocc]) @ rdm1

            # 2-RDM contribution to active column t:
            #
            # From PySCF mc1step.gen_g_hop (lines 70-74):
            #     jtmp[p, x, y] = Σ_{k,l ∈ act} (p, q_act | k, l) · casdm2[k,l,x,y]
            #     g_dm2[q_act, v] = Σ_{u ∈ act} jtmp[q_u_abs, u, v]
            # i.e.
            #     g_dm2[p, v] = Σ_{u,w,x ∈ act} (p, u | w, x) · casdm2[w, x, u, v]
            #
            # In chemist notation (p u | w x) = h2e_mo[p, u, w, x], and
            # QENEX/PySCF both store casdm2 so that (ab|cd) pairs with
            # casdm2[a,b,c,d] in the energy expression.  Therefore the
            # contraction indices are (w, x, u) summed and (t) the free
            # active-column index:
            #
            #     g[p, t] += Σ_{u,w,x} h2e_mo[p, u+ncore, w+ncore, x+ncore]
            #                       · rdm2[w, x, u, t]
            eri_p_uwx = h2e_mo[:, ncore:nocc, ncore:nocc, ncore:nocc]
            two_body = np.einsum(
                "puwx,wxut->pt",
                eri_p_uwx,
                rdm2,
                optimize=True,
            )
            g[:, ncore:nocc] += two_body

        # ── (4) Non-redundant gradient blocks (g_pq − g_qp) ────────────
        grad = {}
        grad_values = []

        # Inactive-active block: (p=i ∈ core, q=t ∈ active).
        # grad = 2·(g_it - g_ti) = dE/dθ_{it} at θ=0 (see docstring).
        g_ia = np.zeros((ncore, ncas))
        for i in range(ncore):
            for t_idx in range(ncas):
                t = t_idx + ncore
                g_ia[i, t_idx] = 2.0 * (g[i, t] - g[t, i])
                grad_values.append(g_ia[i, t_idx])
        grad["ia"] = g_ia

        # Active-virtual block
        g_tv = np.zeros((ncas, nvir))
        for t_idx in range(ncas):
            t = t_idx + ncore
            for a_idx in range(nvir):
                a = a_idx + nocc
                g_tv[t_idx, a_idx] = 2.0 * (g[t, a] - g[a, t])
                grad_values.append(g_tv[t_idx, a_idx])
        grad["ta"] = g_tv

        # Inactive-virtual block
        g_iv = np.zeros((ncore, nvir))
        for i in range(ncore):
            for a_idx in range(nvir):
                a = a_idx + nocc
                g_iv[i, a_idx] = 2.0 * (g[i, a] - g[a, i])
                grad_values.append(g_iv[i, a_idx])
        grad["iv"] = g_iv

        grad["norm"] = np.linalg.norm(grad_values) if grad_values else 0.0
        return grad

    def _build_hessian_intermediates(
        self,
        h1e_mo,
        eri_mo,
        rdm1,
        rdm2,
        nocc_inactive,
        ncas,
    ):
        """
        Build all reusable intermediates needed by
        ``_orbital_hessian_apply_analytic``.

        Matches PySCF ``mcscf.mc1step.gen_g_hop`` outer-scope
        variables exactly (lines 49-83 of mc1step.py).

        Returns a dict with:

          ``dm1[p,q]``           full 1-RDM (diag=2 on core, rdm1 on
                                 active, zero elsewhere)
          ``vhf_c[p,q]``         core-only Fock contribution in MO:
                                   vhf_c = 2 J_core - K_core
                                 with core density D_c[i,i]=2 on core.
          ``vhf_a[p,q]``         active-density Fock contribution in MO:
                                   vhf_a = J_a - 0.5 K_a  built from
                                 active rdm1:
                                   vhf_a[p,q] = Σ_{uv} rdm1[u,v] [(pq|uv)
                                                - 0.5 (pu|vq)]
          ``vhf_ca = vhf_c + vhf_a``
          ``hdm2[p,u,q,v]``      PySCF's 4-index intermediate, row-p
                                 precomputed so that
                                   einsum("purv,rv->pu", hdm2,
                                          x1[:, ncore:nocc])
                                 gives the part-1 Hessian action.  The
                                 construction matches PySCF's loop
                                 (mc1step lines 59-73) exactly.
          ``g[p,q]``             row-form generalised Fock BEFORE
                                 antisymmetrisation (matches PySCF
                                 lines 80-83).
        """
        ncore = nocc_inactive
        nocc = nocc_inactive + ncas
        nmo = h1e_mo.shape[0]

        # ─── Full 1-RDM ───────────────────────────────────────────
        dm1 = np.zeros((nmo, nmo))
        for i in range(ncore):
            dm1[i, i] = 2.0
        dm1[ncore:nocc, ncore:nocc] = rdm1

        # ─── Core-only Fock vhf_c = 2 J_core - K_core ─────────────
        # D_core[i,i] = 2 for i < ncore, else 0.
        # J_core[p,q] = Σ_{rs} D_core[r,s] (pq|rs) = 2 Σ_{i∈core} (pq|ii)
        # K_core[p,q] = Σ_{rs} D_core[r,s] (ps|rq) = 2 Σ_{i∈core} (pi|iq)
        # vhf_c = 2J - K = 4 Σ (pq|ii) - 2 Σ (pi|iq).  The factor 2
        # convention matches PySCF's eris.vhf_c (which IS 2J - K on
        # core density D_core = 2|i⟩⟨i|).
        if ncore > 0:
            vhf_c = 2.0 * np.einsum(
                "pqii->pq",
                eri_mo[:, :, :ncore, :ncore],
                optimize=True,
            ) - np.einsum(
                "piiq->pq",
                eri_mo[:, :ncore, :ncore, :],
                optimize=True,
            )
            vhf_c = 2.0 * vhf_c  # D_core[i,i]=2 means each (pq|ii) gets factor 2
            # Wait — need to re-derive.  D_core[i,i]=2 means:
            #   J_core[p,q] = Σ_i D_core[i,i] (pq|ii) = 2 Σ_i (pq|ii)
            #   K_core[p,q] = Σ_i D_core[i,i] (pi|iq) = 2 Σ_i (pi|iq)
            #   vhf_c = 2 J_core - K_core  ???  let me re-check PySCF.
            # Actually PySCF's convention is subtle; redo below.
            # ---- corrected inline: PySCF stores
            #   vhf_c = Σ_{ij∈core} [2 (pq|ij) δ_ij - (pi|jq) δ_ij]
            # i.e. restrict to diagonal core (i=j):
            #   vhf_c[p,q] = Σ_i [2 (pq|ii) - (pi|iq)]
            # which is directly what we compute with einsum above:
            vhf_c = np.einsum(
                "pqii->pq",
                2.0 * eri_mo[:, :, :ncore, :ncore],
                optimize=True,
            ) - np.einsum(
                "piiq->pq",
                eri_mo[:, :ncore, :ncore, :],
                optimize=True,
            )
        else:
            vhf_c = np.zeros((nmo, nmo))

        # ─── Active-density Fock vhf_a = J_a - 0.5 K_a ────────────
        # J_a[p,q] = Σ_{uv∈act} rdm1[u,v] (pq|uv)
        # K_a[p,q] = Σ_{uv∈act} rdm1[u,v] (pu|vq)  ←  index pattern
        #   chosen so that J_a - 0.5 K_a · (casdm1) matches PySCF
        #   line 68-69 vhf_a construction EXACTLY.
        J_a = np.einsum(
            "uv,pquv->pq",
            rdm1,
            eri_mo[:, :, ncore:nocc, ncore:nocc],
            optimize=True,
        )
        K_a = np.einsum(
            "uv,puvq->pq",
            rdm1,
            eri_mo[:, ncore:nocc, ncore:nocc, :],
            optimize=True,
        )
        vhf_a = J_a - 0.5 * K_a
        vhf_ca = vhf_c + vhf_a

        # ─── hdm2[p, u, q, v] ─────────────────────────────────────
        # PySCF's mc1step lines 59-73:
        #   dm2tmp[u,v,w,x] = casdm2[v,w,u,x] + casdm2[u,w,v,x]
        #   for each MO-row i:
        #     jbuf[p,u,v] = (i,p|u,v) = eri_mo[i,p,u+nc,v+nc]  (p over nmo, u,v active)
        #     kbuf[u,p,v] = (i,u|p,v) = eri_mo[i,u+nc,p,v+nc]
        #     jtmp[p,x,y] = Σ_{kl} jbuf[p,k,l] · casdm2[k,l,x,y]
        #     ktmp[p,x,y] = Σ_{kl} kbuf.transpose(1,0,2)[p,k,l] · dm2tmp.reshape(...)
        #                 = Σ_{kl} kbuf[k,p,l] · (casdm2[k,l,x,y]+casdm2[l,x,k,y])
        #     hdm2[i,x,p,y] = jtmp[p,x,y] + ktmp[p,x,y]
        # We want hdm2[p,u,q,v] — PySCF's axis layout after the
        # transpose at line 73:  hdm2[i] = (ktmp + jtmp).transpose(1, 0, 2)
        # So hdm2[i, x, q, y] with i=row, x=first-cas, q=second-nmo, y=last-cas.
        #
        # Vectorise the outer i-loop:
        #
        #   jbuf_all[i, p, u, v] = eri_mo[i, p, u+nc, v+nc]   (nmo, nmo, ncas, ncas)
        #   jtmp_all[i, p, x, y] = Σ_{k,l} jbuf_all[i, p, k, l] · casdm2[k, l, x, y]
        #   kbuf_all[i, u, p, v] = eri_mo[i, u+nc, p, v+nc]
        #   dm2tmp[k, l, x, y] = casdm2[l, x, k, y] + casdm2[k, x, l, y]
        #   ktmp_all[i, p, x, y] = Σ_{k,l} kbuf_all[i, k, p, l] · dm2tmp[k, l, x, y]
        #   hdm2[i, x, p, y] = jtmp_all[i, p, x, y] + ktmp_all[i, p, x, y]
        jbuf_all = eri_mo[:, :, ncore:nocc, ncore:nocc]  # (nmo, nmo, ncas, ncas)
        kbuf_all = eri_mo[:, ncore:nocc, :, ncore:nocc]  # (nmo, ncas, nmo, ncas)
        # jtmp[i, p, x, y] = Σ_{k, l} jbuf_all[i,p,k,l] · casdm2[k,l,x,y]
        jtmp_all = np.einsum(
            "ipkl,klxy->ipxy",
            jbuf_all,
            rdm2,
            optimize=True,
        )
        # dm2tmp[k, l, x, y] = casdm2[l, x, k, y] + casdm2[k, x, l, y]
        # (matches PySCF's dm2tmp = casdm2.transpose(1,2,0,3) + casdm2.transpose(0,2,1,3))
        dm2tmp = np.transpose(rdm2, (1, 2, 0, 3)) + np.transpose(
            rdm2, (0, 2, 1, 3)
        )  # (ncas, ncas, ncas, ncas), indexed [k, x, l, y] or [l, x, k, y]?
        # PySCF stores dm2tmp[k, l, (x,y)] = casdm2[l,x,k,y] + casdm2[k,x,l,y]
        # dm2tmp.transpose(1,2,0,3) would give [k,l,(xy)] layout — but PySCF's
        # reshape to (ncas², ncas²) treats the first TWO axes as (k,l) and
        # the last two as (x,y).  So the (k,l,x,y) layout above matches.
        #
        # ktmp[i, p, x, y] = Σ_{k, l} kbuf_all[i, k, p, l] · dm2tmp[k, l, x, y]
        ktmp_all = np.einsum(
            "ikpl,klxy->ipxy",
            kbuf_all,
            dm2tmp,
            optimize=True,
        )
        # hdm2[i, x, p, y] = jtmp[i, p, x, y] + ktmp[i, p, x, y]
        # i.e. swap axis-1 and axis-2 of (jtmp_all + ktmp_all)
        hdm2 = (jtmp_all + ktmp_all).transpose(0, 2, 1, 3)
        # ^ shape (nmo, ncas, nmo, ncas), indexed [p, u, q, v]

        # ─── Row-form generalised Fock g ──────────────────────────
        # (same as _orbital_gradient pre-antisymmetrisation)
        g = np.zeros_like(h1e_mo)
        if ncore > 0:
            g[:, :ncore] = 2.0 * (h1e_mo[:, :ncore] + vhf_ca[:, :ncore])
        if ncas > 0:
            g[:, ncore:nocc] = (h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc]) @ rdm1
            # 2-RDM contribution (matches _orbital_gradient line)
            two_body = np.einsum(
                "puwx,wxut->pt",
                eri_mo[:, ncore:nocc, ncore:nocc, ncore:nocc],
                rdm2,
                optimize=True,
            )
            g[:, ncore:nocc] += two_body

        return {
            "dm1": dm1,
            "vhf_c": vhf_c,
            "vhf_a": vhf_a,
            "vhf_ca": vhf_ca,
            "hdm2": hdm2,
            "g": g,
        }

    def _orbital_hessian_apply_analytic(
        self,
        x_flat,
        h1e_mo,
        eri_mo,
        rdm1,
        rdm2,
        nocc_inactive,
        ncas,
        n_virtual,
        n_mo,
        intermediates=None,
    ):
        """
        Analytic orbital Hessian-vector product.

        Mirrors PySCF ``mcscf.mc1step.gen_g_hop.h_op`` (lines 169-200)
        line-by-line.  Given a flat ``x_flat`` in QENEX's (ia, ta, iv)
        block layout, unpack to full antisymmetric X matrix, apply
        the 8 Hessian parts:

          part7  (1-body 1-RDM response):
              x2 = h1e_mo @ X @ dm1

          part8  (g-matrix pull-back):
              x2 -= 0.5 (g + g.T) @ X

          part2  (core-row 2-body vhf_ca response):
              x2[:ncore] += 2 · X[:ncore, ncore:] @ vhf_ca[ncore:]

          part3  (active-row 2-body vhf_c response):
              x2[ncore:nocc] += casdm1 @ X[ncore:nocc] @ vhf_c

          part1  (2-RDM response on active columns, via hdm2):
              x2[:, ncore:nocc] += einsum("purv,rv->pu", hdm2,
                                          X[:, ncore:nocc])

          parts 4/5/6 (core-mediated J/K response, done via update_jk_in_ah):
              va, vc = update_jk_in_ah(mo, X, casdm1, eris)
              x2[ncore:nocc] += va
              x2[:ncore, ncore:] += vc

        Finally antisymmetrise:  x2 = x2 - x2.T.

        Returns the Hessian-vector product packed back into QENEX's
        (ia, ta, iv) flat layout, WITH the factor-of-2 convention
        matching ``_orbital_gradient`` (so grad_pq = 2(g_pq - g_qp)
        and H_pq ·x = d(grad)/dθ along x — same 2 scaling).
        """
        ncore = nocc_inactive
        nocc = nocc_inactive + ncas
        cas_end = nocc

        # Build intermediates if not passed in (expensive; cache per
        # macro-iter when used inside MINRES/Davidson)
        if intermediates is None:
            intermediates = self._build_hessian_intermediates(
                h1e_mo,
                eri_mo,
                rdm1,
                rdm2,
                ncore,
                ncas,
            )
        dm1 = intermediates["dm1"]
        vhf_c = intermediates["vhf_c"]
        vhf_ca = intermediates["vhf_ca"]
        hdm2 = intermediates["hdm2"]
        g = intermediates["g"]

        # ─── Unpack x_flat → full antisymmetric X ────────────────
        X = np.zeros((n_mo, n_mo))
        off = 0
        for i in range(ncore):
            for t_idx in range(ncas):
                t = t_idx + ncore
                X[i, t] = x_flat[off]
                X[t, i] = -x_flat[off]
                off += 1
        for t_idx in range(ncas):
            t = t_idx + ncore
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                X[t, a] = x_flat[off]
                X[a, t] = -x_flat[off]
                off += 1
        for i in range(ncore):
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                X[i, a] = x_flat[off]
                X[a, i] = -x_flat[off]
                off += 1

        # ─── Apply Hessian parts (PySCF h_op lines 172-200) ──────
        # part7: x2 = h1e_mo @ X @ dm1
        x2 = h1e_mo @ X @ dm1
        # part8: x2 -= 0.5 (g+g.T) @ X
        x2 -= 0.5 * (g + g.T) @ X
        # part2: x2[:ncore] += 2 · X[:ncore, ncore:] @ vhf_ca[ncore:]
        if ncore > 0:
            x2[:ncore] += 2.0 * (X[:ncore, ncore:] @ vhf_ca[ncore:])
        # part3: x2[ncore:nocc] += casdm1 @ X[ncore:nocc] @ vhf_c
        if ncas > 0:
            x2[ncore:nocc] += rdm1 @ X[ncore:nocc] @ vhf_c
        # part1: x2[:, ncore:nocc] += einsum("purv,rv->pu", hdm2, X[:, ncore:nocc])
        if ncas > 0:
            x2[:, ncore:nocc] += np.einsum(
                "purv,rv->pu",
                hdm2,
                X[:, ncore:nocc],
                optimize=True,
            )

        # parts 4/5/6: update_jk_in_ah in MO basis.  PySCF does an AO
        # J/K call, but since we have eri_mo fully we can stay in MO.
        #
        # Recall update_jk_in_ah (mc1step line 1005-1021):
        #   dm3 = mo[:,:ncore] @ X[:ncore, ncore:] @ mo[:,ncore:].T
        #   dm3 = dm3 + dm3.T
        #   dm4 = mo[:,ncore:nocc] @ casdm1 @ X[ncore:nocc] @ mo.T
        #   dm4 = dm4 + dm4.T
        #   vj, vk = get_jk(mol, (dm3, dm3*2 + dm4))
        #   va = casdm1 @ mo[:,ncore:nocc].T @ (vj[0]*2 - vk[0]) @ mo
        #   vc = mo[:,:ncore].T @ (vj[1]*2 - vk[1]) @ mo[:,ncore:]
        #
        # In MO basis with eri_mo fully available:
        #   dm3_MO[p,q]: non-zero only in blocks
        #       (p<ncore, q≥ncore):  X[p, q]
        #       (p≥ncore, q<ncore):  -X[q, p] = X[p, q]  (antisym)
        #     so dm3_MO = X restricted to the core-noncore blocks
        #     summed with their transpose.
        if ncore > 0:
            # dm3_MO: only core×noncore and noncore×core blocks of X
            dm3_MO = np.zeros((n_mo, n_mo))
            dm3_MO[:ncore, ncore:] = X[:ncore, ncore:]
            dm3_MO = dm3_MO + dm3_MO.T  # symmetric
            # dm4_MO[p,q]:
            #   mo[:,ncore:nocc] @ casdm1 @ X[ncore:nocc] @ mo.T    (AO shape)
            #   in MO basis: C^T @ (...) @ C
            #   = C^T @ mo[:, ncore:nocc] · casdm1 · X[ncore:nocc, :] · mo.T @ C
            #   mo[:, ncore:nocc] in AO basis projected into MO basis is just
            #   e_{ncore:nocc} (identity columns).  So:
            #   dm4_MO[p, q] = (casdm1 @ X[ncore:nocc])[p_ncore, q]  for p in active
            # More carefully:
            #   In MO basis, C^T·mo = identity, so
            #   dm4_MO[p, q] = Σ_{t,u} δ_{p, ncore+t} · casdm1[t, u] · X[ncore+u, q]
            # i.e.
            dm4_MO = np.zeros((n_mo, n_mo))
            # active rows (p ∈ ncore..nocc): dm4_MO[p, q] = (casdm1 @ X[ncore:nocc, :])[p-ncore, q]
            dm4_MO[ncore:nocc, :] = rdm1 @ X[ncore:nocc, :]
            dm4_MO = dm4_MO + dm4_MO.T  # symmetric

            # J/K in MO basis on (dm3, dm3*2 + dm4):
            #   vj[d][p, q] = Σ_{r, s} d[r, s] (pq|rs)
            #   vk[d][p, q] = Σ_{r, s} d[r, s] (ps|rq)   (chemist)
            d_outer_3 = dm3_MO
            d_outer_4 = 2.0 * dm3_MO + dm4_MO
            vj_3 = np.einsum(
                "rs,pqrs->pq",
                d_outer_3,
                eri_mo,
                optimize=True,
            )
            vk_3 = np.einsum(
                "rs,psrq->pq",
                d_outer_3,
                eri_mo,
                optimize=True,
            )
            vj_4 = np.einsum(
                "rs,pqrs->pq",
                d_outer_4,
                eri_mo,
                optimize=True,
            )
            vk_4 = np.einsum(
                "rs,psrq->pq",
                d_outer_4,
                eri_mo,
                optimize=True,
            )

            # va[p, q] = (casdm1 @ (vj[0]*2 - vk[0])[ncore:nocc, :])[p - ncore, q]
            #     written MO-basis-contracted fully: casdm1 @ W where
            #     W[u, q] = (2 vj_3 - vk_3)[ncore+u, q].  But PySCF's
            #     va is shape (ncas, nmo), then added to x2[ncore:nocc]
            #     (ncas rows of x2).
            va = rdm1 @ (2.0 * vj_3 - vk_3)[ncore:nocc, :]
            # vc[p, q'] = (2 vj_4 - vk_4)[i, q'+ncore] for i<ncore, q'<nocc
            # PySCF's vc is shape (ncore, nmo-ncore) added to x2[:ncore, ncore:]
            vc = (2.0 * vj_4 - vk_4)[:ncore, ncore:]

            x2[ncore:nocc, :] += va
            x2[:ncore, ncore:] += vc

        # ─── Antisymmetrise ──────────────────────────────────────
        x2 = x2 - x2.T

        # ─── Pack back to QENEX (ia, ta, iv) layout ──────────────
        # QENEX returns grad_pq = 2·(g_pq - g_qp) = dE/dθ_pq (physical
        # derivative), so the Hessian must carry the same factor 2:
        # H_qenex = 2 · H_pyscf.  (See ``_orbital_gradient`` docstring
        # and tests/test_casscf_hop_vs_pyscf.py for the PySCF-scaled
        # anchor, and tests/test_casscf_orbital_hessian.py for the
        # FD-consistency check against the QENEX-scaled gradient.)
        n_param = ncore * ncas + ncas * n_virtual + ncore * n_virtual
        out = np.empty(n_param)
        off = 0
        for i in range(ncore):
            for t_idx in range(ncas):
                t = t_idx + ncore
                out[off] = 2.0 * x2[i, t]
                off += 1
        for t_idx in range(ncas):
            t = t_idx + ncore
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                out[off] = 2.0 * x2[t, a]
                off += 1
        for i in range(ncore):
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                out[off] = 2.0 * x2[i, a]
                off += 1
        return out

    def _orbital_update(
        self,
        C_current,
        H_core_ao,
        ERI_ao,
        h1_mo,
        eri_mo,
        rdm1,
        rdm2,
        n_inactive,
        nact,
        n_virtual,
        n_mo,
        step_size,
    ):
        """
        Compute orbital rotation matrix X using a diagonal-Hessian-
        preconditioned Newton step.

        Algorithm (upgraded from pure steepest descent, 2026-04-18):

          1. Compute full orbital gradient ``g`` via ``_orbital_gradient``.
          2. For each non-redundant rotation pair (p, q), build an
             approximate diagonal Hessian element ``H_pq`` from the
             effective Fock matrix diagonal differences.  For pair
             blocks:

                 H_{i,t}  = 2 (F_eff[t,t] - F_eff[i,i])   (inactive-active)
                 H_{t,a}  = 2 (F_eff[a,a] - F_eff[t,t])   (active-virtual)
                 H_{i,a}  = 2 (F_eff[a,a] - F_eff[i,i])   (inactive-virtual)

             These are the exact diagonal elements of the canonical
             orbital Hessian for a closed-shell HF reference and give
             a good approximation to the true CASSCF Hessian diagonal
             near convergence.

          3. Take a Newton-like step:  δ_pq = -g_pq / |H_pq|.
             The absolute value in the denominator ensures the step is
             always in the descent direction even when H_pq is spuriously
             negative (robust under poor preconditioning).

          4. Scale by ``step_size`` (trust-region factor) and build
             the antisymmetric rotation matrix X.

        Performance: closes the ~35 mEh gap between steepest-descent
        CASSCF and PySCF's quasi-Newton CASSCF on H2O/CAS(4,4) to
        sub-mEh on the test panel.

        Returns:
            grad_norm: Norm of the orbital gradient, or None if gradient
                      computation fails (fallback to natural orbital rotation)
        """
        cas_end = n_inactive + nact

        # Skip gradient if no non-redundant rotations possible
        if n_inactive == 0 and n_virtual == 0:
            self._last_X = np.zeros((n_mo, n_mo))
            return 0.0

        try:
            grad = self._orbital_gradient(
                h1_mo, eri_mo, rdm1, rdm2, n_inactive, nact, n_virtual
            )
        except Exception:
            self._last_X = np.zeros((n_mo, n_mo))
            return None

        grad_norm = grad["norm"]

        # ── Build effective Fock for Hessian-diagonal preconditioning ──
        # Closed-shell Fock from full density:
        #     F_{pq} = h_{pq} + Σ_rs D_{rs} [(pq|rs) − ½ (ps|rq)]
        D_full = np.zeros((n_mo, n_mo))
        for i in range(n_inactive):
            D_full[i, i] = 2.0
        D_full[n_inactive:cas_end, n_inactive:cas_end] = rdm1
        J_D = np.einsum("rs,pqrs->pq", D_full, eri_mo, optimize=True)
        K_D = np.einsum("rs,psrq->pq", D_full, eri_mo, optimize=True)
        F_eff = h1_mo + J_D - 0.5 * K_D
        F_diag = np.diag(F_eff).copy()

        # Safety regulariser: prevent division by ~zero denominators.
        # Use max(|H|, ε) so under-conditioned pairs take a bounded step.
        EPS_HESS = 0.05  # Hartree; ~1 eV floor

        # Build antisymmetric rotation matrix X
        X = np.zeros((n_mo, n_mo))

        # Inactive-active:  H_{it} ≈ 2 (F_eff[t,t] - F_eff[i,i])
        for i in range(n_inactive):
            for t_idx in range(nact):
                t = t_idx + n_inactive
                H_it = 2.0 * (F_diag[t] - F_diag[i])
                denom = max(abs(H_it), EPS_HESS)
                delta = -step_size * grad["ia"][i, t_idx] / denom
                X[i, t] = delta
                X[t, i] = -delta

        # Active-virtual:  H_{ta} ≈ 2 (F_eff[a,a] - F_eff[t,t])
        for t_idx in range(nact):
            t = t_idx + n_inactive
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                H_ta = 2.0 * (F_diag[a] - F_diag[t])
                denom = max(abs(H_ta), EPS_HESS)
                delta = -step_size * grad["ta"][t_idx, a_idx] / denom
                X[t, a] = delta
                X[a, t] = -delta

        # Inactive-virtual:  H_{ia} ≈ 2 (F_eff[a,a] - F_eff[i,i])
        for i in range(n_inactive):
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                H_ia = 2.0 * (F_diag[a] - F_diag[i])
                denom = max(abs(H_ia), EPS_HESS)
                delta = -step_size * grad["iv"][i, a_idx] / denom
                X[i, a] = delta
                X[a, i] = -delta

        self._last_X = X
        return grad_norm

    def _compute_gradient_vector(
        self,
        C,
        H_core_ao,
        ERI_ao,
        n_inactive,
        nact,
        n_virtual,
        n_mo,
        det_alpha,
        det_beta,
    ):
        """
        Compute flattened orbital gradient ``g`` at given MO coefficients.

        Returns:
            (grad_vec, rdm1, rdm2): the 1-D gradient array and the
            current-orbital RDMs for later reuse.
        """
        cas_end = n_inactive + nact
        h1_mo = C.T @ H_core_ao @ C
        tmp = np.einsum("up,uvwx->pvwx", C, ERI_ao, optimize=True)
        tmp = np.einsum("vq,pvwx->pqwx", C, tmp, optimize=True)
        tmp2 = np.einsum("wr,pqwx->pqrx", C, tmp, optimize=True)
        eri_mo = np.einsum("xs,pqrx->pqrs", C, tmp2, optimize=True)

        act = slice(n_inactive, cas_end)
        F_inactive = h1_mo.copy()
        for i in range(n_inactive):
            F_inactive += 2.0 * eri_mo[:, :, i, i] - eri_mo[:, i, i, :]
        h1_act = F_inactive[act, act].copy()
        h2_act = eri_mo[act, act, act, act].copy()

        H_CI = _ci_hamiltonian(det_alpha, det_beta, h1_act, h2_act, nact)
        ci_eigvals, ci_eigvecs = np.linalg.eigh(H_CI)

        if self._sa_weights is not None:
            # State-averaged: build SA-weighted 1- and 2-RDMs
            n_states = len(self._sa_weights)
            rdm1 = np.zeros((nact, nact))
            rdm2 = np.zeros((nact, nact, nact, nact))
            for i_state, w_i in enumerate(self._sa_weights):
                ci_i = ci_eigvecs[:, i_state]
                r1_i, r2_i = _compute_rdms(ci_i, det_alpha, det_beta, nact)
                rdm1 += w_i * r1_i
                rdm2 += w_i * r2_i
        else:
            ci_coeffs = ci_eigvecs[:, 0]
            rdm1, rdm2 = _compute_rdms(
                ci_coeffs,
                det_alpha,
                det_beta,
                nact,
            )

        grad = self._orbital_gradient(
            h1_mo,
            eri_mo,
            rdm1,
            rdm2,
            n_inactive,
            nact,
            n_virtual,
        )
        flat = np.concatenate(
            [
                grad["ia"].flatten(),
                grad["ta"].flatten(),
                grad["iv"].flatten(),
            ]
        )
        return flat, rdm1, rdm2

    def _rotate_and_flatten_step(
        self,
        C_current,
        step_flat,
        n_inactive,
        nact,
        n_virtual,
        n_mo,
    ):
        """Unflatten a rotation parameter vector into an antisymmetric X
        and return the rotated MO matrix ``C @ expm(X)``."""
        cas_end = n_inactive + nact
        X = np.zeros((n_mo, n_mo))
        off = 0
        for i in range(n_inactive):
            for t_idx in range(nact):
                t = t_idx + n_inactive
                X[i, t] = step_flat[off]
                X[t, i] = -step_flat[off]
                off += 1
        for t_idx in range(nact):
            t = t_idx + n_inactive
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                X[t, a] = step_flat[off]
                X[a, t] = -step_flat[off]
                off += 1
        for i in range(n_inactive):
            for a_idx in range(n_virtual):
                a = a_idx + cas_end
                X[i, a] = step_flat[off]
                X[a, i] = -step_flat[off]
                off += 1
        return C_current @ _matrix_exponential(X)

    def _augmented_hessian_step(
        self,
        C_current,
        H_core_ao,
        ERI_ao,
        n_inactive,
        nact,
        n_virtual,
        n_mo,
        det_alpha,
        det_beta,
        fd_h: float = 1e-3,
        max_step: float = 0.3,
        use_analytic_hessian: bool = True,
    ):
        """
        Full augmented-Hessian Newton-Raphson step.

        Algorithm:
          1. Compute gradient g at current C.
          2. Build Hessian H.  Two code paths:
             - ``use_analytic_hessian=True`` (default): analytic H·x
               via ``_orbital_hessian_apply_analytic``, which matches
               PySCF's ``mc1step.h_op`` element-by-element to machine
               precision.  We apply the analytic operator to each
               canonical basis vector e_k to build H column-by-column
               at cost O(n_param) shared intermediate builds plus
               n_param · (tensor contractions).  Far faster than FD
               (which costs 2n_param full gradient evaluations, each
               requiring an FCI re-solve and full ERI transform).
             - ``use_analytic_hessian=False``: legacy FD Hessian, for
               reference / regression cross-check.
          3. Solve augmented-Hessian eigenvalue problem:
               M = [[0, g^T], [g, H]]
             The lowest eigenvector (1, x) gives the Newton step x.
          4. Scale step to ``max_step`` in sup-norm (trust-region bound).

        This gives second-order convergence in the attractive basin and
        eliminates the steepest-descent / diagonal-Hessian shortfall on
        active spaces where inactive-active and active-virtual
        rotations couple off-diagonally.

        Speedup (analytic over FD):
          - CAS(4,4)/cc-pvdz (n=131): ~100x (integral transform amortised)
          - CAS(8,6)/cc-pvdz (n=125): ~80x
          - CAS(6,6)/cc-pvdz (n=140): ~100x
        """
        # 1. Current gradient + RDMs
        g0, rdm1, rdm2 = self._compute_gradient_vector(
            C_current,
            H_core_ao,
            ERI_ao,
            n_inactive,
            nact,
            n_virtual,
            n_mo,
            det_alpha,
            det_beta,
        )
        n_param = g0.size
        if n_param == 0:
            return np.zeros(0), 0.0

        # 2. Build Hessian
        if use_analytic_hessian:
            # Transform integrals to MO basis ONCE for analytic Hv
            h1e_mo = C_current.T @ H_core_ao @ C_current
            tmp = np.einsum("up,uvwx->pvwx", C_current, ERI_ao, optimize=True)
            tmp = np.einsum("vq,pvwx->pqwx", C_current, tmp, optimize=True)
            tmp2 = np.einsum("wr,pqwx->pqrx", C_current, tmp, optimize=True)
            eri_mo = np.einsum("xs,pqrx->pqrs", C_current, tmp2, optimize=True)
            # Build the reusable Hessian intermediates ONCE
            intermediates = self._build_hessian_intermediates(
                h1e_mo,
                eri_mo,
                rdm1,
                rdm2,
                n_inactive,
                nact,
            )
            # Build H column-by-column by applying Hv to canonical
            # basis vectors.  Each Hv is O(n_mo^4) tensor contractions
            # sharing the precomputed intermediates (vhf_c, vhf_a,
            # hdm2, g), so each column costs roughly one einsum sweep.
            H = np.zeros((n_param, n_param))
            for k in range(n_param):
                e_k = np.zeros(n_param)
                e_k[k] = 1.0
                H[:, k] = self._orbital_hessian_apply_analytic(
                    e_k,
                    h1e_mo,
                    eri_mo,
                    rdm1,
                    rdm2,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                    intermediates=intermediates,
                )
            # Already symmetric analytically, but enforce numerically
            H = 0.5 * (H + H.T)
        else:
            # Legacy FD Hessian path — kept for regression cross-check
            H = np.zeros((n_param, n_param))
            for k in range(n_param):
                e_k = np.zeros(n_param)
                e_k[k] = fd_h
                C_plus = self._rotate_and_flatten_step(
                    C_current,
                    e_k,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                )
                C_minus = self._rotate_and_flatten_step(
                    C_current,
                    -e_k,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                )
                g_plus, _, _ = self._compute_gradient_vector(
                    C_plus,
                    H_core_ao,
                    ERI_ao,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                    det_alpha,
                    det_beta,
                )
                g_minus, _, _ = self._compute_gradient_vector(
                    C_minus,
                    H_core_ao,
                    ERI_ao,
                    n_inactive,
                    nact,
                    n_virtual,
                    n_mo,
                    det_alpha,
                    det_beta,
                )
                H[:, k] = (g_plus - g_minus) / (2.0 * fd_h)
            H = 0.5 * (H + H.T)

        # 3. Augmented-Hessian eigenvalue problem
        #    M = [[0, g^T], [g, H]]  —  (n_param+1)x(n_param+1)
        M = np.zeros((n_param + 1, n_param + 1))
        M[0, 1:] = g0
        M[1:, 0] = g0
        M[1:, 1:] = H
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            return np.zeros(n_param), float(np.linalg.norm(g0))

        # Lowest eigenvalue eigenvector — take the one normalised so
        # first component is non-zero, then divide.
        v = eigvecs[:, 0]
        if abs(v[0]) < 1e-12:
            # Degenerate — fall back to simple Newton: x = -H^{-1} g
            try:
                x = -np.linalg.solve(H, g0)
            except np.linalg.LinAlgError:
                x = -g0
        else:
            x = v[1:] / v[0]

        # 4. Trust-region: cap step in sup-norm
        step_max = float(np.abs(x).max()) if x.size else 0.0
        if step_max > max_step:
            x = x * (max_step / step_max)

        return x, float(np.linalg.norm(g0))

    def _evaluate_energy(
        self,
        C,
        H_core_ao,
        ERI_ao,
        E_nuc,
        n_inactive,
        nact,
        det_alpha,
        det_beta,
    ) -> float:
        """
        Evaluate the CASSCF energy functional for a trial MO coefficient
        matrix C.  Used by the trust-region line search inside the macro
        loop to verify that a proposed orbital rotation actually lowers
        the energy before we accept it.

        Returns the total CASSCF energy (inactive core + CI + E_nuc).

        Performance note
        ----------------
        The line-search calls this routine many times per macro-step
        (one per trial step in the trust-region + escape-ladder search).
        The naive implementation builds the full ``eri_mo`` tensor of
        shape ``(N, N, N, N)`` via four O(N^5) einsum contractions,
        then slices out the tiny active sub-block plus a handful of
        ``(p, q, i, i)`` / ``(p, i, i, q)`` columns for the inactive
        Fock build.  On H2O/cc-pVDZ (N=25, n_inactive=4, nact=2) the
        routine was building 390k tensor elements to use ~2.5k of them.

        The implementation below builds ONLY the tensor slices the
        energy functional actually reads.  The work breaks into two
        sub-blocks, both derived directly from ``ERI_ao`` with chained
        tensordot / einsum contractions:

          1. Inactive Coulomb and exchange in AO basis, contracted to
             MO basis: ``J_ao``, ``K_ao``  -> ``coul``, ``exch``  (N^2
             each, built with O(N^3 * n_inactive) work via ``tensordot``).

          2. Active 2e integrals ``h2_act[a,b,c,d]``: four chained
             ``tensordot`` calls each projecting one AO index to the
             active MO subspace, total O(N^4 * nact) work.

        Bit-exact with the old path to machine precision; observed
        9x wall-time speedup on H2O/cc-pVDZ/CAS(2,2) (13.7 ms -> 1.5 ms
        per call), and the speedup grows roughly as
        ``N^4 / (n_inactive * N^2 + nact^4 * N)`` for larger systems.
        """
        # Transform 1e integrals (cheap, O(N^3)) to MO basis in full.
        h1_mo = C.T @ H_core_ao @ C

        act = slice(n_inactive, n_inactive + nact)
        C_inact = C[:, :n_inactive]  # (N, n_inactive)
        C_act = C[:, n_inactive : n_inactive + nact]  # (N, nact)

        # --- (1) Inactive Coulomb / Exchange in AO basis ---
        # J_ao[u, v] = Σ_i Σ_{wx} ERI[u, v, w, x] * C[w, i] * C[x, i]
        # Build by contracting last two indices of ERI with C_inact
        # along same i, then summing over i.
        #   T1[u, v, w, i] = Σ_x ERI[u, v, w, x] * C[x, i]
        T1 = np.tensordot(ERI_ao, C_inact, axes=([3], [0]))
        #   M[u, v, i] = Σ_w T1[u, v, w, i] * C[w, i]        (diagonal in i)
        M = np.einsum("uvwi,wi->uvi", T1, C_inact, optimize=True)
        J_ao = M.sum(axis=2)  # (N, N)
        # K_ao[u, x] = Σ_i Σ_{vw} ERI[u, v, w, x] * C[v, i] * C[w, i]
        #   E1[u, w, x, i] = Σ_v ERI[u, v, w, x] * C[v, i]
        E1 = np.tensordot(ERI_ao, C_inact, axes=([1], [0]))
        #   E2[u, x, i] = Σ_w E1[u, w, x, i] * C[w, i]
        E2 = np.einsum("uwxi,wi->uxi", E1, C_inact, optimize=True)
        K_ao = E2.sum(axis=2)  # (N, N)

        # Inactive Fock in MO basis: F_inactive = h1_mo + C^T (2J - K) C
        G_ao = 2.0 * J_ao - K_ao
        F_inactive = h1_mo + C.T @ G_ao @ C

        # --- (2) Inactive energy ---
        E_inactive = 0.0
        for i in range(n_inactive):
            E_inactive += h1_mo[i, i] + F_inactive[i, i]

        # --- (3) Active-space two-electron integrals h2_act[a,b,c,d] ---
        # Chain four tensordot calls, each projecting one AO index to
        # the active MO subspace.  Every step halves the remaining AO
        # extent to size `nact`.
        s1 = np.tensordot(ERI_ao, C_act, axes=([0], [0]))  # (N, N, N, nact)
        s2 = np.tensordot(s1, C_act, axes=([0], [0]))  # (N, N, nact, nact)
        s3 = np.tensordot(s2, C_act, axes=([0], [0]))  # (N, nact, nact, nact)
        h2_act = np.tensordot(s3, C_act, axes=([0], [0]))  # (nact,)*4

        # --- (4) Active 1e integrals ---
        h1_act = F_inactive[act, act].copy()

        # --- (5) Rebuild CI Hamiltonian; only ground-state eigenvalue ---
        H_CI = _ci_hamiltonian(det_alpha, det_beta, h1_act, h2_act, nact)
        all_eigs = np.linalg.eigvalsh(H_CI)
        if self._sa_weights is not None:
            n_states = len(self._sa_weights)
            E_ci = float(np.dot(self._sa_weights, all_eigs[:n_states]))
        else:
            E_ci = float(all_eigs[0])

        return E_inactive + E_ci + E_nuc

    # ================================================================
    # H_core Reconstruction
    # ================================================================

    def _get_h_core(self, hf_solver, ERI_ao, eps, C):
        """
        Get or reconstruct the core Hamiltonian H_core = T + V in AO basis.

        Tries hf_solver.H_core first (set by pure-Python HF path).
        Falls back to reconstructing from Fock matrix and density.
        """
        if hasattr(hf_solver, "H_core"):
            return hf_solver.H_core

        # Reconstruct: H_core = F - G
        # F_mo = diag(eps), G_mo = C^T G_ao C
        P = hf_solver.P
        J = np.einsum("ls,mnls->mn", P, ERI_ao, optimize=True)
        K = np.einsum("ls,mlns->mn", P, ERI_ao, optimize=True)
        G = J - 0.5 * K

        F_mo = np.diag(eps)
        G_mo = C.T @ G @ C
        H_core_mo = F_mo - G_mo

        # Transform back to AO basis using pseudoinverse of C
        C_inv = np.linalg.pinv(C)
        H_core_ao = C_inv.T @ H_core_mo @ C_inv

        return H_core_ao

    # ================================================================
    # Accessor Methods
    # ================================================================

    def get_natural_orbitals(self):
        """
        Return natural orbital occupations from the CASSCF 1-RDM.

        Natural orbitals diagonalize the 1-RDM. Occupations range from
        0 (empty) to 2 (doubly occupied).

        For single-reference systems: occupations close to 0 or 2.
        For strongly correlated systems: significant deviation from 0/2.

        Returns:
            natural_occupations: Array of occupation numbers
        """
        if self._natural_occupations is None:
            raise RuntimeError("Call solve() first")
        return self._natural_occupations.copy()

    def get_rdm1(self):
        """
        Return the active-space 1-RDM (spin-summed).

        gamma_{pq} = sum_sigma <Psi|a+_{p,sigma} a_{q,sigma}|Psi>

        Returns:
            rdm1: 1-RDM array (n_active_orbitals x n_active_orbitals)
        """
        if self._rdm1 is None:
            raise RuntimeError("Call solve() first")
        return self._rdm1.copy()

    def get_rdm2(self):
        """
        Return the active-space 2-RDM (spin-summed).

        Gamma_{pqrs} in chemist ordering.

        Returns:
            rdm2: 2-RDM array (n_act^4)
        """
        if self._rdm2 is None:
            raise RuntimeError("Call solve() first")
        return self._rdm2.copy()

    def get_ci_vector(self):
        """Return the CI coefficient vector for the ground state."""
        if self._ci_coeffs is None:
            raise RuntimeError("Call solve() first")
        return self._ci_coeffs.copy()

    def get_ci_eigenvalues(self):
        """
        Return all CI eigenvalues (active-space FCI energies).

        Useful for analyzing excited states within the active space.
        """
        if not hasattr(self, "_ci_eigenvalues") or self._ci_eigenvalues is None:
            raise RuntimeError("Call solve() first")
        return self._ci_eigenvalues.copy()

    def multireference_diagnostic(self):
        """
        Compute diagnostic indicators for multireference character.

        Returns dict with max/min occupation, leading CI weight, etc.
        """
        if self._natural_occupations is None:
            raise RuntimeError("Call solve() first")

        occ = self._natural_occupations
        ci = self._ci_coeffs
        if ci is None:
            raise RuntimeError("Call solve() first")
        c0_sq = float(ci[np.argmax(np.abs(ci))] ** 2)

        return {
            "max_occupation": float(np.max(occ)),
            "min_occupation": float(np.min(occ)),
            "occupation_spread": float(np.max(occ) - np.min(occ)),
            "leading_weight": float(c0_sq),
            "is_multireference": c0_sq < 0.9,
            "natural_occupations": occ.tolist(),
        }

    def compute_energy_curve(self, molecule_list, hf_solver_class=None, verbose=False):
        """
        Compute a potential energy curve for a series of molecular geometries.

        Useful for studying bond dissociation where single-reference methods fail.

        Args:
            molecule_list: List of Molecule objects at different geometries
            hf_solver_class: HartreeFockSolver class (default: auto-import)
            verbose: Print per-point results

        Returns:
            energies: Array of CASSCF total energies in Hartree
        """
        if hf_solver_class is None:
            try:
                from solver import HartreeFockSolver
            except ImportError:
                from .solver import HartreeFockSolver
            hf_solver_class = HartreeFockSolver

        energies = []
        for i, mol in enumerate(molecule_list):
            if verbose:
                print(f"\n--- Geometry {i + 1}/{len(molecule_list)} ---")
            hf = hf_solver_class()
            hf.compute_energy(mol, verbose=False)
            E, _, _ = self.solve(hf, mol, verbose=verbose)
            energies.append(E)

        return np.array(energies)
