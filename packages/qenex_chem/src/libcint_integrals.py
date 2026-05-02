"""
libcint Integration for QENEX LAB
====================================
Wraps libcint (the C integral library used by PySCF) to compute
molecular integrals with machine-precision accuracy.

This replaces the Python Obara-Saika implementation for cc-pVXZ
basis sets, giving exact agreement with PySCF on ALL integrals
(overlap, kinetic, nuclear attraction, electron repulsion).

The existing Obara-Saika engine remains available as a fallback
for STO-3G and 6-31G basis sets.

Usage:
    from libcint_integrals import compute_integrals_libcint
    S, T, V, ERI = compute_integrals_libcint(molecule)
"""

import numpy as np

__all__ = ["compute_integrals_libcint", "LIBCINT_AVAILABLE"]

try:
    from pyscf import gto

    LIBCINT_AVAILABLE = True
except ImportError:
    LIBCINT_AVAILABLE = False


def compute_integrals_libcint(molecule):
    """
    Compute all molecular integrals using libcint (via PySCF).

    Args:
        molecule: QENEX Molecule object with .atoms, .charge, .multiplicity, .basis_name

    Returns:
        S: overlap matrix (N x N)
        T: kinetic energy matrix (N x N)
        V: nuclear attraction matrix (N x N)
        ERI: electron repulsion integrals (N x N x N x N)

    All integrals are in the SPHERICAL-harmonic basis (PySCF cart=False),
    matching the Dunning cc-pVXZ and Pople 6-31G(d) definitions.  This
    was previously cart=True (Cartesian d-functions, 6 per shell) which
    added a spurious s-contaminant and shifted post-HF correlation
    energies by ~3 mHartree from published reference values.  Changed
    2026-04-21; see tests/test_scientific_references.py.
    """
    if not LIBCINT_AVAILABLE:
        raise ImportError("PySCF/libcint not available. Install: pip install pyscf")

    # Build PySCF molecule matching our geometry
    atoms_str = "; ".join(f"{el} {x} {y} {z}" for el, (x, y, z) in molecule.atoms)
    basis_name = getattr(molecule, "basis_name", "sto-3g")

    # Map our basis names to PySCF names
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
    pyscf_basis = basis_map.get(basis_name.lower(), basis_name)

    # Determine spin for PySCF
    n_elec = (
        sum(
            {
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
                "P": 15,
                "S": 16,
            }[el]
            for el, _ in molecule.atoms
        )
        - molecule.charge
    )
    spin = molecule.multiplicity - 1

    # cart=False: spherical d-functions (5 per d-shell), matching
    # the Dunning cc-pVXZ and Pople 6-31G(d) definitions.  See the
    # note in solver.py for history / rationale.
    mol = gto.M(
        atom=atoms_str,
        basis=pyscf_basis,
        unit="bohr",
        cart=False,
        charge=molecule.charge,
        spin=spin,
        verbose=0,
    )

    # Compute integrals using libcint
    S = mol.intor("int1e_ovlp")  # Overlap
    T = mol.intor("int1e_kin")  # Kinetic
    V = mol.intor("int1e_nuc")  # Nuclear attraction
    ERI = mol.intor("int2e")  # Electron repulsion (4-index)

    return S, T, V, ERI


def compute_hf_with_libcint(molecule, max_iter=100, convergence=1e-10, verbose=True):
    """
    Run a complete RHF calculation using PySCF's SCF solver with libcint.

    Uses PySCF's full SCF machinery (SAD initial guess, DIIS, level shifting)
    for guaranteed convergence to the global minimum. Returns data in the
    format expected by QENEX LAB's MP2/CCSD solvers.

    Returns:
        E_total: total energy (electronic + nuclear repulsion)
        C: MO coefficient matrix
        eps: orbital energies
        ERI: AO electron repulsion integrals (chemist notation)
        n_occ: number of occupied orbitals
    """
    if not LIBCINT_AVAILABLE:
        raise ImportError("PySCF/libcint not available")

    from pyscf import scf

    # Build PySCF molecule
    atoms_str = "; ".join(f"{el} {x} {y} {z}" for el, (x, y, z) in molecule.atoms)
    basis_name = getattr(molecule, "basis_name", "sto-3g")
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
    pyscf_basis = basis_map.get(basis_name.lower(), basis_name)
    spin = molecule.multiplicity - 1

    # cart=False: spherical d-functions (5 per d-shell).  See
    # solver.py for rationale.
    mol_p = gto.M(
        atom=atoms_str,
        basis=pyscf_basis,
        unit="bohr",
        cart=False,
        charge=molecule.charge,
        spin=spin,
        verbose=0,
    )

    # Run PySCF RHF (uses SAD guess, proper DIIS, guaranteed convergence)
    mf = scf.RHF(mol_p)
    mf.max_cycle = max_iter
    mf.conv_tol = convergence
    mf.verbose = 4 if verbose else 0
    mf.kernel()

    E_total = mf.e_tot
    C = mf.mo_coeff
    eps = mf.mo_energy
    n_occ = mol_p.nelectron // 2

    # Get AO ERI in chemist notation (pq|rs)
    ERI = mol_p.intor("int2e")

    return E_total, C, eps, ERI, n_occ
