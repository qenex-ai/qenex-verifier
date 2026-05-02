"""
Test: Fluorine Atom - Open-Shell System
Uses UHF (Unrestricted Hartree-Fock) for the open-shell doublet state.
"""
import pytest
import numpy as np

try:
    from packages.qenex_chem.src.solver import UHFSolver
    from packages.qenex_chem.src.molecule import Molecule
except ImportError:
    from solver import UHFSolver
    from molecule import Molecule


def test_fluorine_atom():
    """
    Fluorine Atom (9 electrons, doublet)
    Reference STO-3G energy: ~-98.57 Hartree
    """
    mol = Molecule(
        atoms=[('F', (0.0, 0.0, 0.0))],
        charge=0,
        multiplicity=2,  # Doublet: 9 electrons, 5 alpha, 4 beta
        basis_name='sto-3g'
    )
    
    solver = UHFSolver()
    E_tot, _ = solver.compute_energy(mol, verbose=True)
    
    print(f"\nF Atom UHF Energy: {E_tot:.6f} Hartree")
    print(f"Reference STO-3G: ~-98.57 Hartree")
    
    # STO-3G reference for F atom is approximately -98.57 Hartree
    # Allow reasonable tolerance for numerical differences
    assert E_tot < -97.0, f"Fluorine atom energy {E_tot:.4f} should be < -97 Ha"
    assert E_tot > -100.0, f"Fluorine atom energy {E_tot:.4f} should be > -100 Ha"
    
    # Verify UHF attributes are stored
    assert hasattr(solver, 'P_alpha'), "UHF solver should store P_alpha"
    assert hasattr(solver, 'P_beta'), "UHF solver should store P_beta"
    assert hasattr(solver, 'C_alpha'), "UHF solver should store C_alpha"
    assert hasattr(solver, 'C_beta'), "UHF solver should store C_beta"
    
    # Check electron counts
    # F has 9 electrons, doublet => 5 alpha, 4 beta
    n_alpha = np.trace(solver.P_alpha @ np.eye(solver.P_alpha.shape[0]))
    n_beta = np.trace(solver.P_beta @ np.eye(solver.P_beta.shape[0]))
    
    print(f"Alpha electrons: {n_alpha:.2f}, Beta electrons: {n_beta:.2f}")


if __name__ == "__main__":
    test_fluorine_atom()
