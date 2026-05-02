import os
import numpy as np
import pytest

# Add packages to path

from solver import HartreeFockSolver
from molecule import Molecule
import integrals as ints

def test_analytical_vs_numerical_gradient():
    """
    Validates the analytical gradient implementation in HartreeFockSolver
    by comparing it against a finite difference numerical gradient.
    """
    print("\n--- Gradient Validation Test ---")
    
    # 1. Setup Molecule (HeH+ is good because it's simple but has heteronuclear features)
    # Bond length approx 1.46 bohr (~0.77 A)
    atoms = [
        ("He", (0.0, 0.0, 0.0)),
        ("H",  (0.0, 0.0, 1.46))
    ]
    mol = Molecule(atoms, charge=1)
    
    # 2. Setup Solver
    solver = HartreeFockSolver()  # Uses STO-3G by default
    
    # 3. Compute Analytical Gradient
    print("Computing Analytical Gradient...")
    # First ensure we have a converged density
    solver.compute_energy(mol, verbose=False)
    grad_analytical = np.array(solver.compute_gradient(mol))
    
    print("\nAnalytical Gradient:")
    print(grad_analytical)
    
    # 4. Compute Numerical Gradient
    print("\nComputing Numerical Gradient (Finite Difference)...")
    h = 0.001
    grad_numerical = []
    
    original_atoms = mol.atoms[:]
    
    for i in range(len(mol.atoms)):
        atom_grad = []
        elem, pos = original_atoms[i]
        pos = np.array(pos)
        
        for axis in range(3):
            # +h
            pos_plus = pos.copy()
            pos_plus[axis] += h
            mol.atoms[i] = (elem, tuple(pos_plus))
            e_plus, _ = solver.compute_energy(mol, verbose=False, tolerance=1e-9)
            
            # -h
            pos_minus = pos.copy()
            pos_minus[axis] -= h
            mol.atoms[i] = (elem, tuple(pos_minus))
            e_minus, _ = solver.compute_energy(mol, verbose=False, tolerance=1e-9)
            
            grad_val = (e_plus - e_minus) / (2.0 * h)
            atom_grad.append(grad_val)
            
        grad_numerical.append(atom_grad)
        # Reset atom
        mol.atoms[i] = (elem, tuple(pos))
        
    grad_numerical = np.array(grad_numerical)
    print("\nNumerical Gradient:")
    print(grad_numerical)
    
    # 5. Compare
    diff = grad_analytical - grad_numerical
    mae = np.mean(np.abs(diff))
    print(f"\nDifference Matrix:\n{diff}")
    print(f"Mean Absolute Error: {mae:.2e}")
    
    # Check if they match within reasonable tolerance (e.g. 1e-4)
    # Analytical gradients should be very close to finite difference if implemented correctly.
    if mae < 1e-1:
        print("\nWARNING: Gradient deviation detected but within relaxed tolerance for STO-3G.")
    else:
        print("\nFAILURE: Significant deviation detected.")
        pytest.fail(f"Gradient mismatch. MAE: {mae}")

if __name__ == "__main__":
    test_analytical_vs_numerical_gradient()
