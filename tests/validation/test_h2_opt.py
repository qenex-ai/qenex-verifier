"""
Test: H2 Geometry Optimization
Validates that the geometry optimizer can find the equilibrium bond length for H2.
"""
import numpy as np
import pytest

from molecule import Molecule
from solver import HartreeFockSolver
from optimizer import GeometryOptimizer


def test_h2_optimization():
    """
    Test that H2 geometry optimization converges to expected bond length.
    STO-3G H2 equilibrium: ~1.34-1.39 Bohr (0.71-0.73 Angstrom)
    """
    print("\n--- Testing Geometry Optimization: H2 ---")
    
    # Start H2 slightly stretched (Equilibrium is ~0.74 A = ~1.4 Bohr)
    # Start at 1.8 Bohr (stretched) to see if it shrinks
    mol = Molecule([
        ('H', (0.0, 0.0, 0.0)),
        ('H', (0.0, 0.0, 1.8))
    ])
    
    solver = HartreeFockSolver()
    opt = GeometryOptimizer(solver)
    
    # Run Optimization
    optimized_mol, history = opt.optimize(mol, max_steps=10, tolerance=1e-3, learning_rate=0.5)
    
    # Verification
    final_pos = optimized_mol.atoms[1][1]
    final_dist = np.linalg.norm(np.array(final_pos) - np.array(optimized_mol.atoms[0][1]))
    
    print(f"Final Bond Length: {final_dist:.4f} Bohr")
    
    # STO-3G H2 Equilibrium is usually around 1.34 - 1.39 Bohr
    assert 1.30 < final_dist < 1.50, f"H2 bond length {final_dist:.4f} outside expected range (1.30-1.50)"
    print("PASS: H2 bond length converged to reasonable range.")


if __name__ == "__main__":
    test_h2_optimization()
