"""Hydrogen Atom Validation Tests"""
import numpy as np
import integrals as ints
from solver import HartreeFockSolver
from integrals import ContractedGaussian
from molecule import Molecule

# Define H atom with multiplicity 2 (doublet)
mol = Molecule([('H', (0.0, 0.0, 0.0))], multiplicity=2)

# Build Basis (manually or via solver)
solver = HartreeFockSolver()
basis = solver.build_basis(mol)
bf = basis[0] # H 1s

# Compute T and V
T = 0.0
V = 0.0
S = 0.0

for p1 in bf.primitives:
    for p2 in bf.primitives:
        S += ints.overlap(p1, p2)
        T += ints.kinetic(p1, p2)
        V += ints.nuclear_attraction(p1, p2, np.array([0.0, 0.0, 0.0]), 1.0)

print(f"H Atom STO-3G:")
print(f"Overlap: {S:.6f} (Should be 1.0)")
print(f"Kinetic: {T:.6f}")
print(f"Potential: {V:.6f}")
print(f"Total (H core): {T+V:.6f}")

# STO-3G H atom result:
# T = 0.7600
# V = -1.2266
# E = -0.4666
