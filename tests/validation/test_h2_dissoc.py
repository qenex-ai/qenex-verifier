
import numpy as np
import integrals as ints
from solver import HartreeFockSolver
from integrals import ContractedGaussian
from molecule import Molecule

# Define 2 H atoms far apart (20 Bohr)
# Energy should be 2 * E(H) = 2 * (-0.4666) = -0.9332
# Nuclear repulsion ~ 0.
mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (20.0, 0.0, 0.0))])

# Run SCF
solver = HartreeFockSolver()
E_nuc = solver.compute_nuclear_repulsion(mol)
E_elec, E_tot = solver.compute_energy(mol)

print(f"H2 (R=20): E_tot={E_tot:.6f} (Ref: -0.93316)")
