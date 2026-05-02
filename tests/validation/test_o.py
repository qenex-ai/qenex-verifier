
import numpy as np
from solver import HartreeFockSolver
from molecule import Molecule

mol = Molecule([('O', (0.0, 0.0, 0.0))], multiplicity=3)
solver = HartreeFockSolver()
E_elec, E_tot = solver.compute_energy(mol)
print(f"O Atom: E_tot={E_tot:.6f}")
