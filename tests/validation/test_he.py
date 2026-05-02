
import numpy as np
from solver import HartreeFockSolver
from molecule import Molecule

mol = Molecule([('He', (0.0, 0.0, 0.0))])
solver = HartreeFockSolver()
E_elec, E_tot = solver.compute_energy(mol)
print(f"He Atom: E_tot={E_tot:.6f} (Ref: -2.807)")
