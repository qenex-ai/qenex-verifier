
import numpy as np
from solver import HartreeFockSolver
from molecule import Molecule

# Ne Atom (10 e)
mol = Molecule([('Ne', (0.0, 0.0, 0.0))])
solver = HartreeFockSolver()
# No nuclear repulsion
E_elec, E_tot = solver.compute_energy(mol)
print(f"Ne Atom: E_tot={E_tot:.6f} (Ref: ~ -127.8)")
