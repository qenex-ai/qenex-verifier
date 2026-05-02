
import numpy as np
import integrals as ints
from solver import HartreeFockSolver
from integrals import ContractedGaussian
from molecule import Molecule

# H atom s-s kinetic check
# Just create 1s orbital and print T
# Normalization: N
# Primitive: exp(-alpha r^2)
# alpha = 1.0
bf = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (0,0,0))
T = ints.kinetic(bf, bf)
print(f"Kinetic s (alpha=1): {T:.6f} (Ref: 1.5)")

bf_p = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (1,0,0))
T_p = ints.kinetic(bf_p, bf_p)
print(f"Kinetic p (alpha=1): {T_p:.6f} (Ref: 2.5)")
