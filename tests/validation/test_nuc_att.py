
import numpy as np
import integrals as ints
from solver import HartreeFockSolver
from integrals import ContractedGaussian
from molecule import Molecule

# Check Nuclear Attraction
# (s|1/r|s) normalized
# Formula: 2 * sqrt(alpha/pi) * F0(...) ?
# For H atom (Z=1, R=0).
# <s|1/r|s> = 2 * alpha / pi ?? No.
# <phi|1/r|phi> = integral |phi|^2 / r.
# phi = (2a/pi)^0.75 e^-ar^2
# phi^2 = (2a/pi)^1.5 e^-2ar^2
# Integral 4pi r^2 dr * phi^2 * 1/r = 4pi Integral r e^-2ar^2
# = 4pi * [ -1/(4a) e^-2ar^2 ]_0^inf = 4pi * (1/4a) = pi/a
# (2a/pi)^1.5 * pi/a = 2^1.5 a^1.5 / pi^1.5 * pi / a = 2^1.5 a^0.5 / pi^0.5
# = 2 * sqrt(2a/pi)
# For a=1: 2 * sqrt(2/pi) = 2 * 0.7978 = 1.5957

bf = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (0,0,0))
# nuclear_attraction returns -Z * <...>. So pass Z=-1 to get positive integral?
# No, we want to check magnitude.
# Z=1.
v_val = ints.nuclear_attraction(bf, bf, np.array([0.0,0.0,0.0]), 1.0)
print(f"Nuclear Attraction (s|1/r|s) Z=1: {-v_val:.6f} (Ref: 1.595769)")

