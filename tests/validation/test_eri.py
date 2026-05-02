"""ERI (Electron Repulsion Integral) Validation Tests"""
import numpy as np
import integrals as ints
from integrals import ContractedGaussian
from solver import HartreeFockSolver
from molecule import Molecule

# Check ERI magnitude
# (ss|ss) for alpha=1.0
bf = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (0,0,0))
eri_val = ints.eri(bf, bf, bf, bf)
# Ref: 2 pi^2.5 / (2*2*sqrt(4)) = 2 * 17.49 / 8 = 4.37
# Formula: 2 pi^2.5 / ( (a+b)(c+d) sqrt(a+b+c+d) )
# a=b=c=d=1.
# p=2, q=2. p+q=4.
# Denom = 2 * 2 * 2 = 8.
# Num = 2 * pi^2.5 ~ 2 * 17.49 = 35.
# Val = 35/8 = 4.37
# With normalization:
# N = (2/pi)^0.75
# N^4 = (2/pi)^3 = 8 / pi^3
# Integral * N^4 = 4.37 * 8 / 31 = 1.128 ?
# Analytical (ss|ss) normalized = 2 * sqrt(alpha/pi) ? No.
# For a=1, (ss|ss) = 2/sqrt(pi) * 1 = 1.128379.

print(f"ERI (ss|ss) raw integral: {eri_val / (bf.N**4):.6f} (Ref: ~4.37)")
print(f"ERI (ss|ss) normalized: {eri_val:.6f} (Ref: 1.128379)")
