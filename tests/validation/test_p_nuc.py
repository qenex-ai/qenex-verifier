
import numpy as np
import integrals as ints

# Check (p|1/r|p)
bf = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (1,0,0)) # px
# Z=1
val = -ints.nuclear_attraction(bf, bf, np.array([0.0,0.0,0.0]), 1.0)

# Analytic Estimate
# N^2 * pi/6
# N = (128/pi^3)^0.25 = 1.425
# N^2 = 2.031
# Val = 2.031 * 3.14159 / 6 = 2.031 * 0.5236 = 1.063

print(f"Nuclear Attraction (p|1/r|p): {val:.6f} (Exp: ~1.063)")

# Check (p|1/r|s) ? Zero by symmetry if centered at origin.
bf_s = ints.BasisFunction([0.0,0.0,0.0], 1.0, 1.0, (0,0,0))
val_ps = -ints.nuclear_attraction(bf, bf_s, np.array([0.0,0.0,0.0]), 1.0)
print(f"Nuclear Attraction (p|1/r|s): {val_ps:.6f} (Exp: 0.0)")

# Check (p|1/r|p) where C is shifted
# C = (1,0,0)
val_shift = -ints.nuclear_attraction(bf, bf, np.array([1.0,0.0,0.0]), 1.0)
print(f"Nuclear Attraction (p|1/r|p) C=(1,0,0): {val_shift:.6f}")
