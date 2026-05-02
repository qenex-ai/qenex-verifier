
import numpy as np
import integrals as ints

alpha = 1.0
# P-x orbital
# coeff=1, norm=calculated
bf = ints.BasisFunction([0.0,0.0,0.0], alpha, 1.0, (1,0,0))

S = ints.overlap(bf, bf)
T = ints.kinetic(bf, bf)

print(f"P-orbital alpha={alpha}")
print(f"Overlap: {S:.6f}")
print(f"Kinetic: {T:.6f}")
print(f"Expected T (2.5*alpha): {2.5*alpha:.6f}")

# P-y orbital
bf_y = ints.BasisFunction([0.0,0.0,0.0], alpha, 1.0, (0,1,0))
T_y = ints.kinetic(bf_y, bf_y)
print(f"Kinetic Py: {T_y:.6f}")

# D-orbital (L=2) check just in case (xy)
# T for d-orbital?
# l=2. T = (2l+3)/2 * alpha ??
# For s (l=0): 1.5 alpha
# For p (l=1): 2.5 alpha
# For d (l=2): 3.5 alpha?
bf_d = ints.BasisFunction([0.0,0.0,0.0], alpha, 1.0, (1,1,0))
T_d = ints.kinetic(bf_d, bf_d)
print(f"Kinetic D(xy): {T_d:.6f}")
print(f"Expected T D (3.5*alpha): {3.5*alpha:.6f}")
