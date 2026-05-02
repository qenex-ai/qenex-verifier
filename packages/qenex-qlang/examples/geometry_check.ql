# Q-Lang Geometry Validation Demo
# This script proves that the solver is now geometry-sensitive for multi-atom systems.

# 1. Linear H3 Chain (Equally spaced)
# H -- 0.74 -- H -- 0.74 -- H
print "--- Simulating Linear H3 (Equally Spaced) ---"
simulate chemistry H 0,0,0 H 0.74,0,0 H 1.48,0,0 sto-3g
define E_linear = last_energy

# 2. Triangular H3 (Equilateral)
# H
# | \
# H--H
# All sides ~0.74
print "--- Simulating Triangular H3 (Equilateral) ---"
# Coordinates for equilateral triangle with side 0.74
# H1: 0, 0, 0
# H2: 0.74, 0, 0
# H3: 0.37, 0.6408, 0 (height = sqrt(0.74^2 - 0.37^2) = 0.6408)
simulate chemistry H 0,0,0 H 0.74,0,0 H 0.37,0.6408,0 sto-3g
define E_triangle = last_energy

# 3. Comparison
# If the solver was fake, these energies would be identical or close to H2 energy.
# If real, the Triangular H3 should likely be less stable (higher energy) or significantly different due to ring strain/frustration in s-orbitals.
print "Linear Energy:"
print E_linear
print "Triangular Energy:"
print E_triangle

define Diff = E_linear - E_triangle
print "Stability Difference:"
print Diff
