# Q-Lang Water Geometry Optimization
# Goal: Find the equilibrium H-O-H bond angle for Water.
# Target: ~104.5 degrees, ~0.96 Angstrom bond length.

print "--- Q-Lang Water Geometry Optimization ---"

# Initial Parameters
# Start with linear-ish or 90 deg structure to see if it bends correctly.
# Try 90 degrees (pi/2) approx.
# We optimize Cartesian coordinates directly because Q-Lang optimizer
# works best on independent variables in the geometry string.

# O at Origin
# H1 restricted to X axis (optimizes x1) -> This defines the first bond and removes rotation.
# H2 free in XY plane (optimizes x2, y2) -> This allows angle and second bond to change.

define x1 = 1.6   # Initial bond length guess (Bohr)
define x2 = 0.0   # H2 x-coord
define y2 = 1.6   # H2 y-coord (Start at 90 degrees)

print "Initial Guess:"
print "H1 (x):"
print x1
print "H2 (x,y):"
print x2
print y2

print "Starting Optimization..."

# Run BFGS Optimization
# Note: We use basis 'sto-3g' which is minimal but fast.
# O at (0,0,0) fixed.
optimize geometry O 0,0,0 H $x1,0,0 H $x2,$y2,0 sto-3g

print "--- Optimization Complete ---"
print "Final Geometry Parameters:"
print "x1 (Bond 1):"
print x1
print "x2 (H2 x):"
print x2
print "y2 (H2 y):"
print y2

# Result Analysis
# Bond 1 Length = x1
# Bond 2 Length = sqrt(x2^2 + y2^2)
# Angle Cos(theta) = (x1*x2) / (x1 * Bond2) = x2 / Bond2

print "Please calculate Angle = acos(x2 / sqrt(x2^2 + y2^2)) manually from output."
