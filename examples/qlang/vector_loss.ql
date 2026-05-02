# The Silent Vector Data Loss
# Demonstrating how Q-Lang silently discards uncertainty when using Vectors.

# Scenario: A velocity vector with precise components, multiplied by a time scalar with uncertainty.
# v = [10, 20] m/s
# t = 5.0 +/- 1.0 s
# Expected Position: [50 +/- 10, 100 +/- 20] m

define v = [10.0, 20.0]
define t = 5.0 +/- 1.0

define pos = v * t

print "--- Vector Uncertainty Loss Test ---"
print "Velocity:"
print v
print "Time (with uncertainty):"
print t
print "Position (Result):"
print pos

# If result shows "± 0.00e+00", the uncertainty was lost.
