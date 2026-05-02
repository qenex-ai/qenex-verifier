# Q-Lang Native Optimization Challenge
# Goal: Find equilibrium H2 bond length using native scipy optimizer

print "--- Q-Lang Native Optimization ---"

# 1. Define initial guess
define r = 2.0

# 2. Run Native Optimization
# This replaces the manual loop.
# Syntax: optimize geometry <Molecule Definition with $variables>
print "Starting Optimization..."
optimize geometry H 0,0,0 H $r,0,0 sto-3g

# 3. Output Results
print "--- Optimization Complete ---"
print "Final r (Bohr):"
print r

define r_angstrom = r * 0.529177
print "Final r (Angstroms):"
print r_angstrom
