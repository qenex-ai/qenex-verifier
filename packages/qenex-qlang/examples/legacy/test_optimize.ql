# Q-Lang Optimization Demo
# Replaces 100 lines of manual loop with the 'optimize' command

print "--- Q-Lang Auto-Optimizer Test ---"

# 1. Define initial guess
define r = 2.0

# 2. Run Optimization
# Syntax: optimize <var> minimize "<command>" using gradient_descent tolerance=1e-4
# Note: The command inside quotes uses $r which is substituted each step.
optimize r minimize "simulate chemistry H 0,0,0 H $r,0,0 sto-3g" using gradient_descent tolerance=0.0001

print "--- Final Result ---"
print "Equilibrium Bond Length:"
print r
print "Target: ~0.74 (depends on basis set)"
