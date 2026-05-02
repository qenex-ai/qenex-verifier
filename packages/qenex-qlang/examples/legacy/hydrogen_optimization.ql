# hydrogen_optimization.ql
# Optimizing Hydrogen Molecule (H2) using the new RHF solver.
# Equilibrium is expected around 1.4 Bohr (approx 0.74 Angstrom).
# Target Energy: -1.117 Hartree (STO-3G).

# Note: The solver expects atomic units (Bohr).

define r = 1.4

print "--- Initial State ---"
print "Bond Length r:"
print $r

print "--- Starting Optimization ---"
# optimize geometry <Type> <Coords> ...
# H at origin, H at x=r
optimize geometry H 0,0,0 H $r,0,0

print "--- Optimization Complete ---"
print "Final Bond Length:"
print $r
print "Final Energy:"
print $last_energy

# Verification
if $last_energy < -1.1*kg*m*m/s/s:
    print "✅ Energy matches STO-3G benchmark (~ -1.117 Eh)"
else:
    print "⚠️  Energy is incorrect."
