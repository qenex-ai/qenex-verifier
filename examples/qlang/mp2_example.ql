# MP2 Correlation Energy Example
# Demonstrates Møller-Plesset 2nd Order Perturbation Theory
# which provides electron correlation beyond Hartree-Fock

# Run MP2 calculation on H2 at equilibrium (R=0.74 Angstrom)
print "=== H2 Molecule MP2 ==="
simulate chemistry H 0,0,0 H 0,0,0.74 method=mp2

# Access results
print "H2 MP2 Calculation Complete"
print "Correlation energy captures electron-electron avoidance"

# Compare with water molecule
print ""
print "=== Water Molecule MP2 ==="
simulate chemistry O 0,0,0 H 0,0.757,0.587 H 0,-0.757,0.587 method=mp2
