# Q-Lang Discovery Demo 2.0: Unit-Correct Force Analysis

# 1. Simulate Ground State H2 (Equilibrium)
print "Simulating Ground State H2..."
simulate chemistry H 0,0,0 H 0.74,0,0 sto-3g
define E_ground = last_energy

# 2. Simulate Stretched H2 (Bond breaking)
print "Simulating Stretched H2..."
simulate chemistry H 0,0,0 H 1.5,0,0 sto-3g
define E_stretched = last_energy

# 3. Calculate Force
# E is now correctly in Joules (thanks to the interpreter patch)
# Distance change: 0.76 Angstroms -> meters
define dx = 0.76e-10 [m]

define Delta_E = E_stretched - E_ground
define Force = Delta_E / dx

print "Energy Difference (Joules):"
print Delta_E

print "Molecular Restoring Force (Newtons):"
print Force

# 4. Check Dimensional Consistency
# Force should be [M L T^-2] (Newtons)
# Q-Lang will auto-verify this on print or calculation
