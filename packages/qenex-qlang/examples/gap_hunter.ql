# --- Q-Lang Gap Hunter Protocol ---
# Target: Electron Correlation Energy Error (The "RHF Gap")
# Theory: At r=infinity, E(H2) must equal 2 * E(H) = -1.0 Hartree.
# RHF often incorrectly predicts ionic states (H+ H-) at limits.

# Define Unit for consistency
J_unit = 1.0 * kg * m^2 / s^2

print "--- INITIATING GAP SEARCH ---"
print "Target: Hydrogen Molecule Dissociation Limit"

# 1. Measure Baseline (Equilibrium)
# At 0.74 Angstroms, RHF is usually accurate.
r_eq = 0.74
print "1. Probing Equilibrium State (r=0.74 A)..."
# [FIX] Added '$' for variable interpolation
simulate chemistry H 0,0,0 H $r_eq,0,0 sto-3g
E_eq = last_energy

# 2. Measure The Limit (Dissociation)
# At 10.0 Angstroms, the bond should be broken.
r_limit = 10.0
print "2. Probing Dissociation Limit (r=10.0 A)..."

print "   [A] Standard RHF Method..."
simulate chemistry H 0,0,0 H $r_limit,0,0 sto-3g method=RHF
E_limit_RHF = last_energy

print "   [B] Advanced CI Method..."
simulate chemistry H 0,0,0 H $r_limit,0,0 sto-3g method=CI
E_limit_CI = last_energy

# 3. Reference Value (Exact Quantum Mechanics)
# 1 Hartree = 4.3597e-18 Joules
# Exact Electronic Energy for 2 H atoms = -1.0 Hartree
# Nuclear Repulsion at distance r must be added: +1/r
exact_electronic = -1.0
nuclear_repulsion = 1.0 / r_limit
exact_limit_hartrees = exact_electronic + nuclear_repulsion

J_per_Hartree = 4.3597447222071e-18

# [FIX] Apply units to the scalar so subtraction works
exact_limit_joules = exact_limit_hartrees * J_per_Hartree * J_unit

print "--- GAP ANALYSIS ---"
print "Exact Quantum Limit (J):"
print exact_limit_joules

print "RHF Dissociation Energy (J):"
print E_limit_RHF
gap_rhf = E_limit_RHF - exact_limit_joules
print "RHF Error Gap:"
print gap_rhf

print "CI Dissociation Energy (J):"
print E_limit_CI
gap_ci = E_limit_CI - exact_limit_joules
print "CI Error Gap:"
print gap_ci

# 5. Conclusion
# 1e-19 J is significant. We use a tight tolerance now.
# Note: RHF (Hückel) coincidentally gets this right due to lack of repulsion terms.
# CI gets it right by correctly treating correlation.
if gap_ci < 1e-20 * J_unit:
    print ">>> SUCCESS: Solver converges to exact dissociation limit."
    print ">>> Quantum Chemistry Accuracy Verified."
else:
    print ">>> WARNING: Gap still persists (Check Solver Implementation)."
