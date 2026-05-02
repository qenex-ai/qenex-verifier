# Universal Science Audit
# Checking for subtle flaws across Physics, Chemistry, Biology, and Quantum Mechanics.

# --- 1. PHYSICS: Dimensional Integrity ---
# Claim: Energy / c^2 should equal Mass.
# E = 1.0e17 J (M L^2 T^-2) (Approx 1 kg * c^2)
# c = 2.9979e8 m/s (L T^-1)
# Result should be Mass [M^1]
print "--- Physics: Dimensional Integrity ---"
# define c = 2.99792458e8  <-- Uses built-in protected 'c'
define m_true = 1.0 +/- 0.01
define E = m_true * c^2
define m_derived = E / c^2

print "Original Mass:"
print m_true
print "Derived Mass (E/c^2):"
print m_derived
# Flaw check: Does m_derived retain the [Mass] dimension and uncertainty?

# --- 2. CHEMISTRY: Fractional Reaction Orders ---
# Rate Law: Rate = k * [A]^1.5
# [A] has 10% uncertainty.
# Expected Uncertainty in Rate: 1.5 * 10% = 15% (Linear approx)
print "\n--- Chemistry: Fractional Powers ---"
define conc_A = 2.0 +/- 0.2
define rate = conc_A ^ 1.5
print "Concentration [A]:"
print conc_A
print "Rate ([A]^1.5):"
print rate
# Flaw check: Does uncertainty propagate correctly for non-integer powers?

# --- 3. BIOLOGY: Dimensional Arguments in Exponentials ---
# N = N0 * exp(r * t)
# r = 0.5 [1/s]
# t = 2.0 [s]
# r*t is dimensionless. exp(r*t) should work.
# Flaw check: Does Q-Lang validate that the argument to exp() is dimensionless?
print "\n--- Biology: Dimensional Exponentials ---"
define r = 0.5
define t = 2.0
define exponent = r * t
define growth = exp(exponent)
print "Exponent (r*t):"
print exponent
print "Growth Factor:"
print growth

# --- 4. QUANTUM: Complex Probability ---
# psi = 1 + 1i
# Prob = |psi|^2 = (sqrt(1^2 + 1^2))^2 = 2
# We need to ensure we can take magnitude of complex QValues.
print "\n--- Quantum: Probability Amplitude ---"
define psi = sqrt(-1) + 1
print "Wavefunction (psi):"
print psi
# Magnitude check
define prob = abs(psi)^2
print "Probability (|psi|^2):"
print prob

# --- 5. VECTORS: Correlated Uncertainty ---
# v = [10.0, 20.0] +/- [1.0, 2.0] (10% relative error)
# v_sq = v * v = [100.0, 400.0]
# Correlated Error: 20% (Linear addition of relative errors)
#   -> [20.0, 80.0]
# Uncorrelated (Wrong): 14.14% (Quadrature)
#   -> [14.14, 56.56]
print "\n--- Vectors: Correlated Uncertainty ---"
define v = [10.0, 20.0] +/- [1.0, 2.0]
define v_sq = v * v
print "Vector v:"
print v
print "Vector v*v (Expect 20% error):"
print v_sq

