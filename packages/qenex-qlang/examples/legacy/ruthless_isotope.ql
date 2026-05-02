# Q-Lang "Ruthless" Evaluation: H2/D2 Isotope Effect
# Task: 
# 1. Autonomous geometry optimization of H2
# 2. Calculation of Force Constant (k) via second derivative
# 3. Prediction of Vibrational Frequency for H2 and D2
# 4. Validation of Isotope Shift (should be ~1.414)

print "--- Starting Ruthless Isotope Evaluation ---"

# --- Constants ---
define pi = 3.14159265359
define Angstrom = 1e-10 * m
define amu = 1.66054e-27 * kg
define mass_H = 1.00784 * amu
define mass_D = 2.01410 * amu
define c_light = 2.9979e8 * m / s

# --- Optimization Setup ---
# We start close to the answer to focus on the precision of the frequency calculation
define r = 0.70
define dr = 0.005
define learning_rate_scaled = 2e-2
define tolerance = 0.0001
define max_steps = 15
define step = 0
define diff = 1.0

# J_unit used to strip units for scalar math inside the loop
define J_unit = 1.0 * kg * m^2 / s^2

print "Phase 1: Geometry Optimization"

while diff > tolerance:
    if step > max_steps:
        define diff = 0.0
        print "Warning: Max steps reached."
    end

    if diff > tolerance:
        # Energy at r
        simulate chemistry H 0,0,0 H $r,0,0 sto-3g
        define E_center = last_energy / J_unit
        
        # Energy at r + dr (Gradient probe)
        define r_probe = r + dr
        simulate chemistry H 0,0,0 H $r_probe,0,0 sto-3g
        define E_probe = last_energy / J_unit
        
        # Gradient = dE/dr
        define grad = (E_probe - E_center) / dr
        
        # Update
        define change = learning_rate_scaled * grad
        define r = r - change
        
        # Convergence check
        define diff_sq = change * change
        define diff = sqrt(diff_sq)
        
        define step = step + 1
        print step
        print r
    end
end

print "Optimization Converged."
print "Equilibrium Bond Length (Angstroms):"
print r

# --- Phase 2: Force Constant Calculation ---
print "Phase 2: Calculating Force Constant (k)"

# We need accurate 2nd derivative: E'' = (E(r+d) - 2E(r) + E(r-d)) / d^2
# We use a slightly larger delta for numerical stability of the 2nd derivative
define delta = 0.01

# Point A: r - delta
define r_minus = r - delta
simulate chemistry H 0,0,0 H $r_minus,0,0 sto-3g
define E_minus = last_energy

# Point B: r (Equilibrium)
simulate chemistry H 0,0,0 H $r,0,0 sto-3g
define E_eq = last_energy

# Point C: r + delta
define r_plus = r + delta
simulate chemistry H 0,0,0 H $r_plus,0,0 sto-3g
define E_plus = last_energy

# Calculate k (Force Constant)
# Note: E has units (Joules). delta is dimensionless float (Angstroms).
# We must convert delta to meters to get k in N/m (kg/s^2)
define delta_meters = delta * Angstrom

# Numerator: Joules
define num = E_plus - (2.0 * E_eq) + E_minus
# Denominator: meters^2
define den = delta_meters * delta_meters

define k_force = num / den

print "Force Constant k (N/m):"
print k_force

# --- Phase 3: Frequency Calculation ---
print "Phase 3: Isotope Frequency Analysis"

# Frequency nu = (1/2pi) * sqrt(k / reduced_mass)

# Reduced Mass for H2: (mH * mH) / (mH + mH) = mH / 2
define mu_H2 = mass_H / 2.0
define mu_D2 = mass_D / 2.0

# Calculate Nu H2
# sqrt(k/mu) -> sqrt( (kg/s^2) / kg ) -> sqrt(1/s^2) -> 1/s (Hertz)
define factor_H2 = k_force / mu_H2
define w_H2 = sqrt(factor_H2)
define nu_H2 = w_H2 / (2.0 * pi)

# Calculate Nu D2
define factor_D2 = k_force / mu_D2
define w_D2 = sqrt(factor_D2)
define nu_D2 = w_D2 / (2.0 * pi)

print "Frequency H2 (Hz):"
print nu_H2

print "Frequency D2 (Hz):"
print nu_D2

# --- Phase 4: Validation ---
print "Phase 4: Checking Isotope Effect"

# Theory predicts ratio should be sqrt(mu_D2 / mu_H2) = sqrt(2) approx
# FIX: Handle potential Infinity if nu_D2 is near zero/infinite due to optimization divergence
if nu_D2 > 0:
    define ratio = nu_H2 / nu_D2
else:
    define ratio = 0.0
end

define expected = sqrt(2.0)

print "Calculated Ratio (H2/D2):"
print ratio
print "Expected Ratio:"
print expected

# Define error metric
define error = ratio - expected
define abs_error_sq = error * error
# Ensure we take square root of the value part only if needed, or rely on QValue sqrt
define abs_error = sqrt(abs_error_sq)

print "Absolute Error:"
print abs_error

# Compare scalar values for logic
# If infinity, assume failure
if abs_error < 0.1:
    print "✅ SUCCESS: Isotope effect validated within tolerance."
else:
    print "❌ FAILURE: Isotope effect deviation too high."
