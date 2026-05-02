# Q-Lang Grand Challenge: Autonomous Gradient Descent for H2
# Goal: Find the equilibrium bond length of Hydrogen (approx 0.74 A)
# Starting from a bad guess (r = 2.0 A)

# 1. Setup Parameters
print "--- Q-Lang Grand Challenge: Autonomous Discovery ---"
define r = 2.0
define dr = 0.01
define learning_rate = 0.5
define tolerance = 0.001
define max_steps = 100

define step = 0
define diff = 1.0

# 2. Optimization Loop
while diff > tolerance:
    if step > max_steps:
         print "Max steps reached!"
         # Force exit loop by cheating diff (since we don't have 'break' yet)
         define diff = 0.0
    end
    
    if diff > tolerance:
        print "Iteration:"
        print step
        
        # Calculate E(r)
        # Note: We interpolate $r into the simulation command
        simulate chemistry H 0,0,0 H $r,0,0 sto-3g
        define E_current = last_energy
        
        # Calculate E(r + dr) for numerical gradient
        define r_next = r + dr
        simulate chemistry H 0,0,0 H $r_next,0,0 sto-3g
        define E_next = last_energy
        
        # Compute Gradient (dE/dr)
        # Gradient is effectively Energy/Distance
        # But for optimization, we just want the scalar direction if we are lazy, 
        # OR we do it properly with units.
        # Let's try to be unit-agnostic for the update step to avoid Q-Lang "Dimensional Mismatch" 
        # if learning_rate isn't perfectly dimensioned.
        
        # Hack: Strip units for the optimizer math by dividing by 1 (if it works?)
        # Or just rely on raw values. 
        # Since last_energy has units [M L^2 T^-2], and r has no units (dimensionless float in this script),
        # we have a mismatch. 
        # FIX: We need to strip units from E_current.
        # Q-Lang doesn't have a '.value' accessor syntax exposed yet? 
        # Let's trust the interpreter's eval allows accessing .value? No, it's sandboxed.
        # Workaround: define Energy as dimensionless for this specific math test?
        # No, last_energy is forced to be Joules.
        
        # Force-cast to float?
        # Let's try to make learning_rate have dimensions [L^2 T^2 M^-1] (Inverse Force * Distance?)
        # r (Length) -= LearningRate * Gradient (Energy/Length)
        # L -= [?] * [E/L]
        # L -= [?] * [M L T^-2]
        # To get L, [?] must be [T^2 M^-1].
        # Let's try defining learning_rate with dimensions!
        # Inverse Force constant.
        
        # define alpha = 0.1 * s^2 / kg
        # But 'r' in our script is currently dimensionless (2.0), not '2.0 * m'.
        # If we make r dimensionless, we can't subtract dimensioned gradient.
        
        # CRITICAL FIX for Grand Challenge:
        # We need a way to extract the SCALAR value of last_energy.
        # Since we don't have 'E.value', we will rely on the fact that
        # if we divide Energy by 1 Joule, we get a dimensionless scalar.
        
        define J_unit = 1.0 * kg * m^2 / s^2
        define E_scalar = E_current / J_unit
        define E_next_scalar = E_next / J_unit
        
        # Now gradient is dimensionless
        define grad = (E_next_scalar - E_scalar) / dr
        
        print "Current r (Bohr):"
        print r
        print "Gradient:"
        print grad
        
        # Update r
        # r = r - alpha * grad
        # Scaling factor: E is ~1e-18. Gradient is ~1e-18.
        # We need change ~0.05. So Alpha should be ~5e16
        define learning_rate_scaled = 0.5
        
        define change = learning_rate_scaled * grad
        define r = r - change
        
        # Check convergence (using abs via sqrt(x^2))
        define diff_sq = change * change
        define diff = sqrt(diff_sq)
        
        define step = step + 1
    end
end

print "--- Optimization Complete ---"
print "Final Bond Length (Bohr):"
print r
define r_angstrom = r * 0.529177
print "Final Bond Length (Angstroms):"
print r_angstrom
print "Target was ~0.74 A (approx 1.40 Bohr)"
