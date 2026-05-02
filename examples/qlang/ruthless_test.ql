# Ruthless Cycle Test
# 1. Classical Limit Violation (High Velocity)
# 2. Dimensional Mismatch
# 3. Floating Point Precision Loss

define c_val = 299792458
define v = 0.5 * c_val # 0.5c
define mass = 10.0 * kg

# Classical Kinetic Energy (Wrong at high V)
define E_classical = 0.5 * mass * v^2

# Relativistic Energy (Correct)
define gamma_factor = gamma(v)
define E_relativistic = (gamma_factor - 1) * mass * c_val^2

print "Classical E: "
print E_classical
print "Relativistic E: "
print E_relativistic

# Precision Test
define tiny_mass = 1.0e-30 * kg
define huge_mass = 1.0e30 * kg
define sum_mass = huge_mass + tiny_mass
# If precision is bad, sum_mass == huge_mass
print "Precision check (should show tiny difference):"
print sum_mass - huge_mass

# Dimensional Analysis
print "Testing Invalid Physics (should fail):"
define invalid = mass + (10 * m) 
