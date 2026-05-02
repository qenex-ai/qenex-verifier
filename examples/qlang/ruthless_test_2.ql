# Ruthless Cycle Test 2
# Focus: Non-linear Uncertainty Propagation & Trigonometry

# 1. Define an angle with significant uncertainty
# 0.785398 is approx pi/4 (45 degrees)
# Uncertainty is 0.1 radians
define theta = 0.7854 +/- 0.1

# 2. Calculate Sine
# Expected: sin(0.7854) ~= 0.7071
# Expected Uncertainty: |cos(theta)| * u_theta
# cos(0.7854) ~= 0.7071
# New Unc ~= 0.7071 * 0.1 = 0.0707
define s = sin(theta)

print "Theta:"
print theta
print "Sin(Theta) [Should have uncertainty ~0.07]:"
print s

# 3. Calculate Exponential Decay
# N = N0 * exp(-t/tau)
# t = 1.0 +/- 0.1
define t = 1.0 +/- 0.1
# exp(-t)
# Derivative of exp(-x) is -exp(-x)
# Unc = |-exp(-t)| * dt
# exp(-1) = 0.3678
# Unc = 0.3678 * 0.1 = 0.03678
define decay = exp(-1.0 * t)

print "Decay factor [Should have uncertainty ~0.036]:"
print decay

# 4. Check if uncertainty is ZERO (The Flaw)
# If the interpreter ignores it, the output will look like "0.7071 [Dimensionless]" without the ±
