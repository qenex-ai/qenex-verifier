# Ruthless Cycle Test 2 (Revised)
# Focus: Non-linear Uncertainty Propagation & Trigonometry

# 1. Define an angle with significant uncertainty
define theta = 0.7854 +/- 0.1

# 2. Calculate Sine (Using non-protected variable name)
define val_s = sin(theta)

print "Theta:"
print theta
print "Sin(Theta) [Should have uncertainty ~0.07]:"
print val_s

# 3. Calculate Exponential Decay
define t = 1.0 +/- 0.1
define decay = exp(-1.0 * t)

print "Decay factor [Should have uncertainty ~0.036]:"
print decay
