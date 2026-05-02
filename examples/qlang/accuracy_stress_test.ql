# Q-Lang Accuracy Stress Test
# Targeted search for invalid physical conclusions

print ">>> Test 1: Quantum Probability Logic"
# Hypothesis: Direct multiplication of complex vectors fails to compute probability density |psi|^2
# psi = [i, 1]. |psi|^2 should be [1, 1]. Total Prob = 2.
# If we do psi * psi, we might get [i*i, 1*1] = [-1, 1]. This is physically impossible for probability.

define psi_vec = [1j, 1.0]
print "Psi:"
print psi_vec

define density_naive = psi_vec * psi_vec
print "Naive Density (psi * psi):"
print density_naive

# We need a way to check if this is wrong autonomously.
# In Q-Lang, we can inspect the value.
# The correct operation should be abs(psi)^2 or conj(psi)*psi
define density_correct = abs(psi_vec)^2
print "Correct Density (|psi|^2):"
print density_correct


print "\n>>> Test 2: Gravitational Dimensions"
# F = G * m1 * m2 / r^2
# G = 6.67430e-11 [L^3 M^-1 T^-2]
# m = 100 kg
# r = 5 m
# Result should have dimensions of Force [M L T^-2] (Newtons)

define m1 = 100.0 * kg
define m2 = 100.0 * kg
define dist = 5.0 * m
# Note: G is built-in protected constant
define Force = G * m1 * m2 / dist^2

print "Calculated Force:"
print Force
# We will visually inspect dimensions in output: [M^1 L^1 T^-2]


print "\n>>> Test 3: Relativistic Singularity"
# gamma(c) should be infinite or raise error, not return a finite wrong number.
# define v_crit = 2.99792458e8 * m / s
# define g_crit = gamma(v_crit)
# print g_crit
# Commented out to prevent crash, but we should verify behavior. 
# Let's try 0.9999c
define v_near = 0.9999 * c
define g_near = gamma(v_near)
print "Gamma at 0.9999c:"
print g_near

print "\n>>> Test 4: Uncertainty in Trigonometry"
# theta = pi +/- 0.1
# sin(theta) ~ sin(pi) +/- |cos(pi)*0.1| = 0 +/- 0.1
define pi_val = 3.14159265359
define theta = 3.14159265359 +/- 0.1
define sin_val = sin(theta)
print "Theta:"
print theta
print "Sin(theta):"
print sin_val
