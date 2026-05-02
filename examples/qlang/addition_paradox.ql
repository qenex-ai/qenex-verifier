# The Doppelgänger Addition Paradox
# Demonstrating inconsistency between scalar multiplication and self-addition.

# Define a variable with uncertainty
# x = 10.0 +/- 1.0
define x = 10.0 +/- 1.0

# Method 1: Scalar Multiplication (Correct)
# 2 * x = 20.0 +/- 2.0
define mult_res = x * 2

# Method 2: Self-Addition (The Paradox)
# x + x should equal 2x
# Currently: sqrt(1^2 + 1^2) = 1.414 uncertainty
define add_res = x + x

# Method 3: Self-Subtraction (The Ghost)
# x - x should be exactly 0 (0 +/- 0)
# Currently: sqrt(1^2 + 1^2) = 1.414 uncertainty
define sub_res = x - x

print "--- Paradox Results ---"
print "Original:"
print x
print "x * 2 (Reference):"
print mult_res
print "x + x (Should match x*2):"
print add_res
print "x - x (Should be 0 +/- 0):"
print sub_res
