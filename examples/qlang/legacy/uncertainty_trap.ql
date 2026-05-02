# The Uncertainty Correlation Trap
# Demonstrating how Q-Lang artificially deflates uncertainty in self-multiplication

define x = 10.0 +/- 1.0
# Relative Uncertainty = 1.0 / 10.0 = 10%

# Method 1: Power Operator (Mathematically treated as single variable)
# Expected: 2 * 10% = 20% error -> 100 +/- 20
define via_pow = x ^ 2

# Method 2: Multiplication (Currently treats 'x' and 'x' as independent events)
# Current Flaw: sqrt(10%^2 + 10%^2) = ~14.1% error -> 100 +/- 14.14
define via_mul = x * x

print "--- Uncertainty Trap Results ---"
print "Original: "
print x
print "Via Power (Correct Reference):"
print via_pow
print "Via Mult (The Trap):"
print via_mul

# Check discrepancy
# If via_mul uncertainty < 19.9, the trap is active.
