# Ruthless Cycle 3: Vectors & Quantum Complex Numbers

# 1. Vector Uncertainty Propagation
# We multiply a vector by a scalar that has uncertainty.
# The uncertainty should propagate to the vector components.
define vec = [10.0, 20.0, 30.0]
define scalar_u = 2.0 +/- 0.5
define res_vec = vec * scalar_u

print "Vector * Scalar(Unc) [Expected uncertainty: [5.0, 10.0, 15.0]]:"
print res_vec

# 2. Complex Numbers (Quantum Mechanics check)
# Q-Lang must handle imaginary numbers for Wavefunctions.
# Test: sqrt(-1)
print "Testing sqrt(-1)..."
define i = sqrt(-1)
print i

# Test: Euler's Identity approximation
# e^(i * pi) should be close to -1
define pi = 3.14159
define z = i * pi
define euler = exp(z)

print "Euler Identity (exp(i*pi)) [Expected ~ -1]:"
print euler
