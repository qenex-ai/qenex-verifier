# The Silent Vector Propagation Failure
# Demonstrating how Q-Lang silently discards uncertainty when multiplying two vectors.

# 1. Create a vector with uncertainty
# We do this by multiplying a clean vector by a scalar with uncertainty
# v_raw = [10.0, 20.0]
# s = 1.0 +/- 0.1 (10% uncertainty)
# v_unc = [10.0 +/- 1.0, 20.0 +/- 2.0]
define v_raw = [10.0, 20.0]
define s = 1.0 +/- 0.1
define v_unc = v_raw * s

print "Vector with Uncertainty:"
print v_unc

# 2. Multiply by another vector (Element-wise)
# v2 = [2.0, 2.0]
# Expected Result: [20.0 +/- 2.0, 40.0 +/- 4.0]
define v2 = [2.0, 2.0]
define result = v_unc * v2

print "--- Vector * Vector Result ---"
print "Expected Uncertainty: present (approx 2.0 and 4.0)"
print "Actual Result:"
print result

# If result shows "± 0.00e+00", we have data loss.
