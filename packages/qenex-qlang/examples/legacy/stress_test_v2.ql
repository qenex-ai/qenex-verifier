# Q-Lang Stress Test
# 1. Array Operations (New Feature)
# Should create [1,2,3] * 2 = [2,4,6]
define vec = [1, 2, 3]
define scaled_vec = vec * 2.0

# 2. Math Functions (New Feature)
# sqrt(16) = 4
define root_val = sqrt(16.0)
define sin_val = sin(0.0)

# 3. Robust Error Handling (Should not crash interpreter)
# Dimensional Mismatch
print "--- Triggering Physics Error (Should not crash) ---"
define bad_phys = 1.0*kg + 1.0*s

# 4. Complex Units
define Energy = 10.0 * kg * m^2 / s^2
print "Energy Check:"
print Energy

print "--- Stress Test Complete ---"
