# Q-Lang Physics: Ising Model Phase Transition
# We will simulate a 2D Ferromagnetic Lattice to observe Spontaneous Symmetry Breaking.
# Theoretical Critical Temperature (Onsager): Tc ~ 2.269

print "--- QENEX Physics: Phase Transition Verification ---"

# 1. Low Temperature Phase (Ordered)
# At T=1.0 (well below Tc), spins should align (Ferromagnetism).
# Expect Magnetization -> 1.0
print "Phase 1: Low Temperature (T=1.0)"
simulate physics 20 1000 1.0

# Store the result
define M_low = last_magnetization
print "Magnetization (Low T):"
print M_low

# 2. High Temperature Phase (Disordered)
# At T=4.0 (well above Tc), thermal noise dominates (Paramagnetism).
# Expect Magnetization -> 0.0
print "Phase 2: High Temperature (T=4.0)"
simulate physics 20 1000 4.0

define M_high = last_magnetization
print "Magnetization (High T):"
print M_high

# 3. Validation Logic
if M_low > 0.8:
    if M_high < 0.2:
        print "✅ SUCCESS: Phase Transition Observed."
        print "   - Ordered Phase confirmed (M > 0.8)"
        print "   - Disordered Phase confirmed (M < 0.2)"
    else:
        print "❌ FAILURE: High T phase is too ordered."
else:
    print "❌ FAILURE: Low T phase is disordered."

end
