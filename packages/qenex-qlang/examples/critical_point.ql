
# Critical Point Simulation
# Simulates the 2D Ising model near the critical temperature.
# We expect a sharp drop in magnetization around Tc ~ 2.269.

print "Starting Critical Point Analysis..."

# Parameters
L = 32
sweeps = 500

# Loop through temperatures from 2.0 to 2.5
# We use a manual loop because 'for' is not yet implemented in Q-Lang
T = 2.0
dT = 0.1
target_T = 2.6

# We will check T < target_T
# Note: floating point comparison requires care, but T < 2.6 should work for 2.5

while T < 2.6:
    simulate physics $L $sweeps $T
    
    # Check if we are in the disordered phase (Magnetization near 0)
    # Tc is approx 2.27. So above 2.3 we expect low M.
    if $T > 2.3:
        if $last_magnetization < 0.2:
             print "   -> Phase Transition Detected (Disordered Phase)"
        else:
             print "   -> Still Ordered (or finite size effects)"
        end
    else:
        print "   -> Ordered Phase"
    end

    # Increment T
    # We use Q-Lang's ability to evaluate math in definitions
    T = $T + $dT
    
end

print "Critical Point Analysis Complete."
