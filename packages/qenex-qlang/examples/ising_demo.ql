# Q-Lang Physics Demo: Ising Model Phase Transition
# Goal: Find the Critical Temperature (Tc) where Magnetization drops to zero

print "--- Q-Lang Ising Model Simulation ---"
print "Searching for Critical Temperature (Tc) of 2D Ferromagnet..."

# 1. Setup Parameters
define size = 20
define sweeps = 200

# 2. Temperature Sweep (T from 1.0 to 4.0)
# We want to find where Magnetization M -> 0
# Theoretical Tc ~ 2.269

define T = 1.0
define step_T = 0.2
define max_T = 3.5

# We will store results manually by printing
print "Temp | Magnetization"

while T < max_T:
    # Syntax: simulate physics <size> <sweeps> <temp>
    simulate physics $size $sweeps $T
    
    define M = last_magnetization
    
    # We can't print inline variables yet, so we print on separate lines
    print "T:"
    print T
    print "M:"
    print M
    print "---"
    
    # If M drops below threshold, we found Tc
    if M < 0.15:
         print ">>> Phase Transition Detected!"
         print "Approximate Tc found near:"
         print T
         # Break loop by forcing T to max
         define T = 10.0
    end
    
    define T = T + step_T
end

print "--- Simulation Complete ---"
