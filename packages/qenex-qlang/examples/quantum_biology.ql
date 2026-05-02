
# Quantum Biology Experiment
# We will simulate a simple protein sequence and verify its folded stability
# using an energy threshold check.

print "Initializing Quantum Biology Experiment..."

# Define a Hydrophobic-Polar sequence
# H = Hydrophobic (wants to be inside/compact)
# P = Polar (wants to be outside/indifferent)
define sequence = "HPPHHPHPHHHHPPHH"

# Step 1: Fold the protein using Monte Carlo
# This runs the ProteinFolder kernel in Python
simulate biology folding $sequence temperature=2.5 steps=5000

# Step 2: Analyze the result
# The interpreter stored the energy in $last_energy (Dimensionless for HP model)
print "Folding Energy: "
print $last_energy

# Step 3: Check stability
# For this sequence length (16), a good fold should have Energy <= -4
if $last_energy <= -4:
    print "✅ Stable Conformation Found (Native State Candidate)"
    
    # Let's verify if the energy gap is significant
    # We define a native energy target
    define native_target = -5
    
    if $last_energy <= $native_target:
        print "   -> Highly Stable (Deep Energy Minima)"
    else:
        print "   -> Moderately Stable (Meta-stable)"
    end
else:
    print "⚠️  Unstable Conformation (Molten Globule)"
    print "   -> Optimization failed to find native state."
end

print "Quantum Biology Experiment Complete."
