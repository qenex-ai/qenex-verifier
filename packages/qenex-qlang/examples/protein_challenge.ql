# Q-Lang Biology Challenge: 2D Protein Folding
# Task: Fold a Hydrophobic-Polar (HP) protein sequence
# H = Hydrophobic (loves other H)
# P = Polar (neutral)

print "--- QENEX Biology: 2D Protein Folding Challenge ---"

# 1. Define the HP Sequence
# Simple test case: H-H-P-H-P-H
# Expectation: H's cluster together to minimize energy
define sequence = "HHPHPH"

print "Sequence: "
print sequence

# 2. Simulate Folding
# Options: temperature (for simulated annealing), steps (max Monte Carlo steps)
print "Simulating folding process..."
simulate biology folding $sequence temperature=2.0

# 3. Analyze Results
# The result is stored in 'last_energy' (Dimensionless QValue)
print "Final Energy:"
print last_energy

# 4. Check Success
# Energy should be negative if H-H contacts were found
if last_energy < 0:
    print "✅ SUCCESS: Protein folded into a stable conformation."
else:
    print "❌ FAILURE: Protein did not find a stable fold (Energy >= 0)."
    print "   Consider increasing Monte Carlo steps or adjusting temperature."

end
