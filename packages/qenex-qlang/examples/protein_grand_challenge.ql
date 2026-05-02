# Q-Lang Biology: Grand Challenge
# Sequence: A 20-residue sequence designed to form a hydrophobic core.
# H = Hydrophobic, P = Polar
# Sequence: HPHPPHHPHPPHPHHPHPHH (20 residues)

print "--- QENEX Biology: Grand Challenge (20-mer) ---"

define sequence = "HPHPPHHPHPPHPHHPHPHH"
print "Sequence: "
print sequence

# Increase computational intensity for this harder problem
# steps=50000 ensures the Monte Carlo search explores enough of the landscape
# temperature=3.0 allows escaping local minima early on
print "Simulating deep folding (50k steps)..."
simulate biology folding $sequence steps=50000 temperature=3.0

print "Final Energy:"
print last_energy

print "Final Structure:"
# structure is auto-printed by the kernel, but let's confirm the energy check
if last_energy <= -5:
    print "✅ SUCCESS: Deep folding achieved (Energy <= -5)."
else:
    print "⚠️  WARNING: Fold might be suboptimal (Energy > -5)."

end
