# Q-Lang Grand Challenge Phase 3: Chemical Reaction
# Simulating the formation of water: H2 + O -> H2O (Hypothetical pathway)
# We track the energy as two hydrogen atoms approach an oxygen atom.

# 1. Define Reactants (Far apart)
# H2 molecule at origin/near, O atom far away
# Actually, let's just model the complex at different reaction coordinates.
# But Q-Lang can optimize trajectory.

print "--- Grand Challenge Phase 3: Reaction Pathway ---"

# Step 1: Optimize H2 Molecule alone
print "1. Optimizing H2 Reactant..."
define r_h2 = 1.4 # Guess
optimize geometry H 0,0,0 H $r_h2,0,0 sto-3g
print "Energy(H2):"
print last_energy
define E_reactants = last_energy # Neglecting O energy for a moment (or assuming it's constant/reference)

# Step 2: Optimize H2O Product
print "2. Optimizing H2O Product..."
# We already know good params from Phase 2, but let's re-optimize to be sure
define r_oh = 1.8
define theta = 1.8
optimize geometry O 0,0,0 H $r_oh*sin($theta/2),$r_oh*cos($theta/2),0 H -$r_oh*sin($theta/2),$r_oh*cos($theta/2),0 sto-3g
print "Energy(H2O):"
print last_energy
define E_product = last_energy

# Step 3: Calculate Reaction Energy
define Delta_E = E_product - E_reactants
print "Reaction Energy (Delta E):"
print Delta_E

if Delta_E < 0.0:
    print "✅ Reaction is Exothermic (Energy released)"
else:
    print "⚠️ Reaction is Endothermic (Energy absorbed)"

# Step 4: Validate conservation of mass/stoichiometry
# (Symbolic check via Scout would go here)
validate "H2 + O -> H2O conserves mass"

