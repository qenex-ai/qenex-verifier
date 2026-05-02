# Water Challenge: Optimize H2O Geometry
# Variables: Bond Length (r) and Bond Angle (theta)

# Initial Guesses
# r ~ 0.96 Angstroms (Use 1.0 as guess)
# theta ~ 104.5 degrees = 1.824 radians (Use 1.57 rad = 90 deg as guess)

define r = 1.8
define theta = 1.57

print "Starting Water Optimization..."
print "Initial r: $r"
print "Initial theta: $theta"

# Geometry Definition:
# O  : 0, 0, 0
# H1 : r, 0, 0
# H2 : r*cos(theta), r*sin(theta), 0

optimize geometry O 0,0,0 H $r,0,0 H $r*cos($theta),$r*sin($theta),0 sto-3g

print "Optimization Complete."
print "Final r: $r"
print "Final theta: $theta"

# Convert theta to degrees for readability
define theta_deg = $theta * 180.0 / 3.14159265359
print "Final Angle (degrees): $theta_deg"

# Expected: r ~ 1.8-1.9 Bohr (approx 0.96 A), Angle ~ 100-104 degrees
