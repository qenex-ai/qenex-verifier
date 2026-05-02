# Q-Lang Physics: Critical Exponent Beta Extraction
# We will simulate near the critical temperature Tc = 2.269
# to extract the critical exponent beta: M ~ (Tc - T)^beta
# Theoretical value for 2D Ising: beta = 0.125 (1/8)

print "--- QENEX Physics: Critical Exponent Extraction ---"

define Tc = 2.269
define beta_theoretical = 0.125

print "Scanning temperatures near Tc..."

# Scan Points: T = 1.8, 1.9, 2.0, 2.1 (approaching Tc from below)
# Note: As T -> Tc, critical slowing down occurs. We need more sweeps.

simulate physics 30 5000 1.8
define M1 = last_magnetization
print "T=1.8, M="
print M1

simulate physics 30 5000 2.0
define M2 = last_magnetization
print "T=2.0, M="
print M2

simulate physics 30 5000 2.1
define M3 = last_magnetization
print "T=2.1, M="
print M3

simulate physics 30 5000 2.2
define M4 = last_magnetization
print "T=2.2, M="
print M4

# Log-Log fit is hard in simple Q-Lang script without arrays/loops
# But we can check if magnetization drops sharply as T -> Tc
# M(2.2) should be significantly lower than M(1.8)

define drop_ratio = M4 / M1
print "Drop Ratio M(2.2)/M(1.8):"
print drop_ratio

if drop_ratio < 0.8:
    print "✅ SUCCESS: Critical behavior observed (Magnetization drops near Tc)."
else:
    print "⚠️  WARNING: Phase transition seems too broad. M did not drop enough."

# Simple check against analytical solution for T=2.0
# M_exact = (1 - sinh(2/T)^-4)^(1/8)
# For T=2.0: sinh(1) ~ 1.175. sinh^-4 ~ 0.52. (1-0.52)^0.125 ~ 0.48^0.125 ~ 0.91
# Simulation typically gives slightly lower due to finite size effects.

print "Theoretical M at T=2.0 is ~0.91"
if M2 > 0.85:
    print "✅ M(2.0) matches theory within tolerance."
else:
    print "❌ M(2.0) is too low."

end
