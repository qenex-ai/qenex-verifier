
import numpy as np
import math
from scipy.special import erf, gamma, gammainc

# Import the module to test
import integrals

def reference_boys(n, t):
    if t < 1e-12:
        return 1.0 / (2*n + 1)
    # F_n(t) = 0.5 * t^(-n-0.5) * gamma(n+0.5) * gammainc(n+0.5, t)
    # Note: scipy gammainc is normalized by gamma(a), so we multiply by gamma(a)
    # gammainc(a, x) = (1/gamma(a)) * integral...
    # So scipy_gammainc * scipy_gamma = true incomplete gamma
    val = 0.5 * (t ** (-n - 0.5)) * (gamma(n + 0.5) * gammainc(n + 0.5, t))
    return val

def numerical_kinetic(alpha, beta, A, B, la, lb):
    # 1D kinetic integral <x^la e^-a(x-A)^2 | -0.5 d2/dx2 | x^lb e^-b(x-B)^2>
    
    def bra(x):
        return (x - A)**la * np.exp(-alpha * (x - A)**2)
        
    def ket(x):
        return (x - B)**lb * np.exp(-beta * (x - B)**2)
        
    def laplacian_ket(x):
        # d/dx ket
        # d/dx ( u^l e^-b u^2 ) = l u^{l-1} e - 2b u^{l+1} e
        u = x - B
        term1 = lb * u**(lb-1) if lb > 0 else 0
        term2 = -2 * beta * u**(lb+1)
        d1 = (term1 + term2) * np.exp(-beta * u**2)
        
        # d2/dx2 ket
        # d/dx (term1 e + term2 e)
        # term1' = lb(lb-1) u^{lb-2}
        t1_prime = lb * (lb-1) * u**(lb-2) if lb > 1 else 0
        
        # term2' = -2b(lb+1) u^{lb}
        t2_prime = -2 * beta * (lb+1) * u**lb
        
        poly_prime = t1_prime + t2_prime
        
        # also product rule: (term1 + term2) * (-2b u)
        term_prod = (term1 + term2) * (-2 * beta * u)
        
        return (poly_prime + term_prod) * np.exp(-beta * u**2)

    # Integration limits
    # effectively -inf to inf
    # centers A, B. width ~ 1/sqrt(alpha)
    bounds = min(A, B) - 5, max(A, B) + 5
    x = np.linspace(bounds[0], bounds[1], 10000)
    dx = x[1] - x[0]
    
    vals = bra(x) * (-0.5 * laplacian_ket(x))
    return np.sum(vals) * dx

print("--- Testing Boys Function ---")
t_vals = [0.1, 1.0, 10.0, 100.0]
for t in t_vals:
    ref = reference_boys(0, t)
    jit_val = integrals.boys_jit(0, t)
    print(f"F0({t}): Ref={ref:.6f}, JIT={jit_val:.6f}, Diff={abs(ref-jit_val):.2e}")
    
    ref = reference_boys(1, t)
    jit_val = integrals.boys_jit(1, t)
    print(f"F1({t}): Ref={ref:.6f}, JIT={jit_val:.6f}, Diff={abs(ref-jit_val):.2e}")

print("\n--- Testing Kinetic Jit ---")
alpha = 1.5
beta = 0.8
A = 0.0
B = 1.2
la = 0
lb = 0

# s-s
kin_num = numerical_kinetic(alpha, beta, A, B, la, lb)
# Need to account for 3D overlap in other dims (which are 1)
# kinetic_jit computes 3D.
# Let's mock 1D by setting overlaps in y, z to 1 (conceptually)
# But kinetic_jit adds terms.
# Let's just test 1D term logic by calling get_term_1d indirectly?
# No, easier to test full kinetic_jit for s-s in 3D aligned on X axis.
# In 3D:
# T = Tx Sy Sz + Sx Ty Sz + Sx Sy Tz
# If y, z are centered at 0 with l=0, Sy = S_1d_y.
# Let's match python arguments.
# A=(0,0,0), B=(1.2,0,0)
ra = np.array([0.0, 0.0, 0.0])
rb = np.array([1.2, 0.0, 0.0])

k_jit = integrals.kinetic_jit(0,0,0, 0,0,0, ra, rb, alpha, beta)

# Manual 3D numeric
# x: numeric
# y: overlap(0,0,0,0,a,b) = sqrt(pi/(a+b))
# z: same
norm_y = np.sqrt(np.pi/(alpha+beta))
norm_z = norm_y
# T_num = T_x_num * Sy * Sz + Sx * T_y_num * Sz + Sx * Sy * T_z_num
# T_y_num for 0,0,0,0 is kinetic of 1D gaussians on top of each other
k_y_num = numerical_kinetic(alpha, beta, 0.0, 0.0, 0, 0)
s_x_num = 0.439366 # calculated approximately? No, let's trust overlap_1d for S
s_x_jit = integrals.overlap_1d(0,0, 0.0, 1.2, alpha, beta)

total_k_num = kin_num * norm_y * norm_z + \
              s_x_jit * k_y_num * norm_z + \
              s_x_jit * norm_y * k_y_num

# Don't forget prefactor K_AB in numeric?
# My numerical_kinetic does full integral including exp(-a xA^2...) implicitly in bra/ket defs
# Wait, bra = (x-A)^l exp(-a(x-A)^2).
# This is the primitive.
# So numerical integration GIVES the full integral.
# kinetic_jit returns: pre * (TxSySz...)
# pre = exp(-ab/p R^2).
# Tx from get_term_1d excludes prefactor?
# overlap_1d excludes prefactor?
# Yes.
# So k_jit = pre * (TxSySz...).
# My numerical calculation calculates the actual integral.
# So they should match directly.

print(f"Kinetic 3D s-s: Num={total_k_num:.6f}, JIT={k_jit:.6f}")

# p-s
# la=1 (x), lb=0
kin_num_px = numerical_kinetic(alpha, beta, A, B, 1, 0)
# T = Tx(p,s) Sy Sz + Sx(p,s) Ty Sz + ...
# Sx(p,s) for x axis involves 1D overlap of p and s.
s_x_ps = integrals.overlap_1d(1, 0, 0.0, 1.2, alpha, beta)
total_k_px = kin_num_px * norm_y * norm_z + \
             s_x_ps * k_y_num * norm_z + \
             s_x_ps * norm_y * k_y_num

k_jit_px = integrals.kinetic_jit(1,0,0, 0,0,0, ra, rb, alpha, beta)
print(f"Kinetic 3D p-s: Num={total_k_px:.6f}, JIT={k_jit_px:.6f}")
