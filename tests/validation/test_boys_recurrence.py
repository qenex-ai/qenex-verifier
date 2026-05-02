
import numpy as np
import math
from scipy.special import gammainc, gamma

def reference_boys(n, t):
    if t < 1e-12:
        return 1.0 / (2*n + 1)
    val = 0.5 * (t ** (-n - 0.5)) * (gamma(n + 0.5) * gammainc(n + 0.5, t))
    return val

def upward_recurrence(n_target, t):
    # F0 is accurate
    if t < 1e-8:
        return 1.0 / (2*n_target + 1)
        
    f_curr = 0.5 * np.sqrt(np.pi / t) * math.erf(np.sqrt(t))
    if n_target == 0:
        return f_curr
        
    exp_t = np.exp(-t)
    
    for n in range(n_target):
        # F_{n+1} = ( (2n+1)F_n - exp(-t) ) / (2t)
        f_next = ( (2*n + 1)*f_curr - exp_t ) / (2*t)
        f_curr = f_next
        
    return f_curr

print("Testing Upward Recurrence Stability:")
for t in [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]:
    print(f"t = {t}")
    for n in [1, 2, 3, 4, 10]:
        ref = reference_boys(n, t)
        up = upward_recurrence(n, t)
        err = abs(ref - up)
        print(f"  n={n}: Ref={ref:.6e}, Up={up:.6e}, Err={err:.6e}")
