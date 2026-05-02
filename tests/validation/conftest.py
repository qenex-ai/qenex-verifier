"""
QENEX Validation Test Configuration

Path setup is handled by the root conftest.py.
NOTE: np.seterr(all='raise') removed — DFT has legitimate underflows in
exp(-alpha*r^2) that trigger FloatingPointError. Individual tests that
need strict mode should use np.seterr locally.
"""
