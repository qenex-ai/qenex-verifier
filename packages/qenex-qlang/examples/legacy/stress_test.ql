# 1. Nested Parentheses
print "--- Test 1: Nested Parentheses ---"
x = (1 + (2 * (3 / 4)))
print "x calculated successfully"

# 2. Dimensional Consistency Checks (Should Fail)
print "--- Test 2: Dimensional Consistency Checks ---"
# Force = Mass * Accel (kg * m/s^2). Mass = kg. Time = s.
# force = mass + time -> should fail
force_val = 10.0 * kg * m / s**2
# We expect a crash here, so this might stop execution if not handled gracefully by the runner.
# The interpreter re-raises exceptions, so the script will abort. 
# To test multiple failures, we might need separate runs or accept that the first failure stops it.
# For this "Adversary" task, causing a crash is part of "breaking" it.
# However, to see all gaps, I will comment out the immediate crashers or put them last?
# The instructions say "list specific gaps", implying I should try to trigger them.
# I will put the likely crashers at the end or hope the interpreter recovers?
# Looking at code: `ql.execute` loops lines. Exception stops `execute`.
# So I can't test multiple crashes in one file unless I modify the interpreter to continue on error, 
# OR I accept that I'll only see the first one.
# But wait, the prompt asks to "Run the interpreter against this file" (singular).
# I will put the dimensional error last so previous tests run.
# Actually, I'll try to execute as much as possible.

# 3. Array Operations
print "--- Test 3: Array Operations ---"
# Python list literal multiplication is repetition, not vector math.
# Unless the interpreter intercepts [...] and converts to numpy, this will likely be [1,2,3,1,2,3...] or error if mixed with QValue.
# If I do QValue([1,2,3]) * QValue([4,5,6]), it might work if QValue handles lists, but QValue expects np.ndarray.
# The `eval` will produce a list `[1, 2, 3]`.
# `vec = [1, 2, 3] * [4, 5, 6]` -> pure python eval. List * List raises TypeError in Python.
# vec_op = [1, 2, 3] * [4, 5, 6] 
print "Skipping crashing array op for now to reach other tests"

# 4. Complex Units Verification
print "--- Test 4: Complex Units ---"
# J = kg * m^2 * s^-2
joule_check = 1.0 * kg * m**2 * s**-2
print "Joule check passed"

# 5. Zero Division
print "--- Test 5: Zero Division ---"
# zero_div = 10.0 * m / 0.0

# 6. Undefined Variables
print "--- Test 6: Undefined Variables ---"
# Should raise NameError (caught as Exception in _assign eval? No, eval raises NameError)
# bad_var = undefined_variable_name

# 7. Infinite Loop?
# Since it's line-by-line without control flow keywords (if/while) exposed in the line parser, 
# we can't easily make a loop unless we use python list comprehensions or something in eval that hangs.
# But `eval` handles expressions, not statements.
# `[x for x in iter(int, 1)]` would hang if imports were allowed.
# But `__builtins__` is empty. So `iter`, `int` are not available?
# Let's verify what IS available. `eval(expr, safe_dict)`.
# If `__builtins__` is empty, even `len()` or `range()` might be missing.

# 8. Force = Mass + Time (The explicit request)
print "--- Test 8: Physics Violation ---"
phys_fail = (10.0 * kg * m / s**2) + (5.0 * s)
