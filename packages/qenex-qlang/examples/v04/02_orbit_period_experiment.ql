# Q-Lang v0.4 example 2 — Experiment block with non-skippable invariants.
#
# Derive the orbital period from Kepler's third law for a circular
# orbit and enforce several sanity checks as STRUCTURAL invariants of
# the experiment.  If any invariant fails the experiment raises
# InvariantViolation and NO result is emitted — you cannot
# accidentally use a result computed from nonsensical inputs.
#
# Kepler III:  T^2 = (4 * pi^2 / (G * M)) * r^3
# For circular orbit at radius r around mass M.
#
# Expected stdout:
#   T(Earth @ Sun) = 1 year (to within rounding)
#
# What this example demonstrates:
#   * experiment NAME { given: ...  let ...  invariant: ...  result: ... }
#   * given: with declared unit (compile-time dim check of caller's arg)
#   * invariant: clauses that MUST pass before result: is emitted
#   * Passing an experiment with a failing invariant raises cleanly
#     (see example 02b below — this example uses valid inputs)

let G = 6.67430e-11 [m^3/kg/s^2]       # gravitational constant
let pi2 = 9.8696044                    # pi^2, scalar

experiment orbital_period {
    given: M in [kg]                   # central mass
           r in [m]                    # orbital radius

    # All three invariants must hold or no result is emitted.
    invariant: M > 0.0 [kg]
    invariant: r > 0.0 [m]
    invariant: r < 1.0e13 [m]          # inside solar system-ish

    let r_cubed = r * r * r
    let T_sq = (4.0 * pi2 / (G * M)) * r_cubed
    let T = sqrt(T_sq)

    result: T
}

# Solve for Earth: M_sun ~ 1.989e30 kg, r ~ 1.496e11 m (1 AU)
let T_earth = orbital_period(M: 1.989e30 [kg], r: 1.496e11 [m])
print "T(Earth @ Sun) = {T_earth}"
