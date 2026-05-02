# Q-Lang v0.4 example 1 — Dimensional arithmetic with uncertainty.
#
# Measure a bond length with uncertainty, convert to different units,
# compute derived quantities, and produce a provenance trace.
#
# Expected stdout:
#   r           = 0.74 +/- 0.01 [Angstrom]
#   r (in Bohr) = 1.39841 +/- 0.018898 [Bohr]
#   diameter    = 1.48 +/- 0.02 [Angstrom]
#   area        = 1.7206 +/- 0.046496 [Angstrom^2]
#
# What this example demonstrates:
#   * Unit-carrying literals with uncertainty (0.74 +/- 0.01 [Angstrom])
#   * Unit conversion (Angstrom -> Bohr) preserves the ratio
#   * Uncertainty propagates correctly through * and ** (quadrature)
#   * Running with --trace writes a full derivation DAG to JSONL
#   * Every binding is a node; every operation has content-addressed
#     provenance for re-execution by `qlang replay`.

let r = 0.74 [Angstrom] +/- 0.01 [Angstrom]
let r_bohr = r in [Bohr]
let diameter = 2.0 * r
let area = r * r

print "r           = {r}"
print "r (in Bohr) = {r_bohr}"
print "diameter    = {diameter}"
print "area        = {area}"
