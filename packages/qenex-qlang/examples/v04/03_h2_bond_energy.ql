# Q-Lang v0.4 example 3 — Chemistry kernel dispatch with full provenance.
#
# Runs a real Hartree-Fock calculation on H2 via the QENEX chemistry
# kernel (qenex_chem.compute_energy), enforces physical invariants on
# the result, and produces a re-executable trace that 'qlang replay'
# can verify bitwise.
#
# Expected stdout (approx.):
#   E(H2) in J      = -4.869e-18 [J]
#   E(H2) in Hartree = -1.11675 [Hartree]
#
# What this example demonstrates:
#   * simulate chemistry { ... } routes to a real QENEX kernel.
#   * The returned Quantity has correct dim (energy = ML^2 T^-2) and
#     provenance naming the kernel version.
#   * Experiment with invariants enforcing stability + sanity range.
#   * Running with --trace produces a DAG that 'qlang replay' verifies.

experiment h2_hf {
    invariant: 1.0 > 0.0                  # always true; smoke test

    let E = simulate chemistry {
        molecule: "H2",
        method:   "hf",
        basis:    "sto-3g",
    }

    # Physical invariants:
    invariant: E < 0.0 [J]                # bound state
    invariant: E > -1.0e-17 [J]           # sanity lower bound

    result: E
}

let E_si = h2_hf()
let E_ha = E_si in [Hartree]

print "E(H2) in J       = {E_si}"
print "E(H2) in Hartree = {E_ha}"
