# Q-Lang Examples (verifier subset)

Two categories:

## `v04/` — canonical, runnable

Three example programs that use the canonical Q-Lang v0.4 syntax and run
cleanly on the verifier subset's `v04.cli_v04` interpreter. These are
the pedagogical reference for new Q-Lang programs.

```bash
PYTHONPATH=packages/qenex-qlang/src \
  python3 -m v04.cli_v04 run packages/qenex-qlang/examples/v04/03_h2_bond_energy.ql
# -> E(H2) in Hartree = -1.11676 [Hartree]
```

## `legacy/` — pre-v04, illustration only

Twenty Q-Lang programs that pre-date the v0.4 canonical interpreter.
They use the older Q-Lang 2.0 syntax (`define x = ...` rather than
`let x = ...`, `$last_energy` result-binding, and unregistered
simulation domains for biology / physics / general).

These do **not** run on the v04 interpreter shipped with the verifier
subset. They are kept as historical illustration of the language's
research-and-development trajectory; they are not paper-claim-relevant.
None of the v2 abstract numbers depend on them executing.

If a reviewer wants to run them, the lab installation contains the
older interpreter. The verifier subset ships only the canonical v04
implementation.
