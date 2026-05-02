# Supplementary Materials Manifest

This document is a one-page index for reviewers: every numeric claim in
the QENEX LAB v2 abstract is mapped to the **exact reproducer command**
that produces it, the **expected output**, and the **source files** the
command exercises.

## Hard map: claim → command → expected → source

### Computational methods

| Claim                                | Command                                                        | Expected | Source files                                                                 |
| ------------------------------------ | -------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| 6 distinct wired methods             | `method_inventory.py --json` → `headline_v2_claim.wired_count` | `6`      | `molecule.py:Molecule.compute()` (the dispatch tree)                         |
| 7 method names including `rhf` alias | same → `wired_count_raw_with_aliases`                          | `7`      | same                                                                         |
| 6 standalone solver classes          | same → `headline_v2_claim.standalone_count`                    | `6`      | `casscf.py`, `eomccsd.py`, `tddft.py`, `uccsd.py`, `dlpno_ccsd.py`, `qmc.py` |
| 12 user-callable methods             | same → `user_callable_total`                                   | `12`     | union of the two above                                                       |
| 13 method-implementation modules     | same → `module_count_upper_bound`                              | `13`     | walks `qenex_chem/src/*.py` for solver-named files                           |

### Precision

| Claim                                       | Command                                                                                                                                                 | Expected                      | Source                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| 6 basis sets                                | `precision_matrix.py --json` → `headline_v2.basis_sets_total`                                                                                           | `6`                           | `basis_*.py` files, `molecule.py:Molecule(basis_name=…)`             |
| Frozen PySCF references for sto-3g, cc-pvdz | same → `basis_sets_with_reference`                                                                                                                      | `["sto-3g", "cc-pvdz"]`       | `certification.py:PYSCF_RHF_*`                                       |
| 5 of 6 reference-validated tuples sub-nano  | same → `tuples_sub_nanohartree` of `tuples_with_reference`                                                                                              | `5` / `6`                     | full HF kernel (`solver.py`, `integrals.py`, `libcint_integrals.py`) |
| H2 HF/STO-3G energy = -1.1167593074 Ha      | direct: `python3 -c "from molecule import Molecule; m=Molecule([('H',(0,0,0)),('H',(0,0,1.3984))],basis_name='sto-3g'); E,_=m.compute('hf'); print(E)"` | sub-nanohartree drift         | same                                                                 |
| ULP noise <1e-12 Ha suppressed by rounding  | `precision_matrix.py` rounds to 11 decimals                                                                                                             | bit-identical SHA across runs | `scripts/precision_matrix.py:128-138`                                |

### Q-Lang scientific protocol

| Claim                                  | Command                                                                                                                 | Expected                                | Source                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| 16 canonical v04 modules               | `qlang_inventory.py --json` → `canonical_v04.modules`                                                                   | `16`                                    | `packages/qenex-qlang/src/v04/*.py`                                                       |
| 5,593 lines in canonical v04           | same → `canonical_v04.lines`                                                                                            | `5,593`                                 | same                                                                                      |
| 3 simulate domains registered          | same → `headline_v2.registered_domains_count`                                                                           | `3`                                     | `simulate_dispatch_v04.py:_register_defaults()`                                           |
| Domains: chemistry, chemistry_geom, md | same → `registered_domains_list`                                                                                        | `["chemistry","chemistry_geom","md"]`   | same                                                                                      |
| End-to-end Q-Lang execution            | `PYTHONPATH=packages/qenex-qlang/src python3 -m v04.cli_v04 run packages/qenex-qlang/examples/v04/03_h2_bond_energy.ql` | "E(H2) in Hartree = -1.11676 [Hartree]" | full v04 stack: lexer → parser → interp → simulate_dispatch → `qenex_chem.compute_energy` |

### Drug discovery (sovereignty boundary)

| Claim                       | Command                                                 | Verifier subset value | Lab value | Reason for difference                                                  |
| --------------------------- | ------------------------------------------------------- | --------------------- | --------- | ---------------------------------------------------------------------- |
| 18 drug-discovery modules   | `module_inventory.py --json` → `drug_discovery_modules` | `0`                   | `18`      | drug-discovery code is sovereign QENEX LTD IP; not shipped in verifier |
| 167 chemistry-package files | same → `qenex_chem_total_files`                         | `28`                  | `167`     | verifier ships only the chemistry-validation subset                    |
| Workspace Python modules    | same → `workspace_total_python_modules`                 | `50`                  | `330`     | verifier ships chem + qlang + qenex-core/constants only                |

### Audit & evaluator (sovereignty boundary)

| Claim                                       | Verifier subset                                     | Lab                                        | Difference                                                                                                                                   |
| ------------------------------------------- | --------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Evaluator                                   | `verifier_evaluator.py full` → `22/22 ZERO DEFECTS` | `evaluator.py full` → `58/58 ZERO DEFECTS` | Verifier runs 8 of 12 chemistry levels; excludes discovery (L7), scientific_integrity (L10), tamper_proof (L11), universe_capabilities (L12) |
| Standards audit (TRL/MRL/NPR/FAIR/ISO/WIPO) | not shipped                                         | `standards_audit.py` → `111/113`           | audit references files that are sovereign lab IP (auto_patent.py, wipo_pdf.py, brain proxy installation evidence)                            |

### Tests

| Claim                                         | Command                       | Verifier subset            | Lab           |
| --------------------------------------------- | ----------------------------- | -------------------------- | ------------- |
| pytest collection                             | `pytest --collect-only -q`    | `304 tests`                | `5,731 tests` |
| Validation tests (chemistry frozen-reference) | `pytest tests/validation/ -q` | `192 pass, 5 skip, 0 fail` | (full set)    |
| Top-level chemistry tests                     | `pytest tests/test_*.py -q`   | `24 pass, 2 skip, 0 fail`  | (full set)    |

## One-command verification

A reviewer who wants a single yes/no answer runs:

```bash
bash verify.sh
```

This script runs all four reproducer scripts in `--json` mode, writes
their outputs to a temporary directory, and verifies each output's
SHA-256 against `MANIFEST.sha256`. Exit code 0 means all four claims
reproduce bit-for-bit; non-zero means the reviewer's environment
produces different bytes (typically because their numerical-library
build aggregates BLAS reductions in a different order).

## How to interpret a SHA mismatch

If `verify.sh` fails, the **scientific** claim (sub-nanohartree HF
agreement) usually still holds — the _bit-level_ identity claim does
not. Inspect the failing JSON and check whether the energy values
are within the validation tolerance:

```bash
python3 packages/qenex_chem/src/scripts/precision_matrix.py --json \
  | python3 -c "import sys,json; d=json.load(sys.stdin); \
    [print(r['molecule'], r['basis'], r['drift_hartree']) \
     for r in d['results'] if r['classification']!='no-reference-available']"
```

If every drift is < 1e-6 Ha, the chemistry is correct on your platform
even if the bit-level reproducibility claim does not apply.

## Source files manifest

```
qenex-verifier/
├── 88 Python files
├── 35 .ql Q-Lang files (3 v04-runnable + 32 legacy illustration)
├── 32 chemistry test files
├── 3 Lebedev grid data files (.npz)
├── 6 top-level files (README, LICENSE, NOTICE, CITATION.cff,
│                       CHANGELOG.md, MANIFEST.sha256, SUPPLEMENTARY.md,
│                       conftest.py, pyproject.toml, verify.sh)
└── ~2.2 MB total source size
```

Every file is either:

1. **Computation source** — chemistry solvers, Q-Lang interpreter, basis
   sets, Lebedev grids. These are the same files (modulo the rebranding
   of the docstrings noted in `CHANGELOG.md`) as the lab's chemistry
   layer.
2. **Reproducer infrastructure** — the four `scripts/*_inventory.py`
   files, `verifier_evaluator.py`, `verify.sh`, `MANIFEST.sha256`.
3. **Tests** — chemistry validation against PySCF references.
4. **Documentation** — `README.md`, `CHANGELOG.md`, `SUPPLEMENTARY.md`
   (this file), `V2_ABSTRACT.md` (the paper draft itself), and
   `V1_VS_V2_RECONCILIATION.md` (honest v1→v2 numeric diff).

Nothing in this subset performs network access, reads `~/.qenex/`,
contacts a brain proxy, or invokes an MCP tool. The verifier is a
self-contained, air-gapped, deterministic reproducer.
