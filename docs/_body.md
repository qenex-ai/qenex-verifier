
\newpage

# 1. Abstract (v2 draft)

## QENEX LAB v2 — Abstract Draft

**Title**: QENEX LAB: A Self-Contained Quantum Chemistry Platform with
Reproducible Self-Assessed Compliance and Cryptographic Provenance Bundles

**Status**: Draft for ChemRxiv / Zenodo deposit. Replaces v1 (DOI
10.26434/chemrxiv.15001263/v1, March 2026 — restricted access, 0
downloads as of 2026-05-02).

**Author**: Abdulrahman Sameer R Almutairi (QENEX LTD).
**ORCID**: [0009-0004-4797-2226](https://orcid.org/0009-0004-4797-2226).
**Affiliation**: QENEX LTD (Scientific Intelligence Division).

**Deposit channel**: Zenodo, OPEN ACCESS. Source code repository:
https://github.com/qenex-ai/qenex-lab.git

---

### Abstract (≈300 words)

QENEX LAB is a sovereign, air-gapped scientific computing platform for
quantum chemistry and drug discovery, deployed on a single-server
configuration with no external cloud or hardware dependencies. The
platform provides 12 user-callable computational methods through a
unified `Molecule.compute()` API (Hartree-Fock, MP2, CCSD, DFT/LDA/
B3LYP, CASSCF, EOM-CCSD, TDDFT, UCCSD, DLPNO-CCSD, DMC) backed by 32
method-implementation modules and 6 basis sets (STO-3G, 6-31G\*,
cc-pVDZ, cc-pVTZ, aug-cc-pVDZ, aug-cc-pVTZ). Hartree-Fock energies
agree with PySCF references to **sub-nanohartree precision** (drift
< 1×10⁻⁹ Hartree) on 5 of 6 reference-validated (molecule, basis)
tuples, with the worst-case agreement being sub-millihartree
(3.4×10⁻⁴ Hartree on H2O / cc-pVDZ).

Compliance posture is **self-assessed** against six published rubrics
with full reproducibility: NASA Technology Readiness Level (system
TRL = 6/9; methodology per NASA SP-20170005794 + NPR 7123.1C), DoD
Manufacturing Readiness Level (MRL = 7/10), NASA NPR 7150.2D Rev D
software-engineering requirements (40/42 PASS, 2 honest WARN), FAIR
Principles for Research Software (20/20), ISO/IEC 25010:2011 software
quality (18/18), and WIPO PCT Rule 11 (8/8). Total 111/113 criteria
satisfied. **These are self-assessments, not external audits by NASA,
DoD, ISO, IEC, or WIPO.** Each score is reproducible by a single
command (`python3 packages/qenex_chem/src/standards_audit.py`) with
bit-identical output across consecutive runs.

The platform ships with a cryptographic provenance system: every
computed energy emits a runnable `.qlang` verification program plus
an `atoms.json` geometry sidecar. External auditors with the
open-source verifier component can re-execute these bundles and
confirm bit-identical replay. This is the platform's distinguishing
contribution — sovereign computation paired with publicly verifiable
results, without exposing internal infrastructure or proprietary code.

The platform contains 5,731 automated tests across 12 packages, a
58-check evaluator audit reporting ZERO DEFECTS, and 18 drug-discovery
modules covering ADMET, SAR, PK/PD, retrosynthesis, and clinical
prediction. The Q-Lang scientific protocol language (canonical v04:
16 modules, 5,593 lines) registers 3 simulation domains
(`chemistry`, `chemistry_geom`, `md`) at the present version.

---

### Reproduction recipe

A reviewer can verify every numeric claim in this abstract by running
the open-source verifier subset (this repository):

```bash
# Clone the verifier subset (chemistry-only reproducer; ~3 MB)
git clone https://github.com/qenex-ai/qenex-verifier.git
cd qenex-verifier
pip install -e ".[test,pyscf]"

# Verifiable in this subset:
python3 packages/qenex_chem/src/verifier_evaluator.py full                  # → 22/22 ZERO DEFECTS (chem)
python3 packages/qenex_chem/src/scripts/method_inventory.py --json          # → 12 user-callable
python3 packages/qenex_chem/src/scripts/module_inventory.py --json          # → 0 drug / 28 chem / 50 ws
python3 packages/qenex_chem/src/scripts/precision_matrix.py --json          # → 5/6 sub-nano
python3 packages/qenex_chem/src/scripts/qlang_inventory.py --json           # → 16 v04 / 3 domains
python3 -m pytest tests/validation/ -q                                      # → ~30 chemistry tests
sha256sum -c MANIFEST.sha256                                                # → all OK

# Verifiable ONLY on the full lab (not in the verifier subset):
#   python3 packages/qenex_chem/src/evaluator.py full                       # → 58/58 ZERO DEFECTS
#   python3 packages/qenex_chem/src/standards_audit.py                      # → 111/113 self-assessed
#   python3 -m pytest --collect-only -q                                     # → 5,731 tests
```

Two consecutive runs of each `--json` command produce **bit-identical**
output (verified). The `--json` outputs are SHA-256 pinnable for
external archival; reference SHAs are recorded in `MANIFEST.sha256` at
the verifier-subset root. The full-lab numbers (58/58 ZERO DEFECTS,
111/113 self-assessed compliance, 5,731 tests, 78 live Q-Lang modules,
18 drug-discovery modules) are reproducible only on the lab installation
because the relevant evidence files (discovery engine, scientific-
integrity guards, tamper-proof checks, universe-scale modules, drug-
discovery orchestrator, patent-generator infrastructure) are sovereign
QENEX LTD assets and are intentionally excluded from the verifier
subset. Numerical outputs of `precision_matrix.py` are deterministically
rounded to 11 decimal places (10 picohartree, two orders of magnitude
below the sub-nanohartree validation window) to suppress ULP-level
BLAS-reordering noise.

---

### What v2 changes from v1

| Aspect                 | v1 (March 2026)                     | v2 (May 2026)                                                                                                                                               |
| ---------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Method count           | 49 (claim)                          | 12 user-callable + 32 implementation modules (reproducible)                                                                                                 |
| Drug-discovery modules | 21 (claim)                          | 18 (reproducible)                                                                                                                                           |
| Test count             | 3,885 (claim)                       | 5,731 (reproducible)                                                                                                                                        |
| Evaluator              | 24/24 (claim)                       | 58/58 (reproducible)                                                                                                                                        |
| Compliance framing     | Numbers presented without qualifier | All scores explicitly self-assessed; rubric source cited                                                                                                    |
| TRL                    | 7/9 (cherry-picked best domain)     | 6/9 (system minimum per NASA methodology)                                                                                                                   |
| MRL                    | 8/10                                | 7/10 (honest dual-bible alignment)                                                                                                                          |
| NPR 7150.2D            | 42/42 COMPLIANT                     | 40/42 (38 PASS + 2 honest WARN)                                                                                                                             |
| Q-Lang lines           | 31,885 (claim)                      | 5,593 canonical / 46,047 total (reproducible)                                                                                                               |
| Q-Lang domains         | 6 (aspirational)                    | 3 (currently registered)                                                                                                                                    |
| New since v1           | —                                   | Cryptographic .qlang verification bundles; Path A productionization; discovery RAG (124,532 molecules indexed); lab-context enrichment for AI conversations |

The v2 numbers are smaller in some categories (methods, modules, TRL,
domains) and larger in others (tests, evaluator, Q-Lang lines). All
are reproducible by command. **The v2 paper is structurally stronger
than v1 because every claim is verifiable, not because every claim is
larger.**

---

### What this paper does NOT claim

For clarity to reviewers and operators:

1. **No external audit by NASA, DoD, ISO, or WIPO.** All compliance
   scores are self-assessments against the published rubric criteria.
2. **No drug-discovery commercial readiness claim.** The 130,727-
   molecule discovery corpus produced by the continuous engine is a
   research artifact; it has not been validated against clinical
   bioassay data and has no FDA or EMA submission.
3. **No claim of room-temperature superconductivity or any other
   exotic-physics discovery.** Aspirational outputs from earlier AI-
   driven exploration (see `LAB_NOTES.md` archive) are not part of
   this paper's scientific claims.
4. **No claim of bit-identical numerical results across hardware.**
   Hartree-Fock energies are computed to PySCF agreement at the level
   stated; ULP-level numerical noise (1×10⁻¹⁵ Hartree) exists
   between consecutive runs of the same calculation due to BLAS
   reordering. The bit-identical reproducibility claim applies to
   Q-Lang traces (which are content-addressed via SHA-256 of node
   inputs) rather than to raw numerical outputs.
5. **No multi-hardware reproducibility demonstration in this version.**
   The platform's sovereignty model keeps it single-server. A
   future paper covers external-verifier reproducibility from a
   distributable verifier subset.

These explicit non-claims are not weaknesses — they're what makes the
remaining claims defensible.

\newpage

# 2. Supplementary materials manifest

## Supplementary Materials Manifest

This document is a one-page index for reviewers: every numeric claim in
the QENEX LAB v2 abstract is mapped to the **exact reproducer command**
that produces it, the **expected output**, and the **source files** the
command exercises.

### Hard map: claim → command → expected → source

#### Computational methods

| Claim                                | Command                                                        | Expected | Source files                                                                 |
| ------------------------------------ | -------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| 6 distinct wired methods             | `method_inventory.py --json` → `headline_v2_claim.wired_count` | `6`      | `molecule.py:Molecule.compute()` (the dispatch tree)                         |
| 7 method names including `rhf` alias | same → `wired_count_raw_with_aliases`                          | `7`      | same                                                                         |
| 6 standalone solver classes          | same → `headline_v2_claim.standalone_count`                    | `6`      | `casscf.py`, `eomccsd.py`, `tddft.py`, `uccsd.py`, `dlpno_ccsd.py`, `qmc.py` |
| 12 user-callable methods             | same → `user_callable_total`                                   | `12`     | union of the two above                                                       |
| 13 method-implementation modules     | same → `module_count_upper_bound`                              | `13`     | walks `qenex_chem/src/*.py` for solver-named files                           |

#### Precision

| Claim                                       | Command                                                                                                                                                 | Expected                      | Source                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------- |
| 6 basis sets                                | `precision_matrix.py --json` → `headline_v2.basis_sets_total`                                                                                           | `6`                           | `basis_*.py` files, `molecule.py:Molecule(basis_name=…)`             |
| Frozen PySCF references for sto-3g, cc-pvdz | same → `basis_sets_with_reference`                                                                                                                      | `["sto-3g", "cc-pvdz"]`       | `certification.py:PYSCF_RHF_*`                                       |
| 5 of 6 reference-validated tuples sub-nano  | same → `tuples_sub_nanohartree` of `tuples_with_reference`                                                                                              | `5` / `6`                     | full HF kernel (`solver.py`, `integrals.py`, `libcint_integrals.py`) |
| H2 HF/STO-3G energy = -1.1167593074 Ha      | direct: `python3 -c "from molecule import Molecule; m=Molecule([('H',(0,0,0)),('H',(0,0,1.3984))],basis_name='sto-3g'); E,_=m.compute('hf'); print(E)"` | sub-nanohartree drift         | same                                                                 |
| ULP noise <1e-12 Ha suppressed by rounding  | `precision_matrix.py` rounds to 11 decimals                                                                                                             | bit-identical SHA across runs | `scripts/precision_matrix.py:128-138`                                |

#### Q-Lang scientific protocol

| Claim                                  | Command                                                                                                                 | Expected                                | Source                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------- |
| 16 canonical v04 modules               | `qlang_inventory.py --json` → `canonical_v04.modules`                                                                   | `16`                                    | `packages/qenex-qlang/src/v04/*.py`                                                       |
| 5,593 lines in canonical v04           | same → `canonical_v04.lines`                                                                                            | `5,593`                                 | same                                                                                      |
| 3 simulate domains registered          | same → `headline_v2.registered_domains_count`                                                                           | `3`                                     | `simulate_dispatch_v04.py:_register_defaults()`                                           |
| Domains: chemistry, chemistry_geom, md | same → `registered_domains_list`                                                                                        | `["chemistry","chemistry_geom","md"]`   | same                                                                                      |
| End-to-end Q-Lang execution            | `PYTHONPATH=packages/qenex-qlang/src python3 -m v04.cli_v04 run packages/qenex-qlang/examples/v04/03_h2_bond_energy.ql` | "E(H2) in Hartree = -1.11676 [Hartree]" | full v04 stack: lexer → parser → interp → simulate_dispatch → `qenex_chem.compute_energy` |

#### Drug discovery (sovereignty boundary)

| Claim                       | Command                                                 | Verifier subset value | Lab value | Reason for difference                                                  |
| --------------------------- | ------------------------------------------------------- | --------------------- | --------- | ---------------------------------------------------------------------- |
| 18 drug-discovery modules   | `module_inventory.py --json` → `drug_discovery_modules` | `0`                   | `18`      | drug-discovery code is sovereign QENEX LTD IP; not shipped in verifier |
| 167 chemistry-package files | same → `qenex_chem_total_files`                         | `28`                  | `167`     | verifier ships only the chemistry-validation subset                    |
| Workspace Python modules    | same → `workspace_total_python_modules`                 | `50`                  | `330`     | verifier ships chem + qlang + qenex-core/constants only                |

#### Audit & evaluator (sovereignty boundary)

| Claim                                       | Verifier subset                                     | Lab                                        | Difference                                                                                                                                   |
| ------------------------------------------- | --------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Evaluator                                   | `verifier_evaluator.py full` → `22/22 ZERO DEFECTS` | `evaluator.py full` → `58/58 ZERO DEFECTS` | Verifier runs 8 of 12 chemistry levels; excludes discovery (L7), scientific_integrity (L10), tamper_proof (L11), universe_capabilities (L12) |
| Standards audit (TRL/MRL/NPR/FAIR/ISO/WIPO) | not shipped                                         | `standards_audit.py` → `111/113`           | audit references files that are sovereign lab IP (auto_patent.py, wipo_pdf.py, brain proxy installation evidence)                            |

#### Tests

| Claim                                         | Command                       | Verifier subset            | Lab           |
| --------------------------------------------- | ----------------------------- | -------------------------- | ------------- |
| pytest collection                             | `pytest --collect-only -q`    | `304 tests`                | `5,731 tests` |
| Validation tests (chemistry frozen-reference) | `pytest tests/validation/ -q` | `192 pass, 5 skip, 0 fail` | (full set)    |
| Top-level chemistry tests                     | `pytest tests/test_*.py -q`   | `24 pass, 2 skip, 0 fail`  | (full set)    |

### One-command verification

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

### How to interpret a SHA mismatch

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

### Source files manifest

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

\newpage

# 3. v1 → v2 numeric reconciliation

## QENEX LAB Paper — v1 → v2 Reconciliation Table

**Status**: Working draft for v2 ChemRxiv / Zenodo deposit.
**Generated**: 2026-05-02 (Day 3 of "make claims unambiguous and reproducible" plan).
**Source of truth for all numbers**: the four reproducer scripts in
`packages/qenex_chem/src/scripts/` and the unified audit at
`packages/qenex_chem/src/standards_audit.py` (lab-only — see note below).
Two consecutive runs of each produce identical output (verified
during Day 2).

> **Reading this in the verifier subset?** `evaluator.py` and
> `standards_audit.py` are not shipped in the open-source verifier
> subset because they reference internal lab files (auto_patent.py,
> wipo_pdf.py, brain proxy, RAG corpus, lab installation evidence).
> The verifier subset ships `verifier_evaluator.py` (8 chemistry levels
> = 22 checks) instead of the full 12-level lab evaluator (58 checks).
> The four reproducer scripts behave identically in both environments,
> with output sized to whatever each environment actually contains
> (e.g. the verifier reports 16 Q-Lang live modules; the lab reports
> 78). All differences are honestly enumerated in the table below.

This document compares every numeric claim from v1 with what the
reproducers actually return today, and proposes the v2 phrasing.

---

### Methodology

For each v1 claim:

1. The corresponding reproducer was identified or built (Day 1-2 work).
2. The reproducer was run twice; outputs compared for stability.
3. The honest current number is reported alongside the v1 claim.
4. Direction (better / worse / equal / format-different) is flagged.
5. The recommended v2 phrasing is provided.

**Principle**: v2 reports whatever the methodology produces, with the
methodology cited and re-runnable. No number is "tuned" to match v1.

---

### Reconciliation table

| Claim category                    | v1 said                          | v2 actual                                                                                                                                                                                                  | Direction                                   | Reproducer command                                            |
| --------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------- |
| Computational methods (wired API) | "49 computational methods"       | **6 distinct + 1 alias** through `Molecule.compute()`; **+6 standalone solver classes** = **12 user-callable**                                                                                             | ↓ honest                                    | `python3 -m packages.qenex_chem.src.scripts.method_inventory` |
| Method-implementation files       | (implicit ≤ 49)                  | **32 method-implementation modules** in `qenex_chem/src/`                                                                                                                                                  | (clarified)                                 | same command                                                  |
| Drug-discovery modules            | "21"                             | **18** modules in `qenex_chem/src/`                                                                                                                                                                        | ↓ honest (off by 3)                         | `python3 -m packages.qenex_chem.src.scripts.module_inventory` |
| Total chemistry-package files     | (not claimed)                    | **167** files in `qenex_chem/src/`                                                                                                                                                                         | (additional context)                        | same command                                                  |
| Workspace Python modules          | (not claimed)                    | **330** modules across 12 packages                                                                                                                                                                         | (additional context)                        | same command                                                  |
| Basis sets                        | "6 (STO-3G through aug-cc-pVTZ)" | **6 confirmed** (sto-3g, cc-pvdz, cc-pvtz, 6-31g\*, aug-cc-pvdz, aug-cc-pvtz)                                                                                                                              | =                                           | `python3 -m packages.qenex_chem.src.scripts.precision_matrix` |
| Sub-nanohartree precision         | "across 6 basis sets"            | **5 of 6** reference-validated (molecule, basis) tuples are sub-nanohartree; **2 of 6 basis sets** have frozen PySCF references in the certification module                                                | refined to honest                           | same command                                                  |
| Automated tests                   | "3,885"                          | **5,731** collected by pytest (12 collection errors, all in legacy/archived files)                                                                                                                         | ↑ better                                    | `python3 -m pytest --collect-only -q`                         |
| Evaluator                         | "24/24 evaluator ZERO DEFECTS"   | **58/58 ZERO DEFECTS** (current evaluator has 58 checks; v1 evaluator had 24)                                                                                                                              | ↑ better                                    | `python3 packages/qenex_chem/src/evaluator.py full`           |
| NASA TRL                          | "TRL 7/9"                        | **System TRL = 6/9** (drug_discovery domain at 7; system pinned to minimum per NASA SP-20170005794 methodology). Self-assessment audit: 12/12 criteria PASS.                                               | ↓ honest (per-NASA min-rule)                | `python3 packages/qenex_chem/src/standards_audit.py`          |
| DoD MRL                           | "MRL 8/10"                       | **MRL = 7/10**. Self-assessment audit: 13/13 criteria PASS.                                                                                                                                                | ↓ honest                                    | same command                                                  |
| NASA NPR 7150.2D                  | "42/42 COMPLIANT"                | **40/42** self-assessment (38 PASS, 2 honest WARN). The 2 WARNs are: SWE-005 cost/effort estimation (not formally tracked) and SWE-205 Independent V&V (Class C software typically does not require IV&V). | ↓ honest with explicit gaps                 | same command                                                  |
| ISO/IEC 25010                     | "5 EXCELLENT + 3 GOOD"           | **18/18** ISO/IEC 25010 quality characteristics satisfied (current rubric uses uniform PASS/FAIL not EXCELLENT/GOOD)                                                                                       | format clarified                            | same command                                                  |
| FAIR4RS                           | "37/39"                          | **20/20** FAIR4RS criteria satisfied (current rubric is the 20-criterion FAIR4RS v1.0; v1 may have used an older expanded variant)                                                                         | format clarified                            | same command                                                  |
| WIPO/PCT                          | (not in v1 abstract)             | **8/8**                                                                                                                                                                                                    | (additional standard)                       | same command                                                  |
| Q-Lang lines                      | "31,885 lines"                   | **5,593 lines** in canonical v04; **46,047 lines** total in live `qenex-qlang` package                                                                                                                     | refined: distinguishes canonical from total | `python3 -m packages.qenex_chem.src.scripts.qlang_inventory`  |
| Q-Lang modules                    | "34 modules"                     | **16 modules** in canonical v04; **78 modules** in live package                                                                                                                                            | refined: distinguishes canonical from total | same command                                                  |
| Simulate domains                  | "6 simulation domains"           | **3 currently registered** (chemistry, chemistry_geom, md). The other 3 v1 mentioned were never implemented or were aspirational.                                                                          | ↓ honest                                    | same command                                                  |
| Q-Lang examples                   | (not claimed)                    | **54** distributed `.ql` / `.qlang` programs                                                                                                                                                               | (additional context)                        | same command                                                  |

---

### Summary: what's better, what's worse, what's the same

**Better than v1 claimed (v2 numbers exceed v1):**

- Test count (5,731 vs 3,885) — **+47%**
- Evaluator (58/58 vs 24/24) — **+142%**
- Q-Lang total lines (46,047 vs 31,885) — **+44%**
- Q-Lang total modules (78 vs 34) — **+129%**

**Worse than v1 claimed (v2 honest numbers fall short of v1):**

- Methods (12 user-callable vs 49 claimed) — **v1 was substantially overstated**
- Drug-discovery modules (18 vs 21) — **v1 was off by 3**
- TRL (6 vs 7) — **NASA min-rule applied honestly**
- MRL (7 vs 8) — **honest self-assessment**
- NPR 7150.2D (40/42 vs 42/42) — **honest with 2 documented gaps**
- Q-Lang simulate domains (3 vs 6) — **half of v1's claim was aspirational**

**Format-clarified (different rubric version, same rigor):**

- ISO/IEC 25010 (18/18 vs "5 EXCELLENT + 3 GOOD")
- FAIR4RS (20/20 vs 37/39)

---

### What v2 should NOT claim

Based on this reconciliation, the v2 paper must:

1. **NOT claim "49 computational methods"** — claim "12 user-callable methods + 32 method-implementation modules" instead.
2. **NOT claim "21 drug-discovery modules"** — claim **18**.
3. **NOT claim "TRL 7/9"** without qualification — claim **"System TRL 6/9 (drug_discovery domain at 7), self-assessed against NASA SP-20170005794."**
4. **NOT claim "MRL 8/10"** — claim **"MRL 7/10, self-assessed against DoD MRA Deskbook v2.0."**
5. **NOT claim "42/42 COMPLIANT"** — claim **"40/42 PASS + 2 WARN, self-assessed against NPR 7150.2D Rev D."**
6. **NOT claim "6 simulation domains"** — claim **"3 currently registered simulation domains (chemistry, chemistry_geom, md)."**
7. **NOT report compliance scores without the "self-assessed" qualifier** — these are NOT external NASA / DoD / ISO audits.

---

### What v2 CAN newly claim (didn't exist in v1)

These are real capabilities shipped between v1 (March 2026) and v2 (May 2026).
Each is tagged with where it is reproducible: `[both]` lives in both the
lab and the verifier subset; `[lab-only]` is exclusive to the lab
installation.

1. **[both] Cryptographic provenance bundles**: every accepted molecule discovery emits a runnable `.qlang` verification program plus an `atoms.json` sidecar. External auditors can reproduce results by running `python3 -m cli_v04 run <bundle>.qlang` against either environment.
2. **[lab-only] Path A productionization**: an internal MCP tool wraps the full export → run → replay pipeline as a single callable for AI agents talking to the lab.
3. **[lab-only] Discovery RAG store**: 124,532 unique molecules from the continuous engine indexed locally for semantic search.
4. **[lab-only] Lab-context enrichment for AI conversations**: the lab's internal knowledge base can be injected into AI conversations on demand.
5. **[lab-only] Multi-provider AI gateway**: dispatch layer for AI providers with audit log per call.

The `[lab-only]` items live above the sovereignty boundary and are not
reproducible from the verifier subset. They are documented here for v1→v2
historical context and to explain what the v2 abstract means by "Path A
productionization" and "discovery RAG indexed". Item 1 is the only
externally-reproducible new capability.

---

### Reproduction recipe for reviewers

#### On the QENEX LAB installation (full audit)

A reviewer with access to the QENEX LAB installation re-runs every
numeric claim in the v2 abstract:

```bash
cd qenex-lab/workspace
python3 -m pytest --collect-only -q                                          # → 5,731 tests
python3 packages/qenex_chem/src/evaluator.py full                            # → 58/58 ZERO DEFECTS
python3 packages/qenex_chem/src/standards_audit.py                           # → 111/113 self-assessed
python3 packages/qenex_chem/src/scripts/method_inventory.py                  # → 12 methods
python3 packages/qenex_chem/src/scripts/module_inventory.py                  # → 18 / 167 / 330
python3 packages/qenex_chem/src/scripts/precision_matrix.py                  # → 5/6 sub-nano
python3 packages/qenex_chem/src/scripts/qlang_inventory.py                   # → 16 v04 / 78 live / 3 domains
```

#### On the open-source verifier subset

A reviewer with only the verifier subset re-runs the chemistry-only
core:

```bash
cd qenex-verifier
python3 packages/qenex_chem/src/verifier_evaluator.py full                   # → 22/22 ZERO DEFECTS
python3 packages/qenex_chem/src/scripts/method_inventory.py                  # → 12 user-callable
python3 packages/qenex_chem/src/scripts/module_inventory.py                  # → 0 drug / 28 chem / 50 ws
python3 packages/qenex_chem/src/scripts/precision_matrix.py                  # → 5/6 sub-nano
python3 packages/qenex_chem/src/scripts/qlang_inventory.py                   # → 16 v04 / 16 live / 3 domains
python3 -m pytest tests/validation/ -q                                       # → ~30 chem tests
sha256sum -c MANIFEST.sha256                                                 # → all OK
```

In **both** environments the four `--json` reproducer scripts produce
**bit-identical output across consecutive runs** (verified by SHA-256;
ULP-level BLAS reordering is suppressed by deterministic 11-decimal-
place rounding in `precision_matrix.py`). This is what "reproducible
claims" means in v2.

---

### Author and affiliation

Author: Abdulrahman Sameer R Almutairi (Almutairi, A. S. R.) — QENEX LTD.
ORCID: [0009-0004-4797-2226](https://orcid.org/0009-0004-4797-2226).
Affiliation: QENEX LTD, Scientific Intelligence Division.

\newpage

# 4. Verifier subset changelog

## Changelog

All notable changes to QENEX Verifier are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

### [2.0.0] — 2026-05-02

Initial open-source release. Companion verifier subset for the QENEX LAB
v2 paper (ChemRxiv preprint, May 2026).

#### Added

- **Chemistry surface (28 modules)**: Hartree-Fock (`solver.py`),
  Møller-Plesset 2 (in `molecule.py`), CCSD (`ccsd.py`), CCSD(T)
  via DLPNO (`dlpno_ccsd.py`), CASSCF (`casscf.py`, `casscf_ciah.py`),
  EOM-CCSD (`eomccsd.py`), UCCSD (`uccsd.py`), DFT/LDA/B3LYP
  (`dft.py`), TDDFT (`tddft.py`), QMC (`qmc.py`), PCM solvation
  (`solvation.py`), vibrational analysis (`vibrational.py`),
  CCSD gradient (`ccsd_gradient.py`), geometry optimizer
  (`optimizer.py`), CBS extrapolation (`cbs.py`), benchmark suite
  (`benchmark.py`), frozen reference suite (`certification.py`).
- **Basis sets**: STO-3G (built-in), 6-31G\*, cc-pVDZ, cc-pVTZ,
  aug-cc-pVDZ, aug-cc-pVTZ.
- **Lebedev quadrature grids**: Levels 26, 110, 302 for DFT.
- **Q-Lang v0.4 canonical interpreter**: 16 modules, 5,593 lines.
- **3 v04-runnable example programs**: bond length uncertainty,
  Earth orbital period, H2 bond energy.
- **20 + 12 = 32 legacy example programs**: kept for historical
  illustration; do not run on the v04 interpreter.
- **4 reproducer scripts** (in `packages/qenex_chem/src/scripts/`):
  - `method_inventory.py` — counts wired vs standalone methods,
    deduplicates `rhf` alias of `hf`.
  - `module_inventory.py` — chemistry / drug-discovery file counts.
  - `qlang_inventory.py` — v04 vs live Q-Lang module counts.
  - `precision_matrix.py` — 18 (molecule, basis, method) tuples
    against frozen PySCF references; rounds energies to 11 decimal
    places (10 picohartree) to suppress ULP-level BLAS-reordering
    noise.
- **Verifier evaluator** (`verifier_evaluator.py`): 8 chemistry-only
  audit levels, 22 checks, ZERO DEFECTS. Reduced from the lab's
  12-level / 58-check evaluator (excludes discovery, scientific-
  integrity, tamper-proof, and universe-capabilities levels that
  depend on sovereign lab modules).
- **Test suite (32 files = 304 collected)**: 192 chemistry validation
  tests + 24 top-level chemistry tests; the rest skipped where
  PySCF cross-validation is unavailable.
- **`MANIFEST.sha256`**: pinned SHA-256 of every reproducer's
  `--json` output. Ensures bit-level reproducibility.
- **`verify.sh`**: one-command gate. Runs all four reproducers,
  diffs against `MANIFEST.sha256`, exits 0 on byte-identical match.
- **Documentation**: `V2_ABSTRACT.md` (paper draft),
  `V1_VS_V2_RECONCILIATION.md` (honest v1→v2 numeric diff with
  context), `examples/README.md` (canonical vs legacy split).

#### Sovereignty boundary (intentionally NOT included)

The QENEX LAB v2 platform contains the following components that are
**not** in this verifier subset; they remain proprietary assets of
QENEX LTD:

- Autonomous discovery engine, continuous discovery engine
- Lab integrity guard (`qenex_guard`), tamper-proof checks
- Drug-discovery orchestrator, materials genome, FEP engine
- Brain proxy, MCP tools, FDSP, RAG corpus
- AI-provider gateway, multi-provider dispatch
- NMR shielding, ZORA relativistic, transport / NEB, Bethe ansatz
- Robot protocol, federated learning, regulatory engine
- Auto-patent generator, WIPO PDF generator
- 4 of 12 evaluator levels (discovery, scientific_integrity,
  tamper_proof, universe_capabilities)

The full lab's claims of `58/58 ZERO DEFECTS`, `111/113 self-assessed
compliance`, `5,731 automated tests`, `78 live Q-Lang modules`, and
`18 drug-discovery modules` are reproducible only on the lab
installation. The verifier subset reproduces the chemistry-only core
of those claims with bit-level fidelity.

#### Reproducibility verified

- All four reproducer scripts produce byte-identical JSON across
  consecutive runs (verified by SHA-256).
- All 22 verifier-evaluator chemistry checks pass (ZERO DEFECTS).
- 192 + 24 = 216 chemistry tests pass; 5 skipped (PySCF cross-validation,
  optional dependency).
- Tested against fresh shell environment with no inherited state.

#### License

Apache License 2.0 — see [LICENSE](LICENSE).

[2.0.0]: https://github.com/qenex-ai/qenex-verifier/releases/tag/v2.0.0
