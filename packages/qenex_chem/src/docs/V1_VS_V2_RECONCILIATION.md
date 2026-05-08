# QENEX LAB Paper — v1 → v2 Reconciliation Table

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

## Methodology

For each v1 claim:

1. The corresponding reproducer was identified or built (Day 1-2 work).
2. The reproducer was run twice; outputs compared for stability.
3. The honest current number is reported alongside the v1 claim.
4. Direction (better / worse / equal / format-different) is flagged.
5. The recommended v2 phrasing is provided.

**Principle**: v2 reports whatever the methodology produces, with the
methodology cited and re-runnable. No number is "tuned" to match v1.

---

## Reconciliation table

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

## Summary: what's better, what's worse, what's the same

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

## What v2 should NOT claim

Based on this reconciliation, the v2 paper must:

1. **NOT claim "49 computational methods"** — claim "12 user-callable methods + 32 method-implementation modules" instead.
2. **NOT claim "21 drug-discovery modules"** — claim **18**.
3. **NOT claim "TRL 7/9"** without qualification — claim **"System TRL 6/9 (drug_discovery domain at 7), self-assessed against NASA SP-20170005794."**
4. **NOT claim "MRL 8/10"** — claim **"MRL 7/10, self-assessed against DoD MRA Deskbook v2.0."**
5. **NOT claim "42/42 COMPLIANT"** — claim **"40/42 PASS + 2 WARN, self-assessed against NPR 7150.2D Rev D."**
6. **NOT claim "6 simulation domains"** — claim **"3 currently registered simulation domains (chemistry, chemistry_geom, md)."**
7. **NOT report compliance scores without the "self-assessed" qualifier** — these are NOT external NASA / DoD / ISO audits.

---

## What v2 CAN newly claim (didn't exist in v1)

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

## Reproduction recipe for reviewers

### On the QENEX LAB installation (full audit)

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

### On the open-source verifier subset

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

## Author and affiliation

Author: Abdulrahman Sameer R Almutairi (Almutairi, A. S. R.) — QENEX LTD.
ORCID: [0009-0004-4797-2226](https://orcid.org/0009-0004-4797-2226).
Affiliation: QENEX LTD, Scientific Intelligence Division.
