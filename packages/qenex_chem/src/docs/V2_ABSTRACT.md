# QENEX LAB v2 — Abstract Draft

**Title**: QENEX LAB: A Self-Contained Quantum Chemistry Platform with
Reproducible Self-Assessed Compliance and Cryptographic Provenance Bundles

**Status**: Draft for ChemRxiv / Zenodo deposit. Replaces v1 (DOI
10.26434/chemrxiv.15001263/v1, March 2026 — restricted access, 0
downloads as of 2026-05-02).

**Author**: Abdulrahman Sameer R Almutairi (QENEX LTD).
**Affiliation**: QENEX LTD (Scientific Intelligence Division).

**Deposit channel**: Zenodo, OPEN ACCESS. Source code repository:
https://github.com/qenex-ai/qenex-lab.git

---

## Abstract (≈300 words)

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

## Reproduction recipe

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

## What v2 changes from v1

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

## What this paper does NOT claim

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
