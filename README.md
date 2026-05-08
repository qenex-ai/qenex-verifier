# QENEX Verifier

Open-source reproduction subset for the QENEX LAB v2 paper (ChemRxiv, May
2026). This repository ships exactly the chemistry-validation surface
required to verify every numeric claim in the paper abstract — and
nothing else.

The full QENEX LAB platform is the sovereign, air-gapped scientific
computing system of QENEX LTD. The verifier subset is the strict
open-source slice of that platform that an external auditor can run
without access to the lab's brain server.

**For reviewers in a hurry**: read [`docs/QENEX_LAB_v2_Summary_Card.pdf`](docs/QENEX_LAB_v2_Summary_Card.pdf)
(2 pages) before diving in. The full Supplementary Information is at
[`docs/QENEX_LAB_v2_Supplementary.pdf`](docs/QENEX_LAB_v2_Supplementary.pdf)
(14 pages: abstract, claim manifest, v1→v2 reconciliation, changelog).
Both PDFs are regeneratable from markdown sources via
[`docs/build_pdf.sh`](docs/build_pdf.sh) (requires pandoc + xelatex).

---

## What is reproducible here

Every numeric claim in the v2 abstract has a single command behind it,
producing **bit-identical JSON output across consecutive runs**:

| Claim                                                                      | Reproducer command                                                   |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| 12 user-callable computational methods                                     | `python3 packages/qenex_chem/src/scripts/method_inventory.py --json` |
| 13 method-implementation modules in this subset                            | same as above                                                        |
| 6 basis sets (STO-3G, 6-31G\*, cc-pVDZ, cc-pVTZ, aug-cc-pVDZ, aug-cc-pVTZ) | `python3 packages/qenex_chem/src/scripts/precision_matrix.py --json` |
| 5 of 6 reference-validated tuples sub-nanohartree                          | same as above                                                        |
| 16 canonical Q-Lang v04 modules / 5,593 lines                              | `python3 packages/qenex_chem/src/scripts/qlang_inventory.py --json`  |
| 3 simulation domains (chemistry, chemistry_geom, md)                       | same as above                                                        |
| 3 v04-runnable .ql examples (+ 32 legacy)                                  | same as above                                                        |
| 22/22 chemistry-validation checks pass                                     | `python3 packages/qenex_chem/src/verifier_evaluator.py full`         |
| 304 chemistry tests pass (192 in tests/validation/)                        | `python3 -m pytest -q`                                               |

Two consecutive runs of any `--json` reproducer command produce
**bit-identical output** (verified — see `MANIFEST.sha256`).

---

## What this subset does NOT contain

The QENEX LAB v2 abstract makes additional claims that are reproducible
**only on the full lab installation**, not in this verifier subset:

- **`58/58 ZERO DEFECTS`** — full lab evaluator runs 12 levels including
  discovery audit (L7), scientific-integrity guards (L10), tamper-proof
  checks (L11), and universe-scale capabilities (L12: NMR shielding,
  ZORA relativistic, NEB transport, Bethe ansatz, robot protocol).
  The verifier subset's `verifier_evaluator.py` runs the 8 chemistry
  levels: edge_cases, precision, physics, properties, excited,
  vibrational, new_methods, benchmark — for **22/22 ZERO DEFECTS**
  in this subset.
- **`111/113 self-assessed compliance criteria`** — across the 6
  rubrics (NASA TRL, DoD MRL, NPR 7150.2D, FAIR4RS, ISO/IEC 25010,
  WIPO PCT). The compliance audit references files that live on the
  lab installation (auto_patent.py, wipo_pdf.py, brain proxy,
  installation evidence, etc.) and is not part of this subset.
- **`5,731 automated tests`** — the full lab. This subset ships **304
  curated tests** sufficient to verify the precision matrix and full
  chemistry surface (HF, MP2, CCSD, DFT, EOM-CCSD, UCCSD, TDDFT,
  CASSCF, DLPNO-CCSD, QMC, vibrational, solvation).
- **`78 live Q-Lang modules / 46,047 lines total`** — the full lab.
  This subset ships only the canonical v04 (16 modules, 5,593 lines).
- **`18 drug-discovery modules`** — the full lab. This subset ships
  zero drug-discovery modules — they are sovereign lab IP.
- **`124,532 indexed molecules in the discovery RAG`** — the full lab.
- **`Cryptographic .qlang verification bundles`** — the lab emits
  these; this subset can replay them via the `cli_v04 run` pathway,
  but the emission infrastructure (continuous discovery engine,
  brain proxy) is not included.

These are not weaknesses of the verifier subset — they are the result
of a deliberate sovereignty boundary. The subset is published so
reviewers can verify the **published numeric claims** without QENEX LTD
having to expose its full research infrastructure.

---

## Quick start

```bash
# Clone
git clone https://github.com/qenex-ai/qenex-verifier.git
cd qenex-verifier

# Install (Python 3.10–3.12)
pip install -e ".[test,pyscf]"

# Verify
python3 packages/qenex_chem/src/verifier_evaluator.py full      # 22/22 ZERO DEFECTS
python3 packages/qenex_chem/src/scripts/method_inventory.py     # 12 user-callable methods
python3 packages/qenex_chem/src/scripts/module_inventory.py     # 0 drug-discovery (this subset)
python3 packages/qenex_chem/src/scripts/precision_matrix.py     # 5/6 sub-nanohartree
python3 packages/qenex_chem/src/scripts/qlang_inventory.py      # 16 v04 modules
python3 -m pytest tests/validation/ -q                          # ~30 chemistry tests pass
```

`pyscf` is optional. Without it, `tests/validation/test_cross_validation.py`
will skip; all other tests pass without external chemistry packages.

---

## Reproducibility guarantee

All four reproducer scripts produce **byte-identical JSON output** on
consecutive runs (verified by SHA-256). The `precision_matrix.py` script
rounds Hartree-Fock energies to 11 decimal places (10 picohartree, 100×
below the sub-nanohartree validation window) to suppress ULP-level
BLAS-reordering noise without weakening any scientific claim.

To verify the SHAs:

```bash
sha256sum -c MANIFEST.sha256
```

Expected output: `qenex-verifier-claims-* : OK` for each of the four
reproducer outputs. If your numbers differ, your numerical-library
build (BLAS / LAPACK) is producing a different rounding pattern than
ours and the v2 paper's bit-level claim does not apply on your platform.
The scientific claim (sub-nanohartree agreement) remains valid.

---

## Subset boundary

```
qenex-verifier/                       (this repo)
├── conftest.py                       (sys.path injection — mirrors lab)
├── pyproject.toml                    (Python 3.10–3.12 + numpy/scipy/numba)
├── LICENSE                           (Apache 2.0)
├── README.md                         (this file)
├── MANIFEST.sha256                   (pinned SHAs of reproducer outputs)
├── packages/
│   ├── qenex_chem/src/
│   │   ├── molecule.py, solver.py, ccsd.py, …  (28 chemistry source files)
│   │   ├── basis_*.py + lebedev_*.npz           (basis sets, DFT grids)
│   │   ├── certification.py                     (frozen PySCF references)
│   │   ├── verifier_evaluator.py                (8 chem audit levels, 22 checks)
│   │   ├── docs/V2_ABSTRACT.md                  (the paper draft itself)
│   │   ├── docs/V1_VS_V2_RECONCILIATION.md      (v1 → v2 honest reconciliation)
│   │   └── scripts/{method,module,qlang,precision}_inventory.py
│   └── qenex-qlang/
│       ├── src/v04/                             (16 canonical interpreter modules)
│       ├── src/qlang_v04.py                     (top-level re-export shim)
│       ├── examples/v04/                        (3 v04-runnable .ql programs)
│       ├── examples/legacy/                     (20 pre-v04 illustration only)
│       └── tests/                               (canonical v04 tests)
├── examples/qlang/legacy/                       (12 pre-v04 language tests)
└── tests/                                       (32 chemistry validation tests = 304 collected)
```

What is intentionally **NOT** in this subset:

- `discovery_engine.py`, `continuous_engine.py` — autonomous discovery
- `qenex_guard.py`, `tamper_proof.py`, `physics_guardrail.py` — lab integrity
- `genomics.py`, `nmr.py`, `relativistic.py`, `tensor.py` — physics modules
- `inverse_design.py`, `dispersion.py`, `molecular_dynamics.py` — additional methods
- `robot_protocol.py` — laboratory automation
- `auto_patent.py`, `wipo_pdf.py` — IP / patent generators
- ALL of `discovery/`, MCP tools, brain proxy, RAG corpus
- 4 of 12 evaluator levels (7 discovery, 10 scientific_integrity,
  11 tamper_proof, 12 universe_capabilities)
- All examples/ scripts depicting aspirational/exotic results that
  the v2 abstract explicitly disavows (theory_of_everything.ql,
  cancer_grand_challenge.ql, biological_rejuvenation.ql, etc.)

---

## Citing

```bibtex
@article{almutairi_qenex_lab_v2_2026,
  title   = {QENEX LAB: A Self-Contained Quantum Chemistry Platform with
             Reproducible Self-Assessed Compliance and Cryptographic
             Provenance Bundles},
  author  = {Almutairi, Abdulrahman Sameer R},
  year    = {2026},
  organization = {QENEX LTD},
  doi     = {<v2-pending>},
  note    = {Verifier subset: \url{https://github.com/qenex-ai/qenex-verifier}}
}
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The verifier subset is open-source. The full QENEX LAB platform is
proprietary; for commercial licensing, contact QENEX LTD.

---

## Honest framing

This is a verifier, not a demo. Every command in this README runs the
real solvers, computes real Hartree-Fock / CCSD / DFT energies, and
agrees with PySCF references at the precision stated in the v2 abstract.
The subset does not stub, mock, or shortcut any computation. If a
reviewer's run of `precision_matrix.py` produces a different
classification ("sub-microhartree" instead of "sub-nanohartree", say),
the answer is in their numerical-library build — not in any difference
between the subset and the lab. The chemistry is identical.
