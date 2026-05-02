# Changelog

All notable changes to QENEX Verifier are documented here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-05-02

Initial open-source release. Companion verifier subset for the QENEX LAB
v2 paper (ChemRxiv preprint, May 2026).

### Added

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

### Sovereignty boundary (intentionally NOT included)

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

### Reproducibility verified

- All four reproducer scripts produce byte-identical JSON across
  consecutive runs (verified by SHA-256).
- All 22 verifier-evaluator chemistry checks pass (ZERO DEFECTS).
- 192 + 24 = 216 chemistry tests pass; 5 skipped (PySCF cross-validation,
  optional dependency).
- Tested against fresh shell environment with no inherited state.

### License

Apache License 2.0 — see [LICENSE](LICENSE).

[2.0.0]: https://github.com/qenex-ai/qenex-verifier/releases/tag/v2.0.0
