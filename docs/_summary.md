---
title: "QENEX LAB v2 — One-Page Summary Card"
author: "QENEX LTD"
date: "2026-05-08"
geometry:
  - margin=1.8cm
fontsize: 9.5pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
linkcolor: blue
urlcolor: blue
colorlinks: true
papersize: a4
header-includes:
  - \pagestyle{empty}
  - \usepackage{fvextra}
  - \fvset{breaklines=true,fontsize=\footnotesize}
  - \usepackage{microtype}
---

\begin{center}
\textbf{\Large QENEX LAB v2 — Reviewer Summary Card}\\[3pt]
\textit{One-page cheat sheet for the v2 paper; full SI is the companion PDF.}
\end{center}

## What this paper is

QENEX LAB is a sovereign quantum-chemistry platform. The **v2** preprint
(May 2026) supersedes v1 (March 2026, ChemRxiv 10.26434/chemrxiv.15001263/v1).
The improvement over v1 is **not larger numbers** — most numbers fell
when audited honestly — but **reproducibility**: every numeric claim in
the v2 abstract has a single command behind it that produces
bit-identical JSON output across consecutive runs.

## How to verify

Clone the open-source verifier subset (Apache 2.0):

```bash
git clone https://github.com/qenex-ai/qenex-verifier.git
cd qenex-verifier
pip install -e ".[test,pyscf]"
bash verify.sh                              # exit 0 = all 4 SHAs match
python3 packages/qenex_chem/src/verifier_evaluator.py full   # 22/22 ZERO DEFECTS
```

## Numbers, source-checked

| Claim                                    | Verifier subset                                | Lab installation                                   |
| ---------------------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| User-callable methods (Molecule.compute) | **12** (verified)                              | 12                                                 |
| Method-implementation modules            | 13 (verifier ships subset)                     | **32**                                             |
| Basis sets with frozen PySCF references  | **5/6 sub-nanohartree** on (mol, basis) tuples | identical                                          |
| Q-Lang v04 modules / lines               | **16 / 5,593**                                 | 16 / 5,593                                         |
| Q-Lang simulation domains                | **3** (chemistry, chemistry_geom, md)          | 3                                                  |
| Evaluator                                | **22/22 ZERO DEFECTS** (8 chem levels)         | 58/58 (12 levels, lab-only)                        |
| Self-assessed compliance                 | n/a (audit lab-only)                           | **111/113** (NASA TRL/MRL/NPR + FAIR + ISO + WIPO) |
| Tests collected                          | 304                                            | 5,731                                              |

## Sovereignty boundary (intentionally absent from verifier)

Discovery engine, drug-discovery orchestrator, brain proxy, RAG corpus,
MCP tools, lab-integrity guard, robot protocol, NMR/ZORA/Bethe/genomics
modules, auto-patent generator, FEP engine, materials genome,
4-of-12 evaluator levels (discovery / scientific_integrity /
tamper_proof / universe_capabilities). These remain proprietary
QENEX LTD assets; the v2 paper does not require them for any
externally-checkable claim.

## What v2 does NOT claim

1. No external NASA / DoD / ISO / WIPO audit. All compliance scores
   are **self-assessed** against published rubrics.
2. No drug-discovery commercial readiness. The 130,727-molecule
   discovery corpus is a research artifact; no FDA / EMA submission.
3. No room-temperature superconductivity, theory-of-everything,
   biological rejuvenation, or any other exotic-physics discovery.
4. No bit-identical numerical results across hardware. Hartree-Fock
   energies match PySCF references at the precision stated;
   ULP-level ($\sim 10^{-15}$ Ha) BLAS-reordering noise is
   suppressed by deterministic 11-decimal rounding in the
   `precision_matrix.py` reproducer to make JSON output SHA-pinnable.

## Citation

```
QENEX LTD (2026). QENEX LAB: A Self-Contained Quantum Chemistry
Platform with Reproducible Self-Assessed Compliance and Cryptographic
Provenance Bundles. ChemRxiv preprint, DOI: <v2-pending>.
Verifier subset: https://github.com/qenex-ai/qenex-verifier
```

\vspace{1em}
\hrule
\vspace{0.3em}
\footnotesize \textit{This summary card and the full Supplementary
Information PDF are reviewer-facing artefacts for the QENEX LAB v2
preprint. The companion verifier subset is licensed under Apache 2.0;
the full QENEX LAB platform is the proprietary asset of QENEX LTD.}
