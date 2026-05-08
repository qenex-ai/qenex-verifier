---
title: |
  QENEX LAB v2 — Supplementary Information
subtitle: |
  Reproducer subset, claim manifest, v1→v2 reconciliation, changelog
author:
  - QENEX LTD
date: "2026-05-08"
abstract: |
  This document collates the reviewer-facing materials for the QENEX
  LAB v2 paper (ChemRxiv preprint, May 2026).  It contains: the v2
  abstract draft; the supplementary materials manifest mapping every
  numeric claim in the abstract to a single reproducer command and
  the source files that implement it; the v1→v2 numeric reconciliation
  (a transparent diff between the v1 March-2026 preprint claims and
  the v2 numbers as produced by deterministic reproducer scripts);
  and the changelog for the open-source verifier subset that
  accompanies this paper at https://github.com/qenex-ai/qenex-verifier.
documentclass: article
geometry:
  - margin=2.2cm
  - top=2.5cm
fontsize: 10pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
linkcolor: blue
urlcolor: blue
toccolor: black
colorlinks: true
toc: true
toc-depth: 2
numbersections: true
papersize: a4
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{QENEX LAB v2 — Supplementary Information}
  - \fancyhead[R]{\thepage}
  - \fancyfoot[L]{\small QENEX LTD · 2026 · Apache 2.0 (verifier subset)}
  - \fancyfoot[R]{\small \today}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \renewcommand{\footrulewidth}{0.2pt}
  - \usepackage{microtype}
  - \usepackage{longtable}
  - \usepackage{booktabs}
  - \usepackage{fvextra}
  - \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\},fontsize=\small}
  - \fvset{breaklines=true,fontsize=\small}
---
