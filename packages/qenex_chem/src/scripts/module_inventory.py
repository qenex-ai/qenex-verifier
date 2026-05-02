"""
QENEX Module Inventory — Reproducer Script (paper claim verification).

Counts the lab's module ecosystem along well-defined dimensions so v2
paper claims have a single command behind them. Two consecutive runs
produce identical output.

Usage:
    python3 packages/qenex_chem/src/scripts/module_inventory.py [--json]

Categories reported (curated, stable, every entry justified):

  drug_discovery
        Drug-discovery modules in qenex_chem/src/. The v1 claim of
        "21 drug discovery modules" is reproduced here as a transparent
        list, not a number.

  qm_methods
        Quantum-chemistry method/solver modules (HF, DFT, CCSD, ...)

  qm_infrastructure
        Basis sets, integrals, geometry, common QM machinery.

  scientific_kernels
        Domain-specific scientific computation outside the chemistry
        package: physics, biology, climate, neuroscience, astrophysics,
        tissue, mathematics. Counted by package, not by file.

  cross_domain
        Modules that explicitly bridge multiple domains.

  ip_provenance
        Modules supporting reproducibility, provenance, certification,
        patents — the "auditable science" infrastructure.

  ai_orchestration
        Modules that wire AI / agents / FDSP / decomposition into the
        scientific stack.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent  # workspace/packages/qenex_chem/src/
_PACKAGE_ROOT = _SRC.parent  # workspace/packages/qenex_chem/
_WORKSPACE = _PACKAGE_ROOT.parent.parent  # workspace/

# Categorization rules. Every file under qenex_chem/src/ is assigned to
# exactly one category by the first matching rule. Files matching no rule
# are placed in 'other' for review.
_CATEGORY_RULES: List = [
    (
        "drug_discovery",
        re.compile(
            r"^("
            r"admet|assay_data|binding|chemical_space|clinical_predictor|"
            r"crisis_molecules|docking|drug_|lead_optimization|"
            r"materials_discovery_pipeline|pharmacophore|pkpd|polypharmacology|"
            r"retrosynthesis|sar|schwarz_screening|screening_library|toxicity|"
            r"virtual_screening"
            r")\.py$"
        ),
    ),
    (
        "qm_methods",
        re.compile(
            r"^("
            r"casscf|ccsd|dft|eomccsd|f12_methods|nevpt2|qmc|tddft|uccsd|"
            r"dlpno_ccsd|cbs|double_hybrid|density_fitting|dispersion|"
            r"relativistic|solver|auto_method"
            r")\.py$|^casscf_"
        ),
    ),
    (
        "qm_infrastructure",
        re.compile(
            r"^("
            r"basis_|molecule|integrals|aimd|alchemical_md|conformer|"
            r"geometry|enhanced_sampling|defect_transport|md|vibrational"
            r")\.py$|^basis_"
        ),
    ),
    (
        "ip_provenance",
        re.compile(
            r"^("
            r"auto_patent|certification|discovery_ledger|ip_landscape|"
            r"patent_prior_art|provenance|regulatory_engine|regulatory_package|"
            r"regulatory_presub|reproducibility_engine|standards_audit|"
            r"standards_evaluator|wipo_pdf|tamper_proof"
            r")\.py$"
        ),
    ),
    (
        "ai_orchestration",
        re.compile(
            r"^("
            r"agent_science_bridge|causal_engine|continuous_engine|"
            r"continuous_discovery|cross_domain|cross_domain_pipeline|"
            r"discovery_dashboard|discovery_engine|discovery_handoff|"
            r"drug_discovery_orchestrator|enterprise_api|experiment_designer|"
            r"hypothesis_engine|inverse_design|materials_genome|molecule_database|"
            r"nlq|paper_generator|polyglot_bridge|qenex_cache|qenex_parallel|"
            r"qenex_precision|rest_server|result_explainer|science_agent|"
            r"targeted_discovery|drug_pipeline|drug_repurposing|drug_targets"
            r")\.py$"
        ),
    ),
    (
        "evaluation",
        re.compile(
            r"^("
            r"accuracy_report|benchmark|benchmark_export|competitive_benchmark|"
            r"cross_validation|evaluator|standards_audit|novelty"
            r")\.py$"
        ),
    ),
    (
        "utility",
        re.compile(
            r"^(ascii_visualizer|wipo_pdf|lab_report|prometheus_backend|"
            r"reaction|reaction_simulator|diffusion_molecule|manufacturing|"
            r"cost_estimator|materials_genome|materials_discovery|"
            r"qenex_chem_legacy|crisis_molecules)\.py$"
        ),
    ),
]


def categorize_file(filename: str) -> str:
    for category, rule in _CATEGORY_RULES:
        if rule.search(filename):
            return category
    return "other"


def inventory_qenex_chem() -> Dict[str, Any]:
    files = sorted(f.name for f in _SRC.glob("*.py") if f.name != "__init__.py")
    by_category: Dict[str, List[str]] = {}
    for f in files:
        cat = categorize_file(f)
        by_category.setdefault(cat, []).append(f)
    counts = {cat: len(fs) for cat, fs in by_category.items()}
    return {
        "total_files": len(files),
        "counts_by_category": dict(sorted(counts.items())),
        "files_by_category": {cat: by_category[cat] for cat in sorted(by_category)},
    }


def inventory_workspace_packages() -> Dict[str, Any]:
    """Count Python module files in each top-level workspace package."""
    pkg_root = _WORKSPACE / "packages"
    if not pkg_root.exists():
        return {"packages": [], "total_modules": 0}
    packages = []
    total = 0
    for pkg_dir in sorted(pkg_root.iterdir()):
        if not pkg_dir.is_dir():
            continue
        # Look for src/ subdir (most QENEX packages use this)
        src = pkg_dir / "src"
        candidates = [src, pkg_dir]
        py_files = 0
        chosen = None
        for c in candidates:
            if c.exists():
                files = [
                    p
                    for p in c.rglob("*.py")
                    if "__pycache__" not in p.parts
                    and "_archive" not in p.parts
                    and "archive" not in p.parts
                ]
                if files:
                    py_files = len(files)
                    chosen = c
                    break
        packages.append(
            {
                "name": pkg_dir.name,
                "src_path": str(chosen.relative_to(_WORKSPACE)) if chosen else None,
                "py_modules": py_files,
            }
        )
        total += py_files
    return {"packages": packages, "total_modules": total}


def build_report() -> Dict[str, Any]:
    chem = inventory_qenex_chem()
    ws = inventory_workspace_packages()
    drug = chem["files_by_category"].get("drug_discovery", [])
    return {
        "schema_version": 1,
        "qenex_chem": chem,
        "workspace_packages": ws,
        "headline_v2": {
            "drug_discovery_modules": len(drug),
            "drug_discovery_files": drug,
            "qenex_chem_total_files": chem["total_files"],
            "workspace_total_python_modules": ws["total_modules"],
            "recommended_paper_phrasing": (
                f"{len(drug)} drug-discovery modules in qenex_chem/src/; "
                f"{chem['total_files']} total files in the chemistry package; "
                f"{ws['total_modules']} Python modules across "
                f"{len(ws['packages'])} top-level workspace packages."
            ),
        },
    }


def print_human(report: Dict[str, Any]) -> None:
    print("=" * 70)
    print("  QENEX Module Inventory")
    print("=" * 70)

    h = report["headline_v2"]
    print(
        f"\n  qenex_chem/src/      {report['qenex_chem']['total_files']} Python files"
    )
    print(
        f"  workspace total      {h['workspace_total_python_modules']} Python modules across {len(report['workspace_packages']['packages'])} packages"
    )

    print(f"\n  By category in qenex_chem/src/:")
    for cat, n in report["qenex_chem"]["counts_by_category"].items():
        print(f"    {cat:<22s} {n}")

    print(f"\n  Drug discovery modules ({len(h['drug_discovery_files'])}):")
    for f in h["drug_discovery_files"]:
        print(f"    {f}")

    print(f"\n  Workspace packages:")
    for p in report["workspace_packages"]["packages"]:
        print(f"    {p['name']:<25s} {p['py_modules']:>4} modules  ({p['src_path']})")

    print()
    print("=" * 70)
    print("  HEADLINE FOR v2 PAPER:")
    print("=" * 70)
    print(f"  Drug discovery modules:       {h['drug_discovery_modules']}")
    print(f"  qenex_chem total files:       {h['qenex_chem_total_files']}")
    print(f"  Workspace total modules:      {h['workspace_total_python_modules']}")
    print()
    print("  Recommended phrasing:")
    print(f"    {h['recommended_paper_phrasing']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Module inventory reproducer.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    # When emitting JSON, redirect stdout to stderr during report construction
    # so import-time banners cannot corrupt the JSON output.
    if args.json:
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            report = build_report()
        finally:
            sys.stdout = _real_stdout
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        report = build_report()
        print_human(report)


if __name__ == "__main__":
    main()
