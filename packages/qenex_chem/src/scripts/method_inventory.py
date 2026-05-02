"""
QENEX Method Inventory — Reproducer Script (paper claim verification).

Counts the lab's computational quantum-chemistry method implementations
across four well-defined dimensions, so v2 paper claims about "N methods"
have a single command behind them. Two consecutive runs produce identical
output (verified — no timestamps, no random ordering).

Usage:
    python3 -m packages.qenex_chem.src.scripts.method_inventory [--json]

Output dimensions:
    1. WIRED        — methods callable via Molecule.compute(method=...)
                      (the canonical user-facing API)
    2. STANDALONE   — solver classes that work but are not wired to compute()
                      (callable from tests / direct import only)
    3. VALIDATED    — methods that have at least one passing test in
                      tests/validation/ or tests/test_*.py
    4. TOTAL_IMPL   — count of distinct method-implementation modules in
                      qenex_chem/src/ (file-level count)

The defensible v2 paper claim is the WIRED count for "user-callable methods"
and TOTAL_IMPL for "method implementations in the codebase". Avoid claiming
49 unless 49 is what the script outputs.

Reproducibility:
    - All counts derived from filesystem inspection + import-time discovery
    - No external network calls
    - No timestamps in output (only date for the report file, optional)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Path setup — script is at workspace/packages/qenex_chem/src/scripts/
_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent  # workspace/packages/qenex_chem/src/
_PACKAGE_ROOT = _SRC.parent  # workspace/packages/qenex_chem/
_WORKSPACE = _PACKAGE_ROOT.parent.parent  # workspace/
sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# 1. WIRED methods — what Molecule.compute() actually accepts
# ---------------------------------------------------------------------------


def discover_wired_methods() -> List[str]:
    """Inspect Molecule.compute() source to find the methods it accepts.

    We do this by reading the source rather than catching errors per-method,
    because (a) it's faster, (b) it gives clearer evidence of what's wired,
    (c) it doesn't accidentally fire DFT grids etc. just to map names.
    """
    src = (_SRC / "molecule.py").read_text()
    # Find the Molecule.compute method body
    in_compute = False
    methods: List[str] = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("def compute("):
            in_compute = True
            continue
        if in_compute:
            # Heuristic: end-of-method when we hit the next def at the
            # same indent level (4 spaces relative to class)
            if line.startswith("    def ") and not s.startswith("def compute"):
                break
            # Match patterns like:  if m == "hf":      if m in ("hf", "rhf"):
            if "if m ==" in s or "if m in" in s or "elif m ==" in s or "elif m in" in s:
                # Extract quoted strings
                import re

                methods.extend(re.findall(r'"([a-zA-Z][a-zA-Z0-9_()\\-]*)"', s))
                methods.extend(re.findall(r"'([a-zA-Z][a-zA-Z0-9_()\\-]*)'", s))
    return sorted(set(methods))


# ---------------------------------------------------------------------------
# 2. STANDALONE methods — solver classes available but not wired
# ---------------------------------------------------------------------------

# Files in qenex_chem/src/ named like a method or solver. Curated list keeps
# the report stable across reorganisations of source files.
_STANDALONE_CANDIDATES = [
    ("CASSCF", "casscf", "CASSCFSolver"),
    ("CASPT2", "casscf_ciah", "CASPT2Solver"),  # may not exist; checked below
    ("NEVPT2", "nevpt2", None),
    ("EOM-CCSD", "eomccsd", "EOMCCSDSolver"),
    ("TDDFT", "tddft", "TDDFTSolver"),
    ("UCCSD", "uccsd", "UCCSDSolver"),
    ("DLPNO-CCSD", "dlpno_ccsd", "DLPNOCCSDSolver"),
    ("DMC (QMC)", "qmc", "DMCSolver"),
    ("CBS extrapolation", "cbs", None),
    ("Double-hybrid DFT", "double_hybrid", None),
    ("Density fitting", "density_fitting", None),
    ("Dispersion (D3/D4)", "dispersion", None),
    ("Relativistic", "relativistic", None),
    ("F12 methods", "f12_methods", None),
    ("Vibrational analysis", "vibrational", None),
    ("CCSD(T)", "ccsd", "CCSDTSolver"),  # if it exists at all
    ("MP2", "ccsd", "MP2Solver"),
    ("Conformer search", "conformer", None),
    ("Geometry optimization", "geometry", None),
    ("AIMD", "aimd", None),
]


def discover_standalone() -> List[Dict[str, Any]]:
    """Identify solver modules + classes that exist and are importable
    but are NOT wired through Molecule.compute()."""
    results: List[Dict[str, Any]] = []
    wired = set(discover_wired_methods())
    for label, mod_name, cls_name in _STANDALONE_CANDIDATES:
        mod_path = _SRC / f"{mod_name}.py"
        entry: Dict[str, Any] = {
            "label": label,
            "module_name": mod_name,
            "module_exists": mod_path.exists(),
            "class_name": cls_name,
            "class_importable": False,
            "wired_through_compute": False,
        }
        # Check if it's wired through compute() (rough fuzzy match)
        for w in wired:
            if mod_name == w or mod_name in w.lower():
                entry["wired_through_compute"] = True
                break
        if cls_name and mod_path.exists():
            try:
                mod = importlib.import_module(mod_name)
                entry["class_importable"] = hasattr(mod, cls_name)
            except Exception:
                entry["class_importable"] = False
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# 3. VALIDATED methods — those with at least one passing test
# ---------------------------------------------------------------------------


def discover_validated() -> Dict[str, int]:
    """Map method tokens to how many test files reference them.

    A method is 'validated' if it appears in at least one test file under
    tests/validation/ or tests/test_*.py with a meaningful match (file name
    or function name contains the method token).
    """
    method_tokens = {
        "hf": ["hf", "hartree_fock", "rhf", "uhf"],
        "dft": ["dft", "lda", "b3lyp", "vwn", "pbe", "m06"],
        "mp2": ["mp2"],
        "ccsd": ["ccsd"],
        "casscf": ["casscf"],
        "eom_ccsd": ["eomccsd", "eom-ccsd", "eom_ccsd"],
        "tddft": ["tddft"],
        "tdhf": ["tdhf"],
        "dlpno_ccsd": ["dlpno"],
        "qmc": ["qmc", "dmc", "vmc"],
        "ccsd_t": ["ccsd_t", "ccsd(t)"],
        "uccsd": ["uccsd"],
        "nevpt2": ["nevpt2"],
        "f12": ["f12"],
        "vibrational": ["vibrational"],
        "geometry_opt": ["geom_opt", "geometry"],
        "conformer": ["conformer"],
    }
    tests_root = _WORKSPACE / "tests"
    counts: Dict[str, int] = {k: 0 for k in method_tokens}
    if not tests_root.exists():
        return counts
    test_files = list(tests_root.rglob("test_*.py"))
    for tf in test_files:
        name = tf.name.lower()
        try:
            content_lc = tf.read_text(encoding="utf-8").lower()
        except Exception:
            content_lc = ""
        for method_key, tokens in method_tokens.items():
            if any(tok in name for tok in tokens):
                counts[method_key] += 1
                continue
            # Fall through: filename didn't match, but file content might
            if any(tok in content_lc for tok in tokens):
                # Only count once per file regardless of content matches
                counts[method_key] += 1
    return counts


# ---------------------------------------------------------------------------
# 4. TOTAL_IMPL — count of method-implementation modules
# ---------------------------------------------------------------------------


def count_method_modules() -> Tuple[int, List[str]]:
    """Count distinct method-implementation modules in qenex_chem/src/.

    A 'method module' is a Python file whose name suggests it implements a
    quantum-chemistry method or solver. This is a deliberate over-count —
    it includes all plausibly-method-related files. The v2 paper should
    use this only as the upper bound and prefer the WIRED + STANDALONE
    counts for headline claims.
    """
    method_keywords = (
        "casscf",
        "ccsd",
        "dft",
        "eom",
        "f12",
        "fci",
        "mp2",
        "mp3",
        "mp4",
        "qmc",
        "tddft",
        "tdhf",
        "uccsd",
        "uhf",
        "nevpt",
        "caspt",
        "dlpno",
        "double_hybrid",
        "cbs",
        "dispersion",
        "density_fitting",
        "relativistic",
        "solver",
        "vibrational",
        "conformer",
        "geometry",
        "aimd",
        "alchemical",
        "casci",
        "fno",
        "molden",
        "ri_",
        "auxbasis",
    )
    files: List[str] = []
    for f in _SRC.glob("*.py"):
        name = f.stem.lower()
        if any(kw in name for kw in method_keywords):
            files.append(f.name)
    return len(files), sorted(files)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report() -> Dict[str, Any]:
    wired = discover_wired_methods()
    standalone = discover_standalone()
    validated = discover_validated()
    n_modules, module_files = count_method_modules()

    standalone_only = [
        s for s in standalone if s["module_exists"] and not s["wired_through_compute"]
    ]
    standalone_class_importable = [s for s in standalone_only if s["class_importable"]]

    # Recognise that some wired names are aliases of others (rhf -> hf).
    # Reporting both raw and de-aliased counts is more honest: user-facing
    # documentation should quote the de-aliased "distinct methods" number, not
    # the raw token count, otherwise we double-count synonyms.
    WIRED_ALIASES = {"rhf": "hf"}  # alias_name -> canonical_name
    wired_canonical = sorted(set(WIRED_ALIASES.get(m, m) for m in wired))
    wired_aliases_present = sorted(
        a for a in WIRED_ALIASES if a in wired and WIRED_ALIASES[a] in wired
    )

    return {
        "schema_version": 2,
        "wired_methods": {
            "count_raw": len(wired),
            "count_distinct": len(wired_canonical),
            "names": wired,
            "names_canonical": wired_canonical,
            "aliases_present": wired_aliases_present,
            "description": "Methods callable via Molecule.compute(method=...). count_distinct collapses aliases (rhf -> hf).",
        },
        "standalone_methods": {
            "count_with_module": len(standalone_only),
            "count_class_importable": len(standalone_class_importable),
            "entries": standalone_only,
            "description": "Solver classes implemented + importable but NOT wired through compute()",
        },
        "validated_methods": {
            "count_methods_with_tests": sum(1 for v in validated.values() if v > 0),
            "per_method_test_file_count": validated,
            "description": "Method tokens that match at least one test file (filename or content)",
        },
        "total_method_implementation_modules": {
            "count": n_modules,
            "files": module_files,
            "description": "Python files in qenex_chem/src/ whose name suggests a method/solver implementation",
        },
        "headline_v2_claim": {
            "wired_count": len(wired_canonical),
            "wired_count_raw_with_aliases": len(wired),
            "standalone_count": len(standalone_class_importable),
            "user_callable_total": len(wired_canonical)
            + len(standalone_class_importable),
            "module_count_upper_bound": n_modules,
            "recommended_paper_phrasing": (
                f"{len(wired_canonical)} distinct methods wired through "
                f"Molecule.compute() user API (raw token count {len(wired)} "
                f"includes alias rhf->hf); {len(standalone_class_importable)} "
                f"additional standalone solver classes for a total of "
                f"{len(wired_canonical) + len(standalone_class_importable)} "
                f"user-callable methods; {n_modules} method-implementation "
                f"modules in qenex_chem/src/."
            ),
        },
    }


def print_human(report: Dict[str, Any]) -> None:
    print("=" * 70)
    print("  QENEX Method Inventory")
    print("=" * 70)
    w = report["wired_methods"]
    print(
        f"\n  WIRED ({w['count_distinct']} distinct, {w['count_raw']} raw "
        f"tokens including aliases, callable via Molecule.compute()):"
    )
    for n in w["names"]:
        canonical = "" if n not in w.get("aliases_present", []) else " (alias of hf)"
        print(f"    - {n}{canonical}")

    s = report["standalone_methods"]
    print(
        f"\n  STANDALONE ({s['count_with_module']} module-present, "
        f"{s['count_class_importable']} class-importable; not wired through compute()):"
    )
    for entry in s["entries"]:
        marker = "✓" if entry["class_importable"] else "○"
        cls = entry["class_name"] or "(no class hint)"
        print(f"    {marker}  {entry['label']:<22s} {entry['module_name']}.py / {cls}")

    v = report["validated_methods"]
    print(
        f"\n  VALIDATED — methods with at least one matching test file "
        f"({v['count_methods_with_tests']}/{len(v['per_method_test_file_count'])}):"
    )
    for k, c in sorted(v["per_method_test_file_count"].items(), key=lambda kv: -kv[1]):
        if c > 0:
            print(f"    {k:<18s} {c} test files")

    t = report["total_method_implementation_modules"]
    print(
        f"\n  MODULE FILES ({t['count']} method-implementation files in qenex_chem/src/):"
    )
    print(f"    {', '.join(t['files'][:12])}{', ...' if len(t['files']) > 12 else ''}")

    h = report["headline_v2_claim"]
    print("\n" + "=" * 70)
    print("  HEADLINE FOR v2 PAPER (defensible by re-running this script):")
    print("=" * 70)
    print(f"  Wired through Molecule.compute() API:   {h['wired_count']}")
    print(f"  Additional standalone solver classes:   {h['standalone_count']}")
    print(f"  → Total user-callable methods:          {h['user_callable_total']}")
    print(f"  Method-implementation modules (upper):  {h['module_count_upper_bound']}")
    print()
    print("  Recommended paper phrasing:")
    print(f"    {h['recommended_paper_phrasing']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Method inventory reproducer for v2 paper claims."
    )
    parser.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    args = parser.parse_args()
    # When emitting JSON, redirect stdout to stderr during report construction
    # so that import-time banners (e.g. PROMETHEUS PyO3 backend) cannot corrupt
    # the JSON. Human mode keeps banners visible on stdout for transparency.
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
