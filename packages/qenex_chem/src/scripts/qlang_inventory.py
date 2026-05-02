"""
Q-Lang Inventory — Reproducer Script (paper claim verification).

Counts the canonical Q-Lang implementation (v04) and the broader Q-Lang
package's surface area, so v2 paper claims about "Q-Lang lines/modules/
domains" have a single command behind them.

The v1 paper claimed:
    "Q-Lang 2.0 unified scientific language (31,885 lines, 34 modules,
     6 simulation domains)."

The v2 paper should reflect what's actually there NOW:
    - Canonical implementation: workspace/packages/qenex-qlang/src/v04/
    - Total Q-Lang Python lines (live, non-archived)
    - Currently-registered simulate domains (not aspirational)
    - Number of canonical (v04) modules vs broader package

Two consecutive runs produce identical output.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent  # workspace/packages/qenex_chem/src/
_PACKAGE_ROOT = _SRC.parent  # workspace/packages/qenex_chem/
_WORKSPACE = _PACKAGE_ROOT.parent.parent  # workspace/
_QLANG = _WORKSPACE / "packages" / "qenex-qlang"
_QLANG_SRC = _QLANG / "src"
_QLANG_V04 = _QLANG_SRC / "v04"


def count_python_files(root: Path, exclude_archive: bool = True) -> Dict[str, Any]:
    """Walk a directory, count .py files and total lines."""
    if not root.exists():
        return {"files": 0, "lines": 0, "file_list": []}
    files: List[str] = []
    total_lines = 0
    for f in root.rglob("*.py"):
        if "__pycache__" in f.parts:
            continue
        if exclude_archive and ("archive" in f.parts or "_archive" in f.parts):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                lines = sum(1 for _ in fh)
        except Exception:
            lines = 0
        rel = str(f.relative_to(root))
        files.append(rel)
        total_lines += lines
    return {
        "files": len(files),
        "lines": total_lines,
        "file_list": sorted(files),
    }


def discover_registered_domains() -> List[str]:
    """Inspect simulate_dispatch_v04.py source to find which domains are
    actually registered via register_domain() at module load time.

    We do this by static analysis of the source file rather than importing
    and running, to keep the inventory deterministic and side-effect-free.
    """
    f = _QLANG_V04 / "simulate_dispatch_v04.py"
    if not f.exists():
        return []
    src = f.read_text(encoding="utf-8")
    pattern = re.compile(r'register_domain\(\s*"([^"]+)"', re.MULTILINE)
    return sorted(set(pattern.findall(src)))


def discover_v04_modules() -> List[str]:
    """List the canonical v04 module files (not subfolders)."""
    if not _QLANG_V04.exists():
        return []
    return sorted(f.name for f in _QLANG_V04.glob("*.py"))


def count_qlang_examples() -> Dict[str, Any]:
    """Count .ql / .qlang example programs distributed with the package."""
    candidates = [
        _QLANG / "examples",
        _WORKSPACE / "examples" / "qlang",
    ]
    total = 0
    locations: List[Dict[str, Any]] = []
    for c in candidates:
        if not c.exists():
            continue
        files = list(c.rglob("*.ql")) + list(c.rglob("*.qlang"))
        files = [f for f in files if "__pycache__" not in f.parts]
        if files:
            locations.append(
                {
                    "path": str(c.relative_to(_WORKSPACE)),
                    "count": len(files),
                }
            )
            total += len(files)
    return {"total": total, "locations": locations}


def discover_archive_implementations() -> Dict[str, Any]:
    """Identify Q-Lang archived/legacy implementations.

    These are present-but-deprecated: the lab decided in May 2026 to make
    v04 canonical and stop extending the others. The v2 paper should
    mention these honestly so the reader knows v04 is one of multiple
    implementations preserved in the repo.
    """
    archive_dir = _QLANG_SRC / "archive" / "legacy"
    archived = []
    if archive_dir.exists():
        for f in archive_dir.glob("*.py"):
            if f.name.startswith("__"):
                continue
            archived.append(f.name)
    # Also note the orphan TS interpreter we archived
    ts_archive = _WORKSPACE.parent / "_archive" / "qlang-ts-interpreter-2026"
    return {
        "python_legacy_files": sorted(archived),
        "ts_orphan_archived": ts_archive.exists(),
    }


def build_report() -> Dict[str, Any]:
    full_pkg = count_python_files(_QLANG_SRC, exclude_archive=False)
    live_pkg = count_python_files(_QLANG_SRC, exclude_archive=True)
    v04_only = count_python_files(_QLANG_V04, exclude_archive=True)
    domains = discover_registered_domains()
    examples = count_qlang_examples()
    archive = discover_archive_implementations()
    return {
        "schema_version": 1,
        "canonical_v04": {
            "path": str(_QLANG_V04.relative_to(_WORKSPACE)),
            "modules": v04_only["files"],
            "lines": v04_only["lines"],
            "module_files": discover_v04_modules(),
        },
        "qenex_qlang_package": {
            "live_modules": live_pkg["files"],
            "live_lines": live_pkg["lines"],
            "all_modules_including_archive": full_pkg["files"],
            "all_lines_including_archive": full_pkg["lines"],
        },
        "registered_simulate_domains": {
            "count": len(domains),
            "names": domains,
        },
        "examples": examples,
        "archive": archive,
        "headline_v2": {
            "v04_modules": v04_only["files"],
            "v04_lines": v04_only["lines"],
            "live_total_modules": live_pkg["files"],
            "live_total_lines": live_pkg["lines"],
            "registered_domains_count": len(domains),
            "registered_domains_list": domains,
            "recommended_paper_phrasing": (
                f"Canonical Q-Lang implementation (v04) at "
                f"{_QLANG_V04.relative_to(_WORKSPACE)}: "
                f"{v04_only['files']} modules, {v04_only['lines']:,} lines. "
                f"Total live qenex-qlang package (v04 + tooling, excluding "
                f"archive): {live_pkg['files']} modules, "
                f"{live_pkg['lines']:,} lines. "
                f"{len(domains)} simulation domains currently registered "
                f"({', '.join(domains) if domains else 'none'}). "
                f"{examples['total']} example .ql programs distributed."
            ),
        },
    }


def print_human(report: Dict[str, Any]) -> None:
    print("=" * 70)
    print("  Q-Lang Inventory")
    print("=" * 70)

    v = report["canonical_v04"]
    p = report["qenex_qlang_package"]
    d = report["registered_simulate_domains"]
    e = report["examples"]
    a = report["archive"]

    print(f"\n  Canonical v04 ({v['path']}):")
    print(f"    Modules:    {v['modules']}")
    print(f"    Lines:      {v['lines']:,}")
    print(
        f"    Files:      {', '.join(v['module_files'][:6])}{'...' if len(v['module_files']) > 6 else ''}"
    )

    print(f"\n  Full qenex-qlang package (live, excluding archive):")
    print(f"    Modules:    {p['live_modules']}")
    print(f"    Lines:      {p['live_lines']:,}")

    print(f"\n  Including archive/legacy:")
    print(f"    Modules:    {p['all_modules_including_archive']}")
    print(f"    Lines:      {p['all_lines_including_archive']:,}")

    print(f"\n  Registered simulate domains ({d['count']}):")
    if d["names"]:
        for n in d["names"]:
            print(f"    - {n}")
    else:
        print("    (none — domains registered at runtime not detected statically)")

    print(f"\n  Example programs distributed: {e['total']} (.ql / .qlang)")
    for loc in e["locations"]:
        print(f"    {loc['count']} in {loc['path']}/")

    if a["python_legacy_files"]:
        print(f"\n  Archive (preserved but deprecated):")
        print(f"    {len(a['python_legacy_files'])} files in archive/legacy/")
        if a["ts_orphan_archived"]:
            print(
                f"    + TypeScript orphan interpreter at _archive/qlang-ts-interpreter-2026/"
            )

    h = report["headline_v2"]
    print()
    print("=" * 70)
    print("  HEADLINE FOR v2 PAPER:")
    print("=" * 70)
    print(f"  Canonical v04 modules:        {h['v04_modules']}")
    print(f"  Canonical v04 lines:          {h['v04_lines']:,}")
    print(f"  Live package modules:         {h['live_total_modules']}")
    print(f"  Live package lines:           {h['live_total_lines']:,}")
    print(
        f"  Registered simulate domains:  {h['registered_domains_count']} ({', '.join(h['registered_domains_list'])})"
    )
    print()
    print("  Recommended paper phrasing:")
    print(f"    {h['recommended_paper_phrasing']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Q-Lang inventory reproducer.")
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
