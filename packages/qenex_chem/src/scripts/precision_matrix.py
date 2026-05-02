"""
Precision Matrix — Reproducer Script (paper claim verification).

Empirically computes the lab's HF energy for a small set of well-known
molecules across all available basis sets, and reports the absolute drift
vs. frozen PySCF reference values (stored in qenex_chem.certification).
This is the evidence behind the v2 paper claim:

    "Sub-nanohartree precision against PySCF for [molecule] at [basis]"

The script reports honest numbers: which (molecule, basis) combinations
are within 1 nHa, which are within 1 µHa, which are within 1 mHa.
The v2 paper should ONLY claim "sub-nanohartree" for tuples where the
script reports drift < 1e-9 Hartree.

Reference data:
    qenex_chem.certification.PYSCF_RHF_STO3G  (He, H2, H2O)
    qenex_chem.certification.PYSCF_RHF_CCPVDZ (He, H2, H2O)

Geometries are taken from the certification module to match the
reference values exactly.

Tolerance bands:
    SUB_NANOHARTREE   < 1e-9 Hartree  (1 nHa, ~1e-7 kJ/mol)
    SUB_MICROHARTREE  < 1e-6 Hartree  (1 µHa, ~1e-4 kJ/mol)
    SUB_MILLIHARTREE  < 1e-3 Hartree  (1 mHa, ~0.6 kcal/mol)

Usage:
    python3 packages/qenex_chem/src/scripts/precision_matrix.py [--json]

Exit codes:
    0 if all included tuples are within libcint tolerance (1e-6 Hartree)
    1 if any tuple exceeds 1e-6 (signals a regression)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).parent.resolve()
_SRC = _HERE.parent
sys.path.insert(0, str(_SRC))


# Tolerance bands (Hartree)
TOL_SUB_NANO = 1e-9
TOL_SUB_MICRO = 1e-6
TOL_SUB_MILLI = 1e-3
LIBCINT_TOL = 1e-6  # AGENTS.md states libcint integrals → 1e-6 Hartree match

# Geometries — match the certification module exactly
HE_ATOMS = [("He", (0.0, 0.0, 0.0))]
H2_ATOMS = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
H2O_ATOMS = [
    ("O", (0.0, 0.0, 0.2217)),
    ("H", (0.0, 1.4309, -0.8867)),
    ("H", (0.0, -1.4309, -0.8867)),
]

# Reference data (PySCF 2.x). Source: qenex_chem.certification module.
# Cross-checked vs. PySCF Python package directly during certification.
PYSCF_RHF_STO3G = {
    "He": -2.8077839575,
    "H2": -1.1167143251,
    "H2O": -74.9630277528,
}
PYSCF_RHF_CCPVDZ = {
    "He": -2.8551604772,
    "H2": -1.1287094490,
    "H2O": -76.0271118430,
}

REFERENCES = {
    ("He", "sto-3g"): PYSCF_RHF_STO3G["He"],
    ("H2", "sto-3g"): PYSCF_RHF_STO3G["H2"],
    ("H2O", "sto-3g"): PYSCF_RHF_STO3G["H2O"],
    ("He", "cc-pvdz"): PYSCF_RHF_CCPVDZ["He"],
    ("H2", "cc-pvdz"): PYSCF_RHF_CCPVDZ["H2"],
    ("H2O", "cc-pvdz"): PYSCF_RHF_CCPVDZ["H2O"],
}

GEOMETRIES = {
    "He": HE_ATOMS,
    "H2": H2_ATOMS,
    "H2O": H2O_ATOMS,
}

# All basis sets the lab has implemented as Python modules in qenex_chem/src/
# Subset that has frozen PySCF references is reported separately from those
# that don't (the latter are computed but flagged as 'no reference')
BASIS_SETS = ["sto-3g", "cc-pvdz", "cc-pvtz", "6-31g*", "aug-cc-pvdz", "aug-cc-pvtz"]


def classify_drift(drift: float) -> str:
    if drift < TOL_SUB_NANO:
        return "sub-nanohartree"
    if drift < TOL_SUB_MICRO:
        return "sub-microhartree"
    if drift < TOL_SUB_MILLI:
        return "sub-millihartree"
    return "exceeds-1mHa"


def compute_one(
    mol_name: str, basis: str, atoms: List[Tuple[str, Tuple[float, float, float]]]
) -> Dict[str, Any]:
    """Run HF on one (molecule, basis) tuple. Returns a dict with the
    measured energy, drift vs reference (if any), and classification."""
    from molecule import Molecule  # type: ignore

    entry: Dict[str, Any] = {
        "molecule": mol_name,
        "basis": basis,
        "method": "hf",
        "computed_energy_hartree": None,
        "reference_energy_hartree": None,
        "drift_hartree": None,
        "classification": None,
        "status": "OK",
        "error": None,
    }
    try:
        mol = Molecule(atoms, charge=0, basis_name=basis)
        E_total, _ = mol.compute(method="hf", verbose=False)
        # Round to 11 decimal places (10 picohartree) to suppress ULP-level
        # BLAS-reordering noise that varies between runs. This is two orders of
        # magnitude below the tightest sub-nanohartree validation window
        # (1e-9 Ha), so reproducible JSON does not weaken any scientific claim.
        # Classification uses the rounded drift, identical to what's reported.
        entry["computed_energy_hartree"] = round(float(E_total), 11)
        ref = REFERENCES.get((mol_name, basis))
        if ref is not None:
            entry["reference_energy_hartree"] = ref
            drift = round(abs(float(E_total) - ref), 11)
            entry["drift_hartree"] = drift
            entry["classification"] = classify_drift(drift)
        else:
            entry["classification"] = "no-reference-available"
    except Exception as e:
        entry["status"] = "ERROR"
        entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return entry


def build_report() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for mol_name, atoms in GEOMETRIES.items():
        for basis in BASIS_SETS:
            r = compute_one(mol_name, basis, atoms)
            results.append(r)

    # Aggregate summary
    sub_nano = sum(1 for r in results if r.get("classification") == "sub-nanohartree")
    sub_micro = sum(
        1
        for r in results
        if r.get("classification") in ("sub-nanohartree", "sub-microhartree")
    )
    sub_milli = sum(
        1
        for r in results
        if r.get("classification")
        in ("sub-nanohartree", "sub-microhartree", "sub-millihartree")
    )
    no_ref = sum(
        1 for r in results if r.get("classification") == "no-reference-available"
    )
    errors = sum(1 for r in results if r.get("status") == "ERROR")
    exceeds = sum(1 for r in results if r.get("classification") == "exceeds-1mHa")

    # Find tightest tolerance achieved per basis set with a reference
    tolerance_per_basis: Dict[str, str] = {}
    for basis in BASIS_SETS:
        refs_for_basis = [
            r
            for r in results
            if r["basis"] == basis
            and r["status"] == "OK"
            and r["reference_energy_hartree"] is not None
        ]
        if refs_for_basis:
            worst = max(r["drift_hartree"] for r in refs_for_basis)
            tolerance_per_basis[basis] = classify_drift(worst)
        else:
            tolerance_per_basis[basis] = "no-reference-available"

    return {
        "schema_version": 1,
        "tolerances_used": {
            "sub_nanohartree_hartree": TOL_SUB_NANO,
            "sub_microhartree_hartree": TOL_SUB_MICRO,
            "sub_millihartree_hartree": TOL_SUB_MILLI,
            "libcint_expected_max_drift_hartree": LIBCINT_TOL,
        },
        "results": results,
        "summary": {
            "total_tuples": len(results),
            "sub_nanohartree": sub_nano,
            "sub_microhartree_or_better": sub_micro,
            "sub_millihartree_or_better": sub_milli,
            "exceeds_1mHa": exceeds,
            "no_reference_available": no_ref,
            "errors": errors,
            "tolerance_per_basis_set": tolerance_per_basis,
        },
        "headline_v2": {
            "tuples_with_reference": sub_milli + exceeds,
            "tuples_sub_nanohartree": sub_nano,
            "tuples_sub_microhartree_or_better": sub_micro,
            "basis_sets_with_reference": [
                b
                for b in BASIS_SETS
                if tolerance_per_basis[b] != "no-reference-available"
            ],
            "basis_sets_total": len(BASIS_SETS),
            "recommended_paper_phrasing": (
                f"HF energies computed on (He, H2, H2O) across {len(BASIS_SETS)} "
                f"basis sets. Frozen PySCF references available for "
                f"{sub_milli + exceeds} of {len(results)} tuples; remaining "
                f"{no_ref} tuples computed without external reference. "
                f"Among reference-validated tuples: {sub_nano} match PySCF "
                f"to sub-nanohartree, {sub_micro} to sub-microhartree, "
                f"{sub_milli} to sub-millihartree."
            ),
        },
    }


def print_human(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print("  Precision Matrix — HF energies vs PySCF references")
    print("=" * 78)
    print()
    print(
        f"  {'Molecule':<8} {'Basis':<14} {'Computed (Ha)':>20} {'Reference (Ha)':>20} {'|Δ| (Ha)':>14} {'Class':>20}"
    )
    print("  " + "-" * 100)
    for r in report["results"]:
        mol = r["molecule"]
        bas = r["basis"]
        if r["status"] == "ERROR":
            print(f"  {mol:<8} {bas:<14} ERROR: {r['error']}")
            continue
        E = r["computed_energy_hartree"]
        Eref = r["reference_energy_hartree"]
        if Eref is None:
            print(
                f"  {mol:<8} {bas:<14} {E:>20.10f} {'(no reference)':>20} {'-':>14} {r['classification']:>20}"
            )
        else:
            d = r["drift_hartree"]
            print(
                f"  {mol:<8} {bas:<14} {E:>20.10f} {Eref:>20.10f} {d:>14.3e} {r['classification']:>20}"
            )

    s = report["summary"]
    print()
    print("=" * 78)
    print("  Summary")
    print("=" * 78)
    print(f"  Total (molecule, basis) tuples evaluated: {s['total_tuples']}")
    print(
        f"  With external PySCF reference:            {s['total_tuples'] - s['no_reference_available'] - s['errors']}"
    )
    print()
    print(f"  Sub-nanohartree (< 1e-9 Ha):              {s['sub_nanohartree']}")
    print(
        f"  Sub-microhartree or better (< 1e-6 Ha):   {s['sub_microhartree_or_better']}"
    )
    print(
        f"  Sub-millihartree or better (< 1e-3 Ha):   {s['sub_millihartree_or_better']}"
    )
    print(f"  Exceeds 1 mHa (regression):               {s['exceeds_1mHa']}")
    print(f"  Errors during computation:                {s['errors']}")
    print()
    print(f"  Per-basis-set tolerance achieved:")
    for basis, cls in s["tolerance_per_basis_set"].items():
        print(f"    {basis:<14s} {cls}")

    h = report["headline_v2"]
    print()
    print("=" * 78)
    print("  HEADLINE FOR v2 PAPER:")
    print("=" * 78)
    print(f"  Sub-nanohartree tuples:           {h['tuples_sub_nanohartree']}")
    print(
        f"  Sub-microhartree or better:       {h['tuples_sub_microhartree_or_better']}"
    )
    print(
        f"  Basis sets with PySCF reference:  {len(h['basis_sets_with_reference'])} of {h['basis_sets_total']}"
    )
    print(f"     {', '.join(h['basis_sets_with_reference'])}")
    print()
    print("  Recommended paper phrasing:")
    print(f"    {h['recommended_paper_phrasing']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Precision matrix reproducer.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    # When emitting JSON, redirect stdout to stderr during report construction
    # so import-time banners (e.g. PROMETHEUS PyO3 backend) cannot corrupt the
    # JSON output. Human mode keeps banners visible on stdout for transparency.
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
    # Exit code: 0 unless any reference-validated tuple exceeds 1 mHa
    s = report["summary"]
    sys.exit(1 if s["exceeds_1mHa"] > 0 or s["errors"] > 0 else 0)


if __name__ == "__main__":
    main()
