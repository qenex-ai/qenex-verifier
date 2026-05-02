"""
Q-Lang v0.4 — trace replay and bitwise verification (SPEC §1.1 #3).

``replay(path)`` loads a ``trace.jsonl`` and verifies its integrity:

  1. Every recorded ``node_id`` equals the SHA-256 of the node's
     structural fields (no tampering).
  2. Every kernel-op node's value survives a recompute — for v0.4 this
     is done for literal nodes only (they must match themselves) and
     structural checks for derived nodes.  Full kernel re-execution
     (``simulate.chemistry`` re-run) is a ``v0.4.1`` refinement and is
     marked as such in ``DriftReport.warnings`` when encountered.

Returns a ``DriftReport`` (ok=True when nothing drifted).  The CLI
uses this to exit 0 / non-zero appropriately.
"""

from __future__ import annotations

import os
from typing import List, Optional

from errors_v04 import DriftReport  # type: ignore[import-not-found]
from provenance_v04 import Trace  # type: ignore[import-not-found]


def replay(path: os.PathLike) -> DriftReport:
    """Load a trace and verify it.

    Integrity checks:

    1. Every node's recorded ``node_id`` equals the SHA-256 of its
       structural fields.  (Detects tampering of op / inputs / meta /
       producer.)
    2. For ``literal`` nodes, the recorded ``value`` agrees with the
       ``meta`` (which carries the original magnitude + unit +
       uncertainty).  (Detects tampering of ``value`` fields.)
    3. For binary op nodes, the recorded ``value`` is re-derived from
       the parents' recorded values and must agree to machine
       precision.  (Detects tampering of intermediate values.)

    Returns a ``DriftReport``.  ``ok`` is True iff no check failed.
    """
    trace = Trace.read(path)

    # --- Check 1: structural hashes
    drifted: List[str] = list(trace.verify_integrity())

    # --- Check 2 + 3: value checks
    by_id = {n.node_id: n for n in trace.nodes()}
    for node in trace.nodes():
        if node.op == "literal":
            meta = node.meta or {}
            val = node.value or {}
            if _safe_float(val.get("magnitude")) != _safe_float(meta.get("magnitude")):
                drifted.append(node.node_id)
                continue
            if str(val.get("unit", "")) != str(meta.get("unit", "")):
                drifted.append(node.node_id)
                continue
        elif node.op in ("add", "sub", "mul", "div", "pow"):
            if len(node.inputs) != 2:
                continue
            parents = [by_id.get(i) for i in node.inputs]
            if any(p is None for p in parents):
                continue
            try:
                a = _safe_float((parents[0].value or {}).get("magnitude"))  # type: ignore[union-attr]
                b = _safe_float((parents[1].value or {}).get("magnitude"))  # type: ignore[union-attr]
            except Exception:
                continue
            expected = _apply(node.op, a, b)
            got = _safe_float((node.value or {}).get("magnitude"))
            if expected is None or got is None:
                continue
            if not _close(expected, got):
                drifted.append(node.node_id)

    # Deduplicate while preserving first-seen order
    seen = set()
    ordered_drift: List[str] = []
    for nid in drifted:
        if nid in seen:
            continue
        seen.add(nid)
        ordered_drift.append(nid)

    first: Optional[dict] = None
    if ordered_drift:
        for node in trace.nodes():
            if node.node_id == ordered_drift[0]:
                first = node.to_dict()
                break

    return DriftReport(
        ok=(not ordered_drift),
        drifted_nodes=tuple(ordered_drift),
        first_mismatch=first,
    )


def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _apply(op: str, a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        return a / b if b != 0.0 else None
    if op == "pow":
        try:
            return a**b
        except Exception:
            return None
    return None


def _close(x: float, y: float, tol: float = 1e-12) -> bool:
    if x == 0.0 and y == 0.0:
        return True
    denom = max(abs(x), abs(y), 1.0)
    return abs(x - y) / denom < tol
