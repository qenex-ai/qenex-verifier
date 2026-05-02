"""
Q-Lang v0.4 — provenance DAG, content addressing, trace I/O (SPEC §7).

The centerpiece of the third uniqueness guarantee: every Q-Lang
value carries a ``DerivationNode`` in a content-addressed DAG.  The
full DAG can be serialised to a newline-delimited JSON file
(``trace.jsonl``), reloaded, and re-executed bitwise.

Content-addressing (SPEC §7.1)
-------------------------------
``node_id = SHA-256(canonical_json({op, sorted(input_ids), meta}))``

Identical inputs + op + meta → identical node_id.  This gives:

* deduplication of the DAG,
* tamper detection (any bit flip changes node_ids),
* reproducibility check (replay recomputes nodes and compares).

Format
------
Each line of ``trace.jsonl`` is a JSON object with exactly these keys:

    node_id    str (64-char SHA-256 hex)
    op         str
    inputs     list[str]  (parent node_ids, in build order)
    value      {"magnitude": float, "unit": str, "uncertainty": float|None}
               | None     (for nodes whose 'value' is not a Quantity)
    meta       dict       (op-specific fields; see §9 + §3.3)
    producer   str        ("qlang-v0.4" | "qenex_chem@<sha>" | ...)
    ts         str        (ISO-8601, with timezone Z)

Order of emission
-----------------
Nodes are written in topological order (leaves first).  The emitter
records nodes in the order they are created by the evaluator; the
evaluator builds them bottom-up, so that order is already topo.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# Producer string for nodes synthesized by the Q-Lang core itself.
# Kernel-op nodes override this with e.g. "qenex_chem@<git-sha>".
QLANG_PRODUCER = "qlang-v0.4"


# ─────────────────────────────────────────────────────────────────────
# DerivationNode
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DerivationNode:
    """One vertex in the provenance DAG.  Immutable by construction."""

    node_id: str  # 64-char SHA-256 hex
    op: str
    inputs: Tuple[str, ...] = ()
    value: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    producer: str = QLANG_PRODUCER
    ts: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form (used for trace.jsonl emission).
        The dict is exactly what goes on disk for one line."""
        return {
            "node_id": self.node_id,
            "op": self.op,
            "inputs": list(self.inputs),
            "value": _canonical_value(self.value),
            "meta": _canonical_meta(self.meta),
            "producer": self.producer,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DerivationNode":
        return cls(
            node_id=str(d["node_id"]),
            op=str(d["op"]),
            inputs=tuple(d.get("inputs") or ()),
            value=d.get("value"),
            meta=dict(d.get("meta") or {}),
            producer=str(d.get("producer") or QLANG_PRODUCER),
            ts=str(d.get("ts") or ""),
        )


# ─────────────────────────────────────────────────────────────────────
# Canonical JSON serialisation for SHA-256 addressing
# ─────────────────────────────────────────────────────────────────────


def _canonical_meta(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalise meta to a JSON-clean dict with stable key order.

    Values are passed through json.dumps/loads to strip any objects
    that are not natively serialisable (numpy scalars, Decimal, etc.)
    via string coercion — by design, meta is documentation, not
    load-bearing data.
    """
    clean: Dict[str, Any] = {}
    for k in sorted(meta):
        v = meta[k]
        clean[str(k)] = _coerce_json(v)
    return clean


def _canonical_value(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalise value dict keys into fixed order."""
    if value is None:
        return None
    out: Dict[str, Any] = {
        "magnitude": _coerce_json(value.get("magnitude")),
        "unit": _coerce_json(value.get("unit", "")),
    }
    unc = value.get("uncertainty", None)
    out["uncertainty"] = _coerce_json(unc) if unc is not None else None
    return out


def _coerce_json(v: Any) -> Any:
    """Best-effort coercion of Python objects into JSON-native types."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    # Decimal, numpy scalars, fractions.Fraction — preserve precision
    # via str().
    if hasattr(v, "dtype") and hasattr(v, "item"):  # numpy scalar
        try:
            return v.item()
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple)):
        return [_coerce_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce_json(v[k]) for k in sorted(v)}
    return str(v)


def _canonical_dumps(obj: Any) -> str:
    """Deterministic JSON string for hashing.

    * keys sorted
    * no whitespace
    * UTF-8 ensured via ensure_ascii=False (but hash uses .encode)
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────
# Node construction (the only supported way — guarantees hash agreement)
# ─────────────────────────────────────────────────────────────────────


def make_node(
    op: str,
    *,
    inputs: Sequence[str] = (),
    value: Optional[Dict[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
    producer: str = QLANG_PRODUCER,
    ts: Optional[str] = None,
) -> DerivationNode:
    """Compute the SHA-256 content-addressed node_id and return a
    frozen ``DerivationNode``.

    The hash covers only the STRUCTURAL fields (op, sorted inputs,
    meta).  The ``value`` is included in what's written to disk but
    is NOT part of the node_id — otherwise replay drift would change
    node_ids and break the trace's structural integrity check.

    The ``ts`` is not part of the hash either; two nodes produced at
    different times from identical inputs get identical node_ids.
    This preserves SPEC §7's "same inputs → same node_id" contract.
    """
    meta_clean = _canonical_meta(meta or {})
    # NOTE: inputs are hashed in CALL ORDER, not sorted.  For
    # non-commutative operations like `sub` or `div` the operand
    # order is semantically significant: `sub(a,b) != sub(b,a)`.
    # For commutative ops (add, mul) the evaluator can normalise the
    # operand order before calling make_node if canonicalisation is
    # desired; at the trace layer we preserve what the evaluator
    # tells us.
    payload = {
        "op": op,
        "inputs": list(inputs),
        "meta": meta_clean,
        "producer": producer,
    }
    digest = hashlib.sha256(
        _canonical_dumps(payload).encode("utf-8"),
    ).hexdigest()

    if ts is None:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    return DerivationNode(
        node_id=digest,
        op=op,
        inputs=tuple(inputs),
        value=_canonical_value(value),
        meta=dict(meta_clean),
        producer=producer,
        ts=ts,
    )


# ─────────────────────────────────────────────────────────────────────
# Trace — ordered collection of DerivationNodes
# ─────────────────────────────────────────────────────────────────────


class Trace:
    """Append-only, ordered collection of DerivationNodes.

    The evaluator pushes nodes here as values are computed.  Order is
    the order of creation, which for a tree-walking evaluator is
    always topological (leaves before their consumers).

    Deduplication: pushing a node whose ``node_id`` is already present
    is silently ignored — identical computations are represented
    once.  This keeps the trace compact and preserves the §7.1
    content-addressing contract.
    """

    def __init__(self) -> None:
        self._by_id: Dict[str, DerivationNode] = {}
        self._order: List[str] = []

    # -- mutation --

    def push(self, node: DerivationNode) -> DerivationNode:
        if node.node_id in self._by_id:
            # Same ID — this must be an identical computation; the
            # stored node wins (it carries the original ``ts``).
            return self._by_id[node.node_id]
        self._by_id[node.node_id] = node
        self._order.append(node.node_id)
        return node

    # -- read --

    def __iter__(self):
        for nid in self._order:
            yield self._by_id[nid]

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._by_id

    def __getitem__(self, node_id: str) -> DerivationNode:
        return self._by_id[node_id]

    def get(self, node_id: str) -> Optional[DerivationNode]:
        return self._by_id.get(node_id)

    def nodes(self) -> Tuple[DerivationNode, ...]:
        return tuple(self._by_id[nid] for nid in self._order)

    # -- I/O --

    def write(self, path: os.PathLike) -> None:
        """Write the trace as newline-delimited JSON to ``path``."""
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for node in self.nodes():
                fh.write(
                    _canonical_dumps(node.to_dict()) + "\n",
                )

    @classmethod
    def read(cls, path: os.PathLike) -> "Trace":
        """Load a trace from a .jsonl file."""
        trace = cls()
        p = Path(path)
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                node = DerivationNode.from_dict(json.loads(line))
                trace.push(node)
        return trace

    # -- verification --

    def verify_integrity(self) -> List[str]:
        """Check that every recorded node_id equals the SHA-256 of
        its structural fields.  Returns the list of drifted node_ids
        (empty list means the trace is internally consistent).

        This is an OFFLINE check — no kernel calls.  Use ``replay()``
        (see ``cli.py``) for full re-execution verification.
        """
        drift: List[str] = []
        for node in self.nodes():
            payload = {
                "op": node.op,
                "inputs": list(node.inputs),
                "meta": _canonical_meta(node.meta),
                "producer": node.producer,
            }
            expected = hashlib.sha256(
                _canonical_dumps(payload).encode("utf-8"),
            ).hexdigest()
            if expected != node.node_id:
                drift.append(node.node_id)
        return drift


# ─────────────────────────────────────────────────────────────────────
# Convenience constructors for common leaf-op nodes
# ─────────────────────────────────────────────────────────────────────


def literal_node(
    magnitude: Any, unit: str, uncertainty: Optional[float] = None
) -> DerivationNode:
    """Emit a node for a literal NumberLiteral or UncertaintyExpr leaf.

    Leaf literals have no inputs, so the content-address hash has
    nothing distinguishing to work with from the structural fields
    alone.  We therefore embed magnitude + unit + uncertainty into
    ``meta``, which IS part of the hash.  Two distinct literals get
    distinct node_ids; the same literal reappearing in a program
    shares a single node (correct deduplication).
    """
    m = float(magnitude) if not isinstance(magnitude, bool) else magnitude
    return make_node(
        "literal",
        inputs=(),
        value={"magnitude": m, "unit": unit, "uncertainty": uncertainty},
        meta={
            "magnitude": m,
            "unit": unit,
            "uncertainty": uncertainty,
        },
    )


def string_literal_node(s: str) -> DerivationNode:
    return make_node(
        "literal.string",
        inputs=(),
        value={"magnitude": s, "unit": "", "uncertainty": None},
        meta={"value": s},
    )


def bind_node(name: str, input_id: str) -> DerivationNode:
    return make_node(
        "bind",
        inputs=(input_id,),
        value=None,
        meta={"name": name},
    )


def binary_op_node(
    op: str,
    left_id: str,
    right_id: str,
    magnitude: Any,
    unit: str,
    uncertainty: Optional[float] = None,
) -> DerivationNode:
    return make_node(
        op,
        inputs=(left_id, right_id),
        value={
            "magnitude": float(magnitude),
            "unit": unit,
            "uncertainty": uncertainty,
        },
    )


def unary_op_node(
    op: str,
    input_id: str,
    magnitude: Any,
    unit: str,
    uncertainty: Optional[float] = None,
) -> DerivationNode:
    return make_node(
        op,
        inputs=(input_id,),
        value={
            "magnitude": float(magnitude),
            "unit": unit,
            "uncertainty": uncertainty,
        },
    )


def print_node(text: str, input_id: Optional[str] = None) -> DerivationNode:
    return make_node(
        "print",
        inputs=(input_id,) if input_id else (),
        value={"magnitude": text, "unit": "", "uncertainty": None},
    )


def invariant_node(
    experiment: str, expression_source: str, input_id: str, *, passed: bool
) -> DerivationNode:
    return make_node(
        "invariant",
        inputs=(input_id,),
        value=None,
        meta={
            "experiment": experiment,
            "check": expression_source,
            "passed": passed,
        },
    )


def result_node(experiment: str, value_id: str) -> DerivationNode:
    return make_node(
        "experiment.result",
        inputs=(value_id,),
        value=None,
        meta={"experiment": experiment},
    )


def kernel_node(
    op: str,
    input_ids: Sequence[str],
    magnitude: Any,
    unit: str,
    uncertainty: Optional[float],
    producer: str,
    meta: Mapping[str, Any],
) -> DerivationNode:
    """Build a node for a kernel call (``simulate.chemistry`` etc).
    The producer string identifies the kernel + version."""
    return make_node(
        op,
        inputs=tuple(input_ids),
        value={
            "magnitude": float(magnitude)
            if isinstance(magnitude, (int, float))
            else magnitude,
            "unit": unit,
            "uncertainty": uncertainty,
        },
        meta=dict(meta),
        producer=producer,
    )
