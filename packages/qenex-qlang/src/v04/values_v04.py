"""
Q-Lang v0.4 — runtime value types (SPEC §5).

v0.4 runtime has exactly three value kinds:

    Quantity   magnitude: float, dim: Dim, uncertainty: float | None,
               node_id: str        (SHA-256 DAG node id)
    String     value: str, node_id: str
    Host       payload: Any,  dim: Dim, node_id: str

All three share a ``node_id`` so the trace can refer to them
uniformly.  Arithmetic is only defined on ``Quantity``.

Why not the legacy ``qvalue.QValue``
------------------------------------

The pre-existing ``qvalue.QValue`` is a flexible dataclass that
holds ``value: Decimal | float | complex | ndarray`` together with
``dims`` and ``uncertainty``.  It's been a source of bugs in v0.1/v0.2/v0.3
because Decimal and NumPy interact awkwardly (``'Decimal' object is
not subscriptable`` was one of the examples that broke).  For v0.4 we
pin the magnitude representation to ``float`` (IEEE-754 double) and
keep the type deliberately small.

A ``Quantity.to_legacy_qvalue()`` method is provided in case the
legacy ``QValue`` is needed to bridge into existing kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from units_v04 import Dim, SCALAR_DIM  # type: ignore[import-not-found]


# ─────────────────────────────────────────────────────────────────────
# Quantity
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Quantity:
    """A unit + uncertainty bearing scalar.

    ``display_unit`` preserves the unit name the user wrote at the
    literal site (e.g. ``"Bohr"``, ``"Hartree"``).  It's used only
    for rendering; arithmetic is governed by ``dim``.  When missing,
    rendering falls back to the canonical SI unit string for the
    Quantity's ``dim``.
    """

    magnitude: float
    dim: Dim = SCALAR_DIM
    uncertainty: Optional[float] = None  # absolute, same units as magnitude
    node_id: str = ""  # provenance link
    display_unit: str = ""  # user-written unit name; "" == auto

    # --- readable printing --------------------------------------------------

    def _unit_label(self) -> str:
        if self.display_unit:
            return self.display_unit
        if self.dim.is_dimensionless():
            return ""
        return str(self.dim)

    def __repr__(self) -> str:
        parts = [f"{self.magnitude:.6g}"]
        if self.uncertainty is not None:
            parts.append(f"+/- {self.uncertainty:.3g}")
        lbl = self._unit_label()
        if lbl:
            parts.append(f"[{lbl}]")
        return " ".join(parts)

    def __format__(self, spec: str) -> str:
        """Used by f-string interpolation inside ``print``."""
        if not spec:
            s = f"{self.magnitude:g}"
        else:
            s = format(self.magnitude, spec)
        if self.uncertainty is not None:
            s = f"{s} +/- {self.uncertainty:g}"
        lbl = self._unit_label()
        if lbl:
            s = f"{s} [{lbl}]"
        return s

    # --- truth value (for invariant evaluation) ----------------------------

    def __bool__(self) -> bool:
        """A Quantity is truthy if its magnitude is non-zero.  Used
        by the experiment evaluator to check invariant results."""
        return bool(self.magnitude)

    # --- legacy alias so test code written against QValue works ------------

    @property
    def value(self) -> float:
        """Alias for ``magnitude``.  The v0.4 spec calls the numeric
        component ``magnitude`` to avoid overloading the word
        "value"; many tests and legacy callers use ``.value``, so we
        expose the alias.  Setting through this alias is not supported
        (Quantity is frozen)."""
        return self.magnitude

    @property
    def dims(self):
        """Alias for ``dim`` (``QValue`` used the plural)."""
        return self.dim

    # --- with_node helper --------------------------------------------------

    def with_node(self, node_id: str) -> "Quantity":
        """Return a copy with the provenance node_id set."""
        return replace(self, node_id=node_id)


# ─────────────────────────────────────────────────────────────────────
# String
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class String:
    value: str
    node_id: str = ""

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return self.value

    def __format__(self, spec: str) -> str:
        return format(self.value, spec) if spec else self.value

    def __eq__(self, other: object) -> bool:
        """Equal to another ``String`` with the same value, and
        equal to a plain ``str`` that matches the underlying value.
        The node_id is deliberately NOT part of equality: two
        semantically-equal strings from different source locations
        should compare equal."""
        if isinstance(other, String):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("String", self.value))

    def with_node(self, node_id: str) -> "String":
        return replace(self, node_id=node_id)


# ─────────────────────────────────────────────────────────────────────
# Host
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Host:
    """Opaque Python object produced by a kernel (e.g. a
    ``trajectory`` object from ``_sim_md``).  Dim is tracked to
    permit unit-checked field access in later versions."""

    payload: Any
    dim: Dim = SCALAR_DIM
    node_id: str = ""

    def __repr__(self) -> str:
        return f"Host({type(self.payload).__name__})"

    def with_node(self, node_id: str) -> "Host":
        return replace(self, node_id=node_id)


# ─────────────────────────────────────────────────────────────────────
# Union alias
# ─────────────────────────────────────────────────────────────────────


Value = "Quantity | String | Host"


def is_quantity(v: Any) -> bool:
    return isinstance(v, Quantity)


def is_string(v: Any) -> bool:
    return isinstance(v, String)


def is_host(v: Any) -> bool:
    return isinstance(v, Host)


# ─────────────────────────────────────────────────────────────────────
# Helpers used by the evaluator
# ─────────────────────────────────────────────────────────────────────


def quantity_from_literal(
    magnitude: float,
    scale_to_si: float,  # kept for API stability
    dim: Dim,
    uncertainty: Optional[float] = None,
) -> Quantity:
    """Build a Quantity from a parsed literal.

    v0.4 design choice
    ----------------
    We store the magnitude in its DECLARED unit, not SI-normalised.
    A user writes ``1.4 [Bohr]`` and expects ``.value == 1.4`` —
    matching the legacy ``QValue`` and every user-facing unit
    library (pint, Unitful.jl).  Arithmetic on mixed units therefore
    requires explicit conversion via ``expr in [unit]``; the
    ``DimensionMismatchError`` lands at the point of the mixed
    operation.  This is strictly stronger than SI-under-the-hood,
    because the error is visible at the source line of the bad
    operation rather than hidden inside a reformat.

    ``scale_to_si`` is retained in the signature so callers that
    need SI-projection (e.g. trace emission) can compute
    ``magnitude * scale_to_si`` on their own.
    """
    del scale_to_si  # intentionally unused; see docstring
    return Quantity(
        magnitude=float(magnitude),
        dim=dim,
        uncertainty=float(uncertainty) if uncertainty is not None else None,
    )
