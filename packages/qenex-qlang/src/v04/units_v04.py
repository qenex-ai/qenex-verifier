"""
Q-Lang v0.4 — unit registry and conversions (SPEC §4, §5).

The lexer recognises a unit literal as ``[UNIT ( * | / UNIT )^*]``
with optional ``^ INTEGER`` exponents.  The parser produces a
``UnitExpr`` of ``UnitAtom``s.  This module turns a ``UnitExpr``
into a canonical ``(scale, dims)`` pair — the scale factor to
multiply the literal by to convert from the named unit into SI base
units, and the 7-dim exponent vector in SI space.

The SI base dimension order matches the existing
``qenex-qlang/src/dimensions.py`` ``Dimensions`` dataclass fields:

    (mass, length, time, current, temperature, amount, luminous)

Units not in the table are rejected with ``QLangSyntaxError``.  v0.4
is strict; a typo in a unit name is a compile-time error, not a
silent dimensionless fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class Dim:
    """Immutable 7-SI-base-unit exponent vector.

    Mirrors ``dimensions.Dimensions`` but is local to v0.4 so we don't
    have a circular dependency during bootstrap.  ``to_legacy()``
    converts to the existing ``Dimensions`` instance on demand.
    """

    mass: int = 0
    length: int = 0
    time: int = 0
    current: int = 0
    temperature: int = 0
    amount: int = 0
    luminous: int = 0

    # Arithmetic over dimensions

    def __add__(self, other: "Dim") -> "Dim":
        return Dim(
            self.mass + other.mass,
            self.length + other.length,
            self.time + other.time,
            self.current + other.current,
            self.temperature + other.temperature,
            self.amount + other.amount,
            self.luminous + other.luminous,
        )

    def __sub__(self, other: "Dim") -> "Dim":
        return Dim(
            self.mass - other.mass,
            self.length - other.length,
            self.time - other.time,
            self.current - other.current,
            self.temperature - other.temperature,
            self.amount - other.amount,
            self.luminous - other.luminous,
        )

    def __mul__(self, n: int) -> "Dim":
        return Dim(
            self.mass * n,
            self.length * n,
            self.time * n,
            self.current * n,
            self.temperature * n,
            self.amount * n,
            self.luminous * n,
        )

    def is_dimensionless(self) -> bool:
        return (
            self.mass == 0
            and self.length == 0
            and self.time == 0
            and self.current == 0
            and self.temperature == 0
            and self.amount == 0
            and self.luminous == 0
        )

    def __str__(self) -> str:
        if self.is_dimensionless():
            return "1"
        parts = []
        for name, exp in [
            ("kg", self.mass),
            ("m", self.length),
            ("s", self.time),
            ("A", self.current),
            ("K", self.temperature),
            ("mol", self.amount),
            ("cd", self.luminous),
        ]:
            if exp == 0:
                continue
            parts.append(name if exp == 1 else f"{name}^{exp}")
        return "·".join(parts)


# Canonical dimensionless.
SCALAR_DIM = Dim()


# ─────────────────────────────────────────────────────────────────────
# Unit table
#
# Each entry:  NAME -> (scale, Dim)
# ``magnitude_SI = magnitude_in_NAME * scale``
# All scales are CODATA 2018.
# ─────────────────────────────────────────────────────────────────────

_UNIT_TABLE: Dict[str, Tuple[float, Dim]] = {
    # SI base units (scale = 1)
    "kg": (1.0, Dim(mass=1)),
    "m": (1.0, Dim(length=1)),
    "s": (1.0, Dim(time=1)),
    "A": (1.0, Dim(current=1)),
    "K": (1.0, Dim(temperature=1)),
    "mol": (1.0, Dim(amount=1)),
    "cd": (1.0, Dim(luminous=1)),
    # Length
    "m": (1.0, Dim(length=1)),
    "cm": (1e-2, Dim(length=1)),
    "mm": (1e-3, Dim(length=1)),
    "nm": (1e-9, Dim(length=1)),
    "pm": (1e-12, Dim(length=1)),
    "fm": (1e-15, Dim(length=1)),
    "Angstrom": (1e-10, Dim(length=1)),
    "angstrom": (1e-10, Dim(length=1)),
    "A_ngstrom": (1e-10, Dim(length=1)),  # ASCII-safe alias
    "Bohr": (5.29177210903e-11, Dim(length=1)),  # a_0
    "bohr": (5.29177210903e-11, Dim(length=1)),
    "km": (1e3, Dim(length=1)),
    # Time
    "ms": (1e-3, Dim(time=1)),
    "us": (1e-6, Dim(time=1)),
    "ns": (1e-9, Dim(time=1)),
    "ps": (1e-12, Dim(time=1)),
    "fs": (1e-15, Dim(time=1)),
    "min": (60.0, Dim(time=1)),
    "h": (3600.0, Dim(time=1)),
    "hr": (3600.0, Dim(time=1)),
    # Mass
    "g": (1e-3, Dim(mass=1)),
    "mg": (1e-6, Dim(mass=1)),
    "amu": (1.66053906660e-27, Dim(mass=1)),
    "Dalton": (1.66053906660e-27, Dim(mass=1)),
    "Da": (1.66053906660e-27, Dim(mass=1)),
    # Energy  (SI: kg·m²·s⁻²)
    "J": (1.0, Dim(mass=1, length=2, time=-2)),
    "kJ": (1e3, Dim(mass=1, length=2, time=-2)),
    "eV": (1.602176634e-19, Dim(mass=1, length=2, time=-2)),
    "keV": (1.602176634e-16, Dim(mass=1, length=2, time=-2)),
    "MeV": (1.602176634e-13, Dim(mass=1, length=2, time=-2)),
    "Hartree": (4.3597447222071e-18, Dim(mass=1, length=2, time=-2)),
    "hartree": (4.3597447222071e-18, Dim(mass=1, length=2, time=-2)),
    "Eh": (4.3597447222071e-18, Dim(mass=1, length=2, time=-2)),
    "kcal": (4184.0, Dim(mass=1, length=2, time=-2)),
    "cal": (4.184, Dim(mass=1, length=2, time=-2)),
    # Force
    "N": (1.0, Dim(mass=1, length=1, time=-2)),
    "dyn": (1e-5, Dim(mass=1, length=1, time=-2)),
    # Power
    "W": (1.0, Dim(mass=1, length=2, time=-3)),
    "mW": (1e-3, Dim(mass=1, length=2, time=-3)),
    # Pressure
    "Pa": (1.0, Dim(mass=1, length=-1, time=-2)),
    "kPa": (1e3, Dim(mass=1, length=-1, time=-2)),
    "MPa": (1e6, Dim(mass=1, length=-1, time=-2)),
    "bar": (1e5, Dim(mass=1, length=-1, time=-2)),
    "atm": (101325.0, Dim(mass=1, length=-1, time=-2)),
    # Frequency / angular
    "Hz": (1.0, Dim(time=-1)),
    "kHz": (1e3, Dim(time=-1)),
    "MHz": (1e6, Dim(time=-1)),
    "GHz": (1e9, Dim(time=-1)),
    "rad": (1.0, SCALAR_DIM),
    "deg": (math.pi / 180.0, SCALAR_DIM),
    # Charge
    "C": (1.0, Dim(time=1, current=1)),
    "pC": (1e-12, Dim(time=1, current=1)),
    # Voltage
    "V": (1.0, Dim(mass=1, length=2, time=-3, current=-1)),
    # Count-like / dimensionless labels used in .ql programs
    "steps": (1.0, SCALAR_DIM),
    "step": (1.0, SCALAR_DIM),
    "count": (1.0, SCALAR_DIM),
    "iteration": (1.0, SCALAR_DIM),
    "iterations": (1.0, SCALAR_DIM),
    "dimensionless": (1.0, SCALAR_DIM),
}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def lookup_unit(name: str) -> Tuple[float, Dim]:
    """Return ``(scale, dim)`` for a single unit name.  Raises a
    descriptive ``KeyError`` when the name is unknown; callers wrap
    it into ``QLangSyntaxError`` with source location."""
    if name in _UNIT_TABLE:
        return _UNIT_TABLE[name]
    raise KeyError(f"unknown unit {name!r}")


def resolve_unit_expr(atoms: Iterable) -> Tuple[float, Dim]:
    """Combine a parsed ``UnitExpr.atoms`` into a single
    ``(scale, dim)`` pair.

    Scale is multiplicative; exponents combine additively.  Negative
    exponents (from ``/`` in the source) are handled at parse time —
    each atom arrives already signed.

    ``atoms`` is an iterable of objects with ``.name`` (str) and
    ``.exponent`` (int), matching ``ast_nodes.UnitAtom``.
    """
    total_scale = 1.0
    total_dim = SCALAR_DIM
    for atom in atoms:
        scale, dim = lookup_unit(atom.name)
        if atom.exponent > 0:
            total_scale *= scale**atom.exponent
            total_dim = total_dim + (dim * atom.exponent)
        elif atom.exponent < 0:
            n = -atom.exponent
            total_scale /= scale**n
            total_dim = total_dim - (dim * n)
        # exponent 0: identity (shouldn't occur from parser)
    return total_scale, total_dim


def format_unit(dim: Dim) -> str:
    """Human-readable rendering of a dimension vector.  Used in
    error messages and default print formatting."""
    return str(dim)


def registered_unit_names() -> Tuple[str, ...]:
    """Introspection helper for error messages ("did you mean ...")
    and for tests."""
    return tuple(sorted(_UNIT_TABLE))
