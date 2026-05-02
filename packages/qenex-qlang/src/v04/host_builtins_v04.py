"""
Q-Lang v0.4 — host builtin registry (SPEC §9).

Each v0.4 builtin is a Python callable that takes evaluated Q-Lang
``Value``s and returns a ``Value``.  Dimensional checks are enforced
at the point of call; any failure raises ``DimensionMismatchError``.

v0.4 ships a small, focused set:

    sqrt(x)     — scalar square root; dim is inferred (preserves
                  valid integer-exponent dims, else rejected).
    log(x)      — natural log; arg must be dimensionless.
    exp(x)      — exponential; arg must be dimensionless.
    sin, cos, tan  — arg must be dimensionless.
    abs(x)      — absolute value; preserves dim.
    dim(x)      — introspection: returns a dimensionless Quantity
                  whose magnitude is 0 and whose ``meta`` in the
                  trace names the dim (useful for ``invariant:
                  dim(E) == dim([Hartree])`` patterns in v0.5+).
"""

from __future__ import annotations

import math
from typing import Any, Dict

from errors_v04 import DimensionMismatchError  # type: ignore[import-not-found]
from units_v04 import Dim, SCALAR_DIM  # type: ignore[import-not-found]
from values_v04 import Host, Quantity, String  # type: ignore[import-not-found]


def _require_quantity(name: str, arg: Any) -> Quantity:
    if not isinstance(arg, Quantity):
        raise DimensionMismatchError(
            f"builtin {name!r} requires a numeric argument, got {type(arg).__name__}"
        )
    return arg


def _dimless(name: str, q: Quantity) -> Quantity:
    if not q.dim.is_dimensionless():
        raise DimensionMismatchError(
            f"builtin {name!r} requires a dimensionless argument, got [{q.dim}]",
            have=str(q.dim),
            need="1",
        )
    return q


# ─────────────────────────────────────────────────────────────────────
# sqrt: only valid when every dim exponent is even (so integer half is
# representable).
# ─────────────────────────────────────────────────────────────────────


def _builtin_sqrt(x: Any) -> Quantity:
    q = _require_quantity("sqrt", x)
    d = q.dim
    # Every component must be even for dim/2 to be integer
    components = (
        d.mass,
        d.length,
        d.time,
        d.current,
        d.temperature,
        d.amount,
        d.luminous,
    )
    if any(c % 2 != 0 for c in components):
        raise DimensionMismatchError(
            f"sqrt requires all dim exponents to be even, got [{d}]",
            have=str(d),
            need="all even exponents",
        )
    new_dim = Dim(
        mass=d.mass // 2,
        length=d.length // 2,
        time=d.time // 2,
        current=d.current // 2,
        temperature=d.temperature // 2,
        amount=d.amount // 2,
        luminous=d.luminous // 2,
    )
    if q.magnitude < 0.0:
        raise DimensionMismatchError(
            f"sqrt of negative magnitude {q.magnitude} is not real"
        )
    new_mag = math.sqrt(q.magnitude)
    new_unc = None
    if q.uncertainty is not None:
        if q.magnitude > 0.0:
            new_unc = q.uncertainty / (2.0 * new_mag)
        else:
            new_unc = 0.0
    return Quantity(magnitude=new_mag, dim=new_dim, uncertainty=new_unc)


def _builtin_log(x: Any) -> Quantity:
    q = _dimless("log", _require_quantity("log", x))
    return Quantity(
        magnitude=math.log(q.magnitude),
        dim=SCALAR_DIM,
        uncertainty=(
            abs(q.uncertainty / q.magnitude)
            if q.uncertainty is not None and q.magnitude != 0
            else None
        ),
    )


def _builtin_exp(x: Any) -> Quantity:
    q = _dimless("exp", _require_quantity("exp", x))
    m = math.exp(q.magnitude)
    return Quantity(
        magnitude=m,
        dim=SCALAR_DIM,
        uncertainty=(abs(m * q.uncertainty) if q.uncertainty is not None else None),
    )


def _builtin_sin(x: Any) -> Quantity:
    q = _dimless("sin", _require_quantity("sin", x))
    return Quantity(
        magnitude=math.sin(q.magnitude),
        dim=SCALAR_DIM,
        uncertainty=(
            abs(math.cos(q.magnitude) * q.uncertainty)
            if q.uncertainty is not None
            else None
        ),
    )


def _builtin_cos(x: Any) -> Quantity:
    q = _dimless("cos", _require_quantity("cos", x))
    return Quantity(
        magnitude=math.cos(q.magnitude),
        dim=SCALAR_DIM,
        uncertainty=(
            abs(math.sin(q.magnitude) * q.uncertainty)
            if q.uncertainty is not None
            else None
        ),
    )


def _builtin_tan(x: Any) -> Quantity:
    q = _dimless("tan", _require_quantity("tan", x))
    m = math.tan(q.magnitude)
    return Quantity(
        magnitude=m,
        dim=SCALAR_DIM,
        uncertainty=(
            abs(q.uncertainty / math.cos(q.magnitude) ** 2)
            if q.uncertainty is not None
            else None
        ),
    )


def _builtin_abs(x: Any) -> Quantity:
    q = _require_quantity("abs", x)
    return Quantity(
        magnitude=abs(q.magnitude),
        dim=q.dim,
        uncertainty=q.uncertainty,
    )


def _builtin_dim(x: Any) -> String:
    if not isinstance(x, Quantity):
        return String(value=f"[non-Quantity: {type(x).__name__}]")
    return String(value=f"[{x.dim}]")


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────


def default_builtins() -> Dict[str, Any]:
    """Build and return the v0.4 default builtin registry.

    Returns a plain dict mapping Q-Lang name → ``BuiltinEntry``.
    Imported lazily by ``Evaluator.__init__``.
    """
    from evaluator_v04 import BuiltinEntry  # type: ignore[import-not-found]

    reg: Dict[str, Any] = {}

    def add(name: str, fn, *, version: str = "qlang-v0.4"):
        reg[name] = BuiltinEntry(name=name, fn=fn, version=version)

    add("sqrt", _builtin_sqrt)
    add("log", _builtin_log)
    add("exp", _builtin_exp)
    add("sin", _builtin_sin)
    add("cos", _builtin_cos)
    add("tan", _builtin_tan)
    add("abs", _builtin_abs)
    add("dim", _builtin_dim)

    return reg
