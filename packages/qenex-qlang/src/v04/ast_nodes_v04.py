"""
Q-Lang v0.4 — AST node types (SPEC §4).

The parser produces a tree of these dataclasses; the evaluator walks
it.  Every node carries source location (line, col) so error
diagnostics can point back to the offending token.

Kept intentionally small: one dataclass per grammar production in
SPEC §4, no convenience sugar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────
# Unit expressions (only appear inside UnitLiteral)
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnitAtom:
    """Example:  Bohr, m, s^2"""

    name: str
    exponent: int
    line: int
    col: int


@dataclass(frozen=True)
class UnitExpr:
    """Example:  m/s, kg*m^2/s, Hartree.  A product of unit atoms,
    each signed by its cumulative power."""

    atoms: Tuple[UnitAtom, ...]  # exponents may be negative
    line: int
    col: int


# ─────────────────────────────────────────────────────────────────────
# Expression nodes
# ─────────────────────────────────────────────────────────────────────


Expr = Union[
    "NumberLiteral",
    "StringLiteral",
    "Identifier",
    "UnitConversion",
    "UncertaintyExpr",
    "UnaryOp",
    "BinaryOp",
    "Call",
    "SimulateExpr",
    "PipeExpr",
]


@dataclass(frozen=True)
class NumberLiteral:
    """Numeric literal, optionally with an attached unit.  ``text`` is
    the source spelling (preserved so the trace can record exactly
    what the author wrote)."""

    text: str  # canonical source spelling
    unit: Optional[UnitExpr]  # None = dimensionless
    line: int
    col: int


@dataclass(frozen=True)
class StringLiteral:
    value: str  # already-unescaped
    line: int
    col: int


@dataclass(frozen=True)
class Identifier:
    name: str
    line: int
    col: int


@dataclass(frozen=True)
class UnitConversion:
    """``expr in [unit]`` — rescales magnitude, requires dim match."""

    expr: Expr
    target_unit: UnitExpr
    line: int
    col: int


@dataclass(frozen=True)
class UncertaintyExpr:
    """``value +/- uncertainty``"""

    value: Expr
    uncertainty: Expr
    line: int
    col: int


@dataclass(frozen=True)
class UnaryOp:
    op: str  # "-"
    operand: Expr
    line: int
    col: int


@dataclass(frozen=True)
class BinaryOp:
    op: str  # "+", "-", "*", "/", "**", "<", "<=", ">", ">=", "==", "!="
    left: Expr
    right: Expr
    line: int
    col: int


@dataclass(frozen=True)
class Call:
    """Function call.  Positional and/or keyword arguments."""

    callee: str  # v0.4: only direct IDENT(args); no computed callables
    args: Tuple[Expr, ...]
    kwargs: Tuple[Tuple[str, Expr], ...]
    line: int
    col: int


@dataclass(frozen=True)
class SimulateExpr:
    """``simulate DOMAIN { kwargs }``.  ``domain`` is the registered
    domain name; ``kwargs`` carry everything inside the braces
    (including ``conserve:`` and ``tolerance:`` when applicable)."""

    domain: str
    kwargs: Tuple[Tuple[str, Expr], ...]
    line: int
    col: int


@dataclass(frozen=True)
class PipeExpr:
    """``left |> right``.  ``right`` must be a Call; the left-hand
    value is inserted as the first positional argument."""

    left: Expr
    right: Call
    line: int
    col: int


# ─────────────────────────────────────────────────────────────────────
# Statement and top-level nodes
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LetStmt:
    name: str
    value: Expr
    line: int
    col: int


@dataclass(frozen=True)
class PrintStmt:
    value: Expr
    line: int
    col: int


@dataclass(frozen=True)
class GivenParam:
    name: str
    unit: Optional[UnitExpr]
    line: int
    col: int


@dataclass(frozen=True)
class InvariantClause:
    expression: Expr
    source_text: str  # original source (for error messages)
    line: int
    col: int


@dataclass(frozen=True)
class ResultClause:
    expression: Expr
    line: int
    col: int


ExpClause = Union[
    "GivenClause",
    LetStmt,
    InvariantClause,
    ResultClause,
]


@dataclass(frozen=True)
class GivenClause:
    params: Tuple[GivenParam, ...]
    line: int
    col: int


@dataclass(frozen=True)
class ExperimentDef:
    """``experiment NAME { ... }``"""

    name: str
    clauses: Tuple[ExpClause, ...]
    line: int
    col: int


Decl = Union[LetStmt, PrintStmt, ExperimentDef]


@dataclass(frozen=True)
class Program:
    decls: Tuple[Decl, ...]


# ─────────────────────────────────────────────────────────────────────
# Small helpers for debug rendering
# ─────────────────────────────────────────────────────────────────────


def pretty(node, indent: int = 0) -> str:
    """Render an AST node as a readable tree — for debugging."""
    pad = "  " * indent
    if hasattr(node, "__dataclass_fields__"):
        lines = [f"{pad}{type(node).__name__}"]
        for f in node.__dataclass_fields__:
            val = getattr(node, f)
            if f in ("line", "col", "source_text", "text"):
                lines.append(f"{pad}  {f}: {val!r}")
            elif (
                isinstance(val, tuple)
                and val
                and hasattr(val[0], "__dataclass_fields__")
            ):
                lines.append(f"{pad}  {f}:")
                for child in val:
                    lines.append(pretty(child, indent + 2))
            elif isinstance(val, tuple):
                lines.append(f"{pad}  {f}: {val!r}")
            elif hasattr(val, "__dataclass_fields__"):
                lines.append(pretty(val, indent + 1).replace(pad, pad + "  ", 1))
            else:
                lines.append(f"{pad}  {f}: {val!r}")
        return "\n".join(lines)
    return f"{pad}{node!r}"
