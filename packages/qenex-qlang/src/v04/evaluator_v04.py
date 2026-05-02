"""
Q-Lang v0.4 — tree-walking evaluator (SPEC §5, §6, §7).

Takes a ``Program`` AST (from ``parser.parse``) and evaluates it.
Every value produced has a ``node_id`` pointing into the evaluator's
``Trace``, so the full derivation DAG is available after a run.

Evaluation model
----------------
* One mutable environment (SPEC §3.1: one flat scope per program;
  experiments get a child scope for their `given:` params).
* Every expression returns a ``Value`` (Quantity | String | Host) and
  pushes a ``DerivationNode`` into the trace.
* Arithmetic operates on SI-normalised magnitudes; unit literals
  were converted to SI at parse/literal time (see ``values.py``).
* Errors propagate as typed ``QLangError`` subclasses — nothing is
  caught-and-printed.

Delegation to other v0.4 modules
------------------------------
* ``experiment.py`` handles ``experiment NAME { ... }`` declarations
  and their invocation.  The evaluator recognises ``ExperimentDef``
  as a top-level decl and defers to ``experiment.Experiment``.
* ``simulate_dispatch.py`` handles ``SimulateExpr`` nodes — routes
  into ``simulation_handlers._sim_*``.
* ``host_builtins.py`` provides the default builtin registry.

These three hand off back to the evaluator for argument evaluation.
"""

from __future__ import annotations

import io
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from ast_nodes_v04 import (  # type: ignore[import-not-found]
    BinaryOp,
    Call,
    ExperimentDef,
    Identifier,
    LetStmt,
    NumberLiteral,
    PipeExpr,
    PrintStmt,
    Program,
    ResultClause,
    SimulateExpr,
    StringLiteral,
    UnaryOp,
    UncertaintyExpr,
    UnitConversion,
    UnitExpr,
)
from errors_v04 import (  # type: ignore[import-not-found]
    DimensionMismatchError,
    KernelError,
    QLangSyntaxError,
    RebindingError,
    UnboundNameError,
)
from provenance_v04 import (  # type: ignore[import-not-found]
    DerivationNode,
    Trace,
    bind_node,
    binary_op_node,
    literal_node,
    make_node,
    print_node,
    string_literal_node,
    unary_op_node,
)
from units_v04 import Dim, SCALAR_DIM, format_unit, resolve_unit_expr  # type: ignore[import-not-found]
from values_v04 import Host, Quantity, String, quantity_from_literal  # type: ignore[import-not-found]  # noqa: F401


# ─────────────────────────────────────────────────────────────────────
# Interpolation helper  — `"E({mol}) = {E:.4f}"`
# ─────────────────────────────────────────────────────────────────────

_INTERP_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z_0-9]*)(:[^}]*)?\}")


# Domain-specific kwargs that should be treated as symbolic tags
# (not resolved via env lookup) inside a ``simulate DOMAIN {}`` block.
# e.g. ``conserve: total_energy`` — ``total_energy`` is a tag, not a
# reference to a binding.
_SYMBOLIC_TAG_KWARGS: dict[str, frozenset[str]] = {
    "md": frozenset({"conserve"}),
}


def _interpolate(template: str, env: Dict[str, Any]) -> str:
    """f-string-style substitution using the interpreter environment."""

    def repl(m: re.Match) -> str:
        name = m.group(1)
        spec = m.group(2) or ""
        if spec:
            spec = spec[1:]  # drop leading ':'
        if name not in env:
            raise UnboundNameError(name)
        val = env[name]
        if spec:
            return format(val, spec)
        return format(val)

    return _INTERP_PATTERN.sub(repl, template)


# ─────────────────────────────────────────────────────────────────────
# Builtin callable registry
# ─────────────────────────────────────────────────────────────────────


@dataclass
class BuiltinEntry:
    """One registered host callable.

    ``fn`` takes a tuple of evaluated Values (positional) and a dict
    of kwargs (name -> Value); returns a Value.  Signature checking
    lives in the callable itself (it's convenient to have access to
    the evaluator / trace there)."""

    name: str
    fn: Callable[..., Any]
    returns_dim_hint: Optional[Dim] = None
    version: str = "qlang-v0.4"


Builtins = Dict[str, BuiltinEntry]


# ─────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────


class Evaluator:
    """Primary tree walker.  Holds the environment and trace."""

    def __init__(
        self,
        builtins: Optional[Builtins] = None,
        simulate_dispatcher: Optional[Callable] = None,
    ) -> None:
        from host_builtins_v04 import default_builtins  # type: ignore[import-not-found]

        self.env: Dict[str, Any] = {}
        self.trace: Trace = Trace()
        self.builtins: Builtins = builtins or default_builtins()
        # Filled in by Interpreter.__init__ so simulate blocks can
        # route to simulate_dispatch without a circular import here.
        self._simulate_dispatcher = simulate_dispatcher
        # For nested ``experiment(...)`` invocations — map of
        # experiment name -> callable.  Populated as ExperimentDef
        # decls are evaluated.
        self.experiments: Dict[str, Any] = {}
        # Stdout capture — used by ``print`` and surfaced through the
        # interpreter for tests.
        self.stdout = io.StringIO()

    # =================================================================
    # Top-level program
    # =================================================================

    def run(self, program: Program) -> None:
        for decl in program.decls:
            self._run_decl(decl)

    def _run_decl(self, decl) -> None:
        if isinstance(decl, LetStmt):
            self._run_let(decl)
        elif isinstance(decl, PrintStmt):
            self._run_print(decl)
        elif isinstance(decl, ExperimentDef):
            self._run_experiment_def(decl)
        else:
            raise QLangSyntaxError(
                f"unsupported top-level declaration: {type(decl).__name__}",
                line=getattr(decl, "line", None),
                col=getattr(decl, "col", None),
            )

    # =================================================================
    # let / print
    # =================================================================

    def _run_let(self, node: LetStmt) -> None:
        if node.name in self.env:
            raise RebindingError(node.name, line=node.line, col=node.col)
        value = self._eval(node.value)
        # Wrap the value with a bind node
        bn = bind_node(node.name, _nid(value))
        self.trace.push(bn)
        # Stored value's node_id remains the computed one
        self.env[node.name] = value

    def _run_print(self, node: PrintStmt) -> None:
        v = self._eval(node.value)
        text = self._render_for_print(v)
        self.stdout.write(text + "\n")
        # Record the print in the trace; point the input to whatever
        # was printed (so replay can reconstruct it).
        pn = print_node(text, _nid(v))
        self.trace.push(pn)

    def _render_for_print(self, value: Any) -> str:
        """Render a value for stdout.  For strings, do {var}
        interpolation against the current env; for Quantities or
        Hosts, use their native formatting."""
        if isinstance(value, String):
            return _interpolate(value.value, self.env)
        if isinstance(value, Quantity):
            return format(value)
        if isinstance(value, Host):
            return repr(value)
        return str(value)

    # =================================================================
    # Experiments — hand off to v0.4/experiment.py
    # =================================================================

    def _run_experiment_def(self, node: ExperimentDef) -> None:
        from experiment_v04 import ExperimentBlock  # type: ignore[import-not-found]

        eb = ExperimentBlock(node, evaluator=self)
        if node.name in self.env or node.name in self.experiments:
            raise RebindingError(node.name, line=node.line, col=node.col)
        self.experiments[node.name] = eb
        self.env[node.name] = eb

    # =================================================================
    # Expression evaluation
    # =================================================================

    def _eval(self, expr) -> Any:
        if isinstance(expr, NumberLiteral):
            return self._eval_number(expr)
        if isinstance(expr, StringLiteral):
            return self._eval_string(expr)
        if isinstance(expr, Identifier):
            return self._eval_identifier(expr)
        if isinstance(expr, UnitConversion):
            return self._eval_unit_conversion(expr)
        if isinstance(expr, UncertaintyExpr):
            return self._eval_uncertainty(expr)
        if isinstance(expr, UnaryOp):
            return self._eval_unary(expr)
        if isinstance(expr, BinaryOp):
            return self._eval_binary(expr)
        if isinstance(expr, PipeExpr):
            return self._eval_pipe(expr)
        if isinstance(expr, Call):
            return self._eval_call(expr)
        if isinstance(expr, SimulateExpr):
            return self._eval_simulate(expr)

        raise QLangSyntaxError(
            f"evaluator: unsupported node {type(expr).__name__}",
            line=getattr(expr, "line", None),
            col=getattr(expr, "col", None),
        )

    # -----------------------------------------------------------------
    # Literals
    # -----------------------------------------------------------------

    def _eval_number(self, node: NumberLiteral) -> Quantity:
        mag = float(node.text)
        scale = 1.0
        dim = SCALAR_DIM
        if node.unit is not None:
            try:
                scale, dim = resolve_unit_expr(node.unit.atoms)
            except KeyError as e:
                raise QLangSyntaxError(
                    str(e),
                    line=node.unit.line,
                    col=node.unit.col,
                )
        # The literal's source unit name (e.g. "Bohr") is preserved as
        # the Quantity's display_unit, while dim governs arithmetic.
        display_unit = _format_unit_expr(node.unit) if node.unit is not None else ""
        q = quantity_from_literal(mag, scale, dim)
        q = replace(q, display_unit=display_unit)
        # Record literal in trace.  The trace's unit string carries
        # the source name when present, else "1" for dimensionless.
        unit_name = display_unit if display_unit else "1"
        ln = literal_node(q.magnitude, unit_name, q.uncertainty)
        self.trace.push(ln)
        return q.with_node(ln.node_id)

    def _eval_string(self, node: StringLiteral) -> String:
        ln = string_literal_node(node.value)
        self.trace.push(ln)
        return String(node.value, node_id=ln.node_id)

    # -----------------------------------------------------------------
    # Identifiers
    # -----------------------------------------------------------------

    def _eval_identifier(self, node: Identifier) -> Any:
        if node.name not in self.env:
            raise UnboundNameError(node.name, line=node.line, col=node.col)
        return self.env[node.name]

    # -----------------------------------------------------------------
    # Unit conversion: ``expr in [unit]``
    # -----------------------------------------------------------------

    def _eval_unit_conversion(self, node: UnitConversion) -> Quantity:
        inner = self._eval(node.expr)
        if not isinstance(inner, Quantity):
            raise DimensionMismatchError(
                "unit conversion 'in [unit]' requires a Quantity",
                line=node.line,
                col=node.col,
            )
        try:
            target_scale, target_dim = resolve_unit_expr(
                node.target_unit.atoms,
            )
        except KeyError as e:
            raise QLangSyntaxError(
                str(e),
                line=node.target_unit.line,
                col=node.target_unit.col,
            )
        if inner.dim != target_dim:
            raise DimensionMismatchError(
                f"cannot convert [{inner.dim}] to [{target_dim}]",
                have=str(inner.dim),
                need=str(target_dim),
                line=node.line,
                col=node.col,
            )
        # Conversion semantics (v0.4): magnitudes live in their
        # declared unit.  To convert a Quantity that is already
        # dimensionally compatible with the target, we need to know
        # the source unit's scale.  Since the Quantity doesn't carry
        # an explicit source-unit string (just the Dim), we use the
        # trace's last literal/convert node as the source unit hint
        # WHEN the inner expression is a literal — otherwise we
        # require the source to be expressed in SI base units of the
        # target dimension.
        #
        # In practice: for v0.4's grammar ``LITERAL [UNIT] in [UNIT2]``
        # the inner magnitude is in the LITERAL's declared unit.
        # The evaluator needs access to that source scale.  Cleanest
        # route: look through the inner node's trace entry.
        source_scale = _infer_source_scale(self.trace, inner)
        converted_mag = inner.magnitude * source_scale / target_scale
        unc_target = (
            inner.uncertainty * source_scale / target_scale
            if inner.uncertainty is not None
            else None
        )
        new_node = make_node(
            "convert",
            inputs=(inner.node_id,),
            value={
                "magnitude": converted_mag,
                "unit": _format_unit_expr(node.target_unit),
                "uncertainty": unc_target,
            },
            meta={"target_unit": _format_unit_expr(node.target_unit)},
        )
        self.trace.push(new_node)
        return Quantity(
            magnitude=converted_mag,
            dim=target_dim,
            uncertainty=unc_target,
            node_id=new_node.node_id,
            display_unit=_format_unit_expr(node.target_unit),
        )

    # -----------------------------------------------------------------
    # Uncertainty: ``value +/- unc``
    # -----------------------------------------------------------------

    def _eval_uncertainty(self, node: UncertaintyExpr) -> Quantity:
        v = self._eval(node.value)
        u = self._eval(node.uncertainty)
        if not isinstance(v, Quantity) or not isinstance(u, Quantity):
            raise DimensionMismatchError(
                "+/- operator requires numeric operands",
                line=node.line,
                col=node.col,
            )
        if v.dim != u.dim:
            raise DimensionMismatchError(
                f"uncertainty must have same dim as value: [{u.dim}] vs [{v.dim}]",
                have=str(u.dim),
                need=str(v.dim),
                line=node.line,
                col=node.col,
            )
        mag = v.magnitude
        unc = abs(u.magnitude)
        # Preserve the value's display_unit — the ``+/-`` is a pure
        # uncertainty annotation, not a unit change.
        display_unit = v.display_unit or u.display_unit or ""
        unit_for_trace = display_unit if display_unit else format_unit(v.dim)
        n = make_node(
            "uncertainty",
            inputs=(v.node_id, u.node_id),
            value={
                "magnitude": mag,
                "unit": unit_for_trace,
                "uncertainty": unc,
            },
        )
        self.trace.push(n)
        return Quantity(
            magnitude=mag,
            dim=v.dim,
            uncertainty=unc,
            node_id=n.node_id,
            display_unit=display_unit,
        )

    # -----------------------------------------------------------------
    # Unary / binary arithmetic (SPEC §5, §6)
    # -----------------------------------------------------------------

    def _eval_unary(self, node: UnaryOp) -> Quantity:
        x = self._eval(node.operand)
        if not isinstance(x, Quantity):
            raise DimensionMismatchError(
                "unary '-' requires a numeric operand",
                line=node.line,
                col=node.col,
            )
        new_mag = -x.magnitude
        un = unary_op_node(
            "neg",
            x.node_id,
            new_mag,
            format_unit(x.dim),
            x.uncertainty,
        )
        self.trace.push(un)
        return Quantity(
            magnitude=new_mag,
            dim=x.dim,
            uncertainty=x.uncertainty,
            node_id=un.node_id,
        )

    def _eval_binary(self, node: BinaryOp) -> Quantity:
        left = self._eval(node.left)
        right = self._eval(node.right)
        op = node.op

        if op in ("+", "-", "*", "/", "**"):
            return self._eval_arith(node, left, right)
        if op in ("<", "<=", ">", ">=", "==", "!="):
            return self._eval_comparison(node, left, right)

        raise QLangSyntaxError(
            f"unknown binary operator {op!r}",
            line=node.line,
            col=node.col,
        )

    def _eval_arith(self, node: BinaryOp, left, right) -> Quantity:
        if not (isinstance(left, Quantity) and isinstance(right, Quantity)):
            raise DimensionMismatchError(
                f"operator {node.op!r} requires numeric operands",
                line=node.line,
                col=node.col,
            )
        op = node.op
        a, b = left.magnitude, right.magnitude
        sa, sb = left.uncertainty, right.uncertainty

        # Dim rules (SPEC §5)
        if op in ("+", "-"):
            if left.dim != right.dim:
                raise DimensionMismatchError(
                    f"cannot {('add' if op == '+' else 'subtract')} "
                    f"[{left.dim}] and [{right.dim}]",
                    have=str(left.dim),
                    need=str(right.dim),
                    line=node.line,
                    col=node.col,
                )
            new_dim = left.dim
            if op == "+":
                new_mag = a + b
                new_unc = _prop_add(sa, sb)
                tag = "add"
            else:
                new_mag = a - b
                new_unc = _prop_add(sa, sb)
                tag = "sub"
        elif op == "*":
            new_dim = left.dim + right.dim
            new_mag = a * b
            new_unc = _prop_mul(a, b, sa, sb)
            tag = "mul"
        elif op == "/":
            if b == 0.0:
                raise KernelError(
                    "divide",
                    ZeroDivisionError("division by zero"),
                    line=node.line,
                    col=node.col,
                )
            new_dim = left.dim - right.dim
            new_mag = a / b
            new_unc = _prop_div(a, b, sa, sb)
            tag = "div"
        elif op == "**":
            # right must be dimensionless integer-valued scalar
            if not right.dim.is_dimensionless():
                raise DimensionMismatchError(
                    f"exponent must be dimensionless, got [{right.dim}]",
                    have=str(right.dim),
                    need="1",
                    line=node.line,
                    col=node.col,
                )
            if not float(b).is_integer():
                raise DimensionMismatchError(
                    f"exponent must be an integer, got {b}",
                    line=node.line,
                    col=node.col,
                )
            n = int(b)
            new_dim = left.dim * n
            new_mag = a**n
            new_unc = _prop_pow(a, n, sa)
            tag = "pow"
        else:
            raise QLangSyntaxError(
                f"unreachable: arith op {op!r}",
                line=node.line,
                col=node.col,
            )

        # Display-unit propagation rules:
        #   + / -  : result keeps left.display_unit (both sides had
        #            matching dims so they almost always agree anyway).
        #   * / /  : if both sides have the same display_unit we can
        #            render as unit^k; otherwise fall back to SI-canonical.
        #   **     : keep left.display_unit if simple (^1); otherwise SI.
        new_display = ""
        if tag in ("add", "sub"):
            new_display = left.display_unit or right.display_unit or ""
        elif tag in ("mul", "div"):
            if (
                left.display_unit
                and right.display_unit
                and left.display_unit == right.display_unit
            ):
                new_display = f"{left.display_unit}^2" if tag == "mul" else "1"
            elif left.display_unit and right.dim.is_dimensionless():
                new_display = left.display_unit
            elif right.display_unit and left.dim.is_dimensionless():
                new_display = right.display_unit

        bn = binary_op_node(
            tag,
            left.node_id,
            right.node_id,
            new_mag,
            new_display or format_unit(new_dim),
            new_unc,
        )
        self.trace.push(bn)
        return Quantity(
            magnitude=new_mag,
            dim=new_dim,
            uncertainty=new_unc,
            node_id=bn.node_id,
            display_unit=new_display,
        )

    def _eval_comparison(self, node: BinaryOp, left, right) -> Quantity:
        if not (isinstance(left, Quantity) and isinstance(right, Quantity)):
            raise DimensionMismatchError(
                f"comparison {node.op!r} requires numeric operands",
                line=node.line,
                col=node.col,
            )
        if left.dim != right.dim:
            raise DimensionMismatchError(
                f"cannot compare [{left.dim}] with [{right.dim}]",
                have=str(left.dim),
                need=str(right.dim),
                line=node.line,
                col=node.col,
            )
        a, b = left.magnitude, right.magnitude
        op = node.op
        result_bool = {
            "<": a < b,
            "<=": a <= b,
            ">": a > b,
            ">=": a >= b,
            "==": a == b,
            "!=": a != b,
        }[op]
        new_mag = 1.0 if result_bool else 0.0
        tag = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}[op]
        bn = binary_op_node(
            tag,
            left.node_id,
            right.node_id,
            new_mag,
            "1",
            None,
        )
        self.trace.push(bn)
        return Quantity(
            magnitude=new_mag,
            dim=SCALAR_DIM,
            uncertainty=None,
            node_id=bn.node_id,
        )

    # -----------------------------------------------------------------
    # Pipe: ``x |> f(a, b)`` ≡ ``f(x, a, b)``
    # -----------------------------------------------------------------

    def _eval_pipe(self, node: PipeExpr) -> Any:
        left_val = self._eval(node.left)
        call = node.right
        # Pre-pend ``left_val`` to the Call's positional args.
        return self._invoke_call(
            call,
            preposed_positional=[left_val],
        )

    # -----------------------------------------------------------------
    # Call
    # -----------------------------------------------------------------

    def _eval_call(self, node: Call) -> Any:
        return self._invoke_call(node, preposed_positional=[])

    def _invoke_call(self, node: Call, preposed_positional: List[Any]) -> Any:
        # Evaluate all positional and keyword arguments (in source order).
        pos_vals = list(preposed_positional)
        for arg_expr in node.args:
            pos_vals.append(self._eval(arg_expr))
        kw_vals: Dict[str, Any] = {}
        for k, ke in node.kwargs:
            kw_vals[k] = self._eval(ke)

        # Lookup by name: builtins first, then experiments.
        if node.callee in self.builtins:
            be = self.builtins[node.callee]
            try:
                result = be.fn(*pos_vals, **kw_vals)
            except (DimensionMismatchError, QLangSyntaxError, KernelError):
                raise
            except Exception as e:
                raise KernelError(
                    node.callee,
                    e,
                    line=node.line,
                    col=node.col,
                )
            # If the builtin returned a Value already, trust its node.
            # Otherwise, wrap.
            return _ensure_value(
                result,
                self.trace,
                origin=node.callee,
                input_ids=tuple(_nid(v) for v in pos_vals),
                line=node.line,
                col=node.col,
            )

        if node.callee in self.experiments:
            eb = self.experiments[node.callee]
            return eb.invoke(kw_vals, line=node.line, col=node.col)

        raise UnboundNameError(
            node.callee,
            line=node.line,
            col=node.col,
        )

    # -----------------------------------------------------------------
    # simulate DOMAIN { ... }
    # -----------------------------------------------------------------

    def _eval_simulate(self, node: SimulateExpr) -> Any:
        if self._simulate_dispatcher is None:
            raise QLangSyntaxError(
                "internal error: simulate dispatcher not attached",
                line=node.line,
                col=node.col,
            )
        # Validate the domain BEFORE evaluating kwargs so that
        # domain-specific sentinel names (e.g. ``conserve: total_energy``)
        # don't get eagerly resolved as identifier lookups — the
        # dispatcher will interpret them itself.
        from simulate_dispatch_v04 import (  # type: ignore[import-not-found]
            _REGISTRY,
            pre_validate,
        )

        if node.domain not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise QLangSyntaxError(
                f"unknown simulate domain {node.domain!r}; known: {known}",
                line=node.line,
                col=node.col,
            )

        # Domain-specific AST-level validation runs first, using the
        # raw kwargs (un-evaluated identifiers visible as-is).
        pre_validate(node.domain, node.kwargs, line=node.line, col=node.col)

        # Evaluate kwargs eagerly; dispatcher sees resolved Values.
        # For ``conserve:`` the evaluator skips identifier resolution
        # — the dispatcher was validated above, so the identifier is
        # treated as a symbolic tag.
        kw_vals: Dict[str, Any] = {}
        input_ids: List[str] = []
        _tag_kwargs = _SYMBOLIC_TAG_KWARGS.get(node.domain, frozenset())
        for k, ke in node.kwargs:
            if k in _tag_kwargs and type(ke).__name__ == "Identifier":
                # Preserve as a String tag rather than resolving.
                tag_value = String(value=ke.name)
                kw_vals[k] = tag_value
                input_ids.append("")
                continue
            v = self._eval(ke)
            kw_vals[k] = v
            input_ids.append(_nid(v))
        return self._simulate_dispatcher(
            domain=node.domain,
            kwargs=kw_vals,
            input_ids=tuple(input_ids),
            line=node.line,
            col=node.col,
            trace=self.trace,
        )
        # Evaluate kwargs eagerly; dispatcher sees resolved Values.
        kw_vals: Dict[str, Any] = {}
        input_ids: List[str] = []
        for k, ke in node.kwargs:
            v = self._eval(ke)
            kw_vals[k] = v
            input_ids.append(_nid(v))
        return self._simulate_dispatcher(
            domain=node.domain,
            kwargs=kw_vals,
            input_ids=tuple(input_ids),
            line=node.line,
            col=node.col,
            trace=self.trace,
        )

    # =================================================================
    # Scope helper for experiment evaluation (used by experiment.py)
    # =================================================================

    @contextmanager
    def child_scope(self, extras: Dict[str, Any]):
        """Push a scope layer for experiment-local ``given:`` params.
        The v0.4 scope model is one flat env; child_scope just saves
        and restores the overlapping keys."""
        saved: Dict[str, Any] = {}
        for k, v in extras.items():
            if k in self.env:
                saved[k] = self.env[k]
            self.env[k] = v
        try:
            yield
        finally:
            for k in extras:
                if k in saved:
                    self.env[k] = saved[k]
                else:
                    self.env.pop(k, None)


# ─────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────


def _nid(value: Any) -> str:
    return getattr(value, "node_id", "") or ""


def _prop_add(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    ax = a or 0.0
    bx = b or 0.0
    return math.sqrt(ax * ax + bx * bx)


def _prop_mul(
    a: float, b: float, sa: Optional[float], sb: Optional[float]
) -> Optional[float]:
    if sa is None and sb is None:
        return None
    sax = sa or 0.0
    sbx = sb or 0.0
    # σ_f = |f| * sqrt((σa/a)² + (σb/b)²) for f = a*b
    if a == 0.0 or b == 0.0:
        # Fall back to simpler bound
        return math.sqrt((b * sax) ** 2 + (a * sbx) ** 2)
    return abs(a * b) * math.sqrt((sax / a) ** 2 + (sbx / b) ** 2)


def _prop_div(
    a: float, b: float, sa: Optional[float], sb: Optional[float]
) -> Optional[float]:
    if sa is None and sb is None:
        return None
    sax = sa or 0.0
    sbx = sb or 0.0
    if a == 0.0:
        return math.sqrt((sax / b) ** 2 + ((a * sbx) / (b * b)) ** 2)
    return abs(a / b) * math.sqrt((sax / a) ** 2 + (sbx / b) ** 2)


def _prop_pow(a: float, n: int, sa: Optional[float]) -> Optional[float]:
    if sa is None:
        return None
    # σ_{a^n} = |n * a^(n-1)| * σ_a
    return abs(n * a ** (n - 1)) * sa


def _infer_source_scale(trace: Trace, q: Quantity) -> float:
    """Return the scale-to-SI factor for a Quantity's declared unit.

    Priority order:
      1. ``q.display_unit`` (set at literal site, preserved through
         uncertainty and arithmetic that doesn't change units).
      2. The trace node at ``q.node_id`` — used when the Quantity
         was produced by a non-arithmetic path that didn't propagate
         display_unit (e.g. a kernel result).
      3. Fallback 1.0 for pure arithmetic results whose dim is
         the canonical SI rendering.
    """
    from units_v04 import lookup_unit  # type: ignore[import-not-found]

    if q.display_unit:
        try:
            scale, _ = lookup_unit(q.display_unit)
            return scale
        except KeyError:
            pass

    node = trace.get(q.node_id)
    if node is None:
        return 1.0
    if node.op == "literal" and node.value is not None:
        unit_name = node.value.get("unit", "")
        if unit_name:
            try:
                scale, _ = lookup_unit(unit_name)
                return scale
            except KeyError:
                return 1.0
    if node.op == "convert":
        target = node.meta.get("target_unit", "")
        if target:
            try:
                scale, _ = lookup_unit(target)
                return scale
            except KeyError:
                return 1.0
    return 1.0


def _format_unit_expr(uexpr: UnitExpr) -> str:
    """Render a UnitExpr AST node as a string for the trace."""
    parts: List[str] = []
    for atom in uexpr.atoms:
        s = atom.name
        if atom.exponent != 1:
            s = f"{s}^{atom.exponent}"
        parts.append(s)
    return "·".join(parts)


def _ensure_value(
    result,
    trace: Trace,
    *,
    origin: str,
    input_ids: Tuple[str, ...],
    line: Optional[int],
    col: Optional[int],
) -> Any:
    """If ``result`` is already a v0.4 Value (Quantity | String | Host),
    return it.  Otherwise wrap a raw Python scalar."""
    if isinstance(result, (Quantity, String, Host)):
        return result
    if isinstance(result, bool):
        mag = 1.0 if result else 0.0
        n = make_node(
            f"builtin.{origin}.bool",
            inputs=input_ids,
            value={"magnitude": mag, "unit": "1", "uncertainty": None},
        )
        trace.push(n)
        return Quantity(
            magnitude=mag, dim=SCALAR_DIM, uncertainty=None, node_id=n.node_id
        )
    if isinstance(result, (int, float)):
        n = make_node(
            f"builtin.{origin}",
            inputs=input_ids,
            value={"magnitude": float(result), "unit": "1", "uncertainty": None},
        )
        trace.push(n)
        return Quantity(
            magnitude=float(result), dim=SCALAR_DIM, uncertainty=None, node_id=n.node_id
        )
    if isinstance(result, str):
        from provenance_v04 import string_literal_node as sln  # local import

        n = sln(result)
        trace.push(n)
        return String(value=result, node_id=n.node_id)
    # Fallback: wrap in Host
    n = make_node(
        f"builtin.{origin}.host",
        inputs=input_ids,
        value={"magnitude": repr(result), "unit": "", "uncertainty": None},
    )
    trace.push(n)
    return Host(payload=result, dim=SCALAR_DIM, node_id=n.node_id)
