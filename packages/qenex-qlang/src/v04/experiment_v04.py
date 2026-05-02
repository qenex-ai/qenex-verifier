"""
Q-Lang v0.4 — experiment block + invariant enforcement (SPEC §1.1 #2).

An ``experiment NAME { ... }`` compiles to an ``ExperimentBlock``
value in the evaluator's environment.  Invoking it (``NAME(arg: v,
...)``) runs the body in a child scope with the given parameters
bound, evaluates every ``invariant:`` clause, and only then emits
the ``result:`` value.  If any invariant is false — or is not a
Quantity with non-zero magnitude — the experiment raises
``InvariantViolation`` and NO result value is emitted.  The trace
records every invariant, passed or failed.

This is guarantee #2: experiments are a non-skippable protocol.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ast_nodes_v04 import (  # type: ignore[import-not-found]
    ExperimentDef,
    GivenClause,
    GivenParam,
    InvariantClause,
    LetStmt,
    ResultClause,
    UnitExpr,
)
from errors_v04 import (  # type: ignore[import-not-found]
    DimensionMismatchError,
    InvariantViolation,
    QLangSyntaxError,
    UnboundNameError,
)
from provenance_v04 import invariant_node, result_node  # type: ignore[import-not-found]
from units_v04 import resolve_unit_expr  # type: ignore[import-not-found]
from values_v04 import Quantity  # type: ignore[import-not-found]


class ExperimentBlock:
    """Compiled form of an ``experiment`` declaration.  Invoking it
    runs the body under argument-bound given params.

    ``evaluator`` is the parent evaluator.  The experiment body reuses
    its ``_eval`` and environment via ``child_scope()``.
    """

    def __init__(self, node: ExperimentDef, *, evaluator) -> None:
        self.node = node
        self._eval_ref = evaluator
        self._given: tuple[GivenParam, ...] = ()
        # Ordered body = every non-``given``/``result`` clause in source
        # order.  This lets invariants placed high up reject bad inputs
        # before body ``let``s attempt computations that would otherwise
        # raise KernelError (e.g. sqrt of a negative).
        self._ordered_body: list = []
        self._result_expr: Any = None  # set below; validated non-None

        given_seen = False
        for c in node.clauses:
            if isinstance(c, GivenClause):
                if given_seen:
                    raise QLangSyntaxError(
                        f"experiment {node.name!r} has multiple 'given:' clauses",
                        line=c.line,
                        col=c.col,
                    )
                given_seen = True
                self._given = c.params
            elif isinstance(c, (LetStmt, InvariantClause)):
                self._ordered_body.append(c)
            elif isinstance(c, ResultClause):
                if self._result_expr is not None:
                    raise QLangSyntaxError(
                        f"experiment {node.name!r} has multiple 'result:' clauses",
                        line=c.line,
                        col=c.col,
                    )
                self._result_expr = c.expression
            else:
                raise QLangSyntaxError(
                    f"unsupported clause in experiment {node.name!r}: "
                    f"{type(c).__name__}",
                    line=getattr(c, "line", None),
                    col=getattr(c, "col", None),
                )

        if self._result_expr is None:
            raise QLangSyntaxError(
                f"experiment {node.name!r} has no 'result:' clause",
                line=node.line,
                col=node.col,
            )

    # -----------------------------------------------------------------
    # Invocation
    # -----------------------------------------------------------------

    def invoke(
        self,
        kwargs: Dict[str, Any],
        *,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> Any:
        """Execute the experiment body.  Raises InvariantViolation if
        any invariant fails.  Returns the ``result:`` value otherwise."""

        ev = self._eval_ref

        # ---- bind given params ----
        bindings: Dict[str, Any] = {}
        given_names = {p.name for p in self._given}

        # Unknown kwarg?
        for k in kwargs:
            if k not in given_names:
                raise QLangSyntaxError(
                    f"experiment {self.node.name!r} does not declare "
                    f"given parameter {k!r}",
                    line=line,
                    col=col,
                )

        # Missing kwarg?
        for p in self._given:
            if p.name not in kwargs:
                raise QLangSyntaxError(
                    f"experiment {self.node.name!r} missing required "
                    f"given parameter {p.name!r}",
                    line=line,
                    col=col,
                )
            val = kwargs[p.name]
            if p.unit is not None:
                _check_param_dim(p, val, line, col)
            bindings[p.name] = val

        # ---- run body in source order inside a child scope ----
        # Invariants fire at the point they appear in the source, so
        # ``invariant: M > 0.0 [kg]`` placed before any body ``let``
        # rejects bad inputs upstream of downstream computation.
        with ev.child_scope(bindings):
            for stmt in self._ordered_body:
                if isinstance(stmt, LetStmt):
                    ev._run_let(stmt)
                elif isinstance(stmt, InvariantClause):
                    self._check_invariant(stmt)
                else:
                    raise QLangSyntaxError(
                        f"internal error: unexpected body stmt {type(stmt).__name__}",
                        line=getattr(stmt, "line", None),
                        col=getattr(stmt, "col", None),
                    )

            # Result (only reached if all invariants passed).
            # ``self._result_expr`` is guaranteed non-None by __init__.
            result_val = ev._eval(self._result_expr)

        # Record the result emission in the trace
        value_nid = getattr(result_val, "node_id", "") or ""
        rn = result_node(self.node.name, value_nid)
        ev.trace.push(rn)

        return result_val

    # -----------------------------------------------------------------
    # Invariant checking (routed through ConstraintEnforcer where
    # available — the existing physics_guardrail.py has the
    # machinery; for v0.4's MINIMAL set of invariant styles we do the
    # check locally but record the same shape of trace node the
    # larger validator expects).
    # -----------------------------------------------------------------

    def _check_invariant(self, inv: InvariantClause) -> None:
        ev = self._eval_ref
        try:
            val = ev._eval(inv.expression)
        except (InvariantViolation,):
            raise
        except Exception:
            raise

        passed = _truthy(val)

        # Always record the invariant — passed or failed — in the trace.
        in_id = getattr(val, "node_id", "") or ""
        ev.trace.push(
            invariant_node(
                experiment=self.node.name,
                expression_source=inv.source_text,
                input_id=in_id,
                passed=passed,
            )
        )

        if not passed:
            raise InvariantViolation(
                experiment=self.node.name,
                expression_source=inv.source_text,
                node_id=in_id,
                line=inv.line,
                col=inv.col,
            )


def _check_param_dim(
    p: GivenParam,
    val: Any,
    line: Optional[int],
    col: Optional[int],
) -> None:
    """Enforce the dim declared on a ``given: name in [unit]`` param."""
    if p.unit is None:
        return
    try:
        _, required_dim = resolve_unit_expr(p.unit.atoms)
    except KeyError as e:
        raise QLangSyntaxError(
            str(e),
            line=p.unit.line,
            col=p.unit.col,
        )
    if not isinstance(val, Quantity):
        raise DimensionMismatchError(
            f"parameter {p.name!r} requires a Quantity in [{required_dim}]",
            line=line,
            col=col,
        )
    if val.dim != required_dim:
        raise DimensionMismatchError(
            f"parameter {p.name!r}: got [{val.dim}], need [{required_dim}]",
            have=str(val.dim),
            need=str(required_dim),
            line=line,
            col=col,
        )


def _truthy(val: Any) -> bool:
    """Experiment invariants interpret their expression as boolean.

    Comparison operators return a Quantity with magnitude 0 or 1
    (see ``evaluator._eval_comparison``).  A plain Quantity with
    magnitude 0 is false, any other non-zero value is true.  Strings
    and Hosts are never truthy — using them as an invariant is a
    programming error.
    """
    if isinstance(val, Quantity):
        return val.magnitude != 0.0
    return False
