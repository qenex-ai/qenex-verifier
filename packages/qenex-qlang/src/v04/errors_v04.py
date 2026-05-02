"""
Q-Lang v0.4 — typed error hierarchy (SPEC §8).

Every failure surface in v0.4 is a typed subclass of ``QLangError``.
No failure path uses ``print + return sentinel``; every one raises.

Construction convention
-----------------------
Where the error refers to a source location, pass ``line`` and
``col`` (1-indexed) so higher layers can render a caret-annotated
diagnostic.  When the error originates from a wrapped kernel call,
pass the original exception as ``cause`` — it is preserved for
tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


class QLangError(Exception):
    """Root of the Q-Lang v0.4 error hierarchy."""

    def __init__(
        self, message: str, *, line: Optional[int] = None, col: Optional[int] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.col = col

    def __str__(self) -> str:
        if self.line is not None and self.col is not None:
            return f"[line {self.line}, col {self.col}] {self.message}"
        if self.line is not None:
            return f"[line {self.line}] {self.message}"
        return self.message


class QLangSyntaxError(QLangError):
    """Lexer or parser rejected the input.  Raised for any grammar
    violation, and for any attempt to use syntax explicitly deferred
    to v0.5+ (SPEC §3.2).  The ``feature`` slot names the deferred
    construct (e.g. ``"if"``, ``"fn"``) when applicable."""

    def __init__(
        self,
        message: str,
        *,
        line: Optional[int] = None,
        col: Optional[int] = None,
        feature: Optional[str] = None,
    ) -> None:
        super().__init__(message, line=line, col=col)
        self.feature = feature


class DimensionMismatchError(QLangError):
    """Arithmetic or call violates SPEC §5 dimensional rules.
    ``have`` and ``need`` are the two incompatible dimensions, shown
    as e.g. ``"[m]"`` and ``"[kg]"``."""

    def __init__(
        self,
        message: str,
        *,
        have: str = "",
        need: str = "",
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> None:
        super().__init__(message, line=line, col=col)
        self.have = have
        self.need = need


class UnboundNameError(QLangError):
    """An identifier was looked up before being bound (SPEC §8)."""

    def __init__(
        self, name: str, *, line: Optional[int] = None, col: Optional[int] = None
    ) -> None:
        super().__init__(f"unbound name: {name!r}", line=line, col=col)
        self.name = name


class RebindingError(QLangError):
    """Attempt to ``let`` a name that is already bound (SPEC §3.1).
    Q-Lang v0.4 bindings are immutable."""

    def __init__(
        self, name: str, *, line: Optional[int] = None, col: Optional[int] = None
    ) -> None:
        super().__init__(
            f"cannot re-bind {name!r}: Q-Lang v0.4 bindings are immutable",
            line=line,
            col=col,
        )
        self.name = name


@dataclass
class _InvariantInfo:
    experiment: str
    expression_source: str
    node_id: Optional[str] = None


class InvariantViolation(QLangError):
    """An ``invariant:`` clause evaluated to false at ``result:`` time
    (SPEC §1.1 guarantee 2).  The experiment produces NO result."""

    def __init__(
        self,
        experiment: str,
        expression_source: str,
        *,
        node_id: Optional[str] = None,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"invariant failed in experiment {experiment!r}: {expression_source}",
            line=line,
            col=col,
        )
        self.info = _InvariantInfo(
            experiment=experiment,
            expression_source=expression_source,
            node_id=node_id,
        )


class ConservationViolation(QLangError):
    """A ``conserve:`` declaration detected drift > ``tolerance:`` on
    a simulated trajectory (SPEC §3.3)."""

    def __init__(
        self,
        quantity: str,
        drift: float,
        tolerance: float,
        *,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"conservation violated: {quantity!r} drift={drift:.3e} > "
            f"tolerance={tolerance:.3e}",
            line=line,
            col=col,
        )
        self.quantity = quantity
        self.drift = drift
        self.tolerance = tolerance


class KernelError(QLangError):
    """A host builtin raised an exception during evaluation.  The
    original exception is preserved as ``cause`` for full tracebacks."""

    def __init__(
        self,
        builtin: str,
        cause: BaseException,
        *,
        line: Optional[int] = None,
        col: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"kernel {builtin!r} raised {type(cause).__name__}: {cause}",
            line=line,
            col=col,
        )
        self.builtin = builtin
        self.cause = cause
        self.__cause__ = cause  # shows up in traceback


@dataclass
class DriftReport:
    """Returned by ``qlang replay`` when a trace fails to reproduce."""

    ok: bool
    drifted_nodes: Tuple[str, ...] = ()
    first_mismatch: Optional[dict] = None


class ReplayDriftError(QLangError):
    """A ``qlang replay`` detected that the current QENEX no longer
    reproduces the recorded trace bitwise (SPEC §7.3)."""

    def __init__(self, report: DriftReport) -> None:
        super().__init__(
            f"replay drift detected in {len(report.drifted_nodes)} node(s); "
            f"first mismatch: {report.first_mismatch}"
        )
        self.report = report
