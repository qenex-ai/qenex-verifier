"""
Q-Lang v0.4 — public Interpreter facade (SPEC §2.2 public API).

Glue layer that stitches together parser + evaluator + simulate
dispatcher + trace I/O into the single class that users see:

    from qlang_v04 import QLangInterpreter
    interp = QLangInterpreter()
    interp.run(source)
    value = interp.env['my_binding']
    for node in interp.trace:
        ...
    interp.write_trace(path)

The module name is ``interpreter`` but it is imported from the v0.4
package as ``QLangInterpreter`` (see ``__init__.py``).

Design notes
------------

* State is private; inspection is by attribute read, not method.
* Every error path propagates as a typed ``QLangError`` subclass.
* ``stdout`` is captured into a StringIO for test assertions; the
  user's actual terminal is NOT touched by ``interp.run`` — if you
  want to see output, pass ``echo=True`` or use the CLI.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

from errors_v04 import QLangError  # type: ignore[import-not-found]
from evaluator_v04 import Evaluator  # type: ignore[import-not-found]
from parser_v04 import parse  # type: ignore[import-not-found]
from provenance_v04 import Trace  # type: ignore[import-not-found]

# Importing simulate_dispatch registers the default domains.
import simulate_dispatch_v04 as simulate_dispatch  # type: ignore[import-not-found]  # noqa: F401


class QLangInterpreter:
    """Public entry point for Q-Lang v0.4.

    One instance manages one evaluation context: its environment,
    builtins, and trace persist across multiple ``run()`` calls so
    you can add more bindings incrementally.

    Attributes
    ----------
    env : dict[str, Value]
        Top-level bindings.
    trace : Trace
        Ordered collection of ``DerivationNode``s produced during
        every ``run()`` so far.
    stdout : str
        Everything emitted by ``print`` statements, as one string.
    """

    def __init__(self, *, echo: bool = False) -> None:
        self._evaluator = Evaluator(
            simulate_dispatcher=simulate_dispatch.dispatch,
        )
        self._echo = echo

    # -----------------------------------------------------------------
    # Public execution
    # -----------------------------------------------------------------

    def run(self, source: str) -> None:
        """Parse and evaluate a Q-Lang program.

        Raises any ``QLangError`` subclass on failure.  A failure
        leaves the environment in the state it was before the failing
        statement (the evaluator does not partially commit a failing
        ``let``).
        """
        program = parse(source)
        self._evaluator.run(program)
        if self._echo:
            text = self._evaluator.stdout.getvalue()
            if text:
                print(text, end="")
            # Reset buffer so echo doesn't double next run
            self._evaluator.stdout = io.StringIO()

    # -----------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------

    @property
    def env(self) -> Dict[str, Any]:
        return self._evaluator.env

    @property
    def trace(self) -> Trace:
        return self._evaluator.trace

    @property
    def stdout(self) -> str:
        return self._evaluator.stdout.getvalue()

    # -----------------------------------------------------------------
    # Trace I/O
    # -----------------------------------------------------------------

    def write_trace(self, path) -> None:
        """Write the accumulated trace to ``path`` as JSONL.  The
        target directory must exist."""
        self._evaluator.trace.write(Path(path))


__all__ = ["QLangInterpreter", "QLangError"]
