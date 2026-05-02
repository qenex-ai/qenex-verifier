"""
Q-Lang v0.4 — thin re-export shim so ``from qlang_v04 import ...`` works
regardless of whether the caller has the ``src/`` directory on
``sys.path`` (they normally do, courtesy of ``tests/conftest.py``).

Everything lives under ``src/v04/`` but is exported at the top level
via this module for the ergonomic imports that the test suite and
``__init__.py`` rely on.
"""

from __future__ import annotations

import os as _os
import sys as _sys

_V04_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "v04")
if _V04_DIR not in _sys.path:
    _sys.path.insert(0, _V04_DIR)


from errors_v04 import (  # type: ignore[import-not-found]  # noqa: E402
    ConservationViolation,
    DimensionMismatchError,
    DriftReport,
    InvariantViolation,
    KernelError,
    QLangError,
    QLangSyntaxError,
    RebindingError,
    ReplayDriftError,
    UnboundNameError,
)
from interp_v04 import QLangInterpreter  # type: ignore[import-not-found]  # noqa: E402
from provenance_v04 import DerivationNode, Trace  # type: ignore[import-not-found]  # noqa: E402
from replay_v04 import replay  # type: ignore[import-not-found]  # noqa: E402
from cli_v04 import main as cli_main  # type: ignore[import-not-found]  # noqa: E402


__version__ = "0.4.0-dev"
__all__ = [
    "QLangInterpreter",
    "QLangError",
    "QLangSyntaxError",
    "DimensionMismatchError",
    "UnboundNameError",
    "RebindingError",
    "InvariantViolation",
    "ConservationViolation",
    "KernelError",
    "ReplayDriftError",
    "DriftReport",
    "DerivationNode",
    "Trace",
    "replay",
    "cli_main",
]


if __name__ == "__main__":
    raise SystemExit(cli_main())
