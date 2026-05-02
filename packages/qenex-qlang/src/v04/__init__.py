"""
QENEX Q-Lang v0.4 — Scientific Protocol Language
===============================================

v0.4 is a tree-walking interpreter for the Q-Lang v0.4 specification
(see ``docs/SPEC.md``).  Its three core guarantees are:

  1. Cross-domain kernel dispatch native to QENEX.
  2. Experiment as a non-skippable protocol.
  3. Signed, re-executable provenance trace.

Public API (flat namespace)
---------------------------

    from qlang_v04 import (
        QLangInterpreter,
        QLangError, QLangSyntaxError,
        DimensionMismatchError,
        UnboundNameError, RebindingError,
        InvariantViolation, ConservationViolation,
        KernelError, ReplayDriftError, DriftReport,
        DerivationNode, Trace,
    )
"""

from __future__ import annotations

__version__ = "0.4.0-dev"


# ---------------------------------------------------------------------
# Ensure v0.4's own module directory is on sys.path so the intra-package
# imports (``from lexer_v04 import ...`` etc.) work whether v0.4 is imported
# as ``qlang_v0.4`` (via the thin shim in ``src/qlang_v04.py``) or directly
# as ``v0.4`` (e.g. during ``python -m pytest`` from the workspace root
# with conftest.py's sys.path injection).
# ---------------------------------------------------------------------
import os as _os
import sys as _sys

_V04_DIR = _os.path.dirname(_os.path.abspath(__file__))
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
]
