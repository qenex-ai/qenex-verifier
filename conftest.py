"""
QENEX Verifier — pytest sys.path injection.

Mirrors the lab's conftest.py pattern so all chemistry tests use direct
imports (e.g. `from solver import HartreeFockSolver`) without needing a
`pip install -e .` step. This keeps the verifier subset a faithful copy
of the lab's chemistry layer.

The lab equivalent is `qenex-lab/workspace/conftest.py`.
"""

import os
import sys

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))

_PACKAGE_SRC_DIRS = [
    os.path.join(WORKSPACE_ROOT, "packages", "qenex_chem", "src"),
    os.path.join(WORKSPACE_ROOT, "packages", "qenex-core", "src"),
    os.path.join(WORKSPACE_ROOT, "packages", "qenex-qlang", "src"),
    WORKSPACE_ROOT,
]

for _p in _PACKAGE_SRC_DIRS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Pytest exclusions
collect_ignore_glob = [
    ".venv/*",
    "**/__pycache__/*",
]
