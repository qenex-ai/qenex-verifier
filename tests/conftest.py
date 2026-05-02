"""
QENEX Test Configuration (tests/ directory)

Path setup is handled by the root conftest.py.
This file adds tissue package root for tests using `from src.X import ...`.
"""

import sys
import os

# Add tissue package root for legacy `from src.features import ...` style imports
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_tissue_root = os.path.join(WORKSPACE_ROOT, "packages", "qenex-tissue")
if _tissue_root not in sys.path:
    sys.path.insert(0, _tissue_root)
