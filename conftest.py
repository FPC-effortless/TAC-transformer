"""Make the repository root importable for tests.

The test suites import the top-level packages directly (``recon14``,
``tac_transformer``, ``experiments``), which live at the repository root. Bare
``pytest`` does not put the rootdir on ``sys.path`` -- only ``python -m pytest``
does -- so these imports fail on a clean runner unless the root is inserted
explicitly. This module is imported by pytest before any test module, so it is
the right place to do it.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
