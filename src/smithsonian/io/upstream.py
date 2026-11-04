"""Explicit loading of the pinned Spring team's external inference package.

Attribution: Abbas et al., Rice D2K Spring 2026; full source revision in NOTICE.md.
The upstream package is named ``src``. It is registered only when requested.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

SPRING_REVISION = "1b62b68a0d95178e6275ef55c38ee4765317b334"


def load_spring_source(path=None):
    """Register a verified external checkout without changing sys.path.

    Args:
        path: Checkout directory; defaults to SPRING_SOURCE or external/spring.

    Returns:
        Resolved source checkout directory.

    Raises:
        FileNotFoundError: Source checkout is unavailable.
        ValueError: Revision differs or another package occupies the src namespace.
    """
    root = Path(path or os.environ.get("SPRING_SOURCE", "external/spring")).resolve()
    if not (root / "src/annotator/pipeline.py").is_file():
        raise FileNotFoundError("Retrieve the pinned Spring source; see docs/DATA.md")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != SPRING_REVISION:
        raise ValueError("Spring source revision differs from NOTICE.md")
    existing = sys.modules.get("src")
    if existing is not None:
        if str(root / "src") not in list(getattr(existing, "__path__", [])):
            raise ValueError("Another package occupies the upstream 'src' namespace")
        return root
    initializer = root / "src/__init__.py"
    if initializer.is_file():
        spec = importlib.util.spec_from_file_location(
            "src", initializer, submodule_search_locations=[str(root / "src")]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["src"] = module
        spec.loader.exec_module(module)
    else:
        module = types.ModuleType("src")
        module.__path__ = [str(root / "src")]
        sys.modules["src"] = module
    return root
