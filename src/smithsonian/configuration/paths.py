"""Locate repository resources without depending on the caller's directory."""

import os
from pathlib import Path


def repository_root() -> Path:
    """Find the checkout containing the checkpoint configuration.

    Returns:
        Absolute checkout path, optionally selected by SMITHSONIAN_REPOSITORY.

    Raises:
        FileNotFoundError: No source checkout is available. Use an editable install.
    """
    configured = os.environ.get("SMITHSONIAN_REPOSITORY")
    candidates = [Path(configured)] if configured else Path(__file__).resolve().parents
    for candidate in candidates:
        if (candidate / "config/checkpoints.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("Install the source checkout with python -m pip install -e .")
