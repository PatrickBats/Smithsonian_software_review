"""JSON helpers for detector crop preparation."""

from smithsonian.io.artifacts import sha256 as sha

__all__ = ["sha", "load", "write"]

from pathlib import Path
import json


def load(path):
    """Read a JSON artifact.

    Args:
        path: Input file path.

    Returns:
        Decoded JSON value.
    """
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomically write generated metadata without following an output symlink.

    Args:
        path: Destination metadata path in the new run directory.
        value: JSON-serializable metadata; nonfinite numbers are rejected.

    Raises:
        PermissionError: Destination is a symlink.
    """
    path = Path(path)
    if path.is_symlink():
        raise PermissionError(f"Refusing to overwrite linked input: {path}")
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        stream.write(payload)
    temporary.replace(path)
