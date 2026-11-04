"""Verify or copy a named checkpoint from an authorized artifact directory."""

from smithsonian.configuration.options import ConfigArgumentParser
from smithsonian.configuration.paths import repository_root

import hashlib
import json
from pathlib import Path
import shutil

ROOT = repository_root()


def sha256(path):
    """Calculate the SHA-256 identity of an existing artifact.

    Args:
        path: File path to read.

    Returns:
        Lowercase hexadecimal digest.

    Raises:
        FileNotFoundError: The artifact is absent."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    """Require an artifact to match its recorded SHA-256 identity.

    Args:
        path: File to validate.
        expected: Expected hexadecimal SHA-256 digest.

    Returns:
        None when the digest matches.

    Raises:
        ValueError: File contents differ from the expected identity."""
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")


def main():
    """Parse workflow options, validate inputs and write the requested outputs."""
    p = ConfigArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--destination", type=Path)
    args = p.parse_args()
    item = json.loads((ROOT / "config/checkpoints.json").read_text())[args.name]
    source = args.root / item["relative_path"]
    verify(source, item["sha256"])
    if args.destination:
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        with args.destination.open("xb") as out, source.open("rb") as inp:
            shutil.copyfileobj(inp, out)
        verify(args.destination, item["sha256"])
    print(json.dumps({"status": "verified", "name": args.name, "sha256": item["sha256"]}))


if __name__ == "__main__":
    main()
