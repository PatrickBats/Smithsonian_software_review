"""Verify or copy a named checkpoint from an authorized artifact directory."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    """Return a file's SHA-256 using bounded memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, expected):
    """Raise if the supplied file differs from its published fingerprint."""
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
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
