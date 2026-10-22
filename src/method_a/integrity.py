"""Streaming hashes used to bind existing inputs to their recorded bytes."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zlib


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32_file(path: Path) -> str:
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xffffffff:08x}"
