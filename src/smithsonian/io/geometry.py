"""Source integrity and NDPA nanometre-to-pixel geometry.

Coordinate interpretation follows RiceD2KLab/Smithsonian_fossil_Sp26;
see NOTICE.md for source revision and attribution.
"""

from __future__ import annotations
import hashlib
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import openslide


def sha256_file(path: Path) -> str:
    """Hash a source file in bounded memory.

    Args:
        path: Existing Path object.

    Returns:
        SHA-256 hexadecimal digest, reading one MiB at a time."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32_file(path: Path) -> str:
    """Compute the historical CRC32 source-integrity fingerprint.

    Args:
        path: Existing Path object.

    Returns:
        Eight-character hexadecimal CRC32 string."""
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def geometry(slide: openslide.OpenSlide, record: dict) -> tuple[float, float, float, float]:
    """Convert an NDPA circle from nanometers into full-resolution slide pixels.

    Args:
        slide: OpenSlide image with scanner offsets and microns-per-pixel metadata.
        record: Annotation row containing x_nm, y_nm, and radius_nm.

    Returns:
        Center x/y and radii x/y in level-zero pixels.
    """
    props = slide.properties
    xnm = float(props["openslide.mpp-x"]) * 1000
    ynm = float(props["openslide.mpp-y"]) * 1000
    xoff = float(props["hamamatsu.XOffsetFromSlideCentre"])
    yoff = float(props["hamamatsu.YOffsetFromSlideCentre"])
    width, height = slide.dimensions
    x = width / 2 + (int(record["x_nm"]) - xoff) / xnm
    y = height / 2 + (int(record["y_nm"]) - yoff) / ynm
    rx = int(record["radius_nm"]) / xnm
    ry = int(record["radius_nm"]) / ynm
    return x, y, rx, ry
