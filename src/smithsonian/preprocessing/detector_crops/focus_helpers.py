"""Read-only source geometry and Spring 2026 LoG focal-plane fusion."""

# Reused/adapted from Abbas et al., Rice D2K Spring 2026.
# Source: https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26
# Revision: 1b62b68a0d95178e6275ef55c38ee4765317b334. See NOTICE.md.

from __future__ import annotations

from smithsonian.io.geometry import (
    crc32_file as crc32_file,
    geometry as geometry,
    sha256_file as sha256_file,
)


import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def fused_tile(planes, expected_side: int, zero_index: int, zero_reference: np.ndarray):
    """Select each RGB pixel from the plane with maximal absolute LoG response.

    This is equivalent to Sp26 postprocess_tiles.focus_stack(k=5):
    GaussianBlur(gray, (5,5), 0), then Laplacian(CV_64F, ksize=5), with
    np.argmax over source plane order. Streaming keeps memory bounded.
    """
    best_response = np.full((expected_side, expected_side), -1.0, dtype=np.float64)
    fused = np.empty((expected_side, expected_side, 3), dtype=np.uint8)
    winning_plane = np.zeros((expected_side, expected_side), dtype=np.uint8)
    scores = []
    for index, pixels in enumerate(planes):
        if pixels.shape != (expected_side, expected_side, 3) or pixels.dtype != np.uint8:
            raise ValueError("Unexpected source focal tile shape or dtype")
        if index == zero_index and not np.array_equal(pixels, zero_reference):
            raise ValueError("Source plane zero differs from OpenSlide")
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        response = np.abs(cv2.Laplacian(blurred, cv2.CV_64F, ksize=5))
        scores.append(float(response.mean()))
        better = response > best_response
        fused[better] = pixels[better]
        winning_plane[better] = index
        best_response[better] = response[better]
    return fused, scores, np.bincount(winning_plane.ravel(), minlength=len(scores)).tolist()


def save_png_no_replace(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB array atomically without replacing an existing image.

    Args:
        path: New PNG destination.
        rgb: uint8 array with shape (height, width, 3), in RGB order.

    Returns:
        None; an existing destination raises FileExistsError.
    """
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    try:
        Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG", compress_level=3)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
