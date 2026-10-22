"""Spring 2026 LoG fusion and exclusive PNG export; see NOTICE for attribution."""

from __future__ import annotations

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
        if (
            pixels.shape != (expected_side, expected_side, 3)
            or pixels.dtype != np.uint8
        ):
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
    return (
        fused,
        scores,
        np.bincount(winning_plane.ravel(), minlength=len(scores)).tolist(),
    )


def save_png_no_replace(path: Path, rgb: np.ndarray) -> None:
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
