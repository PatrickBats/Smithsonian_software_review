"""Shared data structures for the annotator package."""

# Reused/adapted from Abbas et al., Rice D2K Spring 2026.
# Source: https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26
# Revision: 1b62b68a0d95178e6275ef55c38ee4765317b334. See NOTICE.md.

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """Single detection in both pixel and physical-coordinate spaces."""

    x1_px: float
    y1_px: float
    x2_px: float
    y2_px: float
    score: float
    x_nm: float
    y_nm: float
    width_nm: float
    height_nm: float


@dataclass(frozen=True)
class TileSpec:
    """Grid tile coordinates with both pixel origin and grid index."""

    x: int
    y: int
    w: int
    h: int
    ix: int
    iy: int
