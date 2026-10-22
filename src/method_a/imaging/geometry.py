"""Convert existing NDPA circle geometry to native image pixels."""

from __future__ import annotations

import openslide


def geometry(
    slide: openslide.OpenSlide, record: dict
) -> tuple[float, float, float, float]:
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
