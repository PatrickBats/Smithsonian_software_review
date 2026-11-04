"""Prepare classifier crops in region-local pixel coordinates."""

import math
from PIL import Image, ImageOps


def crop_input(image, box, margin=0.1):
    """Extract a context crop with white padding and bicubic resize.

    Args:
        image: PIL RGB image; coordinates refer to this image, not the whole slide.
        box: Four xyxy pixel coordinates with positive width and height.
        margin: Fraction of the longer box side added on each edge.

    Returns:
        New 224-by-224 RGB PIL image. Source pixels are not modified.

    Raises:
        ValueError: Bounds or margin are nonfinite, inverted or negative."""
    x0, y0, x1, y1 = map(float, box)
    if (
        not all(math.isfinite(v) for v in (x0, y0, x1, y1, margin))
        or x1 <= x0
        or y1 <= y0
        or margin < 0
    ):
        raise ValueError("Invalid classifier crop bounds or margin")
    # Expand the detection box to keep nearby context.
    pad = margin * max(x1 - x0, y1 - y0)
    bounds = (math.floor(x0 - pad), math.floor(y0 - pad), math.ceil(x1 + pad), math.ceil(y1 + pad))
    # Fill any area outside the image with white.
    canvas = Image.new("RGB", (bounds[2] - bounds[0], bounds[3] - bounds[1]), "white")
    clip = (
        max(0, bounds[0]),
        max(0, bounds[1]),
        min(image.width, bounds[2]),
        min(image.height, bounds[3]),
    )
    if clip[2] > clip[0] and clip[3] > clip[1]:
        canvas.paste(image.crop(clip), (clip[0] - bounds[0], clip[1] - bounds[1]))
    # Square padding preserves specimen proportions during resizing.
    side = max(canvas.size)
    return ImageOps.pad(canvas, (side, side), color="white").resize(
        (224, 224), Image.Resampling.BICUBIC
    )
