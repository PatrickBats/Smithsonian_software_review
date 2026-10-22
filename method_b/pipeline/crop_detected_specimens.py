"""Classifier crops from detector-input RGB pixels, with fixed context and padding."""
import math
from PIL import Image, ImageOps

def crop_input(image, box, margin=0.1):
    # Same pixels as detector input; fixed context, no focal-plane re-selection.
    """Crop detector-input RGB pixels with context, white padding and bicubic resizing.

    Args:
        image: RGB PIL image used by the detector.
        box: Local xyxy box in image pixels.
        margin: Context padding as a fraction of the longer box side.

    Returns:
        PIL.Image.Image: RGB classifier crop with size (224, 224).
    """
    x0, y0, x1, y1 = map(float, box)
    if (
        not all(math.isfinite(v) for v in (x0, y0, x1, y1, margin))
        or x1 <= x0
        or y1 <= y0
        or margin < 0
    ):
        raise ValueError("Invalid classifier crop bounds or margin")
    pad = margin * max(x1 - x0, y1 - y0)
    bounds = (math.floor(x0 - pad), math.floor(y0 - pad), math.ceil(x1 + pad), math.ceil(y1 + pad))
    canvas = Image.new("RGB", (bounds[2] - bounds[0], bounds[3] - bounds[1]), "white")
    clip = (
        max(0, bounds[0]),
        max(0, bounds[1]),
        min(image.width, bounds[2]),
        min(image.height, bounds[3]),
    )
    if clip[2] > clip[0] and clip[3] > clip[1]:
        canvas.paste(image.crop(clip), (clip[0] - bounds[0], clip[1] - bounds[1]))
    side = max(canvas.size)
    return ImageOps.pad(canvas, (side, side), color="white").resize(
        (224, 224), Image.Resampling.BICUBIC
    )


def historical_crop_input(image, box):
    """Crop detector-input RGB pixels with context, white padding and bicubic resizing.

    Args:
        image: RGB PIL image used by the detector.
        box: Local xyxy box in image pixels.

    Returns:
        PIL.Image.Image: RGB classifier crop with size (224, 224).
    """
    x0, y0, x1, y1 = box
    pad = 0.1 * max(x1 - x0, y1 - y0)
    b = (math.floor(x0 - pad), math.floor(y0 - pad), math.ceil(x1 + pad), math.ceil(y1 + pad))
    canvas = Image.new("RGB", (b[2] - b[0], b[3] - b[1]), "white")
    clip = (max(0, b[0]), max(0, b[1]), min(image.width, b[2]), min(image.height, b[3]))
    if clip[2] > clip[0] and clip[3] > clip[1]:
        canvas.paste(image.crop(clip), (clip[0] - b[0], clip[1] - b[1]))
    side = max(canvas.size)
    return ImageOps.pad(canvas, (side, side), color="white").resize(
        (224, 224), Image.Resampling.BICUBIC
    )
