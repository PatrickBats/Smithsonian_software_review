"""RGB crop preparation retained from the recorded combined-pipeline evaluation."""

import hashlib
import json
import math
from PIL import Image, ImageOps


SWIN = None
OLD = None
BASE = None


def save(path, value):
    """Write generated JSON metadata.

    Args:
        path: Destination file.
        value: JSON-serializable result.

    Returns:
        None; writes the output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.chmod(0o640)
    temp.replace(path)


def sha(path):
    """Calculate a file SHA-256 digest.

    Args:
        path: Existing input file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crop_input(image, box):
    """Extract a detector box with context and white square padding.

    Args:
        image: PIL RGB region image.
        box: Region-local xyxy box in pixels.

    Returns:
        224-by-224 RGB PIL crop with a 10 percent context margin.
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
