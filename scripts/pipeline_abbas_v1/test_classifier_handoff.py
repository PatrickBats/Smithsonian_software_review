"""Checks for shared artifact permissions, crop validity, and scoring denominators."""

import tempfile, stat, math
from pathlib import Path
from PIL import Image
from pipeline_common import save, crop_input
from compare_classifiers import metric, match_boxes

with tempfile.TemporaryDirectory() as t:
    p = Path(t) / "result.json"
    save(p, {"ok": True})
    assert stat.S_IMODE(p.stat().st_mode) == 0o640
    save(p, {"ok": False})
    assert "false" in p.read_text()
image = Image.new("RGB", (100, 100), "red")
assert crop_input(image, [20, 20, 80, 80]).size == (224, 224)
assert crop_input(image, [-20, -20, 40, 40]).getpixel((0, 0)) == (255, 255, 255)
for box in [[1, 1, 0, 0], [0, 0, float("nan"), 1]]:
    try:
        crop_input(image, box)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid geometry admitted")
refs = [{"box_slide": [0, 0, 10, 10]}, {"box_slide": [20, 20, 30, 30]}]
preds = [{"box_slide": [0, 0, 10, 10]}, {"box_slide": [0, 0, 10, 10]}]
assert len(match_boxes(preds, refs)) == 1
assert match_boxes([], refs) == {}
m = metric(
    [
        {"detected": True, "correct": True},
        {"detected": True, "correct": False},
        {"detected": False, "correct": False},
    ]
)
assert (
    m["classification_accuracy_on_matched"] == 0.5
    and m["end_to_end_known_specimen_success"] == 1 / 3
)
assert metric([])["end_to_end_known_specimen_success"] is None
print("Shared permissions, invalid crops, one-to-one matching and denominator checks passed")
