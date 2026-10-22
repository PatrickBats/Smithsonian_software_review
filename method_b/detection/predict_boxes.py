"""Run a supplied RF-DETR 2XL checkpoint on one RGB image region.

Output boxes use region-local pixel coordinates, ready for infer.py --boxes.
"""

import argparse
import json
from pathlib import Path
from method_b.utils.file_io import verify


def main():
    """Detect specimens in one RGB region using a verified RF-DETR checkpoint.

    Returns:
        None: Writes workflow outputs or prints validation status; failures propagate.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--classes", type=int, choices=[1, 28], required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Choose a new output directory")
    if not 0 <= a.threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    verify(a.checkpoint, a.sha256)
    from PIL import Image
    import torch

    torch.set_num_threads(4)
    from rfdetr_plus.models.detection import RFDETR2XLarge

    model = RFDETR2XLarge(
        pretrain_weights=str(a.checkpoint), num_classes=a.classes, resolution=1000, device=a.device
    )
    with Image.open(a.image) as im:
        predictions = model.predict(im.convert("RGB"), threshold=a.threshold)
    rows = []
    excluded = 0
    for box, score, category in zip(predictions.xyxy, predictions.confidence, predictions.class_id):
        if int(category) == a.classes:
            excluded += 1
            continue
        if not 0 <= int(category) < a.classes:
            raise ValueError("Unexpected detector category")
        rows.append(
            {"box": [float(v) for v in box], "score": float(score), "class_id": int(category)}
        )
    a.output.mkdir(parents=True)
    (a.output / "boxes.json").write_text(
        json.dumps([r["box"] for r in rows], indent=2, allow_nan=False)
    )
    (a.output / "detections.json").write_text(
        json.dumps(
            {
                "predictions": rows,
                "excluded_extra_slot": excluded,
                "checkpoint_sha256": a.sha256,
                "scope": "single RGB region; no whole-slide deduplication",
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(json.dumps({"detections": len(rows)}))


if __name__ == "__main__":
    main()
