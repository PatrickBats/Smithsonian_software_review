"""Classify an RGB crop or supplied region-local detection boxes with a frozen model.

This small demo does not evaluate a dataset or run whole-slide detection.
"""

import argparse
import json
from pathlib import Path
import sys
from artifacts import verify, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/pipeline_abbas_v1"))


def validate_boxes(boxes):
    """Return finite positive-area xyxy boxes; reject malformed detection payloads."""
    import math

    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Each box must have four finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Boxes must have positive area")
    return boxes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["swin", "dino"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--dino-source", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--boxes", type=Path, help="JSON list of xyxy boxes in this image")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Choose a new output file")
    entries = json.loads((ROOT / "config/checkpoints.json").read_text())
    digest = sha256(a.checkpoint)
    known = [v for v in entries.values() if v["architecture"] == a.model and v["sha256"] == digest]
    if not known:
        raise ValueError("Use a verified project checkpoint listed in config/checkpoints.json")
    if a.model == "dino" and (not a.dino_source or not (a.dino_source / "hubconf.py").is_file()):
        raise ValueError("DINO requires --dino-source pointing to the pinned local source")
    import torch
    from PIL import Image
    from compare_classifiers import load_classifier, classify
    from pipeline_common import crop_input

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with Image.open(a.image) as im:
        image = im.convert("RGB")
    boxes = validate_boxes(json.loads(a.boxes.read_text())) if a.boxes else None
    images = [crop_input(image, b) for b in boxes] if boxes is not None else [image]
    model, labels = load_classifier(a.model, a.checkpoint, a.dino_source, a.device)
    outputs = []
    for i, probs in enumerate(classify(model, images)):
        index = max(range(len(probs)), key=probs.__getitem__)
        outputs.append(
            {
                "box": boxes[i] if boxes is not None else None,
                "category": labels[index],
                "score": probs[index],
                "probabilities": probs,
            }
        )
    result = {
        "scope": "single image demonstration; not evaluation",
        "checkpoint_sha256": digest,
        "image_sha256": sha256(a.image),
        "categories": labels,
        "predictions": outputs,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"outputs": len(outputs), "categories": [o["category"] for o in outputs]}))


if __name__ == "__main__":
    main()
