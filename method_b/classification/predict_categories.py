"""Classify an RGB crop or supplied region-local detection boxes with a frozen model.

This small demo does not evaluate a dataset or run whole-slide detection.
"""

import argparse
import json
from pathlib import Path
from method_b.utils.file_io import sha256

ROOT = Path(__file__).resolve().parents[2]



def classify(model, images):
    """Classify RGB crops using the shared 224-pixel preprocessing contract.

    Args:
        model: PyTorch model to inspect or evaluate.
        images: RGB PIL images to pad, resize and classify in input order.

    Returns:
        list[list[float]]: Probabilities in input order and checkpoint class order.
    """
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import functional as TF
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(images), 32):
        xx = []
        for im in images[start : start + 32]:
            im = ImageOps.pad(im, (max(im.size), max(im.size)), color="white").resize(
                (224, 224), Image.Resampling.BICUBIC
            )
            xx.append(TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        with torch.inference_mode():
            logits = model(torch.stack(xx).to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite logits")
        out.extend(logits.softmax(-1).cpu().tolist())
    return out


def validate_boxes(boxes):
    """Reject malformed, nonfinite or nonpositive-area local xyxy boxes.

    Args:
        boxes: Sequence of xyxy boxes in image pixels.

    Returns:
        list: Original validated boxes, including an empty list.
    """
    import math

    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Each box must have four finite coordinates")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Boxes must have positive area")
    return boxes


def main():
    """Classify one RGB crop or supplied local boxes using a verified frozen checkpoint.

    Returns:
        None: Writes workflow outputs or prints validation status; failures propagate.
    """
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
    from method_b.classification.load_classifier import load_classifier
    from method_b.pipeline.crop_detected_specimens import crop_input

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
