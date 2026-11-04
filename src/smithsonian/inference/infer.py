"""Classify an RGB crop or supplied region-local detection boxes with a frozen model.

This demo classifies one image or the crops defined by supplied boxes.
"""

from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config
from smithsonian.configuration.paths import repository_root

import json
from pathlib import Path
from smithsonian.io.artifacts import sha256

ROOT = repository_root()


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
    """Parse workflow options, validate inputs and write the requested outputs."""
    p = ConfigArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["swin", "dino"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--dino-source", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--boxes", type=Path, help="JSON list of xyxy boxes in this image")
    p.add_argument("--output", type=Path, required=True)
    return run_inference(p.parse_args())


def run_inference(args):
    """Execute a verified single-image workflow from resolved command settings.

    Args:
        args: Parsed namespace containing input paths, device and a fresh output path.

    Returns:
        Prediction dictionary with checkpoint identity and region-local boxes.

    Raises:
        FileExistsError: Output already exists.
        ValueError: Inputs or checkpoint identity violate the recorded contract.
    """
    if args.output.exists():
        raise FileExistsError("Choose a new output file")
    # Check the saved model identity before loading its weights.
    entries = json.loads((ROOT / "config/checkpoints.json").read_text())
    digest = sha256(args.checkpoint)
    known = [
        v for v in entries.values() if v["architecture"] == args.model and v["sha256"] == digest
    ]
    if not known:
        raise ValueError("Use a verified project checkpoint listed in config/checkpoints.json")
    if args.model == "dino" and (
        not args.dino_source or not (args.dino_source / "hubconf.py").is_file()
    ):
        raise ValueError("DINO requires --dino-source pointing to the pinned local source")
    import torch
    from PIL import Image
    from smithsonian.models.classifiers import load_classifier, classify
    from smithsonian.preprocessing.crops import crop_input

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with Image.open(args.image) as im:
        image = im.convert("RGB")
    # Classify the supplied boxes or the full input crop.
    boxes = validate_boxes(json.loads(args.boxes.read_text())) if args.boxes else None
    images = [crop_input(image, b) for b in boxes] if boxes is not None else [image]
    model, labels = load_classifier(args.model, args.checkpoint, args.dino_source, args.device)
    # Match each probability vector to the checkpoint category order.
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
        "image_sha256": sha256(args.image),
        "categories": labels,
        "predictions": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    save_effective_config(args, args.output.with_suffix(args.output.suffix + ".config.json"))
    print(json.dumps({"outputs": len(outputs), "categories": [o["category"] for o in outputs]}))

    return result


if __name__ == "__main__":
    main()
