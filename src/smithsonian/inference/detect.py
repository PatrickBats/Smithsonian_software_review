"""Run a supplied RF-DETR 2XL checkpoint on one RGB image region.

Output boxes use region-local pixel coordinates, ready for infer.py --boxes.
"""

from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config

import json
from pathlib import Path
from smithsonian.io.artifacts import verify


def main():
    """Parse workflow options, validate inputs and write the requested outputs."""
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--classes", type=int, choices=[1, 28], required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return run_detection(parser.parse_args())


def run_detection(args):
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
        raise FileExistsError("Choose a new output directory")
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    # Verify the exact trained weights before constructing the detector.
    verify(args.checkpoint, args.sha256)
    from PIL import Image
    import torch

    torch.set_num_threads(4)
    from rfdetr_plus.models.detection import RFDETR2XLarge

    model = RFDETR2XLarge(
        pretrain_weights=str(args.checkpoint),
        num_classes=args.classes,
        resolution=1000,
        device=args.device,
    )
    with Image.open(args.image) as im:
        predictions = model.predict(im.convert("RGB"), threshold=args.threshold)
    # RF-DETR keeps an extra output slot that is not a biological category.
    rows = []
    excluded = 0
    for box, score, category in zip(predictions.xyxy, predictions.confidence, predictions.class_id):
        if int(category) == args.classes:
            excluded += 1
            continue
        if not 0 <= int(category) < args.classes:
            raise ValueError("Unexpected detector category")
        rows.append(
            {"box": [float(v) for v in box], "score": float(score), "class_id": int(category)}
        )
    args.output.mkdir(parents=True)
    (args.output / "boxes.json").write_text(
        json.dumps([r["box"] for r in rows], indent=2, allow_nan=False)
    )
    (args.output / "detections.json").write_text(
        json.dumps(
            {
                "predictions": rows,
                "excluded_extra_slot": excluded,
                "checkpoint_sha256": args.sha256,
                "scope": "single RGB region; no whole-slide deduplication",
            },
            indent=2,
            allow_nan=False,
        )
    )
    save_effective_config(args, args.output / "effective_config.json")
    print(json.dumps({"detections": len(rows)}))

    return json.loads((args.output / "detections.json").read_text())


if __name__ == "__main__":
    main()
