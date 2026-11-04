"""Detect specimens in one prepared RGB region and classify their crops.

The command interface follows the team's Method B organization proposal. This
implementation calls shared detection and classification functions directly.
"""

from argparse import Namespace
from pathlib import Path
import json

from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config
from smithsonian.inference.detect import run_detection
from smithsonian.inference.infer import run_inference
from smithsonian.io.artifacts import verify


def run_pipeline(args):
    """Run saved detection and classification models on the same RGB region.

    Args:
        args: Resolved image, detector/classifier checkpoint paths, verified
            detector hash, model choice, device, threshold, and output directory.

    Returns:
        Classification payload with region-local xyxy boxes, category labels,
        and probabilities. Detector artifacts are saved under output/detection.

    Raises:
        FileExistsError: Output directory already exists.
        FileNotFoundError: An image, checkpoint, or required DINO source is missing.
    """
    if args.output.exists():
        raise FileExistsError("Choose a new combined-pipeline output directory")
    for path in (args.image, args.detector_checkpoint, args.classifier_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}; see docs/DATA.md")
    verify(args.detector_checkpoint, args.detector_sha256)
    if args.model == "dino" and (
        args.dino_source is None or not (args.dino_source / "hubconf.py").is_file()
    ):
        raise FileNotFoundError("DINO requires the pinned source checkout; see docs/DATA.md")

    # Both stages read the same prepared RGB image; no extra focal selection occurs.
    save_effective_config(args, args.output / "effective_config.json")
    run_detection(
        Namespace(
            checkpoint=args.detector_checkpoint,
            sha256=args.detector_sha256,
            image=args.image,
            classes=1,
            threshold=args.threshold,
            device=args.device,
            output=args.output / "detection",
        )
    )
    return run_inference(
        Namespace(
            model=args.model,
            checkpoint=args.classifier_checkpoint,
            image=args.image,
            boxes=args.output / "detection/boxes.json",
            dino_source=args.dino_source,
            device=args.device,
            output=args.output / "classifications.json",
        )
    )


def main():
    """Parse settings and run one detector-to-classifier demonstration."""
    parser = ConfigArgumentParser(description=__doc__)
    for name in ("image", "detector-checkpoint", "classifier-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--model", choices=("swin", "dino"), required=True)
    parser.add_argument("--dino-source", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    result = run_pipeline(args)
    print(json.dumps({"classified_boxes": len(result["predictions"])}))


if __name__ == "__main__":
    main()
