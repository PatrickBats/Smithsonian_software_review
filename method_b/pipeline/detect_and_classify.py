"""Detect a single RGB region and classify its specimen crops with Swin or DINOv2."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def run_pipeline(image, detector_checkpoint, detector_sha256, classifier_checkpoint,
                 model, output, device="cpu", dino_source=None):
    """Run the verified detector and classifier CLIs on the same RGB region.

    Args:
        image: Local RGB image region to process.
        detector_checkpoint: Single-class RF-DETR 2XL checkpoint path.
        detector_sha256: Independently recorded detector fingerprint.
        classifier_checkpoint: Verified Swin or DINOv2 checkpoint path.
        model: Classifier key, swin or dino.
        output: Fresh directory for detection and classification outputs.
        device: Inference device accepted by both model entry points.
        dino_source: Pinned local DINOv2 source; required for model=dino.

    Returns:
        dict: Classification payload with local boxes, labels and probabilities.
            Detection artifacts are retained in output/detection.
    """
    root = Path(__file__).resolve().parents[2]
    if model not in ("swin", "dino"):
        raise ValueError("Classifier must be swin or dino")
    if model == "dino" and dino_source is None:
        raise ValueError("DINOv2 requires its pinned local source checkout")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        sys.executable, "-m", "method_b.detection.predict_boxes",
        "--checkpoint", str(Path(detector_checkpoint).resolve()),
        "--sha256", detector_sha256, "--classes", "1",
        "--image", str(Path(image).resolve()), "--device", device,
        "--output", str(output / "detection"),
    ], cwd=root, check=True)
    command = [
        sys.executable, "-m", "method_b.classification.predict_categories",
        "--model", model, "--checkpoint", str(Path(classifier_checkpoint).resolve()),
        "--image", str(Path(image).resolve()), "--device", device,
        "--boxes", str(output / "detection/boxes.json"),
        "--output", str(output / "classifications.json"),
    ]
    if dino_source is not None:
        command += ["--dino-source", str(Path(dino_source).resolve())]
    subprocess.run(command, cwd=root, check=True)
    return json.loads((output / "classifications.json").read_text())


def main():
    """Parse a single-region inference request and print its output summary.

    Returns:
        None: Writes new pipeline artifacts; subprocess failures propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "detector-checkpoint", "classifier-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--model", choices=("swin", "dino"), required=True)
    parser.add_argument("--dino-source", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_pipeline(args.image, args.detector_checkpoint, args.detector_sha256,
                          args.classifier_checkpoint, args.model, args.output,
                          args.device, args.dino_source)
    print(json.dumps({"classified_boxes": len(result["predictions"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
