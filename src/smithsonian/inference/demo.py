"""Reusable single-region demonstration for the review notebook.

Uses fixed validation examples and verified saved models. No fitting or test-set
selection occurs. Detector boxes use image-local xyxy pixel coordinates.
"""

from argparse import Namespace
import json
from pathlib import Path

from smithsonian.configuration.options import load_config
from smithsonian.configuration.paths import repository_root
from smithsonian.io.artifacts import verify


def prepare_demo(config_path):
    """Check access to every artifact before running models.

    Args:
        config_path: Local JSON configuration with crop, source, and weight paths.

    Returns:
        Pair of resolved configuration and verified validation example metadata.

    Raises:
        FileNotFoundError: Required input is missing; see docs/DATA.md for retrieval.
        ValueError: Checkpoint, sample identity or pinned source revision differs.
    """
    import subprocess

    config = load_config(config_path)
    root = repository_root()
    sample = json.loads((root / "examples/crops.json").read_text())[0]
    region = json.loads((root / "examples/region.json").read_text())
    checkpoints = json.loads((root / "config/checkpoints.json").read_text())
    # Use the fixed validation examples and their recorded file hashes.
    paths = {
        "specimen": (Path(config["crops"]) / sample["image"], sample["sha256"]),
        "image": (Path(config["image"]), region["sha256"]),
        "checkpoint": (Path(config["checkpoint"]), region["checkpoint_sha256"]),
        "swin_checkpoint": (
            Path(config["swin_checkpoint"]),
            checkpoints["swin_separate_slides"]["sha256"],
        ),
        "dino_checkpoint": (
            Path(config["dino_checkpoint"]),
            checkpoints["dino_separate_slides"]["sha256"],
        ),
    }
    for name, (path, digest) in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}. Retrieve the files described in docs/DATA.md and notebooks/README.md, then update the local config."
            )
        verify(path, digest)
    source = Path(config["dino_source"])
    if not (source / "hubconf.py").is_file():
        raise FileNotFoundError("Retrieve the pinned DINOv2 source; see docs/QUICKSTART.md")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
        raise ValueError("Unexpected DINOv2 source revision")
    if sample["split"] != "val" or region["split"] != "val":
        raise ValueError("Only the fixed validation demonstrations are permitted")
    config["specimen"] = str(paths["specimen"][0])
    config["detector_sha256"] = region["checkpoint_sha256"]
    return config, sample


def classify_example(config, model, output, image=None, boxes=None):
    """Run a saved classifier on the specimen or region-local detected boxes.

    Args:
        config: Settings returned by prepare_demo.
        model: 'swin' or 'dino'; both use their saved 28-category checkpoint.
        output: New JSON prediction file.
        image: Optional region path; defaults to the verified specimen crop.
        boxes: Optional JSON file of xyxy boxes for crop extraction.

    Returns:
        Prediction dictionary with all category probabilities and hashes.
    """
    from smithsonian.inference.infer import run_inference

    return run_inference(
        Namespace(
            model=model,
            checkpoint=Path(config[f"{model}_checkpoint"]),
            image=Path(image or config["specimen"]),
            dino_source=Path(config["dino_source"]),
            device=config.get("device", "cpu"),
            boxes=Path(boxes) if boxes else None,
            output=Path(output),
        )
    )


def detect_example(config, output):
    """Detect specimens in the fixed validation region with the saved RF-DETR model.

    Args:
        config: Verified demo configuration.
        output: New directory for detection boxes and provenance.

    Returns:
        Detection dictionary. The extra model slot is excluded consistently.
    """
    from smithsonian.inference.detect import run_detection

    return run_detection(
        Namespace(
            checkpoint=Path(config["checkpoint"]),
            sha256=config["detector_sha256"],
            image=Path(config["image"]),
            classes=1,
            threshold=config.get("threshold", 0.5),
            device=config.get("device", "cpu"),
            output=Path(output),
        )
    )


def prediction_table(results):
    """Convert model outputs to display rows without changing their label order.

    Args:
        results: Mapping from model name to classifier prediction dictionary.

    Returns:
        List of dictionaries containing model, detection index, label and score.
    """
    return [
        {
            "model": model,
            "specimen": index + 1,
            "category": row["category"],
            "score": row["score"],
            "box": row["box"],
        }
        for model, result in results.items()
        for index, row in enumerate(result["predictions"])
    ]


def annotate_predictions(image, result):
    """Draw classifier labels and region-local boxes on a copy of an RGB image.

    Args:
        image: Image path, opened without modifying the source.
        result: Classifier result dictionary with box, category and score fields.

    Returns:
        RGB PIL image with annotations for display only.
    """
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image) as source:
        annotated = source.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=20)
    for row in result["predictions"]:
        if row["box"] is None:
            continue
        x0, y0, x1, y1 = row["box"]
        draw.rectangle((x0, y0, x1, y1), outline="#8747cf", width=4)
        text = f"{row['category']} ({row['score']:.1%})"
        draw.text(
            (max(0, x0), max(0, y0 - 24)),
            text,
            font=font,
            fill="#592a88",
            stroke_width=1,
            stroke_fill="white",
        )
    return annotated


def preview_image(image):
    """Create a compact notebook preview from a copy of an image.

    Args:
        image: PIL image to display. Model inputs retain their original pixels.

    Returns:
        IPython image with an embedded JPEG preview, at most 768 pixels per side.
    """
    from io import BytesIO
    from IPython.display import Image as DisplayImage

    preview = image.convert("RGB")
    preview.thumbnail((768, 768))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=90)
    return DisplayImage(data=buffer.getvalue(), format="jpeg")
