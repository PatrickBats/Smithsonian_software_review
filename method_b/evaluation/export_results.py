"""Export only the recorded Method B classifier and fixed-detection comparisons."""

import argparse
import csv
import json
from pathlib import Path


def export_results(output):
    """Regenerate Method B tables from the included aggregate evidence.

    Args:
        output: Fresh output directory for classification and pipeline CSVs.

    Returns:
        pathlib.Path: Directory containing the generated comparison tables.
            No models are loaded and no datasets are evaluated.
    """
    root = Path(__file__).resolve().parents[2]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model in ("swin", "dino"):
        for split in ("separate_slides", "shared_slides"):
            result = json.loads((root / "results" / f"{model}_{split}.json").read_text())
            rows.append({"model": model, "split": split,
                         "accuracy": result["accuracy"],
                         "macro_f1": result["macro_f1_present"]})
    with (output / "classification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = json.loads((root / "results/rgb_comparison.json").read_text())
    with (output / "detector_crop_comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["classifier", "known_targets", "detected", "correct",
                         "accuracy_on_detected", "end_to_end_success"])
        for model in ("swin", "dino"):
            result = comparison[model]
            writer.writerow([model, result["known_validation_objects"], result["detected"],
                             result["correct"], result["classification_accuracy_on_detected"],
                             result["end_to_end_success"]])
    return output


def main():
    """Parse an output location and export the included Method B evidence.

    Returns:
        None: Writes two CSV tables to a new directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(export_results(args.output))


if __name__ == "__main__":
    main()
