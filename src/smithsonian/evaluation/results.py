"""Regenerate comparison CSVs and a vector chart from saved aggregate metrics."""

from smithsonian.configuration.options import ConfigArgumentParser
from smithsonian.configuration.paths import repository_root

import csv
import html
import json
from pathlib import Path

ROOT = repository_root()


def main():
    """Parse workflow options, validate inputs and write the requested outputs."""
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for model in ("swin", "dino"):
        for split in ("separate_slides", "shared_slides"):
            d = json.loads((ROOT / "results" / f"{model}_{split}.json").read_text())
            rows.append(
                {
                    "model": model,
                    "validation_split": split,
                    "accuracy": d["accuracy"],
                    "macro_f1": d["macro_f1_present"],
                }
            )
    with (args.output / "classification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    detector = json.loads((ROOT / "results/joint_detector.json").read_text())
    with (args.output / "detection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "known_targets", "detected", "correct_end_to_end", "confidence"])
        writer.writerow(
            [
                "joint_rf_detr",
                detector["known_specimens"],
                detector["detected"],
                detector["end_to_end_correct"],
                detector["confidence_threshold"],
            ]
        )
    comparison = json.loads((ROOT / "results/rgb_comparison.json").read_text())
    with (args.output / "rgb_comparison.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "classifier",
                "known_targets",
                "detected",
                "correct",
                "accuracy_on_detected",
                "end_to_end_success",
            ]
        )
        for name in ("swin", "dino"):
            d = comparison[name]
            w.writerow(
                [
                    name,
                    d["known_validation_objects"],
                    d["detected"],
                    d["correct"],
                    d["classification_accuracy_on_detected"],
                    d["end_to_end_success"],
                ]
            )
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="340" viewBox="0 0 900 340">',
        '<rect width="900" height="340" fill="white"/><text x="25" y="35" font-size="22">28-category classifier validation accuracy</text>',
    ]
    for i, r in enumerate(rows):
        y = 75 + i * 55
        label = html.escape(r["model"].upper() + " / " + r["validation_split"].replace("_", " "))
        svg.append(
            f'<text x="25" y="{y + 20}" font-size="17">{label}</text><rect x="330" y="{y}" width="{r["accuracy"] * 450}" height="28" fill="#7553b5"/><text x="{340 + r["accuracy"] * 450}" y="{y + 20}" font-size="17">{r["accuracy"]:.1%}</text>'
        )
    svg.append(
        '<text x="25" y="320" font-size="14">Expert-derived crops; validation only. Different split settings answer different questions.</text></svg>'
    )
    (args.output / "classification.svg").write_text("\n".join(svg))


if __name__ == "__main__":
    main()
