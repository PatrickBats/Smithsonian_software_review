"""Regenerate comparison CSVs and a vector chart from saved aggregate metrics."""

import argparse
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
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
    with (a.output / "classification.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = json.loads(
        (ROOT / "reports/method_a_data_variants_2026-10-18/recorded_comparison.json").read_text()
    )
    with (a.output / "detection.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "known_targets", "detected", "correct_end_to_end"])
        for i, r in enumerate(d["experiments"], 1):
            w.writerow(
                [f"A{i}", d["validation_known_references"], r["detected"], r["end_to_end_correct"]]
            )
    comparison = json.loads((ROOT / "results/rgb_comparison.json").read_text())
    with (a.output / "rgb_comparison.csv").open("w", newline="") as f:
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
            f'<text x="25" y="{y+20}" font-size="17">{label}</text><rect x="330" y="{y}" width="{r["accuracy"]*450}" height="28" fill="#7553b5"/><text x="{340+r["accuracy"]*450}" y="{y+20}" font-size="17">{r["accuracy"]:.1%}</text>'
        )
    svg.append(
        '<text x="25" y="320" font-size="14">Expert-derived crops; validation only. Different split settings answer different questions.</text></svg>'
    )
    (a.output / "classification.svg").write_text("\n".join(svg))


if __name__ == "__main__":
    main()
