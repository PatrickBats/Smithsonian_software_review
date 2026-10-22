"""Export the four recorded Method A configurations without model dependencies."""

import argparse
import csv
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def write_detection_csv(output):
    """Write detection.csv into an existing directory using frozen aggregate counts."""
    source = Path(
        os.path.join(REPOSITORY, "results", "method_a", "recorded_comparison.json")
    )
    recorded = json.loads(source.read_text())

    # The baseline has a different training protocol; retain its recorded identity.
    rows = [("baseline", recorded["baseline"])]
    rows.extend(
        (f"A{i}", result) for i, result in enumerate(recorded["experiments"], 1)
    )
    with (Path(output) / "detection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "known_targets", "detected", "correct_end_to_end"])
        for name, result in rows:
            writer.writerow(
                [
                    name,
                    recorded["validation_known_references"],
                    result["detected"],
                    result["end_to_end_correct"],
                ]
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    write_detection_csv(args.output)


if __name__ == "__main__":
    main()
