"""Read-only classifier dependency, source revision, checkpoint and example checks."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
from artifacts import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--dino-source", type=Path, required=True)
    a = p.parse_args()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy", "Pillow", "omegaconf")
    }
    commit = subprocess.check_output(
        ["git", "-C", str(a.dino_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
        raise ValueError("DINO source revision differs")
    for name, item in json.loads((ROOT / "config/checkpoints.json").read_text()).items():
        if name.endswith("separate_slides"):
            verify(a.artifact_root / item["relative_path"], item["sha256"])
    for row in json.loads((ROOT / "examples/crops.json").read_text()):
        verify(a.crops / row["image"], row["sha256"])
    print(
        json.dumps(
            {
                "status": "passed",
                "dependencies": versions,
                "dino_revision": commit,
                "test_inference": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
