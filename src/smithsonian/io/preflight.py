"""Read-only classifier dependency, source revision, checkpoint and example checks."""

from smithsonian.configuration.options import ConfigArgumentParser
from smithsonian.configuration.paths import repository_root

import importlib.metadata
import json
from pathlib import Path
import subprocess
from smithsonian.io.artifacts import verify

ROOT = repository_root()


def main():
    """Parse workflow options, validate inputs and write the requested outputs."""
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--dino-source", type=Path, required=True)
    args = parser.parse_args()
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "numpy", "Pillow", "omegaconf")
    }
    commit = subprocess.check_output(
        ["git", "-C", str(args.dino_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "7764ea0f912e53c92e82eb78a2a1631e92725fc8":
        raise ValueError("DINO source revision differs")
    for name, item in json.loads((ROOT / "config/checkpoints.json").read_text()).items():
        if name.endswith("separate_slides"):
            verify(args.artifact_root / item["relative_path"], item["sha256"])
    for row in json.loads((ROOT / "examples/crops.json").read_text()):
        verify(args.crops / row["image"], row["sha256"])
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
