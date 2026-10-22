"""Private, verified pretrained copy prevents library recovery paths writing shared weights."""

import shutil
from ..storage import ROOT, WEIGHTS, WEIGHT_SHA, sha, write


def main():

    # Keep library recovery/download behavior away from the shared initialization checkpoint.
    assert sha(WEIGHTS) == WEIGHT_SHA
    p = ROOT / "inputs/rf-detr-xxlarge.pth"
    p.parent.mkdir(parents=True, exist_ok=True)
    with WEIGHTS.open("rb") as src, p.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    assert sha(p) == WEIGHT_SHA

    # Record provenance only after the private copy reproduces the pinned hash.
    write(
        ROOT / "inputs/pretrained_provenance.json",
        {
            "source": str(WEIGHTS),
            "sha256": WEIGHT_SHA,
            "copy": str(p),
            "source_modified": False,
        },
    )


if __name__ == "__main__":
    main()
