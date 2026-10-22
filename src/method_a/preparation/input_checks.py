"""Verify baseline image payloads and freeze external-input fingerprints, without writing to them."""

import argparse, collections, csv, os
import json
from pathlib import Path
from ..storage import BASE, EXPERIMENTS, ROOT, WEIGHTS, load, sha, write


def main(argv=None):
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="initial")
    args = parser.parse_args(argv)
    assert args.label == "initial" or args.label in EXPERIMENTS

    # Reject a changed split or category mapping before inspecting any image pixels.
    assert (
        sha(BASE / "inputs/priority_records.csv")
        == "d2dedb6bbb7e4c0e6c428c0d1cd4fce77d1f34bc6a829a515b172078913628f2"
    )
    assert (
        sha(BASE / "inputs/fine_to_broad.csv")
        == "a2f8da0b40cf9a1a89ea7fffe360770b2b9eae56f9fdd307bbf1b286bd9f793d"
    )

    # Check positive training tiles and the entire validation grid consumed by these runs.
    hashes = load(BASE / "dataset/source_image_hashes.json")
    used = []
    for split in ("train", "valid"):
        coco = load(BASE / f"dataset/{split}/_annotations.coco.json")
        ids = {a["image_id"] for a in coco["annotations"]}
        images = [i for i in coco["images"] if split == "valid" or i["id"] in ids]
        assert len(images) == (884 if split == "train" else 1432)
        for i in images:
            p = BASE / f"dataset/{split}" / i["file_name"]
            assert p.is_symlink()
            expected = hashes[Path(i["file_name"]).stem]
            assert sha(p) == expected
            with Image.open(p) as im:
                assert im.size == (1024, 1024) and im.mode == "RGB"
            st = p.resolve().stat()
            used.append(
                {
                    "path": str(p.resolve()),
                    "sha256": expected,
                    "uid": st.st_uid,
                    "size": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "inode": st.st_ino,
                }
            )

    # Hash compact external inputs in full; retain image metadata for the later integrity audit.
    files = [
        BASE / "inputs/priority_records.csv",
        BASE / "inputs/fine_to_broad.csv",
        BASE / "inputs/broad_category_map.json",
        BASE / "dataset/train/_annotations.coco.json",
        BASE / "dataset/valid/_annotations.coco.json",
        WEIGHTS,
    ]
    snapshot = {
        "status": "verified",
        "baseline_images_checked": len(used),
        "used_images": used,
        "external_files": [
            {
                "path": str(p),
                "sha256": sha(p),
                "uid": p.stat().st_uid,
                "size": p.stat().st_size,
                "mtime_ns": p.stat().st_mtime_ns,
            }
            for p in files
        ],
        "test_images_opened": False,
    }

    # Publish the snapshot only after all protected input checks succeed.
    write(ROOT / "audit" / f"external_inputs_before_{args.label}.json", snapshot)
    print(
        json.dumps(
            {
                "status": "verified",
                "images_checked": len(used),
                "test_images_opened": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
