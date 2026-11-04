"""Create new COCO datasets; Method B and baseline inputs are never modified."""

from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config

import collections
import math
import random
import hashlib
import json
from pathlib import Path
from smithsonian.preprocessing.detector_crops.common import load, sha, write


def link(src, dst):
    """Link a source artifact into a newly created dataset.

    Args:
        src: Existing absolute source path.
        dst: New link path.

    Returns:
        None; existing destinations are rejected.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert not dst.exists() and not dst.is_symlink()
    dst.symlink_to(src)


def coco_base(baseline_root):
    """Read shared COCO metadata and the ordered category vocabulary.

    Args:
        baseline_root: Verified baseline export directory.

    Returns:
        COCO metadata excluding images and annotations.
    """
    return {
        k: v
        for k, v in load(baseline_root / "dataset/train/_annotations.coco.json").items()
        if k not in ("images", "annotations")
    }


def make_dataset(baseline_root, output_root, tiles):
    """Assemble verified crop images and boxes into the current training dataset.

    Args:
        baseline_root: Baseline with category metadata and the fixed validation export.
        output_root: Crop preparation directory containing images and verification records.
        tiles: Planned crops with canvas-relative xywh boxes.

    Returns:
        None; writes a COCO training split and links unchanged validation data.
    """
    dest = output_root / "dataset"
    dest.mkdir(parents=True, exist_ok=False)
    link(baseline_root / "dataset/valid", dest / "valid")
    (dest / "test").mkdir()
    write(
        dest / "test/_annotations.coco.json",
        {**coco_base(baseline_root), "images": [], "annotations": []},
    )
    train = dest / "train"
    train.mkdir()
    images = []
    annotations = []
    for tile in tiles:
        meta = load(output_root / "images/crops" / f"{tile['tile_id']}.json")
        assert meta["status"] == "verified" and meta["sha256"] == sha(
            output_root / "images/crops" / f"{tile['tile_id']}.png"
        )
        image_id = len(images) + 1
        filename = f"{tile['tile_id']}.png"
        images.append({"id": image_id, "file_name": filename, "width": 1024, "height": 1024})
        link(output_root / "images/crops" / filename, train / filename)
        for obj in tile["objects"]:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": obj["category_id"],
                    "bbox": obj["bbox"],
                    "area": obj["bbox"][2] * obj["bbox"][3],
                    "iscrowd": 0,
                    "record_id": obj["record_id"],
                }
            )
    data = {**coco_base(baseline_root), "images": images, "annotations": annotations}
    data["info"] = {"description": "Target-centered native crops on a padded 1024 canvas"}
    write(train / "_annotations.coco.json", data)
    ann_ids = {a["image_id"] for a in data["annotations"]}
    assert len(ann_ids) == len(data["images"])
    records = {a["record_id"] for a in data["annotations"]}
    summary = {
        "status": "verified",
        "experiment": "target_centered_crops",
        "train_images": len(data["images"]),
        "train_boxes": len(data["annotations"]),
        "train_known_records": len(records),
        "train_record_ids": sorted(records),
        "validation_images": 1432,
        "validation_known_records": 274,
        "validation_scope": "unchanged complete fixed grid",
        "test_images_loaded": False,
        "source_assignment_sha256": sha(baseline_root / "inputs/priority_records.csv"),
        "train_coco_sha256": sha(train / "_annotations.coco.json"),
        "valid_coco_sha256": sha(dest / "valid/_annotations.coco.json"),
        "representation": (
            "baseline LoG k5"
            if tiles is None
            else "native LoG k5 context crops, no rescaling, random-offset 1024 canvas padding"
        ),
        "warning": "Incomplete annotations; no unannotated region is certified negative.",
    }
    write(dest / "dataset_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "train_record_ids"}), flush=True)


def intersection(anchor_box, left, top, side):
    """Calculate overlap between a specimen box and a square crop.

    Args:
        anchor_box: Specimen bounds in slide-pixel xyxy format.
        left: Crop left coordinate.
        top: Crop top coordinate.
        side: Square crop width in native pixels.

    Returns:
        Intersection area in square pixels.
    """
    x0, y0 = max(anchor_box[0], left), max(anchor_box[1], top)
    x1, y1 = min(anchor_box[2], left + side), min(anchor_box[3], top + side)
    return max(0, x1 - x0) * max(0, y1 - y0)


def plan_crops(output_root, seed):
    """Plan deterministic specimen windows while avoiding cut neighboring labels.

    Args:
        output_root: Directory containing audited training geometry.
        seed: Seed incorporated into each specimen-specific random generator.

    Returns:
        None; writes per-slide crop plans and a coverage summary.
    """
    slides = load(output_root / "audit/train_source_geometry.json")
    audit = load(output_root / "audit/train_geometry.json")
    if audit["box_max_dimension_percentiles"]["max"] + 64 > 1024:
        raise ValueError("Target needs larger than 1024: review required")
    plans = []
    stats = collections.Counter()
    for slide in slides:
        objects = slide["objects"]
        crops = []
        for anchor in sorted(
            objects, key=lambda specimen: (not specimen["old_grid_covered"], specimen["record_id"])
        ):
            anchor_box = anchor["box_slide"]
            need = max(anchor_box[2] - anchor_box[0], anchor_box[3] - anchor_box[1]) + 64
            side = next(side for side in (512, 640, 768, 896, 1024) if need <= side)
            rng = random.Random(f"{seed}:{anchor['record_id']}")
            # Reuse a prior window only if the whole new anchor has a 16 px margin.
            existing = next(
                (
                    crop
                    for crop in crops
                    if crop["left"] + 16 <= anchor_box[0]
                    and crop["top"] + 16 <= anchor_box[1]
                    and crop["left"] + crop["side"] - 16 >= anchor_box[2]
                    and crop["top"] + crop["side"] - 16 >= anchor_box[3]
                ),
                None,
            )
            if existing:
                existing["anchor_record_ids"].append(anchor["record_id"])
                if anchor["old_grid_covered"]:
                    existing["matched_anchors"].append(anchor["record_id"])
                stats["reused_windows"] += 1
                continue
            lo_x = max(0, math.ceil(anchor_box[2] + 32 - side))
            hi_x = min(slide["width"] - side, math.floor(anchor_box[0] - 32))
            lo_y = max(0, math.ceil(anchor_box[3] + 32 - side))
            hi_y = min(slide["height"] - side, math.floor(anchor_box[1] - 32))
            if lo_x > hi_x or lo_y > hi_y:
                raise ValueError("No feasible crop with target margin")
            candidates = []
            for k in range(128):
                left = rng.randint(lo_x, hi_x)
                top = rng.randint(lo_y, hi_y)
                partial = 0
                for specimen in objects:
                    specimen_box = specimen["box_slide"]
                    area = intersection(specimen_box, left, top, side)
                    full = (specimen_box[2] - specimen_box[0]) * (specimen_box[3] - specimen_box[1])
                    if 0 < area < full - 1e-6:
                        partial += 1
                candidates.append((partial, k, left, top))
            partial, _, left, top = min(candidates)
            pad_left = rng.randint(0, 1024 - side)
            pad_top = rng.randint(0, 1024 - side)
            tile_id = hashlib.sha256(
                f"{slide['source']}:{left}:{top}:{side}:{pad_left}:{pad_top}:log5-sparse-v1".encode()
            ).hexdigest()[:24]
            crop = {
                "tile_id": tile_id,
                "source": slide["source"],
                "left": left,
                "top": top,
                "side": side,
                "pad_left": pad_left,
                "pad_top": pad_top,
                "anchor_record_ids": [anchor["record_id"]],
                "matched_anchors": [anchor["record_id"]] if anchor["old_grid_covered"] else [],
                "objects": [],
                "known_partial_boxes": partial,
            }
            for specimen in objects:
                specimen_box = specimen["box_slide"]
                if intersection(specimen_box, left, top, side) <= 0:
                    continue
                x0 = max(specimen_box[0], left) - left + pad_left
                y0 = max(specimen_box[1], top) - top + pad_top
                x1 = min(specimen_box[2], left + side) - left + pad_left
                y1 = min(specimen_box[3], top + side) - top + pad_top
                crop["objects"].append(
                    {
                        **specimen,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "visible_fraction": intersection(specimen_box, left, top, side)
                        / (
                            (specimen_box[2] - specimen_box[0])
                            * (specimen_box[3] - specimen_box[1])
                        ),
                    }
                )
            crops.append(crop)
            stats[f"side_{side}"] += 1
            stats["known_partial_boxes"] += partial
        plans.append({**slide, "crops": crops})
        write(output_root / "plan" / f"{slide['index']:02d}.json", plans[-1])
    all_crops = [crop for p in plans for crop in p["crops"]]
    matched = [crop for crop in all_crops if crop["matched_anchors"]]
    assert len({r for crop in all_crops for r in crop["anchor_record_ids"]}) == 1605
    assert len({r for crop in matched for r in crop["matched_anchors"]}) == 1061
    report = {
        "status": "planned",
        "train_source_files": 47,
        "all_crop_windows": len(all_crops),
        "matched_crop_windows": len(matched),
        "train_anchor_records": 1605,
        "matched_anchor_records": 1061,
        "matched_label_records": len(
            {specimen["record_id"] for crop in matched for specimen in crop["objects"]}
        ),
        "all_label_records": len(
            {specimen["record_id"] for crop in all_crops for specimen in crop["objects"]}
        ),
        "stats": dict(stats),
        "canvas_side": 1024,
        "padding": "uniform RGB median of crop edge pixels, no real object copied into padding",
        "scale": "native pixels preserved; no crop resizing",
        "partial_known_objects": "avoid boundaries by 128 candidate search; every remaining intersecting YES box is retained clipped",
        "warning": "Matched-anchor crops may incidentally supervise additional known train targets; counts reported explicitly. Padding introduces a train/inference appearance difference. Unknown interior pollen remain incompletely annotated.",
    }
    write(output_root / "plan_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    """Plan training windows or assemble their exported images into COCO format."""
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "finalize"], required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    save_effective_config(args, args.output / f"{args.stage}_config.json")
    if args.stage == "plan":
        plan_crops(args.output, args.seed)
    else:
        plans = [load(path) for path in sorted((args.output / "plan").glob("*.json"))]
        if not plans:
            raise FileNotFoundError("Run crop planning and image export first")
        make_dataset(args.baseline, args.output, [crop for plan in plans for crop in plan["crops"]])


if __name__ == "__main__":
    main()
