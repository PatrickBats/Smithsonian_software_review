"""Frozen Swin/DINO validation comparison; optionally replay identical saved detections.

No detector training, test inference, or threshold search. Run on a UUID-locked GPU.
"""

# Source credit: Fall 2026 Smithsonian team integration; Spring preprocessing,
# tile types and boundary-aware merging derive from RiceD2KLab/Smithsonian_fossil_Sp26
# at 1b62b68a0d95178e6275ef55c38ee4765317b334. Swin: torchvision; DINOv2: Meta.
# See NOTICE.md for upstream links and the distinction between reused and project code.

import argparse, csv, json, math, os, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from method_b.pipeline.crop_detected_specimens import crop_input
from method_b.utils.file_io import sha, save
from method_b.classification.load_classifier import load_classifier
from method_b.classification.predict_categories import classify
from method_b.evaluation.classification_metrics import metric, match_boxes










def main(a):
    """Replay frozen classifiers on validation crops and optional saved detector outputs.

    Args:
        a: Parsed command-line configuration.

    Returns:
        None: Writes workflow outputs or prints validation status; failures propagate.
    """
    started = time.time()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path(a.crop_root)
    allrows = list(csv.DictReader((source / "crop_manifest.csv").open()))
    val = [r for r in allrows if r["split"] == "val"]
    if len(val) != 347 or len({r["record_id"] for r in val}) != 347:
        raise ValueError("Expected frozen separate-slide validation")
    mapping = {r["fine_category"]: r["broad_category"] for r in csv.DictReader(open(a.broad_map))}
    if sha(a.broad_map) != "a2f8da0b40cf9a1a89ea7fffe360770b2b9eae56f9fdd307bbf1b286bd9f793d":
        raise ValueError("Broad map changed")
    frozen_assignments = "da2062a127cf6e14945e50fff097e0db102d7f8203250b47daed6b6bb166478d"
    if sha(source.parent / "record_assignments.csv") != frozen_assignments:
        raise ValueError("Unexpected split")
    originals = []
    tight = []
    tight_rows = []
    for r in val:
        p = source / r["image_relpath"]
        if sha(p) != r["png_sha256"]:
            raise ValueError("Crop changed")
        im = Image.open(p).convert("RGB")
        originals.append(im)
        x = float(r["center_x_px"]) - float(r["crop_left_px"])
        y = float(r["center_y_px"]) - float(r["crop_top_px"])
        rx = float(r["radius_x_px"])
        ry = float(r["radius_y_px"])
        box = [x - rx, y - ry, x + rx, y + ry]
        pad = 0.1 * max(2 * rx, 2 * ry)
        # Avoid inventing missing context beyond the existing crop in the diagnostic.
        if (
            math.floor(box[0] - pad) >= 0
            and math.floor(box[1] - pad) >= 0
            and math.ceil(box[2] + pad) <= im.width
            and math.ceil(box[3] + pad) <= im.height
        ):
            tight.append(crop_input(im, box, 0.1))
            tight_rows.append(r)
    replay = None
    if a.pipeline_root:
        root = Path(a.pipeline_root)
        predictions = json.loads((root / "pipeline_validation/predictions.json").read_text())
        refs = [
            r
            for i in range(4)
            for r in json.loads((root / f"export_shard_{i}_references.json").read_text())
            if r["split"] == "val"
        ]
        allowed = {r["record_id"]: r for r in val}
        sources = {r["image_nots_path"] for r in val}
        if len({r["record_id"] for r in refs}) != len(refs):
            raise ValueError("Duplicate references")
        for r in refs:
            if (
                r["record_id"] not in allowed
                or r["image"] != allowed[r["record_id"]]["image_nots_path"]
                or r["broad_category"] != mapping[allowed[r["record_id"]]["sponsor_category"]]
            ):
                raise ValueError("Reference membership/label changed")
        images = []
        for r in predictions:
            if r["source"] not in sources:
                raise ValueError("Prediction outside validation")
            p = root / "dataset/valid" / Path(r["tile"]).name
            m = json.loads(p.with_suffix(".json").read_text())
            if m["split"] != "val" or m["image"] != r["source"] or sha(p) != m["png_sha256"]:
                raise ValueError("Invalid tile provenance")
            if not np.allclose(
                np.array(r["box_tile"]) + np.array([m["left"], m["top"], m["left"], m["top"]]),
                r["box_slide"],
            ):
                raise ValueError("Coordinate mismatch")
            images.append(crop_input(Image.open(p).convert("RGB"), r["box_tile"], 0.1))
        replay = (predictions, refs, images)
    summaries = {}
    for kind, path in [("swin", a.swin_checkpoint), ("dino", a.dino_checkpoint)]:
        model, labels = load_classifier(kind, path, a.dino_source, a.device)
        condition = {}
        for name, rr, images in [
            ("original_best_plane", val, originals),
            ("tight_box_same_best_plane", tight_rows, tight),
        ]:
            probs = classify(model, images)
            items = []
            for r, pr in zip(rr, probs):
                truth = mapping[r["sponsor_category"]]
                pred = labels[int(np.argmax(pr))]
                items.append(
                    dict(
                        record_id=r["record_id"],
                        truth=truth,
                        prediction=pred,
                        detected=True,
                        correct=truth == pred,
                        probabilities=pr,
                    )
                )
            save(out / f"{kind}_{name}.json", items)
            condition[name] = metric(items)
            if name == "original_best_plane":
                subset = [
                    x for x in items if x["record_id"] in {r["record_id"] for r in tight_rows}
                ]
                condition["original_on_tight_subset"] = metric(subset)
                expected = json.loads(
                    (Path(path).parent / "best_validation_metrics.json").read_text()
                )["accuracy"]
                if abs(condition[name]["classification_accuracy_on_matched"] - expected) > 1e-10:
                    raise ValueError("Original checkpoint result not reproduced")
        if replay:
            predictions, refs, images = replay
            probs = classify(model, images)
            prednames = [labels[int(np.argmax(pr))] for pr in probs]
            items = []
            for source_name in sorted({r["image"] for r in refs}):
                ix = [i for i, r in enumerate(predictions) if r["source"] == source_name]
                ps = [predictions[i] for i in ix]
                rs = [r for r in refs if r["image"] == source_name]
                matched = match_boxes(ps, rs)
                for j, r in enumerate(rs):
                    pred = prednames[ix[matched[j][0]]] if j in matched else None
                    items.append(
                        dict(
                            record_id=r["record_id"],
                            truth=r["broad_category"],
                            prediction=pred,
                            detected=j in matched,
                            correct=pred == r["broad_category"],
                        )
                    )
            condition["same_detector_outputs"] = metric(items)
            save(out / f"{kind}_end_to_end.json", items)
            if kind == "swin":
                prior = json.loads(
                    (Path(a.pipeline_root) / "pipeline_validation/summary.json").read_text()
                )
                if (
                    abs(
                        condition["same_detector_outputs"]["end_to_end_known_specimen_success"]
                        - prior["end_to_end_known_specimen_success"]
                    )
                    > 1e-10
                ):
                    raise ValueError("Original pipeline result not reproduced")
        summaries[kind] = condition
        del model
        torch.cuda.empty_cache()
        save(out / "summary.json", {"status": "running", "results": summaries, "test_used": False})
    save(
        out / "summary.json",
        {
            "status": "complete",
            "results": summaries,
            "test_used": False,
            "end_to_end_executed": replay is not None,
            "scope": "Frozen separate-slide validation; tight-crop diagnostic holds original selected plane fixed, not actual detector outputs",
            "elapsed_seconds": time.time() - started,
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoints": {
                k: sha(v) for k, v in [("swin", a.swin_checkpoint), ("dino", a.dino_checkpoint)]
            },
            "script_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in [
        "crop-root",
        "broad-map",
        "swin-checkpoint",
        "dino-checkpoint",
        "dino-source",
        "output",
    ]:
        p.add_argument("--" + arg, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--pipeline-root",
        help="Local mirrored existing run, including validation tiles and saved predictions; optional",
    )
    a = p.parse_args()
    try:
        main(a)
    except Exception:
        import traceback

        if Path(a.output).exists():
            save(Path(a.output) / "failure.json", {"traceback": traceback.format_exc()})
        raise
