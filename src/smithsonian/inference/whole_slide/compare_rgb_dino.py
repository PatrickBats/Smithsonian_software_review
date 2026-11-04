"""Frozen RGB RF-DETR crops: paired existing Swin/DINO inference, no training."""

import os
import collections
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision.models import swin_t
from torchvision.transforms import functional as TF
from smithsonian.inference.whole_slide import compare_bestplane as prior
from torchvision.ops import box_iou


HERE = Path(__file__).resolve().parent
ORIGINAL = Path(os.environ.get("SPRING_ADAPTER_ROOT", str(HERE)))

BASE = Path("external")
FROZEN = BASE / "model_B/runs/abbas_rgb_input_check_v1"
OUT = Path(os.environ.get("COMPARISON_OUTPUT", str(BASE / "model_B/runs/rfdetr_rgb_dino_v1")))
DINO_RUN = BASE / "runs/broad_dino_2026-10-08/separate_slides/backbone_finetune"
DINO = DINO_RUN / "best.pt"
COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_SOURCE = Path(os.environ.get("DINO_SOURCE", str(HERE / ("dinov2-" + COMMIT))))


def read(path):
    """Load a saved JSON artifact.

    Args:
        path: Existing JSON Path.

    Returns:
        Decoded JSON value.
    """
    return json.loads(path.read_text())


def save(path, value):
    """Write generated JSON metadata.

    Args:
        path: Destination file.
        value: JSON-serializable result.

    Returns:
        None; writes the output file.
    """
    prior.save(path, value)
    path.chmod(0o644)


def load_models(device):
    """Load saved Swin and DINO weights for paired evaluation.

    Args:
        device: Torch execution device.

    Returns:
        Swin model, Swin labels, DINO model, and DINO labels.
    """
    swin_ckpt = torch.load(prior.SWIN, map_location="cpu", weights_only=False)
    dino_ckpt = torch.load(DINO, map_location="cpu", weights_only=False)
    sl = swin_ckpt["categories"]
    dl = dino_ckpt["categories"]
    assert len(sl) == len(dl) == len(set(dl)) == 28
    assert set(sl) == set(dl), "Classifier category sets differ"
    swin = swin_t(weights=None)
    swin.head = nn.Linear(swin.head.in_features, len(sl))
    swin.load_state_dict(swin_ckpt["model"], strict=True)
    from smithsonian.models.dino import DinoClassifier

    dino = DinoClassifier(num_classes=len(dl), source=str(DINO_SOURCE), pretrained=False)
    dino.load_state_dict(dino_ckpt["model"], strict=True)
    for model in (swin, dino):
        model.requires_grad_(False).to(device).eval()
    del swin_ckpt, dino_ckpt
    gc.collect()
    return swin, sl, dino, dl


def match(candidates, truth):
    """Match detected specimens to expert boxes once each at IoU 0.5.

    Args:
        candidates: Predictions with slide-pixel xyxy boxes.
        truth: Reference specimen records in the same coordinate system.

    Returns:
        One evaluation record per known specimen, including detection misses.
    """
    a = torch.tensor([p["box_slide"] for p in candidates], dtype=torch.float32).reshape(-1, 4)
    b = torch.tensor([r["box_slide"] for r in truth], dtype=torch.float32).reshape(-1, 4)
    overlaps = box_iou(a, b)
    pairs = [(float(overlaps[i, j]), i, j) for i, j in torch.nonzero(overlaps >= 0.5).tolist()]
    usedp = set()
    usedt = set()
    assigned = {}
    for overlap, i, j in sorted(pairs, reverse=True):
        if i not in usedp and j not in usedt:
            usedp.add(i)
            usedt.add(j)
            assigned[j] = (i, overlap)
    results = []
    for j, r in enumerate(truth):
        p = candidates[assigned[j][0]] if j in assigned else None
        results.append(
            {
                "record_id": r["record_id"],
                "source": r["image"],
                "true_category": r["broad_category"],
                "detected": p is not None,
                "predicted_category": p["category"] if p else None,
                "classification_correct": p is not None and p["category"] == r["broad_category"],
                "iou": assigned[j][1] if p else None,
            }
        )
    return results


def summarize(rows):
    """Summarize classifier success among known validation specimens.

    Args:
        rows: Reference outcomes containing detected and classification_correct flags.

    Returns:
        Known-target counts and conditional/overall classification rates.
    """
    found = sum(r["detected"] for r in rows)
    correct = sum(r["classification_correct"] for r in rows)
    return {
        "known_validation_objects": len(rows),
        "detected": found,
        "correct": correct,
        "classification_accuracy_on_detected": correct / found,
        "end_to_end_success": correct / len(rows),
    }


def run_workflow(preflight=False):
    """Replay the recorded detector-to-classifier comparison.

    Args:
        preflight: Validate artifacts and exit before full inference when True.

    Returns:
        None; writes paired predictions and known-reference evaluation records.
    """
    from smithsonian.io.upstream import load_spring_source

    load_spring_source()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ["PALYNET_NUM_WORKERS"] = "0"
    from src.annotator.types import TileSpec
    from src.annotator.pipeline import NDPIAnnotator
    from src.annotator.compression import best_focal_plane
    from src.data.ndpi_reader import NDPIData

    device = "cpu" if preflight else "cuda"
    if not preflight:
        assert torch.cuda.is_available(), "GPU required for full inference"
    torch.set_num_threads(2)
    # rfdetr.detr sets this globally at import in the frozen RGB pipeline.
    # Keep the same arithmetic for the Swin replay and paired DINO inference.
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True  # PyTorch default used in saved Swin inference.
    frozen_predictions = FROZEN / "rgb_corrected/predictions.json"
    frozen_matches = FROZEN / "rgb_corrected/known_reference_results.json"
    predictions = read(frozen_predictions)
    baseline = read(frozen_matches)
    assert len(predictions) == 1777 and len(baseline) == 274
    assert sum(r["detected"] for r in baseline) == 258
    assert sum(r["classification_correct"] for r in baseline) == 93
    contract = read(FROZEN / "configuration.json")
    assert prior.sha(prior.SWIN) == contract["swin_sha256"]
    refs = [
        r
        for i in range(4)
        for r in read(prior.OLD / f"export_shard_{i}_references.json")
        if r["split"] == "val"
    ]
    assert len(refs) == 274 and len({r["record_id"] for r in refs}) == 274
    provenance = read(DINO_RUN / "provenance.json")
    assert provenance["source_commit"] == COMMIT
    checkpoint_hash = prior.sha(DINO)
    swin, sl, dino, dl = load_models(device)
    config = {
        "training_performed": False,
        "test_used": False,
        "detector_rerun": False,
        "fixed_detector_confidence": 0.5,
        "fixed_matching_iou": 0.5,
        "frozen_run": str(FROZEN),
        "frozen_predictions_sha256": prior.sha(frozen_predictions),
        "frozen_matches_sha256": prior.sha(frozen_matches),
        "dino_checkpoint": str(DINO),
        "dino_checkpoint_sha256": checkpoint_hash,
        "swin_checkpoint": str(prior.SWIN),
        "swin_checkpoint_sha256": prior.sha(prior.SWIN),
        "dino_source_commit": COMMIT,
        "dino_snapshot_sha256": prior.sha(DINO_RUN / "train_dino_finetune_snapshot.py"),
        "dino_categories": dl,
        "swin_categories": sl,
        "torch_version": torch.__version__,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "crop": "unchanged prior.crop_input: RGB, 10% margin, white square pad, bicubic224, ImageNet normalization",
        "tile": "native Tenengrad Sobel3 at40x; every reconstructed tile must match saved RGB SHA256",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "caveat": "Validation diagnostic: Abbas detector saw 13/14 validation slides in its original training.",
    }
    if preflight:
        print(
            json.dumps({"preflight": "passed", "strict_checkpoint_loading": True, **config}),
            flush=True,
        )
        return
    assert not (OUT / "comparison.json").exists(), "Completed run exists; refusing overwrite"
    save(OUT / "configuration.json", config)
    groups = collections.defaultdict(list)
    for i, p in enumerate(predictions):
        groups[p["source"]].append(i)
    assert len(groups) == 14
    output = [None] * len(predictions)
    traces = []
    for number, (source, indices) in enumerate(sorted(groups.items()), 1):
        slide = FROZEN / "slides" / Path(source).stem
        grid = [TileSpec(**t) for t in read(slide / "tile_grid.json")]
        hashes = {}
        for t in read(slide / "paired_tile_inputs.json"):
            origin = (t["x"], t["y"])
            if origin in hashes:
                assert hashes[origin] == t["rgb_tile_sha256"]
            hashes[origin] = t["rgb_tile_sha256"]
        needed = collections.defaultdict(list)
        for i in indices:
            needed[tuple(predictions[i]["tile_origin"])].append(i)
        crops = {}
        seen = set()
        tile_count = 0
        ndpi = NDPIData(source)
        try:
            # Use original band boundaries so decoder and tile pixels reproduce.
            for bx, by, bw, bh, tiles in NDPIAnnotator._group_into_bands(None, grid, 8192):
                selected = [t for t in tiles if (t.x, t.y) in needed and (t.x, t.y) not in seen]
                if not selected:
                    continue
                band = ndpi.get_tile(x=bx, y=by, w=bw, h=bh, magnification=40)
                for t in selected:
                    origin = (t.x, t.y)
                    if origin in seen:
                        continue
                    tile = best_focal_plane(
                        band[:, t.x - bx : t.x - bx + t.w, :, :], tenengrad_ksize=3
                    )
                    digest = hashlib.sha256(np.ascontiguousarray(tile).tobytes()).hexdigest()
                    assert digest == hashes[origin], (
                        f"Frozen tile pixels changed: {source} {origin}"
                    )
                    image = Image.fromarray(np.asarray(tile, dtype=np.uint8)).convert("RGB")
                    for i in needed[origin]:
                        p = predictions[i]
                        # The original adapter added offsets to np.float32
                        # scalars, rounding slide coordinates to float32.
                        # Reproduce that arithmetic exactly, without changing
                        # either saved detection boxes or crop coordinates.
                        reconstructed = np.array(p["box_tile"], dtype=np.float32) + np.array(
                            [t.x, t.y, t.x, t.y], dtype=np.float32
                        )
                        assert np.array_equal(
                            reconstructed.astype(np.float64), np.array(p["box_slide"])
                        ), "Saved box/origin association changed"
                        crop = prior.crop_input(image, p["box_tile"])
                        crops[i] = (
                            TF.normalize(
                                TF.to_tensor(crop), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                            ),
                            hashlib.sha256(crop.tobytes()).hexdigest(),
                        )
                    traces.append(
                        {"source": source, "tile_origin": list(origin), "rgb_tile_sha256": digest}
                    )
                    seen.add(origin)
                    tile_count += 1
                del band
        finally:
            ndpi.close()
        assert set(crops) == set(indices)
        # Original Swin used groups of32 in detection order for each slide.
        for start in range(0, len(indices), 32):
            batch_indices = indices[start : start + 32]
            x = torch.stack([crops[i][0] for i in batch_indices]).to(device)
            with torch.inference_mode():
                sp = swin(x).softmax(-1).cpu()
                dp = dino(x).softmax(-1).cpu()
            for i, s, d in zip(batch_indices, sp, dp):
                p = predictions[i]
                si = int(s.argmax())
                di = int(d.argmax())
                assert sl[si] == p["category"], f"Swin replay changed at prediction {i}"
                assert abs(float(s[si]) - p["classification_score"]) < 1e-4, (
                    f"Swin score changed at {i}: replay={float(s[si])}, saved={p['classification_score']}, precision={torch.get_float32_matmul_precision()}"
                )
                output[i] = {
                    **p,
                    "category": dl[di],
                    "classification_score": float(d[di]),
                    "swin_category": p["category"],
                    "swin_classification_score": p["classification_score"],
                    "crop_rgb224_sha256": crops[i][1],
                    "frozen_prediction_index": i,
                }
        save(OUT / "slides" / (Path(source).stem + ".json"), [output[i] for i in indices])
        save(
            OUT / "status.json",
            {
                "status": "running",
                "completed_slides": number,
                "total_slides": 14,
                "classified_candidates": sum(p is not None for p in output),
                "tile_hashes_verified": len(traces),
            },
        )
        print(
            json.dumps(
                {
                    "slide": number,
                    "of": 14,
                    "verified_tiles": tile_count,
                    "classified_candidates": len(indices),
                    "swin_replay": "exact categories and scores within1e-4",
                }
            ),
            flush=True,
        )
        del crops
        gc.collect()
    matches = []
    for source in sorted(groups):
        matches.extend(
            match([output[i] for i in groups[source]], [r for r in refs if r["image"] == source])
        )
    bm = {r["record_id"]: r for r in baseline}
    for r in matches:
        before = bm[r["record_id"]]
        assert r["detected"] == before["detected"] and r["iou"] == before["iou"]
    paired = [r for r in matches if r["detected"]]
    transitions = collections.Counter(
        (bool(bm[r["record_id"]]["classification_correct"]), bool(r["classification_correct"]))
        for r in paired
    )
    comparison = {
        "status": "complete",
        "training_performed": False,
        "test_used": False,
        "detector_rerun": False,
        "same_boxes_planes_and_crops_verified": True,
        "all1777_swin_categories_reproduced": True,
        "candidate_predictions": len(output),
        "swin": summarize(baseline),
        "dino": summarize(matches),
        "paired_detected_objects": len(paired),
        "both_correct": transitions[True, True],
        "swin_only_correct": transitions[True, False],
        "dino_only_correct": transitions[False, True],
        "both_wrong": transitions[False, False],
        "caveat": config["caveat"],
    }
    per_class = {}
    for label in sorted(dl):
        rr = [r for r in paired if r["true_category"] == label]
        per_class[label] = {
            "detected_objects": len(rr),
            "swin_correct": sum(bm[r["record_id"]]["classification_correct"] for r in rr),
            "dino_correct": sum(r["classification_correct"] for r in rr),
        }
    save(OUT / "predictions.json", output)
    save(OUT / "known_reference_results.json", matches)
    save(OUT / "verified_tile_inputs.json", traces)
    save(OUT / "per_class_comparison.json", per_class)
    assert prior.sha(frozen_predictions) == config["frozen_predictions_sha256"]
    assert prior.sha(frozen_matches) == config["frozen_matches_sha256"]
    save(OUT / "comparison.json", comparison)
    save(OUT / "status.json", {"status": "complete", "completed_slides": 14})
    print(json.dumps(comparison), flush=True)


def main():
    """Replay paired classifiers on the saved detector boxes and validation regions."""
    from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config

    global FROZEN, OUT, DINO_RUN, DINO, DINO_SOURCE
    parser = ConfigArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--swin-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-run", type=Path, required=True)
    parser.add_argument("--dino-source", type=Path, required=True)
    parser.add_argument("--spring-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    FROZEN, OUT = args.frozen, args.output
    DINO_RUN, DINO_SOURCE = args.dino_run, args.dino_source
    DINO = DINO_RUN / "best.pt"
    prior.SWIN, prior.OLD = args.swin_checkpoint, args.references
    os.environ["SPRING_SOURCE"] = str(args.spring_source)
    if OUT.exists():
        raise FileExistsError("Choose a new replay output directory")
    save_effective_config(args, OUT / "effective_config.json")
    run_workflow(preflight=args.preflight)


if __name__ == "__main__":
    main()
