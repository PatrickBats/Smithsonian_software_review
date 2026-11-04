"""Frozen-split Swin-Tiny baseline. Training/validation only; no test inference.

Outputs must be outside the repository. Each invocation creates a new run.
"""

import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

EXPECTED_SPLIT_SHA = "da2062a127cf6e14945e50fff097e0db102d7f8203250b47daed6b6bb166478d"


def digest(path):
    """Calculate a file SHA-256 digest.

    Args:
        path: Existing input file.

    Returns:
        Hexadecimal SHA-256 string.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    """Read annotation metadata from a CSV file.

    Args:
        path: CSV file with a header row.

    Returns:
        List of row dictionaries.
    """
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    """Write run metadata as readable JSON.

    Args:
        path: Destination file.
        obj: JSON-serializable result; NaN and infinity are rejected.

    Returns:
        None; writes the output file.
    """
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify_inputs(root, assignments, category_map):
    """Validate specimen identities, image bytes, category order, and split membership.

    Args:
        root: Crop dataset directory.
        assignments: Frozen separate-slide assignment CSV.
        category_map: Ordered source-category CSV.

    Returns:
        Crop records, source category names, and an audit dictionary.
    """
    if digest(assignments) != EXPECTED_SPLIT_SHA:
        raise ValueError("Split differs from the adopted assignment")
    source = read_csv(assignments)
    crops = read_csv(root / "crop_manifest.csv")
    classes = read_csv(category_map)
    if [int(c["classifier_class_id"]) for c in classes] != list(range(32)):
        raise ValueError("Expected 32 ordered class IDs")
    labels = [c["sponsor_category"] for c in classes]
    if len(set(labels)) != 32 or any(c["priority"] != "YES" for c in classes):
        raise ValueError("Invalid class vocabulary")
    by_id = {r["record_id"]: r for r in source}
    if len(source) != 2283 or len(by_id) != 2283 or len(crops) != 2283:
        raise ValueError("Unexpected record count")
    if len({r["record_id"] for r in crops}) != 2283 or {r["record_id"] for r in crops} != set(
        by_id
    ):
        raise ValueError("Missing/duplicated crop IDs")
    groups = collections.defaultdict(set)
    images = collections.defaultdict(set)
    for row in crops:
        for key, value in by_id[row["record_id"]].items():
            if row.get(key) != value:
                raise ValueError(f"Crop source field changed: {key}")
        if row["sponsor_category"] not in labels or int(row["source_level"]) != 0:
            raise ValueError("Unknown category or source level")
        groups[row["accession_group_id"]].add(row["split"])
        images[row["image_nots_path"]].add(row["split"])
        expected = (
            row["split"] + "/" + hashlib.sha256(row["record_id"].encode()).hexdigest() + ".png"
        )
        if row["image_relpath"] != expected:
            raise ValueError("Unexpected crop path")
        path = root / expected
        if path.is_symlink():
            raise ValueError("Symlinked crop")
        data = path.read_bytes()
        if (
            len(data) != int(row["png_bytes"])
            or hashlib.sha256(data).hexdigest() != row["png_sha256"]
        ):
            raise ValueError("Crop bytes do not match export manifest")
        with Image.open(io.BytesIO(data)) as image:
            side = int(row["crop_side_px"])
            if image.format != "PNG" or image.mode != "RGB" or image.size != (side, side):
                raise ValueError("Invalid crop format")
            image.verify()
        # Circle must remain wholly inside the exported image.
        for axis, origin in [("x", "left"), ("y", "top")]:
            center = float(row[f"center_{axis}_px"]) - float(row[f"crop_{origin}_px"])
            radius = float(row[f"radius_{axis}_px"])
            if not radius > 0 or center - radius < -1e-6 or center + radius > side + 1e-6:
                raise ValueError("Circle clipped by crop")
    counts = dict(collections.Counter(r["split"] for r in crops))
    if counts != {"train": 1605, "val": 347, "test": 331}:
        raise ValueError("Wrong partition counts")
    if len(groups) != 57 or any(len(s) != 1 for s in list(groups.values()) + list(images.values())):
        raise ValueError("Source leakage across partitions")
    representation = verify_representation(root, crops)
    return (
        crops,
        labels,
        {
            "representation": representation,
            "verified_records": len(crops),
            "split_counts": counts,
            "groups": len(groups),
            "split_sha256": digest(assignments),
            "crop_manifest_sha256": digest(root / "crop_manifest.csv"),
            "category_map_sha256": digest(category_map),
            "test_inference": False,
        },
    )


def verify_representation(root, crops):
    """Admit either the frozen zero-plane export or its verified focus replacement."""
    baseline_sha = "b4d803ca764f9941de38014bc5a67eb797f400a595033118c32512f5701e1373"
    if digest(root / "crop_manifest.csv") == baseline_sha:
        return {"name": "baseline_zero_plane", "crop_manifest_sha256": baseline_sha}
    metadata = json.loads((root / "representation.json").read_text())
    if (
        metadata["representation"] != "crop_wide_tenengrad_best_plane"
        or metadata["baseline_manifest_sha256"] != baseline_sha
    ):
        raise ValueError("Unrecognized image representation")
    if metadata["crop_manifest_sha256"] != digest(root / "crop_manifest.csv") or metadata[
        "focus_scores_sha256"
    ] != digest(root / "focus_scores.json"):
        raise ValueError("Focus provenance fingerprint mismatch")
    if (
        metadata["binding_sha256"] != digest(root.parent / "binding.json")
        or metadata["test_inference"] is not False
    ):
        raise ValueError("Invalid focus run binding")
    baseline_path = root.parent / "baseline_crop_manifest.csv"
    if digest(baseline_path) != baseline_sha:
        raise ValueError("Focus baseline changed")
    baseline = read_csv(baseline_path)
    scores = json.loads((root / "focus_scores.json").read_text())["records"]
    if len(scores) != len(crops) or [r["record_id"] for r in baseline] != [
        r["record_id"] for r in crops
    ]:
        raise ValueError("Focus records or order differ")
    if [r["record_id"] for r in scores] != [r["record_id"] for r in crops]:
        raise ValueError("Score order differs")
    mutable = {"png_bytes", "png_sha256", "rgb_channel_std_mean"}
    for row, old, score in zip(crops, baseline, scores):
        if any(row.get(k) != v for k, v in old.items() if k not in mutable):
            raise ValueError("Focus crop geometry/source changed")
        if (
            row["baseline_png_sha256"] != old["png_sha256"]
            or score["zero_plane_matches_baseline"] is not True
        ):
            raise ValueError("Zero-plane reference mismatch")
        planes = score["planes"]
        if not planes or any(
            not math.isfinite(p["tenengrad"]) or p["tenengrad"] < 0 for p in planes
        ):
            raise ValueError("Invalid focus score")
        best = max(planes, key=lambda p: (p["tenengrad"], -abs(p["z_offset"]), -p["index"]))
        if (
            int(row["selected_plane_index"]) != best["index"]
            or int(row["selected_z_offset"]) != best["z_offset"]
        ):
            raise ValueError("Selected plane is not the fixed-method maximum")
        if score["selected_sha256"] != row["png_sha256"] or score["selected_bytes"] != int(
            row["png_bytes"]
        ):
            raise ValueError("Selected image mismatch")
    return metadata


class Crops(Dataset):
    def __init__(self, root, rows, labels, augment):
        """Initialize the crop dataset without opening images.

        Args:
            root: Crop dataset directory.
            rows: Specimen records for one partition.
            labels: Ordered category names.
            augment: Enable random rotations and flips for training.
        """
        self.rows = rows
        self.targets = [labels.index(r["sponsor_category"]) for r in rows]
        self.augment = augment
        self.images = []
        for row in rows:
            with Image.open(root / row["image_relpath"]) as im:
                side = max(im.size)
                im = ImageOps.pad(im, (side, side), color="white")
                self.images.append(im.resize((224, 224), Image.Resampling.BICUBIC).copy())

    def __len__(self):
        """Report the number of specimen records in this dataset.

        Returns:
            Integer specimen count.
        """
        return len(self.rows)

    def __getitem__(self, index):
        """Load one crop and apply training or validation preprocessing.

        Args:
            index: Specimen index within the dataset.

        Returns:
            Normalized image tensor, category index, and specimen record ID.
        """
        im = self.images[index]
        if self.augment:
            # Quarter-turns and reflections retain the full square specimen crop.
            k = random.randrange(4)
            if k:
                im = im.transpose(
                    [
                        Image.Transpose.ROTATE_90,
                        Image.Transpose.ROTATE_180,
                        Image.Transpose.ROTATE_270,
                    ][k - 1]
                )
            if random.random() < 0.5:
                im = ImageOps.mirror(im)
        x = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return x, self.targets[index], index


def metrics(truth, predictions, labels):
    """Calculate classification metrics over categories with validation support.

    Args:
        truth: Reference category indices.
        predictions: Predicted category indices.
        labels: Ordered category vocabulary.

    Returns:
        Accuracy, confusion counts, per-category scores, and aggregate F1.
    """
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    np.add.at(cm, (truth, predictions), 1)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros(len(labels), float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(labels), float), where=support > 0)
    f1 = np.divide(
        2 * tp,
        support + predicted,
        out=np.zeros(len(labels), float),
        where=(support + predicted) > 0,
    )
    present = support > 0
    return {
        "accuracy": float(tp.sum() / support.sum()),
        "macro_f1_present": float(f1[present].mean()),
        "balanced_accuracy_present": float(recall[present].mean()),
        "evaluated_categories": [labels[i] for i in np.flatnonzero(present)],
        "absent_categories": [labels[i] for i in np.flatnonzero(~present)],
        "per_class": [
            {
                "category": c,
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]) if present[i] else None,
                "f1": float(f1[i]) if present[i] else None,
            }
            for i, c in enumerate(labels)
        ],
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model, loader, device, labels):
    """Evaluate a classifier without updating its weights.

    Args:
        model: Classifier to evaluate.
        loader: Validation DataLoader.
        device: Torch device used for inference.
        labels: Ordered category vocabulary.

    Returns:
        Metrics and specimen prediction records.
    """
    model.eval()
    truth = []
    predictions = []
    probabilities = []
    indices = []
    loss_sum = 0
    with torch.inference_mode():
        for x, y, idx in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite validation logits")
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            p = logits.softmax(1).cpu()
            truth.extend(y.cpu().tolist())
            predictions.extend(p.argmax(1).tolist())
            probabilities.extend(p.tolist())
            indices.extend(idx.tolist())
    result = metrics(truth, predictions, labels)
    result["loss"] = loss_sum / len(truth)
    return result, indices, probabilities
