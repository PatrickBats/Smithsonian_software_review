"""Known-specimen metrics and the frozen greedy IoU matching contract."""

import torch
from torchvision.ops import box_iou


def metric(items):
    """Summarize known-specimen detection and identification outcomes.

    Args:
        items: Records with detected and correct booleans; one per known target.

    Returns:
        Counts and rates. Classification accuracy uses detected targets as its
        denominator; end-to-end success uses all known targets. Empty rates are None."""
    found = sum(x["detected"] for x in items)
    correct = sum(x["correct"] for x in items)
    n = len(items)
    return {
        "known_specimens": n,
        "detected": found,
        "correct": correct,
        "detection_recall_iou50": found / n if n else None,
        "classification_accuracy_on_matched": correct / found if found else None,
        "end_to_end_known_specimen_success": correct / n if n else None,
    }


def match_boxes(predictions, references):
    """Assign predictions to references greedily at IoU of at least 0.5.

    Args:
        predictions: Records containing box_slide xyxy coordinates in slide pixels.
        references: Expert records with the same coordinate convention.

    Returns:
        Mapping from reference index to prediction index and IoU. Each prediction
        and reference can be assigned at most once; category labels are unused.

    Raises:
        ValueError: Coordinates are nonfinite or boxes have nonpositive area."""
    if not predictions or not references:
        return {}
    p = torch.tensor([r["box_slide"] for r in predictions], dtype=torch.float32)
    t = torch.tensor([r["box_slide"] for r in references], dtype=torch.float32)
    if (
        not torch.isfinite(p).all()
        or not torch.isfinite(t).all()
        or (p[:, 2:] <= p[:, :2]).any()
        or (t[:, 2:] <= t[:, :2]).any()
    ):
        raise ValueError("Invalid boxes")
    overlap = box_iou(p, t)
    usedp = set()
    usedt = set()
    assigned = {}
    pairs = [(float(overlap[i, j]), i, j) for i, j in torch.nonzero(overlap >= 0.5).tolist()]
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in usedp and j not in usedt:
            usedp.add(i)
            usedt.add(j)
            assigned[j] = (i, iou)
    return assigned
