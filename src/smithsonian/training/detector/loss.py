"""Ignore-region loss masking and one-class checkpoint initialization.

Adapted from the Fall team implementation, commit 8174c47939430d873a4f42b023f35fe7516aa168.
The initialization detector is credited to Shaikh et al.; see NOTICE.md.
"""

from pathlib import Path


def patch_criterion_for_ignore(criterion, ignore_label, iou_thresh=0.5):
    """Mask unmatched ignore-region queries in the RF-DETR classification loss.

    Args:
        criterion: RF-DETR SetCriterion using the ia_bce_loss branch.
        ignore_label: Reserved remapped category index, 28 in the recorded task.
        iou_thresh: Minimum overlap for ignoring an unmatched predicted box.

    Returns:
        The same criterion with modified forward and loss_labels methods. Boxes
        are normalized cxcywh tensors; labeled-target box losses are unchanged.

    Raises:
        AssertionError: The criterion uses an unsupported classification loss.
    """
    import torch
    import torch.nn.functional as F
    from rfdetr.utilities import box_ops

    assert criterion.ia_bce_loss, "patch written for the ia_bce_loss branch RF-DETR uses by default"
    orig_forward = criterion.forward

    def forward(outputs, targets):
        """Remove ignore targets before matching and expose them to the classification loss.

        Args:
            outputs: Detector predictions including auxiliary decoder outputs.
            targets: Per-image labels and normalized cxcywh boxes.

        Returns:
            Original criterion losses with ignore handling applied.
        """
        kept, ign = [], []
        for t in targets:
            m = t["labels"] != ignore_label
            kept.append({**t, "labels": t["labels"][m], "boxes": t["boxes"][m]})
            ign.append(t["boxes"][~m])
        criterion._ignore_boxes = ign
        criterion._ignore_hits = 0
        try:
            return orig_forward(outputs, kept)
        finally:
            criterion._ignore_boxes = None

    def ignore_mask(pred_boxes):
        """Find queries whose predicted boxes overlap an ignored specimen.

        Args:
            pred_boxes: Normalized cxcywh tensor shaped (batch, queries, 4).

        Returns:
            Boolean (batch, queries) mask at the configured IoU threshold.
        """
        B, Q, _ = pred_boxes.shape
        mask = torch.zeros(B, Q, dtype=torch.bool, device=pred_boxes.device)
        for b, ib in enumerate(criterion._ignore_boxes or []):
            if len(ib) == 0:
                continue
            iou = box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(pred_boxes[b].detach()),
                box_ops.box_cxcywh_to_xyxy(ib.to(pred_boxes.device)),
            )[0]
            mask[b] = iou.max(1).values >= iou_thresh
        return mask

    def loss_labels(outputs, targets, indices, num_boxes, log=True):
        """Compute the original classification loss with unmatched ignored queries masked.

        Args:
            outputs: Prediction logits and normalized boxes.
            targets: Category-labeled targets after ignore removal.
            indices: Hungarian assignments for labeled specimens.
            num_boxes: Loss normalization factor from the original criterion.
            log: Include classification error statistics when True.

        Returns:
            Loss dictionary; box losses and matched labeled queries are unchanged.
        """
        src_logits = outputs["pred_logits"]
        idx = criterion._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        alpha, gamma = criterion.focal_alpha, 2
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_targets = torch.diag(
            box_ops.box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )[0]
        )
        pos_ious = iou_targets.clone().detach()
        prob = src_logits.sigmoid()
        pos_weights = torch.zeros_like(src_logits)
        neg_weights = prob**gamma
        pos_ind = [i for i in idx]
        pos_ind.append(target_classes_o)
        t = torch.clamp(prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha), 0.01).detach()
        pos_weights[tuple(pos_ind)] = t.to(pos_weights.dtype)
        neg_weights[tuple(pos_ind)] = 1 - t.to(neg_weights.dtype)
        loss_ce = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)
        # --- ignore regions: unmatched queries sitting on a pseudo-labeled grain get no loss
        mask = ignore_mask(outputs["pred_boxes"])
        mask[idx] = False
        criterion._ignore_hits += int(mask.sum())
        loss_ce = loss_ce * (~mask).unsqueeze(-1).to(loss_ce.dtype)
        losses = {"loss_ce": loss_ce.sum() / num_boxes}
        if log:
            from rfdetr.models.math import accuracy

            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    criterion.forward = forward
    criterion.loss_labels = loss_labels
    return criterion


def seed_class_head_from_one_class(model, init_path, pollen, n_real=28):
    """Initialize category outputs from the pretrained pollen output row.

    Args:
        model: RF-DETR model with decoder and optional proposal classification heads.
        init_path: Verified one-class checkpoint containing a model state dictionary.
        pollen: Output row identified by the detector's observed class IDs.
        n_real: Number of category outputs to initialize, excluding the unused slot.

    Returns:
        Initialization audit including the selected row and number of updated heads.
        Decoder and proposal head parameters are updated in place.

    Notes:
        Copying the pollen row avoids alternating trained pollen and unused output
        weights when RF-DETR expands a two-row checkpoint head.
    """
    import torch

    sd = torch.load(init_path, map_location="cpu", weights_only=False)["model"]
    b = sd["class_embed.bias"]
    if b.shape[0] != 2:
        return {"seeded": False, "checkpoint_head_rows": int(b.shape[0])}
    assert pollen in (0, 1), pollen
    norms = sd["class_embed.weight"].norm(dim=1)
    w = sd["class_embed.weight"][pollen]
    heads = [("class_embed", model.class_embed, w, b[pollen])]
    if getattr(model, "two_stage", False):
        for g, m in enumerate(model.transformer.enc_out_class_embed):
            key = f"transformer.enc_out_class_embed.{g}"
            heads.append((key, m, sd[f"{key}.weight"][pollen], sd[f"{key}.bias"][pollen]))
    with torch.no_grad():
        for _, m, wr, br in heads:
            m.weight[:n_real] = wr.to(m.weight)
            m.bias[:n_real] = br.to(m.bias)
    return {
        "seeded": True,
        "pollen_row": pollen,
        "pollen_row_source": "pred_class_id_hist / --pollen-row",
        "heads": len(heads),
        "bias_rows": [float(x) for x in b],
        "weight_norm_rows": [float(x) for x in norms],
    }


def pollen_row_for(dataset_dir, override):
    """Read the observed pollen output row from ignore-proposal metadata.

    Args:
        dataset_dir: COCO export containing ignore_labels_summary.json.
        override: Optional explicit row; must agree with recorded detections.

    Returns:
        Integer checkpoint row used for pollen.

    Raises:
        AssertionError: Observed rows are ambiguous or disagree with the override.
    """
    import json

    p = Path(dataset_dir) / "ignore_labels_summary.json"
    if p.exists():
        hist = json.loads(p.read_text())["pred_class_id_hist"]
        rows = [int(k) for k, v in hist.items() if v > 0]
        assert len(rows) == 1, f"one-class model fired on several rows: {hist}"
        if override is not None:
            assert override == rows[0], (override, hist)
        return rows[0]
    assert override is not None, "dataset has no ignore_labels_summary.json; pass --pollen-row"
    return override
