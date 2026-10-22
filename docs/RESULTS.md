# Results and how to reproduce them

There are three different levels of reproduction. Choose the one you need:

1. **Regenerate tables/figures:** read the included aggregate JSON; no model runs.
2. **Replay inference:** use saved checkpoints on the required images.
3. **Retrain:** create new checkpoints using the recorded data and protocol.

The first-run walkthrough covers level 1 and a small inference demonstration.
A successful figure-generation command does not verify a complete training run.

## Results-to-command map

| Result | Evidence and configuration | Code to run | Expected output |
| --- | --- | --- | --- |
| Swin/DINO classifier comparison | `results/{swin,dino}_{separate_slides,shared_slides}.json`; 28-category map; named checkpoints in `config/checkpoints.json` | `python scripts/results.py --output outputs/comparison` | `classification.csv`, `classification.svg` |
| RF-DETR baseline + A1/A2/A3 comparison | `results/method_a/recorded_comparison.json`; confidence 0.5, IoU 0.5 | Same result-generation command | `detection.csv` (all four configurations) |
| Swin/DINO on fixed RGB detector crops | `results/rgb_comparison.json`; same 258 detected objects | Same result-generation command | `rgb_comparison.csv` |
| Separate-slide classifier inference replay | Prepared crop export, broad map, separate-slide Swin/DINO weights | [Classifier replay command](WORKFLOWS.md#replay-classifier-validation-without-training) | Per-model JSON and `summary.json` in the chosen output directory |
| RF-DETR variant inference replay | Configured Method A roots and variant's completed checkpoint | `python scripts/method_a/evaluate.py --config A1` (or A2/A3) | Validation predictions, matching, and summary under the configured results root |
| Fixed RGB comparison replay | Frozen tile/detection artifacts, raw NDPI sources, pinned Spring/DINO source, both weights | [RGB replay commands](WORKFLOWS.md#reproduce-the-frozen-rgb-comparison) | `comparison.json`, predictions, per-class comparison, and input-verification records |
| Retrain classifiers | Frozen 1,605/347/331 split, crop provenance, label map, recorded training schedule | [Classifier training commands](WORKFLOWS.md#crop-preparation-and-classifier-training) | New checkpoint, training history, validation metrics, and provenance |
| Retrain RF-DETR variants | Baseline data plus A1/A2/A3 preparation; 2,400 optimizer steps | [Method A commands](WORKFLOWS.md#rf-detr-method-a) | New final EMA checkpoint and training record |

The classifier replay command is explicitly for **separate-slide** validation.
It does not automatically reproduce shared-slide scores. Shared-slide training
uses the preserved diagnostic split via `--specimen-split`; retain the corresponding
saved results and checkpoint when comparing that setting.

The [Method A result README](../results/method_a/README.md)
explains the baseline/variant evidence files and gives input/output examples for
regenerating their four-row comparison.

## Read the numbers with the right denominator

| Evaluation | Denominator | What a correct result requires |
| --- | --- | --- |
| Expert-crop classification | Evaluated annotated crops | Correct category |
| Known-target detection recovery | Known annotated targets in covered regions | Matched predicted box |
| Classification among detections | Matched targets found by the detector | Correct category for the matched crop |
| End-to-end success | All known targets, including misses | Both a matched location and correct category |

In the recorded RGB comparison, both classifiers receive the same **258 detected
objects** from **274 known targets**. DINO gets 125 correct and Swin gets 93:
48.4% versus 36.0% among detections, or 45.6% versus 33.9% end to end.
These are not the expert-crop accuracies shown on the README.

The reused detector saw 13 of the 14 validation slides during its original
training. The RGB comparison is therefore a diagnostic, not an independent test
of generalization to unseen slides. Incomplete scene labels also prevent treating
all unmatched detections as false positives.

## What is verified

[VALIDATION.md](../VALIDATION.md) distinguishes completed software checks from
workflows that were not rerun during packaging. The final test set remains
reserved. Recorded results are single-run evidence; do not interpret small score
differences as established statistical significance.
