# Evaluation

## Evaluate the joint detector

```bash
python scripts/evaluate_detector.py --config config/detector_evaluation.example.json
```

This runs the saved joint checkpoint on the fixed validation grid and writes
predictions, reference matches, and summary JSON. `detector.py` merges overlapping
tile predictions before scoring. `matching.py` assigns each prediction and reference
at most once using greedy descending IoU matching. Boxes use slide-pixel `xyxy`
coordinates.

The reported joint-detector result uses confidence 0.30 and IoU 0.50. The recorded
combined-pipeline comparison uses confidence 0.50 and IoU 0.50. Its classifiers
receive crops from tile-level best-plane detector inputs. See the
[inference guide](../inference/README.md#recorded-full-slide-classifier-comparison)
for replay commands and required artifacts.

## Metrics

- Classifier accuracy: correct labels among the evaluated crops.
- Detection recall: located targets divided by annotated targets.
- Classification among detections: correct labels divided by matched detections.
- End-to-end success: targets both located and correctly labeled divided by all annotated targets.

With incomplete scene annotations, unmatched predictions require expert review
to determine whether they are real specimens or false positives. The stored
detector metrics report known-target recovery. Classifier macro-F1 averages over
categories with reference examples.

## Regenerate recorded results

`results/` contains classifier, joint-detector, and combined-pipeline aggregate records.
Run this command with a new output directory:

```bash
python scripts/results.py --output outputs/results
```

It generates `classification.csv`, `detection.csv`, `rgb_comparison.csv`, and
`classification.svg` from the included records.

The classifier records use expert-located crops. The joint crop-and-ignore-region
run located 243 of 274 known targets and correctly identified 85. The combined
comparison evaluates both classifiers on 258 matched detected crops from 274 known
targets. Source credits and revisions are in [NOTICE.md](../../../NOTICE.md);
checkpoint identities are in [config/checkpoints.json](../../../config/checkpoints.json).
Historical dataset comparisons remain in the original working project repository.
