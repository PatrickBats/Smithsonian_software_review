# Evaluation

`matching.py` assigns each prediction and reference at most once using greedy descending
IoU matching at 0.5. Coordinates are slide-pixel `xyxy` boxes. `detector.py` applies the
recorded joint-detector validation protocol, including merging overlapping tiles.

```bash
python scripts/evaluate_detector.py --config config/detector_evaluation.example.json
python scripts/results.py --output outputs/results
```

The first command runs the saved joint checkpoint on the fixed validation grid and writes
predictions, reference matches, and summary JSON. The second only converts existing
aggregate records into `classification.csv`, `detection.csv`, `rgb_comparison.csv`, and
`classification.svg`; it needs no models or images.

- Classifier accuracy: correct labels among the evaluated crops.
- Detection recall: located targets divided by annotated targets.
- Classification among detections: correct labels divided by matched detections.
- End-to-end success: targets both located and correctly labeled divided by all annotated targets.

Incomplete scene labels leave unmatched predictions unreviewed; these are not automatically
false positives. The stored detector metrics report known-target recovery. Classifier
macro-F1 averages over categories with reference examples.
