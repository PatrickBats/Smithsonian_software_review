# Method B module responsibilities

The workflow is organized by task: preprocessing, classification, detection,
pipeline execution, evaluation, utilities and attributed reference code.
See README.md for the file map and runnable commands.

## Reusable functions

| Function | Contract |
| --- | --- |
| `classification.load_classifier.load_classifier` | Strict checkpoint loading and ordered class names |
| `classification.predict_categories.classify` | Batched RGB classification, preserving input order |
| `pipeline.crop_detected_specimens.crop_input` | Context crop from detector-input pixels |
| `pipeline.detect_and_classify.run_pipeline` | Single-region detection followed by classification |
| `utils.input_validation.verify_inputs` | Frozen crop inventory, hashes, geometry and split validation |
| `evaluation.classification_metrics.evaluate` | Non-test classifier evaluation |
| `evaluation.classification_metrics.match_boxes` | Greedy one-to-one IoU matching |

The historical full-slide adapter remains in pipeline/reference_slide_pipeline.py.
Its exact crop helper is retained as historical_crop_input so strict replay is not
silently changed to a newer preprocessing implementation. External frozen artifacts
keep their recorded paths and names.

Executable modules use main() and an __main__ guard; helpers are importable modules.
Run from the repository root with python -m method_b.<task>.<module>. Compatibility
shims in scripts/infer.py and scripts/detect.py forward to the new implementations.
Training saves the package source alongside its original single-file snapshots.
