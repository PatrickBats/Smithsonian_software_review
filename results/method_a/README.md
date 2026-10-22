# Method A results: baseline, A1, A2 and A3

These are completed validation results for one RF-DETR 2XL architecture with four
training-data configurations. [Method A](../../src/method_a/README.md) finds
specimens and predicts their identification categories in the same model.
The three newer configurations were completed on October 18, 2026; the original
baseline was completed on October 13.

## Read the four-row comparison

Every row evaluates the same 1,432 original-grid validation tiles containing
274 known specimens, at confidence ≥0.50 and box IoU ≥0.50. IoU measures box
overlap: 0.50 is the required matching threshold.

| Configuration | Training images | Known train specimens | Located / 274 | Located + correct category / 274 | End-to-end recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 5,152 | 1,061 | 24 | 20 | 7.30% |
| A1: positive original tiles | 884 | 1,061 | 84 | 46 | 16.79% |
| A2: original-coverage context crops | 918 | 1,063 | 101 | 46 | 16.79% |
| A3: all-train context crops | 1,423 | 1,605 | 99 | 55 | 20.07% |

A3 has the highest observed end-to-end recovery among the three new runs.
A2 finds more known locations than A1, but both identify 46 correctly.
These are single-seed observations, not evidence of statistical significance.

The historical baseline used different augmentation, learning-rate scheduling,
training duration and AP-based checkpoint selection. A1/A2/A3 share a fixed
2,400-optimizer-step training budget and final EMA weights. Therefore, the
baseline-to-variant comparison does **not** isolate data preparation alone.
The [unified command guide](../../scripts/method_a/README.md) explains both
the crop construction and the distinct training protocols.

## What each number means

| Metric | Calculation | Example: A3 |
| --- | --- | --- |
| Known-target detection recall | Matched known targets / 274 | 99/274 = 36.13% |
| Classification accuracy among matches | Correct categories / matched targets | 55/99 = 55.56% |
| End-to-end recovery | Correct location and category / 274 | 55/274 = 20.07% |

The 274 targets are those covered by the retained grid, not all 347 assigned
validation specimens. The other 73 do not enter this evaluation. Sparse circles
do not label every visible grain. Unmatched predictions are unreviewed, not
established false positives, so full-scene precision/AP is not claimed.
The final test set remains unused.

## Files and examples

| File | What it contains |
| --- | --- |
| [recorded_comparison.json](recorded_comparison.json) | Original aggregate: three experiments plus a separate `baseline` object |
| [baseline_validation.json](baseline_validation.json) | Original PR #5 baseline validation summary, retained byte-for-byte |
| [a1_validation.json](a1_validation.json), [a2_validation.json](a2_validation.json), [a3_validation.json](a3_validation.json) | Primary per-class support/counts, checkpoint hashes and fixed-threshold diagnostics |
| [dataset_counts.json](dataset_counts.json) | Training image, box and unique-specimen counts; frozen data hashes |
| [provenance.json](provenance.json) | Recorded jobs, model settings, evaluation recovery and algorithm fingerprints |

Example fields in A3's `primary` object are:

```json
{"known_specimens": 274, "detected": 99, "end_to_end_correct": 55,
 "known_specimen_recall_iou50": 0.3613138686131387,
 "end_to_end_known_specimen_success": 0.20072992700729927}
```

This is an excerpt, not a replacement for the saved JSON. The new-run evaluator
excluded and audited an unassigned output slot 28; no excluded predictions reached
the primary 0.50 threshold. The recovery changed evaluation handling, not weights.

## Regenerate tables or rerun models

To read the included JSON and generate the four-row CSV without images or a GPU:

```bash
python -m pip install --no-deps -e .
python scripts/method_a/export_results.py --output outputs/method_a_comparison
```

The exact input/output example is in
[src/method_a/examples](../../src/method_a/examples/README.md#1-inspect-all-four-recorded-results).
That command does not rerun inference. To replay full model validation, use the
[variant evaluation instructions](../../scripts/method_a/README.md)
or [baseline evaluation instructions](../../scripts/method_a/README.md),
with the external data and model artifacts listed in the
[data README](../../data/method_a/README.md).

Source evidence comes from
[PR #5](https://github.com/PatrickBats/Smithsonian_fossil_F26/pull/5) and
[PR #7](https://github.com/PatrickBats/Smithsonian_fossil_F26/pull/7).
The six original A1/A2/A3 JSON files are unchanged. The baseline summary was
imported from PR #5 commit `9629be03b886638c983e1c15e2a57ea16f7dca27`.
