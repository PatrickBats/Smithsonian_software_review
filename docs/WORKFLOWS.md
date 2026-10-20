# Workflow commands

Run from the repository root with the documented dependencies installed. Commands
below are separate stages, not a scheduler dependency chain. Use new output
locations and wait for each prerequisite to succeed. Full training requires an
allocated GPU and does not run as part of the software checks.

## Crop preparation and classifier training

`full_tenengrad.py` reads the original crop geometry, scores focal views using mean
squared Sobel gradients, and exports the sharpest plane. It now runs local CPU
workers using the active Python environment; it does not submit cluster jobs.
Source slides must be readable at their recorded manifest paths. The zero-plane
baseline hash must reproduce before a replacement crop is accepted.

```bash
python scripts/classification/full_tenengrad.py --baseline "$ZERO_PLANE_CROPS" --assignments "$ASSIGNMENTS" --categories data/category_map_32.csv --output "$NEW_FOCUS_RUN" --workers 4
```

Use the resulting `dataset/` as `CROP_ROOT`. For the original broad-class runs:

```bash
python scripts/classification/train_swin.py --data "$CROP_ROOT" --assignments "$ASSIGNMENTS" --categories data/category_map_32.csv --broad-map data/fine_to_broad.csv --output "$NEW_SWIN_RUN"
python scripts/classification/train_dino_finetune.py --data "$CROP_ROOT" --assignments "$ASSIGNMENTS" --categories data/category_map_32.csv --broad-map data/fine_to_broad.csv --output "$NEW_DINO_RUN"
```

Outputs must be outside this repository. Use `--smoke --epochs 2 --head-epochs 1`
for a bounded training smoke check in a new output directory. This still trains;
it is not needed for the inference demo. Shared-slide reproduction adds
`--specimen-split "$SHARED_SPLIT_JSON"` using the saved diagnostic assignment.
All trainers preserve the original test membership and select checkpoints using
validation macro-F1. They are strict reproduction entry points for this dataset;
the inference CLI is the reusable interface for other RGB specimens.

## Baseline region preparation

Copy `config/pipeline.example.json` to an ignored local configuration and replace
input and output paths. This defines the retained baseline: fixed annotated ROI
grid, LoG fusion, frozen split, and detector/classifier checkpoint identities.

```bash
python scripts/pipeline_abbas_v1/check_abbas_contract.py
python scripts/pipeline_abbas_v1/prepare_pipeline_data.py --config config/local.json --plan-only
python scripts/pipeline_abbas_v1/prepare_pipeline_data.py --config config/local.json --shard 0 --shards 4
```

Repeat export for shards 1–3. The baseline region grid is retained for reproducing
historical detection work; it does not define which specimens may enter classifier
training. Sparse labels inside these regions remain a limitation.

## RF-DETR Method A

Set paths using `config/environment.example.sh`. To rebuild baseline inputs from
existing exported region PNGs, the ordered stages are:
`build_inputs.py`, `plan_grid.py`, and `build_linked_dataset.py` in
`scripts/method_a_focus_stack_v1/`. These reference the configured
`SMITHSONIAN_ROOT` layout. Their baseline inputs must be prepared before A1/A2/A3.

```bash
python scripts/method_a/data_variants/audit_plan.py
python scripts/method_a/data_variants/prepare_datasets.py m1
python scripts/method_a/data_variants/prepare_datasets.py plan
python scripts/method_a/data_variants/export_crops.py --slide-index 0 --limit 2
```

The limited export checks two crops. For full preparation, export each planned
slide index without `--limit`, then run `prepare_datasets.py finalize`.
A1 keeps positive baseline tiles; A2 uses context windows anchored on originally
covered targets; A3 includes all assigned training targets. All retain the same
28-class task and fixed validation grid.

```bash
python scripts/method_a/data_variants/train.py --variant A1 --smoke
python scripts/method_a/data_variants/train.py --variant A1
python scripts/method_a/data_variants/evaluate.py --variant A1
```

Repeat with A2/A3 as needed. Never rerun against completed shared directories.
The baseline training and evaluation entry points remain available in
`scripts/method_a_focus_stack_v1/` for historical reproduction. New variant runs
use 2,400 optimizer steps and the final EMA checkpoint. No automatic cleanup of
shared checkpoints is included.

## Small detector-to-classifier example

The detector demo accepts an existing RF-DETR 2XL checkpoint and a supplied RGB
image region. Use the class count matching the checkpoint: 1 for the single-class
pipeline or 28 for a Method A model. Supply its recorded SHA-256.

```bash
python scripts/detect.py --checkpoint "$DETECTOR" --sha256 "$DETECTOR_SHA256" --image "$REGION" --classes 1 --output outputs/detection
python scripts/infer.py --model dino --checkpoint weights/dino.pt --dino-source "$DINO_SOURCE" --image "$REGION" --boxes outputs/detection/boxes.json --output outputs/combined.json
```

For Swin change `--model` and `--checkpoint`. Boxes are xyxy pixel coordinates
relative to this region; they are not NDPA nanometres. The handoff uses a 10%
context margin, white square padding, 224-pixel resize, and ImageNet normalization.
This is a local-region demonstration, not the full-slide evaluation implementation.

## Reproduce the frozen RGB comparison

The complete paired comparison retains the recorded detection boxes and uses
identical crops for both classifiers. It requires the external artifacts and
Spring source listed in the data guide.

```bash
python scripts/rfdetr_rgb_classifier/compare_rgb_dino.py --preflight
python scripts/rfdetr_rgb_classifier/compare_rgb_dino.py
```

Set `COMPARISON_OUTPUT` to a new directory. The full run requires a GPU and checks
reconstructed tile hashes, matching, and all saved Swin labels before accepting
its output. Historical BGR/RGB diagnostic scripts are included as technical
provenance, not recommended default preprocessing. Use RGB input for the demo.

## Outputs

Training writes configuration, provenance, metrics, and checkpoints. Validation
writes predictions, matching results, and aggregate scores. `scripts/results.py`
rebuilds the supplied tables and vector chart from aggregate JSON, without rerunning
models. Inspect `VALIDATION.md` for which packaged workflows were exercised.

## Replay classifier validation without training

```bash
python scripts/pipeline_abbas_v1/compare_classifiers.py --crop-root "$CROP_ROOT" --broad-map data/fine_to_broad.csv --swin-checkpoint weights/swin.pt --dino-checkpoint weights/dino.pt --dino-source "$DINO_SOURCE" --device cpu --output "$NEW_VALIDATION_RUN"
```

This evaluates the fixed separate-slide validation crops. It does not evaluate
test specimens. `--pipeline-root` optionally adds the recorded detector-crop
comparison when that mirrored artifact tree is present.

## Reusable interfaces

| Function | Input | Output / checks |
| --- | --- | --- |
| `pipeline_common.crop_input` | RGB PIL image, finite xyxy box, margin | 224-square RGB crop; white padding; invalid geometry rejected |
| `compare_classifiers.load_classifier` | Architecture, trusted checkpoint, DINO source, device | Frozen model and checkpoint-ordered labels; strict weight loading |
| `compare_classifiers.classify` | Model and PIL images | Per-image probabilities in label order; nonfinite logits rejected |
| `compare_classifiers.match_boxes` | Prediction and reference box dictionaries | One-to-one greedy IoU matches; no repeated credit |
| `train_swin.verify_inputs` | Crop root, assignment CSV, category CSV | Verified records, labels, and audit; source hashes and split isolation enforced |
| `broad_labels.apply_broad` | Verified records, fine labels, mapping CSV | Copied records with broad targets; original identities and fine labels retained |
| `specimen_split.apply_split` | Verified records and saved split JSON | Diagnostic memberships; original test set unchanged |
| `tenengrad_pilot.score` | uint8 RGB array | Finite mean squared Sobel-gradient focus score |

The helper `tenengrad_pilot.py` is executed by the full extraction orchestrator
with a supplied record list; it is not a standalone command.
