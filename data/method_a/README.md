# Method A inputs and trained models

Method A reads the existing external image/annotation inputs; this change does
not move them, create an image bundle or introduce a new input format. This
folder makes the input contract and frozen split inspectable in the repository.

| File | What it records |
| --- | --- |
| [input_manifest.json](input_manifest.json) | Required input roles, recorded external paths, available hashes and configuration-specific datasets |
| [split_summary.json](split_summary.json) | Adopted assignment, split counts, grouping checks and effective training/validation coverage |
| [broad_category_map.json](broad_category_map.json) | Original byte-identical model ID 0–27 → category mapping |
| [../fine_to_broad.csv](../fine_to_broad.csv) | Shared 32 → 28 map; referenced rather than duplicated |

The two JSON manifests are compact metadata, not lists of individual specimens
or image payloads. External locations describe recorded artifacts; live NOTS
availability was not re-audited for this reorganization.

For the four configurations, read the [Method A overview](../../src/method_a/README.md).
For commands and output examples, read the [worked examples](../../src/method_a/examples/README.md).

## Which files do I need?

| Task | Required input | Example location or shape |
| --- | --- | --- |
| Regenerate the comparison table | Packaged aggregate results only | [Recorded comparison](../../results/method_a/recorded_comparison.json) |
| Detect specimens in one region | One authorized RGB PNG/JPEG and one trained 28-class checkpoint | `external/method_a_example/region.png` and a named Method A checkpoint |
| Prepare the baseline | Frozen assignment, fine-to-broad map, original NDPA/NDPI metadata and existing fused PNGs | Layout below |
| Prepare A1 | Baseline plans, COCO annotations and linked PNGs | `processed/method_a_focus_stack_v1/` |
| Prepare A2/A3 | Baseline inputs plus original train-slide focal series | Source paths recorded in the frozen manifests |
| Replay full validation | Baseline plans/images, selected model's run status and checkpoint | Existing completed experiment root; separate new results root |

Images and weights require authorized Rice allocation access or an authorized
copy from the project team. The historical shared artifact root is
`/rhf/allocations/dsci435/Smithsonian_F26`. Set `SMITHSONIAN_ROOT` to that mounted
root, or mirror the needed relative directories under a private local directory.
The generic [data guide](../../docs/DATA.md) explains account access.

For a new NOTS run, put derived images, row-level manifests and large checkpoints
under a new user-owned RHF directory; put code, compact results and logs under
the project's `/projects` tree. Do not use completed shared paths as new outputs.

## Frozen data layout

Paths below are relative to `SMITHSONIAN_ROOT`. The baseline code uses this layout.

```text
runs/broad_swin_2026-10-08/separate_slides/ordinary_baseline/
  record_assignments.csv           Frozen original assignment
  fine_to_broad.csv                Original 32 → experimental 28 categories
model_B/data/abbas_focus_stack_v1/
  train/<tile_id>.png              5,152 original fused grid images
  valid/<tile_id>.png              1,432 original fused grid images
processed/method_a_focus_stack_v1/
  inputs/priority_records.csv      Assignment with Method A fields added
  inputs/fine_to_broad.csv         Exact copy of the original map
  inputs/broad_category_map.json   Model ID 0–27 → category name
  plan/<slide_index>.json          Tiles, source coordinates and known boxes
  dataset/train/_annotations.coco.json
  dataset/valid/_annotations.coco.json
  dataset/source_image_hashes.json
  dataset/dataset_summary.json
  dataset/{train,valid}/<tile_id>.png  Read-only links to the fused PNGs
```

The frozen assignment contains 2,283 specimens: 1,605 train, 347 validation and
331 reserved test. The original source inventory has 2,290 priority annotations
and 32 categories; the frozen experiment is a particular historical selection
with an explicit 28-category map, not a replacement source inventory.

Input checks pin the original assignment SHA-256 to
`da2062a127cf6e14945e50fff097e0db102d7f8203250b47daed6b6bb166478d`
and the fine-to-broad map to
`a2f8da0b40cf9a1a89ea7fffe360770b2b9eae56f9fdd307bbf1b286bd9f793d`.
The generated `priority_records.csv` has a different hash because it adds fields.
Original source-image paths inside these files must remain readable; relocating
the artifact root alone does not rewrite them.

## How the split is used

Method A **adopts Patrick's existing assignment**, rather than drawing a new
random split. The original `record_assignments.csv` and its SHA-256 in
[split_summary.json](split_summary.json) are authoritative. A guessed ratio or
seed cannot reproduce it. Preparation verifies those exact bytes, adds the Method A
label fields and renames `val` to `valid`; it does not move any specimen between sets.

| Partition | Assigned records | Exact source images | Frozen split groups |
| --- | ---: | ---: | ---: |
| train | 1,605 | 47 | 45 |
| valid | 347 | 14 | 14 |
| test | 331 | 14 | 14 |

The retained input contains 75 source identities and 73 split groups, including
two groups containing two source identities each. Checks on the retained CSV
found no duplicate record IDs, no source identity crossing partitions, and no
split group crossing partitions. Group membership comes from its existing
`slide_id` / `split_group_id` fields, not newly inferred accession or L/R names.

The original 2,290-record source inventory remains separate. The adopted
2,283-record assignment excludes five repeated same-category records and both
records of one conflicting-label pair; this historical selection is not a rule
for deleting records from the source inventory.

All four configurations retain this assignment. Only training image construction
changes: baseline/A1 cover 1,061 training targets; A2 covers 1,063 because two
known labels occur incidentally inside crops; A3 covers all 1,605. The shared
validation grid covers 274 of 347 assigned references in 1,432 tiles. The other
73 are omitted by the retained grid rules, not reassigned. No test pixels are
exported or evaluated by these workflows; reading test membership from the
assignment does not constitute test inference.

## Which command consumes which input?

```bash
python scripts/method_a/prepare.py --config baseline --stage bind
python scripts/method_a/prepare.py --config A1 --stage check
```

The first reads the original assignment and shared map, verifies their hashes,
and creates new Method A input files under a private baseline destination. The
second requires the completed baseline plans/dataset and verifies the consumed
training/validation PNGs. Both are preparation writers: use only new owned output
roots as explained in the [command guide](../../scripts/method_a/README.md).
Full source geometry/export additionally needs the recorded NDPI/NDPA paths.
Table export needs only saved aggregate JSON; a single-region model demo needs
only your supplied RGB region, matching weights and the category map.

## Example annotation format

The following is a **made-up format example**, not a specimen from the dataset.
In COCO, a box is `[left, top, width, height]` in image pixels:

```json
{
  "images": [{"id": 1, "file_name": "example_tile.png", "width": 1024, "height": 1024}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [100, 120, 120, 120], "area": 14400,
                   "iscrowd": 0, "record_id": "example-only"}],
  "categories": [{"id": 1, "name": "example category", "supercategory": "F26 broad YES"}]
}
```

Actual training files contain all 28 categories, using IDs **1–28**. The loader
remaps them to model IDs **0–27**. Prediction boxes instead use
`[left, top, right, bottom]`; see the image-region example. An unassigned model
output slot 28 is excluded and audited by the variant evaluator.

## Where the four trained checkpoints are

These are recorded artifact locations, not newly verified downloads. Paths are
relative to `SMITHSONIAN_ROOT`; exact hashes are in
[config/checkpoints.json](../../config/checkpoints.json).

| Configuration | Artifact-tool name | Relative checkpoint path |
| --- | --- | --- |
| Baseline | `method_a_baseline` | `processed/method_a_focus_stack_v1/run_artifacts/detector/checkpoint_best_total.pth` |
| A1 | `method_a_a1` | `processed/method_a_sparse_labels_20261017/run_artifacts/m1_positive_grid/training/checkpoint_final_ema.pth` |
| A2 | `method_a_a2` | `processed/method_a_sparse_labels_20261017/run_artifacts/m2_retile_matched/training/checkpoint_final_ema.pth` |
| A3 | `method_a_a3` | `processed/method_a_sparse_labels_20261017/run_artifacts/m2_retile_all/training/checkpoint_final_ema.pth` |

Example: once the A3 checkpoint is accessible at that relative path, verify and
copy it to a local inference location:

```bash
python scripts/artifacts.py --name method_a_a3 --root "$SMITHSONIAN_ROOT" --destination weights/method_a_a3.pth
```

Expected console fields include `"status": "verified"` and `"name": "method_a_a3"`.
The command copies a file already available under `--root`; it does not download
from NOTS. For remote retrieval, copy the selected path from the table with your
own account, preserving its relative layout, then run this verification command.

Full evaluation also requires the matching `training/status.json` (baseline:
`detector/status.json`), plans and dataset links. A weights-only copy is sufficient
for the small inference example, not the full historical validation workflow.
