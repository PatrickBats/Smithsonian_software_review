# Data and checkpoint setup

## Access

The original NDPI microscopy images, verified NDPA annotations, prepared data, and
checkpoints are available to authorized Rice project accounts on NOTS. Contact the
course/project team for allocation membership if access is denied. Use your own
account; no passwords or keys belong in this repository.

The shared project artifact root is `/rhf/allocations/dsci435/Smithsonian_F26`.
This is a documented storage location, not an execution requirement. On a machine
where it is mounted, set `SMITHSONIAN_ROOT` to that root. On a workstation, copy
only the required inputs to a private external directory and retain the relative
layout below. Do not copy whole slides merely to run the crop demo.

```bash
# Example from your workstation; replace YOUR_NETID and destination.
scp -r YOUR_NETID@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_swin_2026-10-08 external/runs/
scp -r YOUR_NETID@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_dino_2026-10-08 external/runs/
```

Create `external/runs` first. For the smallest download, retrieve only `best.pt`
from the run path in `config/checkpoints.json` rather than a full run directory.
The artifact tool verifies the received bytes. The instructor needs an authorized
account or a separately supplied authorized artifact bundle; these are not public
anonymous download URLs.

## Input layout

| Workflow | Required external inputs |
| --- | --- |
| Classifier demo | A checkpoint and RGB PNG crop; DINO also requires its pinned source checkout |
| Classifier training | Frozen assignment CSV, category map, verified zero-plane or Tenengrad crop export |
| Method A | `processed/method_a_focus_stack_v1/` baseline inputs, plans, COCO files and image hashes; generic RF-DETR weights |
| RGB combined comparison | `model_B/runs/abbas_rgb_input_check_v1/`, original validation references, source NDPI images, both classifier checkpoints |

Classifier checkpoint paths are listed in `config/checkpoints.json`. The default
separate-slide runs contain `record_assignments.csv`, `fine_to_broad.csv`, and
category metadata. The crop-wide Tenengrad export requires its `dataset/`,
`binding.json`, and `baseline_crop_manifest.csv` together. The verified export is provided as
`software_review_2026-10-20/classifier_inputs.tar.gz` under the shared project root.
Its fingerprint is recorded in `config/input_bundle.json`. `dataset/crop_manifest.csv` lists per-image paths and
hashes. `examples/crops.json` selects three validation examples from that export.

Training retains exact frozen hashes. Do not edit the assignment CSV to repair
paths. Raw-slide workflows need the original paths recorded in external manifests
or a compatible read-only mount. Portable crop inference does not depend on those
raw-slide paths.

## Upstream sources

```bash
mkdir -p external
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
export DINO_SOURCE="$PWD/external/dinov2"
```

The RGB full-slide comparison additionally uses the existing Spring source:

```bash
git clone https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26.git external/spring
git -C external/spring checkout 1b62b68a0d95178e6275ef55c38ee4765317b334
export SPRING_SOURCE="$PWD/external/spring"
```

Access to that repository may require organization membership. Upstream code is
kept outside the submission, at a fixed revision. DINO inference loads the complete
fine-tuned weights strictly and does not download a new backbone. DINO training
uses the official pretrained backbone and may download those initialization weights.

## Preflight

```bash
python scripts/preflight.py --artifact-root "$SMITHSONIAN_ROOT" --crops "$CROP_ROOT" --dino-source "$DINO_SOURCE"
```

This validates both separate-slide checkpoint hashes, the example images, source
revision, and installed classifier dependencies. It does not assert successful
whole-slide inference or validate source taxonomy. All raw microscopy is read-only.

## Download the crop bundle

```bash
mkdir -p external
scp YOUR_NETID@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/software_review_2026-10-20/classifier_inputs.tar.gz external/
sha256sum external/classifier_inputs.tar.gz
# Compare with config/input_bundle.json before extracting.
tar -xzf external/classifier_inputs.tar.gz -C external
export CROP_ROOT="$PWD/external/classifier_inputs/dataset"
export ASSIGNMENTS="$PWD/external/classifier_inputs/record_assignments.csv"
```

The bundle is approximately 314 MB and contains prepared crops and required
scientific manifests, not whole slides. It includes reserved test files only for
input-integrity checks; the demo manifest selects validation images exclusively.
Do not run model selection or example inference on the test folder.
