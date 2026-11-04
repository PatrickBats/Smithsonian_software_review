# Data and checkpoint setup

For a complete copy-and-run example, use the [first-run walkthrough](QUICKSTART.md).
This page describes the storage layout and additional inputs for larger workflows.

The minimum classifier demo needs **one PNG, one verified checkpoint, and the
classifier environment**. The provided crop bundle contains verified examples
ready for inference.

## Access

The original NDPI microscopy images, verified NDPA annotations, prepared data, and
checkpoints are available to authorized Rice project accounts on NOTS. Contact the
course/project team for allocation membership if access is denied. Use your own
account; no passwords or keys belong in this repository.

The shared project artifact root is `/rhf/allocations/dsci435/Smithsonian_F26`.
On a machine where this location is mounted, set `SMITHSONIAN_ROOT` to that root.
On a workstation, copy the required inputs to a private external directory and
retain the relative layout below. The crop demo uses the prepared PNG images.

```bash
# Optional full run download; the walkthrough downloads only the required weights.
mkdir -p external/runs
scp -r YOUR_NETID@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_swin_2026-10-08 external/runs/
scp -r YOUR_NETID@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_dino_2026-10-08 external/runs/
```

Create `external/runs` first. For the smallest download, retrieve only `best.pt`
from the run path in `config/checkpoints.json`.
The artifact tool verifies the received bytes. The instructor needs an authorized
account or a separately supplied authorized artifact bundle.

## Input layout

| Workflow | Required external inputs |
| --- | --- |
| Classifier demo | A checkpoint and RGB PNG crop; DINO also requires its pinned source checkout |
| Classifier training | Frozen assignment CSV, category map, verified zero-plane or Tenengrad crop export |
| Joint RF-DETR | `processed/method_a_focus_stack_v1/` baseline metadata/validation export; current target crops and ignore boxes; verified best-plane single-class initialization |
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
or a compatible read-only mount. Crop inference uses the local paths in your
configuration.

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
fine-tuned weights from the supplied checkpoint with strict parameter matching.
DINO training uses the official pretrained backbone and may download those
initialization weights.

## Preflight

```bash
python scripts/preflight.py --artifact-root "$SMITHSONIAN_ROOT" --crops "$CROP_ROOT" --dino-source "$DINO_SOURCE"
```

This validates both separate-slide checkpoint hashes, the example images, source
revision, and installed classifier dependencies. Whole-slide inference requires
a separate execution check; category definitions require expert review. Raw
microscopy files are opened read-only.

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
scientific manifests. It includes reserved test files only for
input-integrity checks; the demo manifest selects validation images exclusively.
Do not run model selection or example inference on the test folder.

## Notebook region and detector

The notebook additionally needs the exact prepared validation region and detector
checkpoint listed in [examples/region.json](../examples/region.json). Request these
from the team's authorized artifact custodian, placing them under `external/demo/`
or pointing your local JSON at an existing authorized copy. They are separate from
the classifier bundle. The preflight checks their full SHA-256 hashes before use.
Use the exact checkpoint listed in the manifest to reproduce this demonstration.

## Current joint detector

The initialization and completed joint checkpoint are listed in `config/checkpoints.json`.
Under the RHF root, the completed run is
`model_A/runs/A4_ignore_springinit_20261021_debug/train/`; its training COCO export is
`model_A/runs/A4_ignore_springinit_20261021_debug/data/`. The fixed validation export is
`processed/method_a_focus_stack_v1/`. Retrieve the run's `status.json` with its checkpoint
for evaluation. The crop-generation baseline needs `inputs/priority_records.csv`,
`plan/`, the original source images/annotations, and `dataset/` metadata.

Use local configurations to point to retrieved copies. Source paths inside historical
NDPI manifests remain immutable and require compatible mounts. The PNG region demo
uses locally configured image and checkpoint paths.
