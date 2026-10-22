# Run Method A: baseline, A1, A2 or A3

This folder contains four thin command entry points. The implementation is in
[src/method_a](../../src/method_a/README.md). Choose a configuration with
`--config baseline`, `A1`, `A2` or `A3`; the public interface is the same for all four.
The historical baseline and later variants keep their original scientific protocols.

| Command | What it does | Input → output |
| --- | --- | --- |
| `prepare.py --config NAME --stage STAGE` | Performs one preparation stage | Existing split/images/plans → audited inputs, plans or a COCO dataset |
| `train.py --config NAME [--smoke]` | Trains the selected configuration | Verified dataset + generic initialization → status, contract and checkpoint |
| `evaluate.py --config NAME [--smoke]` | Replays known-reference validation | Completed model/status + original validation grid → predictions and metrics |
| `export_results.py --output DIR` | Reads the four saved results | Packaged JSON → four-row `detection.csv` |

Run from the repository root after `python -m pip install --no-deps -e .`.
All four commands support `--help` without GPU/model libraries. Source preparation
requires the documented OpenSlide/Pillow/NumPy runtime; A2/A3 export also uses
OpenCV, tifffile and Zarr. Training/full validation require RF-DETR 1.6.4,
RF-DETR+ 1.0.2, PyTorch and an allocated CUDA GPU; fixed-step training also uses
PyTorch Lightning. Do not install into or modify a shared environment.

## Keep existing inputs and choose new outputs

The [data README](../../data/method_a/README.md) explains sources, the frozen split,
existing input paths and hashes. No new image download format or data bundle is
introduced. For reads/replay, point the environment at the completed artifact tree.
For rebuilding, use an unused private output root with the required input layout;
never direct preparation writers at the completed shared experiment.

The baseline uses `SMITHSONIAN_ROOT` and its fixed relative layout. A1/A2/A3 also
support the overrides below. These Bash examples show local mirror paths; on NOTS,
new image/row-level data must be under an authorized user-owned RHF directory,
while compact results/logs belong under the project allocation's projects tree.

```bash
export SMITHSONIAN_ROOT="$PWD/external"
export METHOD_A_BASELINE_ROOT="$SMITHSONIAN_ROOT/processed/method_a_focus_stack_v1"
export METHOD_A_DATA_ROOT="$SMITHSONIAN_ROOT/processed/method_a_new"
export METHOD_A_RESULTS_ROOT="$PWD/outputs/method_a_results_new"
export METHOD_A_PRETRAINED_WEIGHTS="$SMITHSONIAN_ROOT/weights/rf-detr-xxlarge.pth"
```

Baseline commands do not use the newer `METHOD_A_*` root overrides. The frozen
source paths inside the original manifests must remain readable for raw-source
operations; changing the artifact root does not rewrite them.

## Preparation stages

Choose the row for your configuration. Stages are explicit so you can inspect each
output before continuing; a failed run may leave partial files. Writers refuse
existing dataset/checkpoint destinations. Do not restart a whole sequence blindly.

| Stage | baseline | A1 | A2 / A3 | Main output |
| --- | --- | --- | --- | --- |
| `bind` | Bind shared assignment and labels | — | — | Baseline `inputs/`, including the exact `fine_to_broad.csv` |
| `check` | — | Verify consumed baseline inputs | Same | Variant root `audit/external_inputs_before_initial.json` |
| `audit` | — | — | Inspect training-source geometry | Variant root `audit/train_source_geometry.json` |
| `weights` | — | Copy verified generic weights | Same | Variant root `inputs/rf-detr-xxlarge.pth` |
| `plan` | Recreate original ROI grid | — | Plan shared context windows | Baseline or variant root `plan/*.json` |
| `export` | — | — | Export one slide's context crops | Variant root `images/crops/`, `export_status/NN.json` |
| `build` | Link original grid images and class boxes | Build positive-grid dataset | Build only the selected context dataset | Selected COCO dataset and `dataset_summary.json` |
| `qa` | — | — | Render sampled exported crops | Variant root `audit/crop_qa_contact_sheet.jpg` and JSON |

A dash means the stage is inapplicable and the command rejects it before data I/O.
A2/A3 share one context pool. Plan/export it once; their `build` stages select
original-coverage anchors or all training anchors respectively.

### Rebuild the grid inputs

On an unused private mirror with the original shared assignment/map, raw metadata
and existing fused PNGs available:

```bash
python scripts/method_a/prepare.py --config baseline --stage bind
python scripts/method_a/prepare.py --config baseline --stage plan
python scripts/method_a/prepare.py --config baseline --stage build
```

`bind` creates `priority_records.csv`, `broad_category_map.json`,
`fine_to_broad.csv` and `summary.json` below the baseline `inputs/`. The map copy
closes the historical gap between input binding and the later variant checks.
`plan` reproduces tile IDs and coordinates; `build` produces COCO labels and
read-only links. It does not copy or overwrite the existing fused images.

Example fields in `dataset/dataset_summary.json` are:

```json
{"status": "verified_shared_grid_28_class_sparse_label_dataset",
 "counts": {"train": {"grid_images": 5152, "unique_known_records": 1061},
            "valid": {"grid_images": 1432, "unique_known_records": 274}}}
```

This is an excerpt, not a complete dataset file.

### Build A1 or the context configurations

All three need the completed baseline tree, including the copied mapping:

```bash
python scripts/method_a/prepare.py --config A1 --stage check
python scripts/method_a/prepare.py --config A1 --stage weights
python scripts/method_a/prepare.py --config A1 --stage build
```

A2/A3 additionally need the original training focal series. After the shared
input check and weight copy, run:

```bash
python scripts/method_a/prepare.py --config A2 --stage audit
python scripts/method_a/prepare.py --config A2 --stage plan
python scripts/method_a/prepare.py --config A2 --stage export --slide-index 0 --limit 1
```

The last command is a one-window pilot, not a complete export. To process the
recorded 47 training-source plans, run one CPU task per index, then build the two
selected datasets. Existing completed pilot crops are rechecked and reused.

```bash
(
  set -e
  for slide_index in $(seq 0 46); do
    python scripts/method_a/prepare.py --config A2 --stage export --slide-index "$slide_index"
  done
  python scripts/method_a/prepare.py --config A2 --stage build
  python scripts/method_a/prepare.py --config A3 --stage build
)
```

Their outputs are `datasets/m2_retile_matched/` and `datasets/m2_retile_all/` under
`METHOD_A_DATA_ROOT`. Each includes COCO labels and a verified summary, and shares
the unchanged validation grid. For example, A1 reports 884 train images,
1,586 box instances, 1,061 known train records and 1,432 validation images.

For sampled visual QA after the complete export (at least four sampled slides):

```bash
python scripts/method_a/prepare.py --config A3 --stage qa
```

## Train a selected configuration

Run a smoke check first. The baseline check is one epoch; A1/A2/A3 use two
optimizer steps. The baseline reads generic weights from the shared artifact root;
variants use the verified private copy created by `--stage weights`.

```bash
python scripts/method_a/train.py --config baseline --smoke
python scripts/method_a/train.py --config baseline
```

Or, after preparing the selected variant:

```bash
python scripts/method_a/train.py --config A1 --smoke
python scripts/method_a/train.py --config A1
```

Use A2 or A3 in the same commands for the other prepared datasets. Each invocation
allocates a unique scratch/cache directory below `SLURM_TMPDIR`, or the operating
system temporary directory if unset. Smoke and full commands can run successively
without colliding even inside the same Slurm allocation. Existing datasets and run
outputs still require unused destinations.

| Configuration | Training/checkpoint rule | Output under its data root |
| --- | --- | --- |
| baseline | At most 40 epochs with early stopping; historical run stopped at 16; AP-selected checkpoint | `run_artifacts/detector/checkpoint_best_total.pth` |
| A1/A2/A3 | Exactly 2,400 optimizer steps; final EMA; no validation-based selection | `run_artifacts/<recorded-ID>/training/checkpoint_final_ema.pth` |

Each also writes status and training-contract JSON. Equal interface does not mean
equal baseline/variant training settings; the saved comparison is not a data-only ablation.

## Replay evaluation

For an existing completed run, use its original data/checkpoint/status tree and
choose a new evaluation destination. Variants use a new `METHOD_A_RESULTS_ROOT`;
the baseline writes to its fixed `run_artifacts/validation/` destination, which
must not already exist. Use a private mirror for baseline replay.

```bash
python scripts/method_a/evaluate.py --config baseline
python scripts/method_a/evaluate.py --config A3
```

Full evaluation uses 1,432 original validation tiles and 274 covered known targets.
It writes predictions, matching evidence and aggregate metrics. A variant's
`--smoke` checks two positive validation images; the baseline has no evaluation
smoke mode. A weights-only copy is enough for single-region inference, but full
replay also needs original plans and matching training status files.

After all three new variant runs finish, advanced users can run
`python -m method_a.reporting` to audit those completed runs and publish their
comparison. `python -m method_a.evaluation.regression` checks synthetic box matching
with the model runtime installed. These operations write audit files and must
use the newly owned experiment roots, not completed shared inputs.

## Export saved results without running a model

```bash
python scripts/method_a/export_results.py --output outputs/method_a_comparison
```

Input: [recorded_comparison.json](../../results/method_a/recorded_comparison.json).
Output: `detection.csv` with four rows. See the exact
[expected output](../../src/method_a/examples/expected_detection.csv) and
[worked examples](../../src/method_a/examples/README.md).
