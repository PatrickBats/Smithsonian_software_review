# Software review validation

Checked on October 20, 2026 using a newly created Python 3.13 Linux environment.
The resolved CPU runtime is recorded in `requirements-cpu-lock.txt`.

## Passed

- Fresh-clone repository checks, all 18 unit tests, image/matching regressions,
  result-table generation, and real Swin inference.

- Dependency installation, `pip check`, and RF-DETR/Swin/DINO imports.
- 18 unit tests: category mapping, checksums, geometry input rejection, Method A
  output boundaries, aggregate consistency, and preserved numerical algorithm ASTs.
- Additional executable regression checks: LoG pixel selection, boundary-aware
  deduplication, classifier crop padding, one-to-one matching, metric denominators,
  and shared-file permissions.
- Original classifier integrity checks over all 2,283 prepared specimens,
  including frozen split and category hashes, crop hashes, and focal-plane metadata.
- Real Swin and DINO CPU inference on a selected validation crop. Both labels
  matched saved validation predictions; confidence differences were below 0.00001.
- Real single-region RF-DETR inference using a verified existing single-class
  checkpoint, followed by both DINO and Swin classification of its detected crop.
- Classifier artifact/example preflight against pinned DINO source and checkpoints.
- Full RGB-comparison CPU preflight, including strict Swin/DINO weight loading
  against the frozen detection artifacts.
- The external crop bundle was copied to shared RHF storage and its SHA-256
  verified remotely. It is approximately 314 MB and group-readable.

The small demos establish executable software paths, not additional model-quality
estimates. Their outputs are not included in the aggregate benchmark comparisons.

## Packaging changes

Machine-specific executable roots were replaced with configuration. The focal
crop worker now executes locally rather than invoking a fixed SSH/Slurm command.
The small LoG regression initially failed with OpenCV 4.13 using 96 default
threads: equal grayscale arrays at different memory alignments produced different
blurred pixels. Setting the test to one OpenCV thread, as the production exporter
already does, restored exact pixel equality without changing the criterion.
Model architectures, label mappings, saved metrics, and numerical algorithms were
preserved. Scripts were formatted, and portable demo/checkpoint/result commands
were added. Automatic checkpoint-pruning scripts are not included.

Only approved code, maps, technical provenance, aggregate evidence, and essential
instructions enter the submission. Fresh Git history has no imported parent
history. Large inputs and weights remain external.

## Limits

No models were retrained for packaging. Full raw-slide extraction, full-grid GPU
inference, and all historical experiments were not rerun. Source path metadata
inside external frozen manifests still requires the original read-only mount or
an equivalent layout. The public crop demo accepts local images independently.

Access to the shared allocation and the Spring source repository is required for
those workflows; clone access alone cannot supply private scientific inputs.
Checkpoint payloads are supplied externally with verified fingerprints. Confirm
with the instructor whether that satisfies the rubric's requested weights file.
The final test set remains unused for model inference and selection.

## Method A organization and interface checks — October 21, 2026

The editable package installs with `python -m pip install --no-deps -e .`.
There are 27 checks under `src/method_a/tests/` and three shared checks under
`tests/`, 30 total. CI installs the package and runs both suites explicitly.

The Method A checks preserve 15 original regression checks and add 12 interface
checks. They cover all four configurations through common training/evaluation
commands, preparation dispatch and dataset selection, invalid-argument rejection
before imports/data I/O, exact input-map copying, isolated scratch directories,
inert imports, packaged category metadata, command help from another directory,
and the expected four-row CSV.

The six original variant report JSONs and original baseline summary retain their
bytes. Six frozen numerical function fingerprints still match the recorded
provenance, including crop planning, fusion, matching and deduplication. Existing
classifier CSV/SVG outputs remain byte-identical after shared report integration.

The four thin commands select baseline/A1/A2/A3 via `--config`; reusable code is
organized by preparation, training and evaluation under `src/method_a/`, with
tests/examples beside it. Input metadata is under `data/method_a/`, and evidence
under `results/method_a/`. Source attribution retains original PR paths/hashes.
Identical duplicated geometry/deduplication helpers are shared. Input binding now
copies the exact map required by variants; each training process receives a unique
scratch directory. Baseline and fixed-step training protocols remain distinct.

The local interface checks ran on macOS/Python 3.13 without imaging/model runtime
installation. No image bundle was created. Existing external image locations and
frozen membership remain unchanged. Raw-source preparation, training, GPU replay,
native Windows execution and live external artifact access were not rechecked;
the historical model-demo checks above were not rerun.
