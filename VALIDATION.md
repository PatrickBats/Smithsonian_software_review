# Validation

Checked November 3, 2026 with Python 3.13 on Linux using CPU models.

## Completed

- Fresh virtual-environment installation from the single requirements file, editable
  package installation, and `pip check`.
- 17 unit tests covering configuration, category mapping, file identity, image geometry,
  focus fusion, matching denominators, ignore-loss gradients, and head initialization.
- All shared modules import; current workflow help commands work. Ruff checks and
  formatting pass. Function docstrings describe inputs and outputs.
- The notebook executed all cells using a real validation specimen and image region.
  Detector outputs were identical to the previous demo; all four Swin/DINO specimen
  and detected-crop probability comparisons had maximum absolute difference **0.0**
  (required tolerance: `1e-5`). Distributed notebook outputs are cleared.
- Synthetic Swin and DINO forward/backward checks produced finite gradients, without
  optimizer updates, downloaded initialization weights, or saved checkpoints.
- The migrated ignore loss, pollen-head initialization, and detector summary match
  the recorded numerical ASTs, excluding docstrings. Crop plans match on 1,605 synthetic
  specimens across 47 slides, including positions, padding, identities, and clipped boxes.
- Existing classifier and combined aggregate result files are byte-identical.

Machine-readable outcomes are in `provenance/validation.json`. Original imports and
source revisions remain in `provenance/`; current file hashes are in `provenance/files.json`.

## Limits

Full raw-slide preparation, full-slide replay, and GPU detector training were not rerun.
Those workflows require the external manifests, images, and runtime documented beside
the code. No scientific experiments, checkpoint changes, or reserved-test evaluation
were performed.

OpenCV 4.13's IPP Gaussian blur produced allocation-dependent pixel differences on this
host, including in the original environment. The focus-fusion equivalence test disables
IPP for both reference and streaming calculations and restores it afterward. Production
preprocessing is unchanged; exact full-slide pixel reproduction should be checked in
the original imaging runtime before regenerating scientific datasets.

The two existing software-review organization PRs were inspected. Their branches remain
intact. The current submission follows one shared package, includes the current joint
crop/ignore workflow, and excludes the older detector-variant runners.
