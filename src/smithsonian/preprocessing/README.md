# Preprocessing

Inputs are frozen specimen assignments, category maps, and externally stored images.
Outputs are image crops, COCO datasets, and metadata recording their source identities.

## Classifier crops

`full_tenengrad.py` coordinates read-only workers in `tenengrad_pilot.py`. Each specimen's
crop is compared across focal planes; the selected RGB crop is saved at native resolution.
The worker name is retained for command compatibility. `crops.py` prepares classifier
inputs from region-local `xyxy` detector boxes using context, white padding, and resizing.

```bash
python -m smithsonian.preprocessing.full_tenengrad --config config/preprocessing.example.json
```

The template needs the verified baseline crop manifest and frozen assignments. Image
preparation preserves split membership; it does not evaluate the reserved test set.

## Detector training crops

The current detector uses the established baseline's metadata and validation grid.
Retrieve that export as described in [data access](../../../docs/DATA.md).
Configure its path and a fresh output directory, then run:

```bash
python -m smithsonian.preprocessing.detector_crops.audit_plan --baseline external/detector_baseline --output outputs/detector_crops
python -m smithsonian.preprocessing.detector_crops.prepare_datasets --config config/detector_crops.example.json --stage plan
python -m smithsonian.preprocessing.detector_crops.export_crops --output outputs/detector_crops --slide-index 0
# Repeat export_crops for each index in outputs/detector_crops/plan/.
python -m smithsonian.preprocessing.detector_crops.prepare_datasets --config config/detector_crops.example.json --stage finalize
python scripts/prepare_ignore_regions.py --config config/ignore_regions.example.json
```

Planning retains the recorded 32-pixel target margin, 16-pixel reuse margin, native
512–1024-pixel windows, and 1024-pixel padded canvas. Export combines focal planes using
the Spring team's LoG focus rule. The ignore-region step runs the verified single-class
detector on training windows and preserves category-labeled boxes.

`detector/` retains baseline input/grid utilities needed to reconstruct the recorded
validation export. Its manifest-bound inputs must be retrieved intact; new category or
split definitions require a separately designed experiment. No old A1/A2/A3 training
commands are included in this submission.
