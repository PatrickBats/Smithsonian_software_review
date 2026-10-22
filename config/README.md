# Checkpoint and path configuration

This folder stores configuration and fingerprints, not model weights.

| File | Purpose |
| --- | --- |
| [checkpoints.json](checkpoints.json) | Named trained models: relative artifact path, SHA-256 and architecture |
| [environment.example.sh](environment.example.sh) | Example input/output roots; replace them with your authorized locations |
| [input_bundle.json](input_bundle.json) | Fingerprint for the external classifier crop bundle |
| [pipeline.example.json](pipeline.example.json) | Existing detector-to-classifier pipeline configuration |

## Method A checkpoint names

`method_a_baseline`, `method_a_a1`, `method_a_a2` and `method_a_a3` select the four
trained 28-class RF-DETR 2XL checkpoints. The model resolution is 1000 pixels.
The baseline entry refers to its selected historical checkpoint; the three new
entries refer to final EMA weights. Their paths and checksums come from the
recorded experiment evidence, not a new model run.

For example, the A3 entry has `architecture: "rfdetr_2xl"`, `categories: 28`,
`data_variant: "A3"` and a `relative_path` ending in
`m2_retile_all/training/checkpoint_final_ema.pth`.

Given an authorized artifact root containing that relative path:

```bash
python scripts/artifacts.py --name method_a_a3 --root "$SMITHSONIAN_ROOT" --destination weights/method_a_a3.pth
```

The output reports `status`, `name` and `sha256`. An unknown name, changed hash
or existing copy destination stops the command. This tool verifies/copies local
or mounted files; it is not a downloader.

The [Method A data README](../data/method_a/README.md) describes input layouts and
external access. The [worked example](../src/method_a/examples/README.md) shows how
to pass the selected checkpoint and hash to `scripts/detect.py --classes 28`.
Existing Swin/DINO names continue to select their classifier checkpoints.
