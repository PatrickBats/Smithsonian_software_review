# Configuration

Copy the needed template to `local.<name>.json` and edit its paths. Local configurations
are ignored by Git. Relative paths in JSON resolve against the JSON file's directory;
explicit command-line paths resolve against the working directory. CLI values take priority.
Unknown keys and malformed inputs fail with an error.

| Template | Command/workflow |
| --- | --- |
| `pipeline.example.json` | One-command detection and classification |
| `demo.example.json` | End-to-end region notebook |
| `inference.example.json` | Crop classifier |
| `preprocessing.example.json` | Classifier focal selection |
| `training.example.json` | Swin or DINO classifier training |
| `detector_crops.example.json` | Detector crop planning/finalization |
| `ignore_regions.example.json` | Training-only ignore proposals |
| `detector_train.example.json` | Current joint RF-DETR training |
| `detector_evaluation.example.json` | Fixed-grid joint validation |
| `whole_slide.example.json` | Paired classifier replay on frozen detections |
| `evaluation.example.json` | Regenerate aggregate result tables |

`checkpoints.json` records checkpoint paths and SHA-256 identities. `input_bundle.json`
identifies the external classifier crop archive. `training.defaults.json` supplies
classifier defaults. New runs save effective settings and source/input identities.
