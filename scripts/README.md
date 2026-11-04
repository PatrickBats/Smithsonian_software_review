# Commands

These files only call shared functions under `src/smithsonian/`. Install the package
first and run commands from the repository root.

| Script | Purpose |
| --- | --- |
| `detect.py` | RF-DETR inference on an RGB region |
| `infer.py` | Swin/DINO inference on a crop or supplied boxes |
| `prepare_ignore_regions.py` | Add training-only ignore boxes |
| `train_detector.py` | Train the current joint detector |
| `evaluate_detector.py` | Evaluate the joint detector on its fixed validation grid |
| `results.py` | Regenerate tables and a chart from saved results |
| `artifacts.py`, `preflight.py` | Verify external inputs/checkpoints |
| `check_repository.py` | Check source syntax, documentation links, and submission contents |
| `classification/` | Compatibility commands for current classifier preparation/training |

Use `--help` for arguments. Package component READMEs give complete command examples.
