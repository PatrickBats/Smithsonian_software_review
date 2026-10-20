# Fossil pollen detection and classification

Two workflows identify fossil palynomorphs in microscope images: RF-DETR detects
and classifies specimens directly; a detector plus Swin-Tiny or DINO classifies
individual crops. The software includes data preparation, fixed-split training,
validation, and small inference demonstrations.

## Install

Use Python 3.13 on Linux. After cloning the repository or extracting the ZIP,
run commands from its root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/check_repository.py
python -m unittest discover -s tests -v
python -m unittest discover -s scripts/method_a/data_variants/tests -v
```

The CPU setup supports the classifier demo. For GPU training install the matching
CUDA torch/torchvision wheels for your machine instead of the CPU wheels.
Install `requirements-imaging.txt` for slide extraction, or
`requirements-detector.txt` for RF-DETR and the full combined pipeline. Keep the
detector runtime separate if it conflicts with another project's environment.
`requirements-cpu-lock.txt` records the complete resolved CPU review environment;
install it to reproduce that environment exactly.

## Data and weights

Images and model weights are external. Follow [data setup](docs/DATA.md) to obtain
access and configure local paths. `config/environment.example.sh` contains the
path settings; `config/checkpoints.json` identifies the exact classifier weights.
A Git checkout alone does not contain the microscope dataset.

```bash
python scripts/artifacts.py --name swin_separate_slides --root "$SMITHSONIAN_ROOT" --destination weights/swin.pt
python scripts/artifacts.py --name dino_separate_slides --root "$SMITHSONIAN_ROOT" --destination weights/dino.pt
```

Each copy is checksum-verified and refuses to replace an existing file. Original
checkpoint category order is preserved. Display-name changes do not reorder classes.

## Try a classifier

Choose a validation crop using `examples/crops.json`; set `CROP` to its local file.

```bash
python scripts/infer.py --model swin --checkpoint weights/swin.pt --image "$CROP" --output outputs/swin_example.json
python scripts/infer.py --model dino --checkpoint weights/dino.pt --dino-source "$DINO_SOURCE" --image "$CROP" --output outputs/dino_example.json
```

Outputs contain category names, scores, probabilities, and input/checkpoint hashes.
Scores are uncalibrated model probabilities. These examples run on CPU by default;
use `--device cuda:0` on an allocated GPU. They do not train or evaluate the test set.

## Workflows

- [Preparation, training, detection, and combined inference](docs/WORKFLOWS.md)
- [Scientific protocol and interpretation](docs/METHODS.md)
- [Validation record](VALIDATION.md)
- [Source attribution](NOTICE.md)

Regenerate compact comparison tables and a vector figure without data or a GPU:

```bash
python scripts/results.py --output outputs/comparison
```

This reads saved aggregate results; it does not rerun inference. The workflow guide
separately describes inference replay and retraining.

## Layout

| Directory | Contents |
| --- | --- |
| `scripts/` | Executable workflows and reusable model/data helpers |
| `config/` | Example paths, pipeline settings, checkpoint fingerprints |
| `data/` | Fixed fine and broad category maps |
| `examples/` | Small validation-example manifest; images retrieved separately |
| `results/`, `reports/` | Aggregate metrics needed to reproduce comparison outputs |
| `tests/` | Input, geometry, matching, and packaging checks |
| `provenance/` | Imported source fingerprints and final file checksums |

Outputs, images, weights, environments, and external source checkouts are ignored
by Git. Use a new output location for each experiment. This submission contains
no meeting records or project-planning archive.
