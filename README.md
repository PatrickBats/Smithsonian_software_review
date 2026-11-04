# Fossil pollen detection and classification

## Introduction

This Rice D2K project helps Smithsonian researchers locate fossil pollen in microscope
images and identify its category. It compares two approaches: RF-DETR predicts locations
and categories together; a single-class detector supplies specimen crops to Swin-Tiny
or DINOv2 for classification.

**Students:** Patrick Batsell, Yun Ying Tsai, Yeonju Kim, Yunfan Bao, Alan Yang.

**Faculty mentor:** Arko Barman. **PhD mentor:** Nhi Le.

**Sponsors:** Ingrid Romero and Scott Wing, Smithsonian Institution.

The project builds on the Spring 2026 team's microscopy pipeline. Adapted code and
model sources are credited in [NOTICE.md](NOTICE.md).

## Repository structure

| Folder | Contents |
| --- | --- |
| `src/smithsonian/data/` | Category grouping and frozen specimen splits |
| `src/smithsonian/preprocessing/` | Focal selection, detector crops, and ignore regions |
| `src/smithsonian/models/` | Swin/DINO models and shared detector structures |
| `src/smithsonian/training/` | Classifier and joint RF-DETR training |
| `src/smithsonian/inference/` | Image detection, crop classification, combined demo |
| `src/smithsonian/evaluation/` | Matching, validation metrics, and saved-result tables |
| `scripts/` | Thin command-line entry points |
| `config/` | JSON settings and checkpoint identities |
| `notebooks/` | One end-to-end region demo |
| `tests/` | Input, geometry, loss, and configuration checks |
| `data/`, `examples/`, `results/` | Small maps, example manifests, and aggregate results |

Each component README describes its interfaces and commands.

## Installation

Use **Python 3.13**. The checked environment is Linux with CPU inference.

```bash
git clone https://github.com/PatrickBats/Smithsonian_software_review.git
cd Smithsonian_software_review
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

On Windows, activate with `.venv\Scripts\activate`. Raw-slide workflows may require
Linux and access to the source paths recorded in their manifests. For GPU training,
install matching CUDA builds of the same PyTorch/torchvision versions in a separate
environment; the supplied requirements select CPU wheels. Training is an explicit command.

## Data

Images and weights remain external. [Data setup](docs/DATA.md) gives retrieval locations,
checkpoint hashes, and required dataset layouts. The modeling task uses 28 categories.
[Category maps and split definitions](src/smithsonian/data/README.md) describe
the labels and specimen assignments.

Copy a [configuration template](config/README.md) to `config/local.<name>.json` and set
its paths. JSON paths resolve relative to that file; command-line paths resolve relative
to the working directory. Command-line options override JSON settings.

## Data preprocessing

Classifier crops use specimen-level focal selection. Joint RF-DETR training uses
focus-stacked, target-centered windows and ignore regions for additional candidate
specimens without category labels. [Preprocessing instructions](src/smithsonian/preprocessing/README.md)
explain both routes and the required inputs.

## Modeling and evaluation

[Training instructions](src/smithsonian/training/README.md) cover RF-DETR, Swin, and DINO.
[Evaluation instructions](src/smithsonian/evaluation/README.md) distinguish classification
accuracy, known-target detection recall, and end-to-end identification.

| Classifier | Separate-slide validation accuracy | Shared-slide validation accuracy |
| --- | ---: | ---: |
| Swin-Tiny | 42.94% | 70.61% |
| DINOv2 | 50.14% | 74.35% |

These classifier results use expert-located crops. The current joint detector located
243/274 known targets and correctly identified 85/274 at confidence 0.30. The combined
pipeline found 258/274; DINO identified 125 and Swin identified 93. These are separate
validation measurements with different input preparation and thresholds.

Regenerate the tables from the included result files:

```bash
python scripts/results.py --output outputs/results
```

## Usage

[notebooks/demo.ipynb](notebooks/demo.ipynb) demonstrates a real specimen, saved Swin/DINO
predictions, RF-DETR detection, extracted crops, and annotated classification outputs.
Follow [notebook setup](notebooks/README.md), then run:

```bash
jupyter lab notebooks/demo.ipynb
```

For individual image commands, see [inference](src/smithsonian/inference/README.md).
For a first run with artifact retrieval, see [the walkthrough](docs/QUICKSTART.md).

## Checks

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
ruff check src scripts tests
ruff format --check src scripts tests
```

[VALIDATION.md](VALIDATION.md) records completed checks and remaining environment limits.
