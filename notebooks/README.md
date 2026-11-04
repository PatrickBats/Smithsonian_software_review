# End-to-end demonstration

`demo.ipynb` calls shared package interfaces to validate inputs, display a real
specimen/reference, run Swin and DINO, detect a region, classify the detections and
render labeled images with a prediction table. All outputs are cleared in Git.

## Setup

Run these commands from the repository root with Python 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m ipykernel install --user --name smithsonian-review --display-name "Smithsonian review"
cp config/demo.example.json config/local.demo.json
```

Retrieve the crop bundle and classifier weights as described in
[the walkthrough](../docs/QUICKSTART.md) and [data access](../docs/DATA.md).
Clone the pinned DINOv2 source following the walkthrough. For the detector stage,
obtain the validation image `f1921738c5ed013195da4ed8.png` and the single-class
`checkpoint_best_total.pth` from the team's authorized artifact custodian. Exact
hashes are in [the region manifest](../examples/region.json). Retrieve these two
artifacts separately from the classifier crop bundle, using the exact checkpoint
listed in the manifest.

Set all locations in `config/local.demo.json`. Configuration paths resolve relative
to that file. Use the actual extracted crop directory containing `val/`, which may
vary with how the bundle was unpacked. Each required file is checked before any
model executes. A missing file fails with its path and retrieval guidance.

```bash
jupyter lab notebooks/demo.ipynb
```

Select **Smithsonian review**, then **Run All**. CPU is the default and is slower
than GPU inference. The notebook reads only fixed validation samples, uses saved
weights, and creates a unique child of the configured output directory. Outputs
include detection JSON, complete classifier probabilities, input hashes and CSV.

For automation, set `SMITHSONIAN_DEMO_CONFIG` to your local JSON path and execute
with `nbclient`; save the executed copy outside the repository. The notebook
demonstrates inference on one prepared region. See the
[evaluation guide](../src/smithsonian/evaluation/README.md) for validation metrics
and the [preprocessing guide](../src/smithsonian/preprocessing/README.md) for NDPI preparation.
