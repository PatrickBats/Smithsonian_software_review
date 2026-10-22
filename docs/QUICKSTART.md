# First-run walkthrough

Follow these steps from top to bottom. Step 1 works without the microscopy data.
Steps 2–3 use existing model weights and do not launch training.

## 1. Check the repository and view results

Use Python 3.13 on Linux. If `python3 --version` reports a different version,
select your Python 3.13 interpreter for the virtual environment.

```bash
git clone https://github.com/PatrickBats/Smithsonian_software_review.git
cd Smithsonian_software_review
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps -e .
python scripts/check_repository.py
python -m unittest discover -s tests -v
python -m unittest discover -s src/method_a/tests -v
python scripts/results.py --output outputs/first_comparison
```

**Expected:** the repository check passes, the two test suites report 3 and 27
passing tests (30 total), and `outputs/first_comparison/` contains:

| File | What to open it for |
| --- | --- |
| `classification.svg` | View the classifier accuracy chart in a browser |
| `classification.csv` | Swin/DINO accuracy and macro-F1 by validation split |
| `detection.csv` | Baseline and A1/A2/A3 detected and correctly identified target counts |
| `rgb_comparison.csv` | Swin/DINO comparison on identical detector crops |

These commands regenerate outputs from recorded aggregate metrics. They do not
rerun the models. See [the results guide](RESULTS.md) for inference replay commands.

For Method A, continue with the [Method A worked examples](../src/method_a/examples/README.md):
first the four-row CSV, then a 28-class detector on an authorized image region.

## 2. Run Swin on a real specimen

### Install the classifier environment

Keep the virtual environment from step 1 active:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

This is the smaller CPU classifier setup. The complete tested CPU environment is
recorded in `requirements-cpu-lock.txt`; imaging/detector dependencies are not
needed for this single-crop example.

### Retrieve the inputs

You need an authorized Rice account with project-allocation access and network/VPN
access to NOTS. Replace `YOUR_NETID` below. These commands copy files only; they do
not submit a cluster job. If you cannot access NOTS, ask the project team for an
authorized copy of these exact files, then continue with verification.

```bash
export RICE_ACCOUNT=YOUR_NETID
export SMITHSONIAN_ROOT="$PWD/external"
mkdir -p "$SMITHSONIAN_ROOT/runs/broad_swin_2026-10-08/separate_slides/ordinary_baseline"
scp "${RICE_ACCOUNT}@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/software_review_2026-10-20/classifier_inputs.tar.gz" "$SMITHSONIAN_ROOT/"
scp "${RICE_ACCOUNT}@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_swin_2026-10-08/separate_slides/ordinary_baseline/best.pt" "$SMITHSONIAN_ROOT/runs/broad_swin_2026-10-08/separate_slides/ordinary_baseline/"
```

The crop bundle is approximately 314 MB. Verify it against the published manifest
before extracting:

```bash
python - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
p = Path(os.environ['SMITHSONIAN_ROOT']) / 'classifier_inputs.tar.gz'
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
expected = json.loads(Path('config/input_bundle.json').read_text())['sha256']
assert h.hexdigest() == expected, 'Crop bundle checksum mismatch'
print('Crop bundle verified')
PYCODE
tar -xzf "$SMITHSONIAN_ROOT/classifier_inputs.tar.gz" -C "$SMITHSONIAN_ROOT"
export CROP_ROOT="$SMITHSONIAN_ROOT/classifier_inputs/dataset"
python scripts/artifacts.py --name swin_separate_slides --root "$SMITHSONIAN_ROOT" --destination weights/swin.pt
```

**Expected:** the artifact tool prints `"status": "verified"`. It refuses to
replace an existing destination. If you already copied the checkpoint, omit
`--destination weights/swin.pt` to verify the source without copying again.

### Select and verify the example

Use the first validation example listed in the repository. This selects the file
and checks its image hash, so no filename needs to be guessed:

```bash
export CROP="$(python - <<'PYCODE'
import hashlib, json, os
from pathlib import Path
row = json.loads(Path('examples/crops.json').read_text())[0]
p = Path(os.environ['CROP_ROOT']) / row['image']
assert row['split'] == 'val'
assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
print(p)
PYCODE
)"
python scripts/infer.py --model swin --checkpoint weights/swin.pt --image "$CROP" --output outputs/first_swin.json
```

**Expected:** one prediction is written to `outputs/first_swin.json`. In the tested
CPU environment this example predicts **Taxodium sp.**, with a score of about
**0.767**, matching its reference category. Small numeric differences can occur
across hardware and runtimes.

### Read the output

```bash
python - <<'PYCODE'
import json
from pathlib import Path
r = json.loads(Path('outputs/first_swin.json').read_text())
p = r['predictions'][0]
print('Predicted category:', p['category'])
print('Model score:', round(p['score'], 4))
print('Class probabilities:', len(p['probabilities']))
PYCODE
```

| JSON field | Meaning |
| --- | --- |
| `predictions[].category` | Model's highest-scoring category |
| `predictions[].score` | Uncalibrated model probability; not a guarantee of correctness |
| `predictions[].probabilities` | Scores in the order of the top-level `categories` list |
| `predictions[].box` | `null` for an already cropped specimen; xyxy coordinates when supplying boxes |
| `image_sha256`, `checkpoint_sha256` | Exact input and weight identities |

This command produces JSON, not an annotated image. Open the input PNG separately
to inspect the specimen. A successful run does not mean every prediction is correct.

## 3. Compare DINO on the same crop

Continue in the same shell so `CROP` and `SMITHSONIAN_ROOT` remain defined:

```bash
mkdir -p "$SMITHSONIAN_ROOT/runs/broad_dino_2026-10-08/separate_slides/backbone_finetune"
scp "${RICE_ACCOUNT}@nots.rice.edu:/rhf/allocations/dsci435/Smithsonian_F26/runs/broad_dino_2026-10-08/separate_slides/backbone_finetune/best.pt" "$SMITHSONIAN_ROOT/runs/broad_dino_2026-10-08/separate_slides/backbone_finetune/"
python scripts/artifacts.py --name dino_separate_slides --root "$SMITHSONIAN_ROOT" --destination weights/dino.pt
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
export DINO_SOURCE="$PWD/external/dinov2"
python scripts/preflight.py --artifact-root "$SMITHSONIAN_ROOT" --crops "$CROP_ROOT" --dino-source "$DINO_SOURCE"
python scripts/infer.py --model dino --checkpoint weights/dino.pt --dino-source "$DINO_SOURCE" --image "$CROP" --output outputs/first_dino.json
```

**Expected:** preflight passes and DINO writes one prediction. For this example,
DINO predicted **Platycarya platycarioides** with a score of about **0.694** in the
tested CPU environment—an incorrect prediction. DINO's better aggregate validation
accuracy does not mean it wins on every specimen. This example illustrates that
correctly running software and correctly identifying a specimen are different checks.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Permission denied or connection failure on NOTS | Verify your Rice account, network/VPN, and allocation access; obtain an authorized artifact bundle if needed |
| A path variable is empty | Re-run the `export` lines in the same shell; all commands assume the repository root |
| Output already exists | Choose a new output filename/directory; completed outputs are protected |
| Checksum mismatch | Re-copy the named artifact; keep the published fingerprint unchanged |
| Unknown checkpoint | Use a checkpoint in `config/checkpoints.json`; the demo deliberately checks its identity |
| DINO source is missing or has the wrong revision | Check out the exact commit above and set `DINO_SOURCE` |
| xFormers unavailable warning on CPU | The tested CPU demo works without xFormers |
| A prediction disagrees with the reference | Record it as a model error; inspect the image and labels rather than changing the reference to match the prediction |

Next: use the [workflow guide](WORKFLOWS.md) for detector integration and the
[results guide](RESULTS.md) for full validation replay. No step above uses the
reserved test specimens for inference.
