# Method A worked examples

Run these commands from the **repository root**. Example 1 works with Python
alone. Example 2 runs a trained model and needs external scientific inputs.
Shell blocks using `export` and heredocs require Bash (Linux, macOS or WSL);
they are not PowerShell commands. The model runtime must also support the chosen OS.

## 1. Inspect all four recorded results

**What the script does:** `scripts/method_a/export_results.py` reads the packaged aggregate JSON
and writes the four-row Method A comparison table.
No images, weights, GPU or training are needed.

**Input:** the baseline and A1/A2/A3 entries in
[recorded_comparison.json](../../../results/method_a/recorded_comparison.json).
For example, its baseline entry reports `detected: 24` and
`end_to_end_correct: 20` out of `validation_known_references: 274`.

**Run:** install the local package, then choose an output directory that does not
already exist. Installing this package does not install GPU/model dependencies.

```bash
python -m pip install --no-deps -e .
python scripts/method_a/export_results.py --output outputs/method_a_comparison
```

**Output:** `outputs/method_a_comparison/detection.csv` must contain exactly:

```csv
variant,known_targets,detected,correct_end_to_end
baseline,274,24,20
A1,274,84,46
A2,274,101,46
A3,274,99,55
```

The same expected file is checked in as [expected_detection.csv](expected_detection.csv).
Compare parsed CSV rows if your platform writes different newline characters.
Reading this table does not replay inference or prove that a new training run
will reproduce its counts. The shared `scripts/results.py` command can separately
export all team comparisons; this example produces only Method A's `detection.csv`.
See the [result README](../../../results/method_a/README.md)
for denominators and comparison limits.

## 2. Run a trained model on one image region

**What the script does:** `scripts/detect.py` runs RF-DETR on one RGB image region
and returns boxes with category IDs. Method A already predicts categories, so
this example does not add a Swin/DINO classification stage.

**Inputs:** an authorized PNG/JPEG region, the chosen trained checkpoint, and its
published hash. A 1024 × 1024 region matches the native validation tile size.
The repository does not include a public Method A specimen image; obtain one
from the team's validation images using authorized access. The example path
below must be replaced with a real local image. Do not select a reserved test image.

After obtaining the environment and A3 checkpoint described in the
[data README](../../../data/method_a/README.md), run:

```bash
export REGION="$PWD/external/method_a_example/region.png"
export MODEL_NAME=method_a_a3
python scripts/artifacts.py --name "$MODEL_NAME" --root "$SMITHSONIAN_ROOT" --destination "weights/$MODEL_NAME.pth"
export MODEL_SHA="$(python - <<'PY'
import json, os
from pathlib import Path
print(json.loads(Path(os.path.join('config', 'checkpoints.json')).read_text())[os.environ['MODEL_NAME']]['sha256'])
PY
)"
python scripts/detect.py --checkpoint "weights/$MODEL_NAME.pth" --sha256 "$MODEL_SHA" --image "$REGION" --classes 28 --threshold 0.5 --device cpu --output "outputs/${MODEL_NAME}_region"
```

Use an isolated detector environment with RF-DETR 1.6.4, RF-DETR+ 1.0.2 and their
runtime dependencies; [requirements-detector.txt](../../../requirements-detector.txt)
records the detector package versions. This model example was not rerun while
writing these instructions. If the checkpoint was already copied in the data
guide, verify it without `--destination` rather than copying it a second time.

Change `MODEL_NAME` to `method_a_baseline`, `method_a_a1` or `method_a_a2` to use
another model. Use a new output directory each time. The original baseline and
three new models are distinct checkpoints of the same 28-class architecture.

**Outputs:**

| File | Contents |
| --- | --- |
| `boxes.json` | A list of region-local `[left, top, right, bottom]` pixel boxes |
| `detections.json` | Boxes, scores, zero-based category IDs, checkpoint hash and excluded extra-slot count |

An **illustrative output format**, not a measured prediction, is:

```json
{
  "predictions": [{"box": [100.0, 120.0, 220.0, 240.0], "score": 0.8, "class_id": 0}],
  "excluded_extra_slot": 0,
  "checkpoint_sha256": "<the selected model's verified SHA-256>",
  "scope": "single RGB region; no whole-slide deduplication"
}
```

The actual number of boxes may be zero or more. Scores are not guarantees of
correctness. To name an ID, use the original model mapping included as
[data/method_a/broad_category_map.json](../../../data/method_a/broad_category_map.json);
do not use the original 32-category CSV as the model's index order. This CLI writes JSON, not a box-overlay image.

## 3. Replay the full known-reference evaluation

This requires an allocated CUDA GPU, all 1,432 linked validation tiles and the
completed run's checkpoint/status files. Follow the
[unified evaluation instructions](../../../scripts/method_a/README.md#replay-evaluation)
and choose `--config baseline`, `A1`, `A2` or `A3`.
The single-region example does not reproduce the 274-target benchmark: that
evaluation additionally merges overlapping-tile detections and matches known boxes.
