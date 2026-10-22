# Method A: find and identify pollen

Method A predicts both specimen boxes and their categories with RF-DETR 2XL.
It has **four data configurations of one 28-class architecture**, selected with
`--config baseline`, `A1`, `A2` or `A3`. They are not four model architectures.

An ROI is an expert-selected rectangular image region. A tile is a square cut
from that region; a context crop is a window placed around a known specimen.
Focus fusion combines sharp pixels from the source's focal planes.

## Choose a configuration

| Configuration | Training input | Images | Known targets | Located / 274 | Located + correct category / 274 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | All original ROI-grid tiles | 5,152 | 1,061 | 24 | 20 |
| A1 | Original tiles containing a retained YES box | 884 | 1,061 | 84 | 46 |
| A2 | Context crops anchored on originally covered train targets | 918 | 1,063 | 101 | 46 |
| A3 | Context crops anchored on all assigned train targets | 1,423 | 1,605 | 99 | 55 |

A2 retains two additional known labels incidentally inside its windows. All four
use the same 1,432 validation tiles covering 274 known specimens, at confidence
and matching IoU 0.50. The baseline used different training and checkpoint selection;
this is not a four-way data-only ablation. Test remains reserved.

## Start here

| Goal | Read | Input → output |
| --- | --- | --- |
| Understand data and split | [Data README](../../data/method_a/README.md) | Existing inputs and frozen membership → documented model inputs |
| Run preparation, training or full validation | [Command guide](../../scripts/method_a/README.md) | Existing scientific inputs → selected configuration's artifacts |
| Try commands and inspect output formats | [Worked examples](examples/README.md) | Saved metrics → CSV; supplied RGB region + model → predictions |
| Interpret recorded evidence | [Results](../../results/method_a/README.md) | Validation aggregates → four-configuration comparison and limits |
| Check the software | [Tests](tests/README.md) | Code, saved aggregates and temporary fixtures → pass/fail |

From the repository root, install the local package once:

```bash
python -m pip install --no-deps -e .
python scripts/method_a/export_results.py --output outputs/method_a_comparison
```

This example needs no images, model weights or GPU. It reads recorded metrics
and writes `detection.csv` with baseline/A1/A2/A3; it does not run a model.
Model and source-processing commands require the runtime in the command guide.

## Modules and reusable functions

| Location | Responsibility | Examples |
| --- | --- | --- |
| [configurations.py](configurations.py) | Names, artifact IDs and configuration identity | `resolve_configuration()`, `resolve_variant()` |
| [preparation/](preparation) | Bind inputs, plan grid/context windows, export images and build COCO datasets | `variants.plan_crops()`, `variants.make_dataset()` |
| [training/](training) | Historical grid training or the matched fixed-step protocol | `commands.main()` |
| [evaluation/](evaluation) | Model replay, boundary-aware duplicate removal and known-target matching | `known_references.summarize()`, `deduplication.deduplicate()` |
| [imaging/](imaging) | NDPA geometry and LoG focal fusion | `geometry.geometry()`, `focus_fusion.fused_tile()` |
| [integrity.py](integrity.py), [storage.py](storage.py) | Hashes, configured roots and bounded metadata writes | `sha256_file()`, `write()` |
| [results.py](results.py), [reporting.py](reporting.py) | Export saved comparisons; audit newly completed experiment runs | `write_detection_csv()`, `reporting.main()` |

The four thin entry points in `scripts/method_a/` call these modules. Imports do
not parse arguments, read project data or launch work; optional imaging/model
libraries are needed only by the implementation being used. The command help and
recorded-table export need standard-library Python only.

Example function call:

```python
from method_a.configurations import resolve_configuration, resolve_variant

print(resolve_configuration("a3"))
print(resolve_variant("A3"))
```

Output:

```text
A3
m2_retile_all
```

`m1_positive_grid`, `m2_retile_matched`, `m2_retile_all` and external
`method_a_focus_stack_v1` paths are frozen artifact IDs. They remain unchanged so
existing manifests, images and checkpoints resolve. Code is organized by function,
not by old semester or experiment-folder names.

## Sources and platform scope

The implementation and recorded results derive from Yunfan Bao's working-repository
[PR #5](https://github.com/PatrickBats/Smithsonian_fossil_F26/pull/5) and
[PR #7](https://github.com/PatrickBats/Smithsonian_fossil_F26/pull/7).
[Source fingerprints](../../provenance/import_manifest.json) retain original
paths and versions. Spring fusion/deduplication attribution is in
[NOTICE](../../NOTICE.md).

Python paths use `os.path.join` or pathlib; recorded external source identities
retain their required format. Full historical workflows also need POSIX ownership
checks, symlinks, original source mounts, CUDA and Bash. Local interface checks
ran on macOS/Python 3.13; native Windows and GPU replay were not checked here.
