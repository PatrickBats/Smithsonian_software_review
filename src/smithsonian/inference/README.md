# Inference

Inputs are RGB images and verified checkpoints. Outputs retain source identities,
boxes, category scores, and effective settings. Use a new output path for each run.

## Run both stages

```bash
python -m smithsonian.inference.combined --config config/pipeline.example.json
```

This writes detection boxes and classifier predictions for the same region.

## Detect or classify one image

```bash
python scripts/detect.py --checkpoint external/weights/rfdetr\(2xl\)_best_plane.pth --sha256 3bec840a984efc5b56471f1a4c3222e445661dda61ef69ea18a38cff1978a246 --image external/region.png --classes 1 --threshold 0.5 --output outputs/detection
python scripts/infer.py --model swin --checkpoint weights/swin.pt --image external/region.png --boxes outputs/detection/boxes.json --output outputs/classification.json
```

For an existing crop, omit `--boxes`. Use `--model dino --dino-source external/dinov2`
with the DINO checkpoint. To use the joint detector, supply its verified checkpoint
and hash, `--classes 28`, and the recorded threshold `--threshold 0.3`.

`run_detection` returns region-local `xyxy` boxes. `run_inference` crops and classifies
those boxes. The region image is already prepared: these commands do not select focal
planes or perform whole-slide merging. [The notebook](../../../notebooks/README.md)
connects these functions and displays their outputs.

## Recorded full-slide classifier comparison

`whole_slide/compare_rgb_dino.py` replays the classifiers on saved detector predictions,
using the original NDPI sources, tile-wide focal selection, and fixed matching inputs.
It requires the frozen RGB comparison export and pinned Spring checkout described in
[data access](../../../docs/DATA.md).

```bash
python -m smithsonian.inference.whole_slide.compare_rgb_dino --config config/whole_slide.example.json --preflight
```

Remove `--preflight` only to run the full validation replay on CUDA. This command consumes
existing detector outputs; it does not retrain or rerun the detector. The region notebook
is the small end-to-end detection/classification demonstration.
