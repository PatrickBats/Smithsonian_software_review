# Training

The entry points train models on frozen training specimens and use the documented
validation protocol. They never run reserved-test inference. Run these commands only
when training is intended; the notebook uses saved checkpoints.

## Swin and DINO

Inputs: verified classifier crops, assignment CSV, source category map, and broad-category
mapping. Outputs: checkpoint, validation predictions, training history, and effective settings.

```bash
python -m smithsonian.training.train_swin --config config/training.example.json
python -m smithsonian.training.train_dino_finetune --config config/training.example.json --dino-source external/dinov2 --output ../dino_training
```

Use separate fresh output directories. Optional `--specimen-split` selects the frozen
shared-slide assignment; omission uses separate slides. `common.py` provides dataset
loading, augmentations, validation, and metrics. Hyperparameters are in JSON and can
be overridden explicitly on the command line.

## Joint RF-DETR

Inputs: the target-centered COCO dataset with ignore boxes and the verified Spring
single-class initialization checkpoint. Prepare these using the
[preprocessing commands](../preprocessing/README.md).

```bash
python scripts/train_detector.py --config config/detector_train.example.json
```

This command requires CUDA. The current recipe uses RF-DETR 2XL, 28 category outputs,
1000-pixel model inputs, and 2,400 optimizer updates. The one-class pollen weights seed
each category output. `detector/loss.py` excludes ignored boxes from target matching
and masks classification loss for unmatched predictions overlapping ignored boxes.
Matched labeled objects retain their original classification and box supervision.

`--resume` requires a compatible run record and resume checkpoint. Runtime state and
source/input identities are recorded beside outputs. There is no default elapsed-time
cutoff; optional segment duration is used only when explicitly requested.
