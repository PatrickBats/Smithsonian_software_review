# Saved model interfaces

`load_classifier(kind, path, dino_source, device)` constructs a frozen classifier. `classify(model, images)` returns probability vectors in checkpoint label order. `DinoClassifier` supplies the recorded partial-backbone training architecture.

**Inputs:** Trusted verified checkpoints, RGB crops, optional pinned DINO source.

**Outputs:** Torch models, ordered labels and probabilities. Model loading occurs
when `load_classifier` is called.


See the [root README](../../../README.md) for installation and workflow entry points.
