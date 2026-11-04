# Saved model interfaces

`load_classifier(kind, path, dino_source, device)` constructs a frozen classifier. `classify(model, images)` returns probability vectors in checkpoint label order. `DinoClassifier` supplies the recorded partial-backbone training architecture.

**Inputs:** Trusted verified checkpoints, RGB crops, optional pinned DINO source.

**Outputs:** Torch models, ordered labels and probabilities; importing the package starts no downloads.


See the [root README](../../../README.md) for installation and workflow entry points.
