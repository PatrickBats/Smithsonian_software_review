"""Synthetic CPU forward/backward checks; no specimens, fitting runs or weights saved."""

import argparse


def main():
    """Exercise Swin and optional local DINO construction with synthetic RGB tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-source", help="Pinned local DINOv2 checkout; never downloads")
    args = parser.parse_args()
    import torch
    from torch import nn
    from torchvision.models import swin_t
    from smithsonian.models.dino import DinoClassifier, adaptable

    torch.set_num_threads(4)
    torch.manual_seed(549)
    images = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 27])
    swin = swin_t(weights=None)
    swin.head = nn.Linear(swin.head.in_features, 28)
    models = [("swin", swin)]
    if args.dino_source:
        dino = DinoClassifier(28, source=args.dino_source, pretrained=False)
        dino.configure(False)
        assert all(p.requires_grad == adaptable(name) for name, p in dino.named_parameters())
        models.append(("dino", dino))
    for name, model in models:
        model.train()
        logits = model(images)
        assert logits.shape == (2, 28) and torch.isfinite(logits).all()
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)
        print(f"{name}: finite 28-category forward/backward; no optimizer step or saved weights")


if __name__ == "__main__":
    main()
