"""DINOv2 backbone and classification head used by the recorded adaptation runs.

Uses Meta's DINOv2 implementation (Oquab et al.); see NOTICE.md. Construction
loads the explicit local source. Importing this module never downloads weights.
"""

import hashlib
import os
import torch
from torch import nn

DINO_SOURCE = os.environ.get("DINO_SOURCE", "./external/dinov2")


def adaptable(name):
    """Identify parameters updated during partial DINO backbone fine-tuning.

    Args:
        name: Fully qualified model parameter name.

    Returns:
        True for the head, final two blocks, or final normalization.
    """
    return name.startswith(
        ("head.", "backbone.blocks.10.", "backbone.blocks.11.", "backbone.norm.")
    )


def frozen_digest(model):
    """Hash parameters that must stay frozen during DINO adaptation.

    Args:
        model: DINO classifier whose parameters are read without modification.

    Returns:
        SHA-256 over parameter names and tensor bytes.
    """
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if not adaptable(n):
            h.update(n.encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class DinoClassifier(nn.Module):
    def __init__(self, num_classes=32, source=None, pretrained=True):
        """Construct the pretrained backbone and category prediction head.

        Args:
            num_classes: Number of output categories.
            source: Local pinned DINOv2 checkout.
            pretrained: Load pretrained backbone weights when True.
        """
        super().__init__()
        self.backbone = torch.hub.load(
            source or DINO_SOURCE, "dinov2_vitb14", source="local", pretrained=pretrained
        )
        if len(self.backbone.blocks) != 12:
            raise ValueError("Expected 12 DINO blocks")
        self.head = nn.Linear(1536, num_classes)
        self.configure(True)

    def configure(self, head_only):
        """Choose head-only training or partial backbone adaptation.

        Args:
            head_only: True to train only the classifier head.

        Returns:
            None; updates parameter requires_grad flags in place.
        """
        for n, p in self.named_parameters():
            p.requires_grad_(n.startswith("head.") or (not head_only and adaptable(n)))

    def train(self, mode=True):
        """Set training mode while keeping frozen backbone layers in evaluation mode.

        Args:
            mode: Whether to enable training for the adaptable layers.

        Returns:
            This classifier instance.
        """
        super().train(mode)
        self.backbone.eval()
        if mode:
            self.backbone.blocks[10].train()
            self.backbone.blocks[11].train()
            self.head.train()
        return self

    def forward(self, x):
        """Compute category logits from normalized specimen tensors.

        Args:
            x: Float tensor shaped (batch, 3, 224, 224).

        Returns:
            Float logits shaped (batch, number of categories).
        """
        features = self.backbone.forward_features(x)
        return self.head(
            torch.cat([features["x_norm_clstoken"], features["x_norm_patchtokens"].mean(1)], dim=1)
        )
