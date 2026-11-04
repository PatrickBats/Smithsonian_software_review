"""Load frozen Swin/DINO classifiers and predict RGB crop probabilities.

Swin: torchvision (Liu et al.); DINOv2: Meta (Oquab et al.). See NOTICE.md.
"""

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision.models import swin_t
from torchvision.transforms import functional as TF


def load_classifier(kind, path, dino_source, device):
    """Construct a saved classifier without downloading pretrained weights.

    Args:
        kind: Model family, either swin or dino.
        path: Trusted project checkpoint file containing model and categories.
        dino_source: Pinned local DINOv2 source checkout; unused for Swin.
        device: Torch device such as cpu or cuda:0.

    Returns:
        Evaluation-mode model and the ordered list of 28 category names.

    Raises:
        ValueError: Checkpoint categories are duplicated or have the wrong count.
        RuntimeError: Model tensors do not match the architecture."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = ckpt["categories"]
    if len(labels) != 28 or len(set(labels)) != 28:
        raise ValueError("Expected unique broad28 checkpoint labels")
    if kind == "swin":
        model = swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(labels))
    else:

        class Dino(nn.Module):
            def __init__(self):
                """Construct the pretrained backbone and category prediction head."""
                super().__init__()
                self.backbone = torch.hub.load(
                    str(dino_source), "dinov2_vitb14", source="local", pretrained=False
                )
                self.head = nn.Linear(1536, len(labels))

            def forward(self, x):
                """Compute category logits from normalized specimen tensors.

                Args:
                    x: Float tensor shaped (batch, 3, 224, 224).

                Returns:
                    Float logits shaped (batch, number of categories).
                """
                f = self.backbone.forward_features(x)
                return self.head(
                    torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], dim=1)
                )

        model = Dino()
    model.load_state_dict(ckpt["model"], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    return model, labels


def classify(model, images):
    """Predict category probabilities with the frozen preprocessing contract.

    Args:
        model: Evaluation-mode torch classifier on its intended device.
        images: Sequence of PIL RGB images. Each is square-padded in white,
            resized to 224-by-224 and normalized with ImageNet channel statistics.

    Returns:
        List of probability vectors, one per input, in checkpoint category order.

    Raises:
        FloatingPointError: Model produces nonfinite logits."""
    device = next(model.parameters()).device
    out = []
    for start in range(0, len(images), 32):
        xx = []
        for im in images[start : start + 32]:
            im = ImageOps.pad(im, (max(im.size), max(im.size)), color="white").resize(
                (224, 224), Image.Resampling.BICUBIC
            )
            xx.append(TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        with torch.inference_mode():
            logits = model(torch.stack(xx).to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite logits")
        out.extend(logits.softmax(-1).cpu().tolist())
    return out
