"""Regression checks for the shared RGB classifier inference module."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from PIL import Image
    from method_b.classification.predict_categories import classify
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install requirements.txt for model-runtime tests")
class RuntimeContracts(unittest.TestCase):
    def test_rgb_normalization_and_output_order_across_batches(self):
        """Retain RGB channel order and all predictions across a 32-image boundary."""

        class Recorder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.batches = []

            def forward(self, images):
                self.batches.append(images.detach().clone())
                return images.mean((2, 3))

        model = Recorder().eval()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        images = [Image.new("RGB", (224, 224), colors[i % 3]) for i in range(35)]
        probabilities = classify(model, images)
        self.assertEqual([len(x) for x in model.batches], [32, 3])
        self.assertEqual(len(probabilities), 35)
        red = torch.tensor([(1 - .485) / .229, - .456 / .224, - .406 / .225])
        torch.testing.assert_close(model.batches[0][0, :, 100, 100], red)
        self.assertEqual([max(range(3), key=p.__getitem__) for p in probabilities],
                         [i % 3 for i in range(35)])
        self.assertFalse(any(x.requires_grad for x in model.batches))

    def test_nonfinite_logits_stop_inference(self):
        """Reject numerical failures instead of serializing invalid scores."""

        class Broken(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(self, images):
                return torch.full((len(images), 28), float("nan"))

        with self.assertRaises(FloatingPointError):
            classify(Broken(), [Image.new("RGB", (224, 224))])

    def test_empty_detections_return_no_classifications(self):
        """Allow detector regions with no boxes without invoking a forward pass."""
        self.assertEqual(classify(torch.nn.Linear(3, 28), []), [])


if __name__ == "__main__":
    unittest.main()
