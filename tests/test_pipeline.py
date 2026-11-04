"""Scientific contracts for crop handling, focus selection, and ignore supervision."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "PIL", "cv2"))
HAS_DETECTOR = HAS_MODELS and importlib.util.find_spec("rfdetr") is not None


@unittest.skipUnless(HAS_MODELS, "Install requirements.txt for image/model checks")
class ImageContracts(unittest.TestCase):
    def test_crop_color_padding_and_geometry(self):
        import numpy as np
        from PIL import Image
        from smithsonian.preprocessing.crops import crop_input

        source = Image.new("RGB", (20, 20), (255, 0, 0))
        crop = np.asarray(crop_input(source, (-10, -10, 10, 10), margin=0))
        self.assertEqual(crop.shape, (224, 224, 3))
        self.assertTrue((crop[10, 10] == [255, 255, 255]).all())
        self.assertTrue((crop[-10, -10] == [255, 0, 0]).all())
        for box in ([0, 0, 0, 10], [0, 0, float("nan"), 10]):
            with self.assertRaises(ValueError):
                crop_input(source, box)

    def test_focus_stack_matches_batch_reference(self):
        import cv2
        import numpy as np
        from smithsonian.preprocessing.detector_crops.focus_helpers import fused_tile

        # IPP Gaussian blur can vary with allocation on this host, including in
        # the pre-refactor environment. Compare algorithms with that backend off.
        original_ipp = cv2.ipp.useIPP()
        original_threads = cv2.getNumThreads()
        self.addCleanup(cv2.ipp.setUseIPP, original_ipp)
        self.addCleanup(cv2.setNumThreads, original_threads)
        cv2.ipp.setUseIPP(False)
        cv2.setNumThreads(1)
        planes = np.random.default_rng(14).integers(0, 256, (3, 24, 24, 3), dtype=np.uint8)
        responses = np.stack(
            [
                abs(
                    cv2.Laplacian(
                        cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (5, 5), 0),
                        cv2.CV_64F,
                        ksize=5,
                    )
                )
                for x in planes
            ]
        )
        selected = responses.argmax(0)
        yy, xx = np.indices(selected.shape)
        expected = planes[selected, yy, xx]
        actual, scores, counts = fused_tile(iter(planes), 24, 0, planes[0])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(sum(counts), 24 * 24)
        # Ties retain the earlier plane rather than changing focal identity.
        _, _, counts = fused_tile([planes[0], planes[0]], 24, 0, planes[0])
        self.assertEqual(counts, [24 * 24, 0])

    def test_matching_is_one_to_one_and_misses_stay_in_denominator(self):
        from smithsonian.evaluation.matching import match_boxes, metric

        predictions = [{"box_slide": [0, 0, 10, 10]}, {"box_slide": [0, 0, 10, 10]}]
        references = [{"box_slide": [0, 0, 10, 10]}, {"box_slide": [30, 30, 40, 40]}]
        self.assertEqual(len(match_boxes(predictions, references)), 1)
        result = metric(
            [{"detected": True, "correct": True}, {"detected": False, "correct": False}]
        )
        self.assertEqual(result["classification_accuracy_on_matched"], 1)
        self.assertEqual(result["end_to_end_known_specimen_success"], 0.5)


@unittest.skipUnless(HAS_DETECTOR, "Install requirements.txt for RF-DETR loss checks")
class IgnoreContracts(unittest.TestCase):
    def criterion(self):
        import torch
        from rfdetr.models.lwdetr import SetCriterion

        class FixedMatcher(torch.nn.Module):
            def forward(self, outputs, targets, **kwargs):
                if targets[0]["labels"].tolist() != [0]:
                    raise AssertionError("Ignore labels reached target matching")
                return [(torch.tensor([0]), torch.tensor([0]))]

        return SetCriterion(28, FixedMatcher(), {}, 0.25, ["labels", "boxes"], ia_bce_loss=True)

    def test_ignore_masks_only_unmatched_classification_gradients(self):
        import torch
        from smithsonian.training.detector.loss import patch_criterion_for_ignore

        criterion = patch_criterion_for_ignore(self.criterion(), ignore_label=28)
        logits = torch.zeros(1, 3, 29, requires_grad=True)
        boxes = torch.tensor(
            [[[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1]]], requires_grad=True
        )
        targets = [{"labels": torch.tensor([0, 28]), "boxes": boxes[0, :2].detach().clone()}]
        losses = criterion({"pred_logits": logits, "pred_boxes": boxes}, targets)
        losses["loss_ce"].backward()
        self.assertGreater(logits.grad[0, 0].abs().sum(), 0)
        self.assertEqual(logits.grad[0, 1].abs().sum(), 0)
        self.assertGreater(logits.grad[0, 2].abs().sum(), 0)
        self.assertIn("loss_bbox", losses)
        self.assertIsNone(criterion._ignore_boxes)

    def test_no_ignore_matches_original_loss(self):
        import torch
        from smithsonian.training.detector.loss import patch_criterion_for_ignore

        original, masked = self.criterion(), patch_criterion_for_ignore(self.criterion(), 28)
        outputs = {
            "pred_logits": torch.full((1, 2, 29), 0.2),
            "pred_boxes": torch.tensor([[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]]),
        }
        targets = [{"labels": torch.tensor([0]), "boxes": outputs["pred_boxes"][0, :1]}]
        expected, actual = original(outputs, targets), masked(outputs, targets)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-6)

    def test_pollen_weights_seed_all_real_categories(self):
        import torch
        from smithsonian.training.detector.loss import seed_class_head_from_one_class

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.class_embed = torch.nn.Linear(4, 29)
                self.two_stage = False

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "init.pth"
            weight = torch.arange(8, dtype=torch.float32).reshape(2, 4)
            torch.save(
                {
                    "model": {
                        "class_embed.weight": weight,
                        "class_embed.bias": torch.tensor([3.0, 9.0]),
                    }
                },
                path,
            )
            model = Model()
            unused = model.class_embed.weight[28].detach().clone()
            seed_class_head_from_one_class(model, path, pollen=0)
            torch.testing.assert_close(model.class_embed.weight[:28], weight[0].expand(28, 4))
            torch.testing.assert_close(model.class_embed.weight[28], unused)


class RecordedResults(unittest.TestCase):
    def test_current_detector_counts_are_consistent(self):
        from smithsonian.configuration.paths import repository_root

        row = json.loads((repository_root() / "results/joint_detector.json").read_text())
        self.assertEqual(
            (row["known_specimens"], row["detected"], row["end_to_end_correct"]), (274, 243, 85)
        )
        self.assertEqual(sum(x["support"] for x in row["per_class"]), 274)
        self.assertAlmostEqual(row["end_to_end_known_specimen_success"], 85 / 274)


def main():
    """Run image, matching, and detector supervision contracts."""
    unittest.main()


if __name__ == "__main__":
    main()
