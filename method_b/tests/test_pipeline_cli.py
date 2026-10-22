"""Check region-to-crop orchestration without loading model weights."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from method_b.pipeline.detect_and_classify import run_pipeline


class PipelineContracts(unittest.TestCase):
    def test_detector_and_classifier_share_region_and_local_boxes(self):
        """Pass the same image and generated boxes to the DINOv2 classifier."""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            output = base / "new_run"
            image = base / "region.png"
            payload = {"predictions": []}

            def execute(command, **kwargs):
                if command[2] == "method_b.classification.predict_categories":
                    (output / "classifications.json").write_text(json.dumps(payload))

            with patch("method_b.pipeline.detect_and_classify.subprocess.run",
                       side_effect=execute) as run:
                self.assertEqual(run_pipeline(image, base / "detector.pth", "a" * 64,
                                              base / "dino.pt", "dino", output,
                                              dino_source=base / "dinov2"), payload)
            detector, classifier = [call.args[0] for call in run.call_args_list]
            for command in (detector, classifier):
                self.assertEqual(command[command.index("--image") + 1], str(image.resolve()))
            self.assertEqual(classifier[classifier.index("--boxes") + 1],
                             str(output.resolve() / "detection/boxes.json"))
            self.assertIn("--dino-source", classifier)
            self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))

    def test_missing_dino_source_fails_before_creating_outputs(self):
        """Reject incomplete DINO configuration before invoking the detector."""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new_run"
            with patch("method_b.pipeline.detect_and_classify.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    run_pipeline("image.png", "detector.pth", "a" * 64,
                                 "dino.pt", "dino", output)
                run.assert_not_called()
            self.assertFalse(output.exists())

    def test_detector_failure_prevents_classification(self):
        """Propagate detector failure instead of producing a classification result."""
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new_run"
            with patch("method_b.pipeline.detect_and_classify.subprocess.run",
                       side_effect=subprocess.CalledProcessError(1, "detector")) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_pipeline("image.png", "detector.pth", "a" * 64,
                                 "swin.pt", "swin", output)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "classifications.json").exists())


if __name__ == "__main__":
    unittest.main()
