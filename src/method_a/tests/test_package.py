"""Check the public four-configuration interface without scientific data or a GPU."""

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from method_a.configurations import CONFIGURATIONS
from method_a.evaluation import commands as evaluate
from method_a.preparation import commands as prepare
from method_a.training import commands as train

REPOSITORY = Path(__file__).resolve().parents[3]


class MethodAPackageTests(unittest.TestCase):
    def test_input_binding_copies_the_exact_mapping_required_by_variants(self):
        from method_a.preparation import inputs

        mapping = REPOSITORY / "data" / "fine_to_broad.csv"
        with mapping.open(newline="") as stream:
            categories = [row["fine_category"] for row in csv.DictReader(stream)]
        with tempfile.TemporaryDirectory() as cwd:
            assignment = Path(cwd) / "assignment.csv"
            records = []
            for split, count, slides in (
                ("train", 1605, 47),
                ("val", 347, 14),
                ("test", 331, 14),
            ):
                for index in range(count):
                    records.append(
                        {
                            "record_id": f"{split}-{index}",
                            "slide_id": f"{split}-{index % slides}",
                            "annotation_type": "circle",
                            "sponsor_classify": "YES",
                            "sponsor_category": categories[index % len(categories)],
                            "split": split,
                        }
                    )
            with assignment.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            destination = Path(cwd) / "inputs"
            with (
                patch.multiple(
                    inputs,
                    ASSIGNMENTS=assignment,
                    FINE_TO_BROAD=mapping,
                    OUT=destination,
                    EXPECTED_ASSIGNMENTS_SHA=hashlib.sha256(
                        assignment.read_bytes()
                    ).hexdigest(),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                inputs.main()
            self.assertEqual(
                (destination / "fine_to_broad.csv").read_bytes(), mapping.read_bytes()
            )
            summary = json.loads((destination / "summary.json").read_text())
            self.assertEqual(
                summary["splits"], {"train": 1605, "valid": 347, "test": 331}
            )

    def test_successive_training_processes_get_distinct_scratch_directories(self):
        from method_a.storage import make_scratch

        with (
            tempfile.TemporaryDirectory() as cwd,
            patch.dict(os.environ, SLURM_TMPDIR=cwd),
        ):
            first = make_scratch("grid")
            second = make_scratch("grid")
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, Path(cwd))
            self.assertTrue(first.is_dir() and second.is_dir())

    def test_all_command_help_works_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            for name in ("prepare", "train", "evaluate", "export_results"):
                with self.subTest(command=name):
                    script = REPOSITORY / "scripts" / "method_a" / f"{name}.py"
                    result = subprocess.run(
                        [sys.executable, "-B", str(script), "--help"],
                        cwd=cwd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertIn("usage:", result.stdout)

    def test_training_dispatch_preserves_four_protocol_choices(self):
        for name in CONFIGURATIONS:
            with (
                self.subTest(configuration=name),
                patch.object(train, "import_module") as load,
            ):
                train.main(["--config", name, "--smoke"])
                expected = "grid" if name == "baseline" else "fixed_steps"
                load.assert_called_once_with(f"method_a.training.{expected}")
                args = (
                    ["--smoke"]
                    if name == "baseline"
                    else ["--variant", name, "--smoke"]
                )
                load.return_value.main.assert_called_once_with(args)

    def test_evaluation_dispatch_preserves_four_protocol_choices(self):
        for name in CONFIGURATIONS:
            with (
                self.subTest(configuration=name),
                patch.object(evaluate, "import_module") as load,
            ):
                evaluate.main(["--config", name])
                expected = "grid" if name == "baseline" else "known_references"
                load.assert_called_once_with(f"method_a.evaluation.{expected}")
                if name == "baseline":
                    load.return_value.main.assert_called_once_with()
                else:
                    load.return_value.main.assert_called_once_with(["--variant", name])

    def test_invalid_stages_fail_before_importing_or_running_a_workflow(self):
        cases = (
            (prepare, ["--config", "A1", "--stage", "export", "--slide-index", "0"]),
            (prepare, ["--config", "baseline", "--stage", "weights"]),
            (prepare, ["--config", "A2", "--stage", "export"]),
            (prepare, ["--config", "A3", "--stage", "build", "--limit", "1"]),
            (evaluate, ["--config", "baseline", "--smoke"]),
            (train, ["--config", "Method B"]),
        )
        for command, args in cases:
            with (
                self.subTest(args=args),
                patch.object(command, "import_module") as load,
            ):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    command.main(args)
                load.assert_not_called()

    def test_baseline_preparation_dispatch(self):
        for stage, module in prepare.GRID_STAGES.items():
            with (
                self.subTest(stage=stage),
                patch.object(prepare, "import_module") as load,
            ):
                prepare.main(["--config", "baseline", "--stage", stage])
                load.assert_called_once_with(f"method_a.preparation.{module}")
                load.return_value.main.assert_called_once_with()

    def test_context_dataset_build_selects_only_requested_configuration(self):
        crops = [{"matched_anchors": ["known"]}, {"matched_anchors": []}]
        with tempfile.TemporaryDirectory() as cwd:
            plan = Path(cwd) / "plan"
            plan.mkdir()
            (plan / "00.json").write_text("{}")
            for name, experiment, selected in (
                ("A2", "m2_retile_matched", crops[:1]),
                ("A3", "m2_retile_all", crops),
            ):
                backend = Mock(ROOT=Path(cwd))
                backend.load.return_value = {"crops": crops}
                with (
                    self.subTest(configuration=name),
                    patch.object(prepare, "import_module", return_value=backend),
                ):
                    prepare.main(["--config", name, "--stage", "build"])
                    backend.make_dataset.assert_called_once_with(experiment, selected)

    def test_importing_workflows_does_not_open_project_data(self):
        code = """
from unittest.mock import patch
with patch('pathlib.Path.open', side_effect=AssertionError('Import opened project data')):
    from method_a.preparation import weights, input_checks, qa
    from method_a import reporting
    assert all(callable(module.main) for module in (weights, input_checks, qa, reporting))
"""
        with tempfile.TemporaryDirectory() as cwd:
            subprocess.run([sys.executable, "-B", "-c", code], cwd=cwd, check=True)

    def test_result_entry_point_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            output = Path(cwd) / "tables"
            script = REPOSITORY / "scripts" / "method_a" / "export_results.py"
            subprocess.run(
                [sys.executable, "-B", str(script), "--output", str(output)],
                cwd=cwd,
                check=True,
            )
            with (output / "detection.csv").open(newline="") as stream:
                actual = list(csv.reader(stream))
            expected = (
                REPOSITORY / "src" / "method_a" / "examples" / "expected_detection.csv"
            )
            with expected.open(newline="") as stream:
                self.assertEqual(actual, list(csv.reader(stream)))
            self.assertEqual(
                {path.name for path in output.iterdir()}, {"detection.csv"}
            )

    def test_shared_deduplication_removes_overlap(self):
        from method_a.evaluation.deduplication import deduplicate
        from method_a.evaluation.detection_types import Detection, TileSpec

        first = Detection(10, 10, 30, 30, 0.5, 0, 0, 0, 0)
        second = Detection(10, 10, 30, 30, 0.9, 0, 0, 0, 0)
        tiles = {
            TileSpec(0, 0, 100, 100, 0, 0): [first],
            TileSpec(5, 0, 100, 100, 1, 0): [second],
        }
        self.assertEqual(deduplicate(tiles, 0.5), [second])

    def test_packaged_category_map_matches_shared_mapping(self):
        path = REPOSITORY / "data" / "method_a"
        categories = json.loads((path / "broad_category_map.json").read_text())
        with (REPOSITORY / "data" / "fine_to_broad.csv").open(newline="") as stream:
            expected = {
                row["broad_class_id"]: row["broad_category"]
                for row in csv.DictReader(stream)
            }
        self.assertEqual(categories, expected)
        split = json.loads((path / "split_summary.json").read_text())
        self.assertEqual(
            sum(item["records"] for item in split["partitions"].values()), 2283
        )
        self.assertFalse(split["test_pixels_used"])


if __name__ == "__main__":
    unittest.main()
