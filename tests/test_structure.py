"""Check package boundaries, guarded execution and distributable notebook state."""

import ast
import json
import unittest

from smithsonian.configuration.paths import repository_root

ROOT = repository_root()


class StructureTests(unittest.TestCase):
    def test_entrypoints_have_guarded_main(self):
        for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
            for path in base.rglob("*.py"):
                tree = ast.parse(path.read_text())
                guards = [
                    n
                    for n in tree.body
                    if isinstance(n, ast.If) and "__name__" in ast.unparse(n.test)
                ]
                mains = [
                    n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
                ]
                if guards or mains:
                    self.assertTrue(mains and guards, str(path))
                    self.assertEqual(ast.unparse(guards[0].body[0]), "main()", str(path))

    def test_no_script_imports_or_path_injection(self):
        for path in (ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith("scripts"), str(path))
                if isinstance(node, ast.Call):
                    self.assertNotIn(
                        ast.unparse(node.func), ("sys.path.insert", "sys.path.append"), str(path)
                    )

    def test_notebook_has_executed_outputs_and_pipeline_calls(self):
        notebook = json.loads((ROOT / "notebooks/demo.ipynb").read_text())
        code = ""
        images = 0
        tables = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsInstance(cell["execution_count"], int)
                for output in cell["outputs"]:
                    self.assertNotEqual(output["output_type"], "error")
                    data = output.get("data", {})
                    images += int("image/png" in data or "image/jpeg" in data)
                    tables += int("text/html" in data)
                code += "".join(cell["source"])
        self.assertGreaterEqual(images, 3)
        self.assertGreaterEqual(tables, 1)
        for interface in (
            "prepare_demo",
            "classify_example",
            "detect_example",
            "annotate_predictions",
        ):
            self.assertIn(interface, code)


def main():
    """Run repository structure tests."""
    unittest.main()


if __name__ == "__main__":
    main()
