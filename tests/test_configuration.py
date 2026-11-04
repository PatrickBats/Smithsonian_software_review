"""Configuration precedence, path origins and immutable metadata contracts."""

import json
from pathlib import Path
import tempfile
import unittest

from smithsonian.configuration.options import ConfigArgumentParser, save_effective_config


class ConfigurationTests(unittest.TestCase):
    def parser(self):
        parser = ConfigArgumentParser()
        parser.add_argument("--image", type=Path, required=True)
        parser.add_argument("--epochs", type=int, default=40)
        return parser

    def test_config_relative_paths_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "settings.json"
            config.write_text(json.dumps({"image": "crop.png", "epochs": 3}))
            parsed = self.parser().parse_args(["--config", str(config), "--epochs", "7"])
            self.assertEqual(parsed.image, Path(folder) / "crop.png")
            self.assertEqual(parsed.epochs, 7)
            parsed = self.parser().parse_args(["--config", str(config), "--image", "other.png"])
            self.assertEqual(parsed.image, Path.cwd() / "other.png")

    def test_unknown_config_keys_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "bad.json"
            config.write_text('{"imag": "typo.png"}')
            with self.assertRaises(SystemExit):
                self.parser().parse_args(["--config", str(config)])

    def test_invalid_types_fail(self):
        with self.assertRaises(SystemExit):
            self.parser().parse_args(["--image", "crop.png", "--epochs", "not-a-number"])

    def test_run_identity_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.png"
            image.write_bytes(b"original")
            output = Path(folder) / "record.json"
            save_effective_config({"image": str(image)}, output)
            save_effective_config({"image": str(image)}, output)
            image.write_bytes(b"changed")
            with self.assertRaises(FileExistsError):
                save_effective_config({"image": str(image)}, output)


def main():
    """Run configuration tests."""
    unittest.main()


if __name__ == "__main__":
    main()
