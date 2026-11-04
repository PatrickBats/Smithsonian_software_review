"""Meaningful input and evaluation regression tests; no images or weights required."""

from pathlib import Path
import tempfile
import unittest
from smithsonian.inference.infer import validate_boxes
from smithsonian.io.artifacts import verify, sha256
from smithsonian.data.broad_labels import apply_broad


ROOT = Path(__file__).resolve().parents[1]


class InputContracts(unittest.TestCase):
    def test_invalid_and_empty_boxes(self):
        self.assertEqual(validate_boxes([]), [])
        for box in ([0, 0, 0, 1], [1, 0, 0, 1], [0, 0, float("nan"), 1], [0, 1, 2]):
            with self.assertRaises(ValueError):
                validate_boxes([box])

    def test_checksum_rejects_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "artifact"
            p.write_bytes(b"first")
            h = sha256(p)
            verify(p, h)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify(p, h)

    def test_broad_mapping_preserves_identity_and_split(self):
        import csv

        with (ROOT / "data/fine_to_broad.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        labels = [r["fine_category"] for r in rows]
        records = [
            {"record_id": str(i), "split": "val", "sponsor_category": n}
            for i, n in enumerate(labels)
        ]
        updated, broad, audit = apply_broad(records, labels, ROOT / "data/fine_to_broad.csv")
        self.assertEqual(len(broad), 28)
        self.assertEqual([r["record_id"] for r in records], [r["record_id"] for r in updated])
        self.assertTrue(all(r["split"] == "val" for r in updated))
        self.assertEqual([r["fine_category"] for r in updated], labels)
        with self.assertRaises(ValueError):
            apply_broad(records, list(reversed(labels)), ROOT / "data/fine_to_broad.csv")


def main():
    """Run input contract checks."""
    unittest.main()


if __name__ == "__main__":
    main()
