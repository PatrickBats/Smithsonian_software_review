"""Compatibility entry point; implementation lives in method_b.detection.predict_boxes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method_b.detection.predict_boxes import main

if __name__ == "__main__":
    main()
