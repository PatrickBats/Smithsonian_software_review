"""Compatibility entry point; implementation lives in method_b.classification.predict_categories."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method_b.classification.predict_categories import main, validate_boxes

if __name__ == "__main__":
    main()
