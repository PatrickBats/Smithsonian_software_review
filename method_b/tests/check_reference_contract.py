"""Regression checks for image selection, LoG pixel provenance and merging."""

# Source credit: Fall 2026 Smithsonian team integration; Spring preprocessing,
# tile types and boundary-aware merging derive from RiceD2KLab/Smithsonian_fossil_Sp26
# at 1b62b68a0d95178e6275ef55c38ee4765317b334. Swin: torchvision; DINOv2: Meta.
# See NOTICE.md for upstream links and the distinction between reused and project code.

import numpy as np
from method_b.preprocessing.prepare_detection_tiles import positions
from method_b.reference.Smithsonian_Fossil_Sp26.focus_fusion import focus_stack, laplacian_of_gaussian
from method_b.reference.Smithsonian_Fossil_Sp26.detection_merging import deduplicate
from method_b.reference.Smithsonian_Fossil_Sp26.detection_types import TileSpec, Detection
import cv2

# Match the extraction entry point: single-thread OpenCV also avoids
# allocation-sensitive GaussianBlur differences in the tiny test fixture.
def main():
    """Run the executable regression checks.

    Returns:
        None: Raises AssertionError if a contract fails.
    """
    cv2.setNumThreads(1)

    assert positions(100, 3100, 1024, 921) == [100, 1021, 1942, 2076]
    assert positions(0, 2048, 1024, 921) == [0, 921, 1024]
    assert positions(0, 500, 1024, 921) == []
    # The grid function has no specimen-coordinate argument, and includes its
    # right/bottom edge positions even if no object lies there.
    rng = np.random.default_rng(549)
    tile = rng.integers(0, 256, (24, 24, 3, 4), dtype=np.uint8)
    rgb, _ = focus_stack(tile, k=5)
    responses = np.stack(
        [
            abs(laplacian_of_gaussian(cv2.cvtColor(tile[:, :, :, z], cv2.COLOR_RGB2GRAY), 5))
            for z in range(4)
        ]
    )
    best = responses.argmax(axis=0)
    expected = tile[np.arange(24)[:, None], np.arange(24)[None, :], :, best]
    assert np.array_equal(rgb, expected), "Stacked pixels must come from max-abs-LoG plane"


    def d(x, score):
        """Construct a small detection fixture for the merging regression.

        Args:
            x: Input tensor or coordinate used by the enclosing operation.
            score: Detection confidence.

        Returns:
            Detection: Axis-aligned synthetic detection.
        """
        return Detection(x, 0, x + 20, 20, score, 0, 0, 0, 0)


    a, b, c = d(0, 0.5), d(2, 0.9), d(4, 0.8)
    tiles = {
        TileSpec(0, 0, 1024, 1024, 0, 0): [a],
        TileSpec(10, 0, 1024, 1024, 10, 0): [b],
        TileSpec(20, 0, 1024, 1024, 20, 0): [c],
    }
    assert deduplicate(tiles, 0.5) == [b]
    # The paper does not merge detections within the same tile in this pass.
    assert len(deduplicate({TileSpec(0, 0, 1024, 1024, 0, 0): [a, b]}, 0.5)) == 2
    assert (
        len(
            deduplicate(
                {TileSpec(0, 0, 1024, 1024, 0, 0): [a], TileSpec(2048, 0, 1024, 1024, 2048, 0): [b]},
                0.5,
            )
        )
        == 2
    )
    print("Abbas grid / LoG pixel identity / boundary-aware union-find checks passed", flush=True)


if __name__ == "__main__":
    main()
