# Scientific protocol

The frozen inventory contains 2,283 specimens. Related Momipites and Caryapollenites
source labels are grouped into 28 modeling categories using `data/fine_to_broad.csv`.
Source categories remain available in the original map.

Separate-slide and shared-slide classifier settings each contain 1,605 training,
347 validation, and 331 reserved test specimens. Shared-slide assignments preserve
test membership. The checkpoint is selected by validation macro-F1.

Classifier training uses specimen-level Tenengrad-selected RGB crops. The joint
detector trains on target-centered, LoG focus-stacked windows with candidate ignore
regions. The recorded combined comparison uses detector boxes from tile-level
best-plane inputs. These preprocessing contracts are preserved independently.

Current joint-detector validation uses confidence 0.30 and IoU 0.50. The recorded
combined diagnostic uses confidence 0.50 and IoU 0.50. Their known-target metrics are
defined in the [evaluation guide](../src/smithsonian/evaluation/README.md).
