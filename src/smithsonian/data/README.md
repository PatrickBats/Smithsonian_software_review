# Frozen identities and category maps

`apply_broad(rows, fine_labels, path)` maps 32 original annotation labels to 28 operational categories. `apply_split(rows, path)` applies the recorded diagnostic assignment while preserving reserved test membership.

**Inputs:** Specimen rows and frozen CSV/JSON mappings.

**Outputs:** Updated records, ordered labels and audit metadata.

## Categories and partitions

The inventory contains 2,283 specimens. `data/fine_to_broad.csv` combines related
Momipites and Caryapollenites labels into 28 modeling categories. The original
labels remain in `data/category_map_32.csv`.

Separate-slide and shared-slide settings each contain 1,605 training, 347 validation,
and 331 reserved test specimens. Related slides stay together in the separate-slide
setting. The shared-slide setting allows different specimens from a slide in training
and validation while preserving test membership.

See the [root README](../../../README.md) for installation and workflow entry points.
