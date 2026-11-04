# Files, coordinates and external sources

`verify(path, expected)` checks SHA-256. `geometry(slide, record)` maps NDPA nanometres to slide pixels using scanner metadata. `load_spring_source` checks and explicitly registers the pinned external source.

**Inputs:** Artifact files, slide properties, annotation coordinates and source checkouts.

**Outputs:** Digests, pixel geometry or verified external package access.


See the [root README](../../../README.md) for installation and workflow entry points.
