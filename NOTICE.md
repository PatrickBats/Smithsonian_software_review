# Source attribution

This package combines the Fall 2026 Rice D2K Smithsonian team's software for
fossil palynomorph detection and classification. Imported source fingerprints and
revisions are recorded in `provenance/import_manifest.json`; final package hashes
are in `provenance/files.json`.

Spring boundary-aware deduplication, tile structures, and LoG focus fusion follow
RiceD2KLab/Smithsonian_fossil_Sp26, revision
`1b62b68a0d95178e6275ef55c38ee4765317b334`:
https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26

DINOv2 is loaded from Meta's official implementation at revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`:
https://github.com/facebookresearch/dinov2

Swin-Tiny uses torchvision's implementation:
https://github.com/pytorch/vision

RF-DETR and RF-DETR+ are installed dependencies, not vendored source:
https://github.com/roboflow/rf-detr

Upstream packages retain their own license notices and terms. No blanket license
is asserted over the sponsor's data or source with no supplied license. The
external source checkouts retain their upstream notices. Raw images and pretrained
weight payloads are not redistributed inside this submission.
