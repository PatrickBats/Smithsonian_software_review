# Source attribution

This package combines the Fall 2026 Rice D2K Smithsonian team's software for
fossil palynomorph detection and classification. Source repositories and revisions
for adapted code are listed below.

Spring boundary-aware deduplication, tile structures, and LoG focus fusion follow
RiceD2KLab/Smithsonian_fossil_Sp26, revision
`1b62b68a0d95178e6275ef55c38ee4765317b334`:
https://github.com/RiceD2KLab/Smithsonian_fossil_Sp26

DINOv2 is loaded from Meta's official implementation at revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`:
https://github.com/facebookresearch/dinov2

Swin-Tiny uses torchvision's implementation:
https://github.com/pytorch/vision

RF-DETR and RF-DETR+ are installed through the requirements file:
https://github.com/roboflow/rf-detr

Upstream packages retain their own license notices and terms. No blanket license
is asserted over the sponsor's data or source with no supplied license. The
external source checkouts retain their upstream notices. Raw images and pretrained
weight payloads are not redistributed inside this submission.

## Adapted code

Tile structures and boundary-aware deduplication are in `src/smithsonian/models/`.
Streaming LoG fusion is in `src/smithsonian/preprocessing/detector_crops/focus_helpers.py`.
These retain credit to Shaikh et al. and the Rice D2K Spring 2026 team. The external
Spring adapter is loaded only for whole-slide replay.

Current ignore-region training and evaluation were adapted from the Fall team's
`method-a-ignore-regions` branch at commit
`8174c47939430d873a4f42b023f35fe7516aa168` in
[PatrickBats/Smithsonian_fossil_F26](https://github.com/PatrickBats/Smithsonian_fossil_F26),
under `scripts/method_a_ignore_regions/`. Historical external artifact paths retain
their original spelling.
