# Detector crops

`audit_plan.py` reads frozen training annotations and source geometry. `prepare_datasets.py`
plans native crop windows and assembles exported images into COCO format. `export_crops.py`
reads the focal stack and pads each selected window onto a 1024-pixel canvas.
`focus_helpers.py` implements the credited Spring LoG focus rule.

Inputs, stages, and example commands are in [preprocessing](../README.md).
