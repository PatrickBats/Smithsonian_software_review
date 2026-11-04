# Joint RF-DETR

`train.py` runs the current 28-category training recipe. `loss.py` provides ignore-region
masking and one-class head initialization. Tests call the loss module directly
to check matching and gradient behavior.

Inputs and commands are in [training](../README.md). The implementation is adapted from
the Fall team's recorded crop-and-ignore-region run; see [attribution](../../../../NOTICE.md).
