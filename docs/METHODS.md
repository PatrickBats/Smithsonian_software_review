# Evaluation and data contracts

The source classification inventory has 2,290 priority annotations. The adopted
experiment excludes seven duplicate/conflicting records and preserves 2,283
specimens. The frozen split has 1,605 training, 347 validation, and 331 reserved
test specimens. Category maps retain original source names and class ordering.

The 28-category experiment groups four Momipites labels and two Caryapollenites
labels from the original 32 categories. This is an explicit experimental mapping,
not a claim that the names are taxonomic synonyms. Keep the original fine label.

Separate-slide validation measures performance on source groups absent from
training. Shared-slide validation uses different specimens but permits shared
slides; it does not demonstrate generalization to unseen slides. Test specimens
remain reserved. Augmentation is limited to training examples.

The original classifier comparison reports Swin 42.94% / 70.61% and adapted DINO
50.14% / 74.35% validation accuracy (separate / shared slides). These use expert
crops, not detector crops. Checkpoint selection uses validation macro-F1.

Method A evaluates its baseline and A1/A2/A3 on the same 274 known validation
targets. The [Method A result README](../results/method_a/README.md)
contains all four rows and metric definitions. The historical baseline differs
in training settings, so comparison with the variants does not isolate data
selection alone. The [unified Method A command guide](../scripts/method_a/README.md)
explains preparation and execution for all four configurations. The
[input and split metadata](../data/method_a/README.md) records how the frozen
assignment is consumed without repartitioning.

The frozen RGB detection-crop comparison reports DINO 125/258 (48.4%) and Swin
93/258 (36.0%) correct among found objects. The reused detector had exposure to
13 of 14 validation slides; this is not an independent unseen-slide detector test.

Missing detections count against end-to-end success. Classification accuracy among
found objects has a different denominator. Unmatched predictions are unreviewed
when annotation coverage is incomplete, so they cannot all be called false
positives. Neither the crop demo nor the saved metrics establish whole-slide AP.

Numerical failures, changed fingerprints, and missing inputs must stop execution.
Do not change split membership, category order, or thresholds to make a run pass.
