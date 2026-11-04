# Recorded results

`results/` contains current classifier, joint-detector, and combined-pipeline aggregate
records. Regenerate tables with `python scripts/results.py --output outputs/results`.
Choose a new output directory for the generated tables.

The joint result is the crop-and-ignore-region run at confidence 0.30: 243 of 274 known
targets located, 85 correctly identified. Its source revision is recorded in
`provenance/cleanup.json`; checkpoint identity is in `config/checkpoints.json`.

The classifier records use expert-located crops. The RGB comparison uses the same
258 matched detected crops for both classifiers, with 274 known targets overall.
See [evaluation](../src/smithsonian/evaluation/README.md) for definitions and replay commands.
Historical dataset comparisons remain in the original working project repository.
