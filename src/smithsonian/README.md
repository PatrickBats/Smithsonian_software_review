# Shared package

The `smithsonian` package follows the Spring team's functional organization:
[data](data/README.md), [preprocessing](preprocessing/README.md),
[models](models/README.md), [training](training/README.md),
[inference](inference/README.md), and [evaluation](evaluation/README.md).
[Configuration](configuration/README.md) and [I/O](io/README.md) contain shared helpers.

Install from the repository root with `python -m pip install --no-deps -e .` after
installing requirements. Scripts and notebooks call these modules directly.
Modules are safe to import. Workflow execution starts when an entry point or
workflow function is called.
