# Configuration and validation

`ConfigArgumentParser` merges JSON defaults and explicit CLI options. `load_config` resolves file-relative paths. `save_effective_config` records resolved settings and hashes. `check_repository` checks submission contents.

**Inputs:** JSON settings, CLI arguments and file paths.

**Outputs:** Validated settings and immutable run metadata; incompatible records raise errors.


See the [root README](../../../README.md) for installation and workflow entry points.
