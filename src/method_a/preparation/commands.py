"""Prepare a selected Method A dataset one explicit stage at a time."""

import argparse
from importlib import import_module

from ..configurations import CONFIGURATIONS, resolve_configuration, resolve_variant

GRID_STAGES = {"bind": "inputs", "plan": "grid", "build": "grid_dataset"}
VARIANT_STAGES = {"check", "audit", "weights", "plan", "export", "build", "qa"}


def build_variant(configuration):
    """Build only the requested dataset from the existing shared image pool."""
    datasets = import_module("method_a.preparation.variants")
    experiment = resolve_variant(configuration)
    if configuration == "A1":
        datasets.make_dataset(experiment)
        return

    plans = [
        datasets.load(path) for path in sorted((datasets.ROOT / "plan").glob("*.json"))
    ]
    if not plans:
        raise FileNotFoundError(
            "No context plans; run --stage plan and complete export first."
        )
    crops = [crop for plan in plans for crop in plan["crops"]]
    if configuration == "A2":
        crops = [crop for crop in crops if crop["matched_anchors"]]
    datasets.make_dataset(experiment, crops)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=resolve_configuration, choices=CONFIGURATIONS, required=True
    )
    parser.add_argument(
        "--stage", choices=sorted(set(GRID_STAGES) | VARIANT_STAGES), required=True
    )
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    # Reject unsupported combinations before importing image libraries or writing outputs.
    allowed = set(GRID_STAGES) if args.config == "baseline" else set(VARIANT_STAGES)
    if args.config == "A1":
        allowed -= {"audit", "plan", "export", "qa"}
    if args.stage not in allowed:
        parser.error(f"{args.config} supports stages: {', '.join(sorted(allowed))}")
    if args.stage != "export" and (
        args.slide_index is not None or args.limit is not None
    ):
        parser.error("--slide-index and --limit apply only to --stage export.")
    if args.stage == "export" and (args.slide_index is None or args.slide_index < 0):
        parser.error("--stage export requires a nonnegative --slide-index.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    if args.config == "baseline":
        import_module(f"method_a.preparation.{GRID_STAGES[args.stage]}").main()
    elif args.stage == "build":
        build_variant(args.config)
    elif args.stage == "plan":
        import_module("method_a.preparation.variants").plan_crops()
    elif args.stage == "export":
        forwarded = ["--slide-index", str(args.slide_index)]
        if args.limit is not None:
            forwarded += ["--limit", str(args.limit)]
        import_module("method_a.preparation.crop_export").main(forwarded)
    else:
        module = {
            "check": "input_checks",
            "audit": "source_audit",
            "weights": "weights",
            "qa": "qa",
        }[args.stage]
        workflow = import_module(f"method_a.preparation.{module}")
        workflow.main([]) if args.stage == "check" else workflow.main()


if __name__ == "__main__":
    main()
