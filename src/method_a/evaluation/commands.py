"""Replay known-reference validation for a selected Method A configuration."""

import argparse
from importlib import import_module

from ..configurations import CONFIGURATIONS, resolve_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=resolve_configuration, choices=CONFIGURATIONS, required=True
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Two-image check for A1/A2/A3"
    )
    args = parser.parse_args(argv)
    if args.config == "baseline":
        if args.smoke:
            parser.error("The baseline has no two-image evaluation mode; omit --smoke.")
        import_module("method_a.evaluation.grid").main()
    else:
        forwarded = ["--variant", args.config]
        if args.smoke:
            forwarded.append("--smoke")
        import_module("method_a.evaluation.known_references").main(forwarded)


if __name__ == "__main__":
    main()
