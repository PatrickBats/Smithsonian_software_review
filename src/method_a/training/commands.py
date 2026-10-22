"""Train any Method A configuration while preserving its recorded protocol."""

import argparse
from importlib import import_module

from ..configurations import CONFIGURATIONS, resolve_configuration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=resolve_configuration, choices=CONFIGURATIONS, required=True
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Run the configuration's short GPU check"
    )
    args = parser.parse_args(argv)

    # Share the interface, not checkpoint selection: the baseline used a different protocol.
    module = "grid" if args.config == "baseline" else "fixed_steps"
    forwarded = [] if args.config == "baseline" else ["--variant", args.config]
    if args.smoke:
        forwarded.append("--smoke")
    import_module(f"method_a.training.{module}").main(forwarded)


if __name__ == "__main__":
    main()
