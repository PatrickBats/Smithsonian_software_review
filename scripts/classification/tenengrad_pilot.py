"""Compatibility entry point for :mod:`smithsonian.preprocessing.tenengrad_pilot`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.preprocessing.tenengrad_pilot import main as run

    return run()


if __name__ == "__main__":
    main()
