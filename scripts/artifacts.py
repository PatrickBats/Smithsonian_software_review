"""Compatibility entry point for :mod:`smithsonian.io.artifacts`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.io.artifacts import main as run

    return run()


if __name__ == "__main__":
    main()
