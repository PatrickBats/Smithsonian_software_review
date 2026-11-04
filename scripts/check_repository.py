"""Compatibility entry point for :mod:`smithsonian.configuration.check_repository`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.configuration.check_repository import main as run

    return run()


if __name__ == "__main__":
    main()
