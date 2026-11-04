"""Compatibility entry point for :mod:`smithsonian.evaluation.results`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.evaluation.results import main as run

    return run()


if __name__ == "__main__":
    main()
