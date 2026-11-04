"""Compatibility entry point for :mod:`smithsonian.preprocessing.full_tenengrad`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.preprocessing.full_tenengrad import main as run

    return run()


if __name__ == "__main__":
    main()
