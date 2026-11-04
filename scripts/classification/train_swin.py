"""Compatibility entry point for :mod:`smithsonian.training.train_swin`. Install with pip install -e ."""


def main():
    """Run the installed package command."""
    from smithsonian.training.train_swin import main as run

    return run()


if __name__ == "__main__":
    main()
