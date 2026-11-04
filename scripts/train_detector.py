"""Command-line entry point for training.detector.train."""


def main():
    """Run the installed package workflow with command-line settings."""
    from smithsonian.training.detector.train import main as run

    return run()


if __name__ == "__main__":
    main()
