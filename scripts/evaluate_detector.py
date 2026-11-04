"""Command-line entry point for evaluation.detector."""


def main():
    """Run the installed package workflow with command-line settings."""
    from smithsonian.evaluation.detector import main as run

    return run()


if __name__ == "__main__":
    main()
