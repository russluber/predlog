"""Predlog package entry point."""

__version__ = "0.1.0"


def main() -> None:
    """Run the Predlog command line interface."""

    from predlog.cli import app

    app()
