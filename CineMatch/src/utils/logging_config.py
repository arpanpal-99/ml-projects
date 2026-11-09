"""Shared logging configuration used by the command-line build and app."""

import logging


def configure_logging() -> None:
    """Set a consistent INFO-level log format for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
