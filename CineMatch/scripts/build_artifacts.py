"""Command-line entry point for building CineMatch's local model artifacts."""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from ml_pipeline import build_artifacts
from src.config import Settings
from src.utils.logging_config import configure_logging


def main() -> int:
    """Load project settings, build artifacts, and report a CLI exit status."""
    load_dotenv()
    configure_logging()
    logger = logging.getLogger(__name__)
    settings = Settings.from_environment()
    try:
        manifest = build_artifacts(settings)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Artifact build failed: %s", exc)
        return 1

    logger.info(
        "Artifact build completed: %d movies, %d features. Start the app with "
        "`streamlit run app.py`.",
        manifest["movie_count"],
        manifest["feature_count"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
