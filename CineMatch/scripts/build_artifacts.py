from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

from src.config import Settings
from src.data.loader import load_and_merge_data
from src.data.preprocessing import preprocess_movies
from src.features.feature_engineering import build_feature_matrix
from src.recommender.artifacts import save_artifacts
from src.utils.logging_config import configure_logging


def main() -> int:
    load_dotenv()
    configure_logging()
    logger = logging.getLogger(__name__)
    settings = Settings.from_environment()
    try:
        movies, load_summary = load_and_merge_data(settings.raw_data_dir)
        movies, preprocessing_summary = preprocess_movies(
            movies,
            principal_cast_size=settings.principal_cast_size,
            max_parse_failure_rate=settings.max_parse_failure_rate,
        )
        vectorizer, matrix = build_feature_matrix(
            movies["tags"].tolist(), max_features=settings.max_features
        )
        summary = {**load_summary, **preprocessing_summary}
        summary["row_count"] = len(movies)
        summary["feature_count"] = matrix.shape[1]
        save_artifacts(
            movies,
            vectorizer,
            matrix,
            settings.artifacts_dir,
            summary,
            settings.max_features,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Artifact build failed: %s", exc)
        return 1

    logger.info(
        "Artifact build completed: %d movies, %d features. Start the app with "
        "`streamlit run app.py`.",
        len(movies),
        matrix.shape[1],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
