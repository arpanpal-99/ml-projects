"""CineMatch's model-building and recommendation entry points.

This file connects the smaller, testable ML/data modules into the end-to-end
workflow. The Streamlit interface belongs in ``app.py``; data validation,
preprocessing, vectorization, artifact checks, and ranking implementations remain
in their focused ``src`` modules.

The recommender is content-based, not a supervised classifier or rating predictor:
there are no labels to train against and it does not predict a user's rating.
Instead, it represents each movie by the words in its metadata and ranks other
movies by how closely those representations match.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import Settings
from src.data.loader import load_and_merge_data
from src.data.preprocessing import preprocess_movies
from src.features.feature_engineering import build_feature_matrix
from src.recommender.artifacts import load_artifacts, save_artifacts
from src.recommender.engine import MovieRecommender

LOGGER = logging.getLogger(__name__)


def build_artifacts(settings: Settings | None = None) -> dict[str, object]:
    """Validate the source CSVs and build the local movie feature model.

    The build has four conceptual stages:

    1. Read and validate movie and credit records, then join them by movie ID.
    2. Parse structured metadata (genres, keywords, cast, and crew), normalize
       the text, and combine it with the overview into one ``tags`` string per
       movie. This creates one consistent text document for each catalogue row.
    3. Fit ``CountVectorizer`` on those strings. Each vocabulary word becomes a
       feature column, and each movie becomes a sparse bag-of-words count vector.
       The vectorizer is fitted only during this build; the saved vocabulary must
       be reused at recommendation time so feature columns keep the same meaning.
       Sparse storage is important because any one movie uses only a small
       fraction of the full vocabulary.
    4. Persist the movie rows, fitted vectorizer, and feature matrix together.
       The manifest records hashes and ordering information so a later load can
       detect mismatched, incomplete, or modified files.

    This is sometimes called "training" the content model, but it does not learn
    from user ratings. Fitting the vectorizer learns only which words occur in the
    supplied catalogue; recommendation ranking happens later at request time.
    """
    load_dotenv()
    settings = settings or Settings.from_environment()

    # Keep a copy of each stage's diagnostics in the manifest. This makes input
    # problems (unmatched IDs, malformed metadata, or empty descriptions) visible
    # after the build rather than silently hiding data loss.
    movies, load_summary = load_and_merge_data(settings.raw_data_dir)
    movies, preprocessing_summary = preprocess_movies(
        movies,
        principal_cast_size=settings.principal_cast_size,
        max_parse_failure_rate=settings.max_parse_failure_rate,
    )

    # CountVectorizer maps each tag string to counts of vocabulary terms. It does
    # not understand plot semantics; shared words (including metadata names) are
    # the similarity signal. The model predicts a ranked list of similar items,
    # not a rating or a guarantee of enjoyment.
    vectorizer, matrix = build_feature_matrix(
        movies["tags"].tolist(),
        max_features=settings.max_features,
    )

    summary = {**load_summary, **preprocessing_summary}
    summary["row_count"] = len(movies)
    summary["feature_count"] = matrix.shape[1]

    # Save all model inputs as one versioned set. The UI loads these files; it
    # never needs the original CSVs when the artifacts already exist.
    manifest = save_artifacts(
        movies,
        vectorizer,
        matrix,
        settings.artifacts_dir,
        summary,
        settings.max_features,
    )
    LOGGER.info(
        "Built the content model with %d movies and %d word features.",
        len(movies),
        matrix.shape[1],
    )
    return manifest


def load_recommender(artifacts_dir: Path | None = None) -> MovieRecommender:
    """Load and validate saved model files, then create the recommendation engine.

    The persisted matrix and vectorizer must be loaded together: matrix columns
    have meaning only according to the vocabulary learned by that vectorizer.
    ``load_artifacts`` checks their hashes, dimensions, schema, and movie order
    before the engine is constructed.
    """
    if artifacts_dir is None:
        load_dotenv()
        directory = Settings.from_environment().artifacts_dir
    else:
        directory = artifacts_dir
    movies, vectorizer, matrix = load_artifacts(directory)
    return MovieRecommender(movies, vectorizer, matrix)


def recommend_movies(
    recommender: MovieRecommender,
    movie_ids: Iterable[int],
    *,
    count: int = 8,
    genre: str | None = None,
    mode: str = "balanced",
) -> pd.DataFrame:
    """Rank movies similar to one seed movie or a small collection of liked films.

    The engine averages the selected movies' sparse feature vectors to form a
    taste profile, then compares that profile with catalogue vectors using cosine
    similarity: the dot product divided by the product of the two vector lengths.
    This scores shared direction/terms rather than letting a long overview win
    merely because it contains more words.

    ``similar`` ranks by cosine similarity alone. The ``balanced`` engine mode
    uses ``0.72 * similarity + 0.20 * quality + 0.08 * popularity``; ``hidden_gems``
    uses ``0.55 * similarity + 0.35 * quality + 0.10 * (1 - popularity)``.
    Quality is based on a vote-count-adjusted catalogue rating. These optional
    ranking signals change the order, but they do not train on user behavior. The
    selected seed titles are excluded, and an optional genre narrows the results.

    The returned DataFrame includes the source movie metadata and ranking scores;
    this function does not render or otherwise depend on the user interface.
    """
    return recommender.recommend_for_movies(
        movie_ids,
        k=count,
        genre=genre,
        mode=mode,
    )
