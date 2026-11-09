"""Movie data processing, feature engineering, and recommendation ranking."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"


def _find_dataset() -> tuple[Path, Path]:
    """Find both required dataset CSVs in this project."""
    data_dirs = (PROJECT_DIR / "data" / "raw", PROJECT_DIR / "data")
    for data_dir in data_dirs:
        movies_path = data_dir / MOVIES_FILE
        credits_path = data_dir / CREDITS_FILE
        if movies_path.is_file() and credits_path.is_file():
            return movies_path, credits_path

    expected = PROJECT_DIR / "data" / "raw"
    raise FileNotFoundError(
        f"Could not find both dataset CSVs. Put {MOVIES_FILE} and {CREDITS_FILE} "
        f"in {expected}."
    )


def _parse_metadata(value: Any) -> list[dict[str, Any]]:
    """Parse a CSV metadata cell into dictionaries, returning empty for invalid data."""
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Safely support Python-style list strings without evaluating arbitrary code.
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _feature_names(value: Any, *, limit: int | None = None) -> list[str]:
    """Normalize metadata names for use as recommendation feature tokens."""
    names = [
        "_".join(str(item.get("name", "")).lower().split())
        for item in _parse_metadata(value)
    ]
    names = [name for name in names if name]
    return names if limit is None else names[:limit]


def display_names(value: Any, *, limit: int | None = None) -> list[str]:
    """Extract names as written in the source metadata for display."""
    names = [
        str(item["name"]).strip()
        for item in _parse_metadata(value)
        if item.get("name")
    ]
    return names if limit is None else names[:limit]


def _feature_directors(value: Any) -> list[str]:
    """Normalize director names into feature tokens."""
    return [
        "_".join(str(person.get("name", "")).lower().split())
        for person in _parse_metadata(value)
        if person.get("job") == "Director" and person.get("name")
    ]


def display_directors(value: Any) -> list[str]:
    """Extract director names without changing their source capitalization."""
    return [
        str(person["name"]).strip()
        for person in _parse_metadata(value)
        if person.get("job") == "Director" and person.get("name")
    ]


@st.cache_resource(show_spinner="Preparing movie features...")
def load_movies_and_features() -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Load movie data, engineer content features, and vectorize the catalogue."""
    movies_path, credits_path = _find_dataset()
    movies = pd.read_csv(movies_path, low_memory=False)
    credits = pd.read_csv(credits_path, low_memory=False)

    required_movie_columns = {"id", "title", "genres", "keywords", "overview"}
    required_credit_columns = {"movie_id", "cast", "crew"}
    if not required_movie_columns.issubset(movies.columns):
        missing = sorted(required_movie_columns - set(movies.columns))
        raise ValueError(f"Movies CSV is missing required columns: {', '.join(missing)}.")
    if not required_credit_columns.issubset(credits.columns):
        missing = sorted(required_credit_columns - set(credits.columns))
        raise ValueError(f"Credits CSV is missing required columns: {', '.join(missing)}.")

    # Join by movie ID and retain only the credit fields used in features/display.
    catalogue = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        how="left",
        left_on="id",
        right_on="movie_id",
        validate="one_to_one",
        sort=False,
    )

    # Normalize structured metadata separately so recommendation tokens stay consistent.
    catalogue["genres_list"] = catalogue["genres"].map(_feature_names)
    catalogue["keywords_list"] = catalogue["keywords"].map(_feature_names)
    catalogue["cast_names"] = catalogue["cast"].map(
        lambda value: _feature_names(value, limit=3)
    )
    catalogue["director_names"] = catalogue["crew"].map(_feature_directors)

    # Combine descriptive fields into one document per movie.
    catalogue["tags"] = catalogue.apply(
        lambda movie: " ".join(
            [
                *movie["genres_list"],
                *movie["keywords_list"],
                *movie["cast_names"],
                *movie["director_names"],
                str(movie["overview"]).lower()
                if pd.notna(movie["overview"])
                else "",
            ]
        ),
        axis=1,
    )

    # Vectorize movie documents without building a dense, all-pairs similarity matrix.
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    try:
        feature_matrix = vectorizer.fit_transform(catalogue["tags"]).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError("No useful text was found in the movie metadata.") from exc
        raise

    # Parse release years once so the UI can display them without reparsing dates.
    catalogue["year"] = pd.to_datetime(
        catalogue["release_date"], errors="coerce"
    ).dt.year
    return catalogue, feature_matrix


def recommend_movies(
    movie_index: int,
    movies: pd.DataFrame,
    feature_matrix: sparse.csr_matrix,
    count: int = 5,
) -> pd.DataFrame:
    """Rank movies by content similarity, excluding the selected movie itself."""
    # Compare one movie vector to the catalogue, avoiding a full similarity matrix.
    scores = cosine_similarity(feature_matrix[movie_index], feature_matrix).ravel()

    # Ignore the selected title and movies without any shared text features.
    candidate_mask = (np.arange(len(movies)) != movie_index) & (scores > 0)
    candidate_indices = np.flatnonzero(candidate_mask)
    if not len(candidate_indices):
        return pd.DataFrame()

    # Use stable tie-breakers so equal scores produce predictable recommendations.
    titles = movies["title"].astype(str).str.casefold().to_numpy()
    movie_ids = movies["id"].to_numpy()
    order = np.lexsort(
        (movie_ids[candidate_indices], titles[candidate_indices], -scores[candidate_indices])
    )
    selected = candidate_indices[order[:count]]

    # Keep IDs for detail lookup and prepare display-friendly genre labels.
    results = movies.iloc[selected][
        ["id", "title", "year", "genres_list", "vote_average"]
    ].copy()
    results["similarity"] = scores[selected]
    results["genres"] = results["genres_list"].map(
        lambda genres: ", ".join(genre.replace("_", " ").title() for genre in genres)
    )
    return results.drop(columns="genres_list").reset_index(drop=True)
