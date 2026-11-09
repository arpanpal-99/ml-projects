"""Convert normalized movie tags into sparse bag-of-words feature vectors."""

from __future__ import annotations

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def build_feature_matrix(
    tags: list[str], max_features: int = 5000
) -> tuple[CountVectorizer, sparse.csr_matrix]:
    """Fit a CountVectorizer and return the vocabulary and movie-by-term matrix."""
    if max_features < 1:
        raise ValueError("max_features must be positive.")
    if not tags:
        raise ValueError("Cannot build movie features from an empty movie table.")

    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    try:
        matrix = vectorizer.fit_transform(tags).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError(
                "No usable words were found in movie tags. Check the source metadata "
                "and preprocessing output before rebuilding."
            ) from exc
        raise
    return vectorizer, matrix
