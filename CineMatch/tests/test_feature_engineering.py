from __future__ import annotations

import pytest

from src.features.feature_engineering import build_feature_matrix


def test_builds_sparse_count_matrix():
    vectorizer, matrix = build_feature_matrix(
        ["space travel galaxy", "space travel stars"], max_features=20
    )

    assert matrix.shape[0] == 2
    assert matrix.shape[1] == len(vectorizer.vocabulary_)
    assert matrix.format == "csr"


def test_empty_vocabulary_gives_actionable_error():
    with pytest.raises(ValueError, match="No usable words"):
        build_feature_matrix(["the and or"])
