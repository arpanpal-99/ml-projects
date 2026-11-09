"""Tests for artifact round-trips and validation of saved model files."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.features.feature_engineering import build_feature_matrix
from src.recommender.artifacts import ArtifactError, load_artifacts, save_artifacts


def test_artifacts_round_trip_and_validate_manifest(tmp_path):
    movies = pd.DataFrame(
        [
            {"id": 1, "title": "Space One", "tags": "space galaxy"},
            {"id": 2, "title": "Space Two", "tags": "space travel"},
        ]
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    save_artifacts(movies, vectorizer, matrix, tmp_path, {"row_count": 2}, 5000)

    loaded_movies, loaded_vectorizer, loaded_matrix = load_artifacts(tmp_path)

    assert loaded_movies["id"].tolist() == [1, 2]
    assert loaded_matrix.shape == matrix.shape
    assert loaded_vectorizer.vocabulary_ == vectorizer.vocabulary_


def test_missing_artifacts_explain_rebuild_command(tmp_path):
    with pytest.raises(ArtifactError, match="python -m scripts.build_artifacts"):
        load_artifacts(tmp_path)


def test_modified_artifact_fails_integrity_check(tmp_path):
    movies = pd.DataFrame([{"id": 1, "title": "Space One", "tags": "space galaxy"}])
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    save_artifacts(movies, vectorizer, matrix, tmp_path, {}, 5000)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ordered_movie_ids_sha256"] = "invalid"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactError, match="ID ordering"):
        load_artifacts(tmp_path)


def test_invalid_manifest_shape_is_reported(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    (tmp_path / "movies.parquet").touch()
    (tmp_path / "vectorizer.joblib").touch()
    (tmp_path / "feature_matrix.npz").touch()

    with pytest.raises(ArtifactError, match="JSON object"):
        load_artifacts(tmp_path)
