"""Persist and validate the movie table, vectorizer, feature matrix, and manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import sparse

ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_FILENAMES = {
    "movies": "movies.parquet",
    "vectorizer": "vectorizer.joblib",
    "matrix": "feature_matrix.npz",
    "manifest": "manifest.json",
}
LOGGER = logging.getLogger(__name__)


class ArtifactError(ValueError):
    """Raised for missing, corrupt, or incompatible generated artifacts."""


def ordered_ids_checksum(ids: pd.Series) -> str:
    """Hash the ordered movie IDs so row-to-matrix alignment can be verified."""
    payload = ",".join(str(int(movie_id)) for movie_id in ids.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest through a temporary file before replacing the target."""
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as file:
        temporary_path = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary_path, path)


def save_artifacts(
    movies: pd.DataFrame,
    vectorizer: Any,
    matrix: sparse.spmatrix,
    artifacts_dir: Path,
    build_summary: dict[str, Any],
    max_features: int,
) -> dict[str, Any]:
    """Write model outputs and a manifest containing dimensions and file hashes."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: artifacts_dir / name for key, name in ARTIFACT_FILENAMES.items()}
    with tempfile.TemporaryDirectory(prefix=".cinematch-build-", dir=artifacts_dir) as temp:
        temp_dir = Path(temp)
        temp_movies = temp_dir / ARTIFACT_FILENAMES["movies"]
        temp_vectorizer = temp_dir / ARTIFACT_FILENAMES["vectorizer"]
        temp_matrix = temp_dir / ARTIFACT_FILENAMES["matrix"]
        movies.to_parquet(temp_movies, index=False)
        joblib.dump(vectorizer, temp_vectorizer, compress=3)
        sparse.save_npz(temp_matrix, matrix.tocsr(), compressed=True)

        manifest: dict[str, Any] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "movie_count": int(len(movies)),
            "feature_count": int(matrix.shape[1]),
            "ordered_movie_ids_sha256": ordered_ids_checksum(movies["id"]),
            "vectorizer": {
                "type": "CountVectorizer",
                "max_features": max_features,
                "stop_words": "english",
                "vocabulary_size": len(vectorizer.vocabulary_),
            },
            "matrix": {"format": "scipy_csr", "dtype": str(matrix.dtype)},
            "build_summary": build_summary,
            "artifacts_sha256": {
                "movies.parquet": _file_sha256(temp_movies),
                "vectorizer.joblib": _file_sha256(temp_vectorizer),
                "feature_matrix.npz": _file_sha256(temp_matrix),
            },
        }

        for key, temp_path in (
            ("movies", temp_movies),
            ("vectorizer", temp_vectorizer),
            ("matrix", temp_matrix),
        ):
            os.replace(temp_path, paths[key])
        _atomic_json(paths["manifest"], manifest)

    LOGGER.info(
        "Wrote %d movies and %d sparse features to %s.",
        len(movies),
        matrix.shape[1],
        artifacts_dir,
    )
    return manifest


def load_artifacts(artifacts_dir: Path) -> tuple[pd.DataFrame, Any, sparse.csr_matrix]:
    """Load artifacts only after checking file integrity and model consistency."""
    paths = {key: artifacts_dir / name for key, name in ARTIFACT_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ArtifactError(
            "CineMatch artifacts are incomplete or missing: "
            f"{', '.join(missing)}. Build them with `python -m scripts.build_artifacts`."
        )

    try:
        with paths["manifest"].open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ArtifactError("Artifact manifest must contain a JSON object; rebuild artifacts.")
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactError(
                f"Unsupported artifact schema version {manifest.get('schema_version')!r}; "
                "rebuild artifacts with the current project version."
            )
        hashes = manifest.get("artifacts_sha256")
        expected_names = {
            ARTIFACT_FILENAMES["movies"],
            ARTIFACT_FILENAMES["vectorizer"],
            ARTIFACT_FILENAMES["matrix"],
        }
        if not isinstance(hashes, dict) or set(hashes) != expected_names:
            raise ArtifactError(
                "Artifact manifest is missing required file integrity checks; rebuild artifacts."
            )
        for name, expected_hash in hashes.items():
            artifact_path = artifacts_dir / name
            if not artifact_path.is_file() or _file_sha256(artifact_path) != expected_hash:
                raise ArtifactError(
                    f"Artifact integrity check failed for {name}; rebuild artifacts."
                )

        movies = pd.read_parquet(paths["movies"])
        vectorizer = joblib.load(paths["vectorizer"])
        matrix = sparse.load_npz(paths["matrix"]).tocsr()
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"Could not load CineMatch artifacts from {artifacts_dir}: {exc}. "
            "Rebuild them with `python -m scripts.build_artifacts`."
        ) from exc

    required = {"id", "title", "tags"}
    if not required.issubset(movies.columns):
        raise ArtifactError(
            f"Movie artifact is missing required columns {sorted(required - set(movies.columns))}; "
            "rebuild artifacts."
        )
    if movies["id"].isna().any() or movies["id"].duplicated().any():
        raise ArtifactError("Movie artifact IDs must be non-null and unique; rebuild artifacts.")
    if len(movies) != matrix.shape[0] or len(movies) != manifest.get("movie_count"):
        raise ArtifactError("Movie and feature-matrix row counts do not match; rebuild artifacts.")
    vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}))
    if (
        matrix.shape[1] != vocabulary_size
        or matrix.shape[1] != manifest.get("feature_count")
        or vocabulary_size != manifest.get("vectorizer", {}).get("vocabulary_size")
    ):
        raise ArtifactError(
            "Feature matrix and vectorizer vocabulary dimensions do not match; rebuild artifacts."
        )
    if ordered_ids_checksum(movies["id"]) != manifest.get("ordered_movie_ids_sha256"):
        raise ArtifactError("Movie ID ordering does not match the manifest; rebuild artifacts.")
    return movies, vectorizer, matrix
