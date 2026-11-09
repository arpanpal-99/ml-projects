"""Tests for CSV validation, movie-credit joins, and source-data diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import DataValidationError, load_and_merge_data


def test_loads_and_merges_on_unique_movie_ids(tmp_path, raw_frames):
    movies, credits = raw_frames
    movies.to_csv(tmp_path / "tmdb_5000_movies.csv", index=False)
    credits.to_csv(tmp_path / "tmdb_5000_credits.csv", index=False)

    merged, summary = load_and_merge_data(tmp_path)

    assert merged["id"].tolist() == [101, 102, 103]
    assert "cast" in merged
    assert summary["movies_without_matching_credits"] == 0


def test_missing_dataset_has_actionable_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="place them in data/raw"):
        load_and_merge_data(tmp_path)


def test_duplicate_ids_are_rejected(tmp_path, raw_frames):
    movies, credits = raw_frames
    movies = pd.concat([movies, movies.iloc[[0]]], ignore_index=True)
    movies.to_csv(tmp_path / "tmdb_5000_movies.csv", index=False)
    credits.to_csv(tmp_path / "tmdb_5000_credits.csv", index=False)

    with pytest.raises(DataValidationError, match="duplicate movie IDs"):
        load_and_merge_data(tmp_path)
