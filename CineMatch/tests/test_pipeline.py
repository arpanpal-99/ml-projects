"""End-to-end tests for artifact building, loading, and recommendation ranking."""

from __future__ import annotations

from ml_pipeline import build_artifacts, load_recommender, recommend_movies
from src.config import Settings


def test_source_to_ranked_recommendation_pipeline(tmp_path, raw_frames):
    """Build from CSV fixtures, reload artifacts, and rank a related movie."""
    raw_dir = tmp_path / "raw"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir()
    movies, credits = raw_frames
    movies.to_csv(raw_dir / "tmdb_5000_movies.csv", index=False)
    credits.to_csv(raw_dir / "tmdb_5000_credits.csv", index=False)

    settings = Settings(
        raw_data_dir=raw_dir,
        artifacts_dir=artifacts_dir,
        max_parse_failure_rate=0.1,
    )
    manifest = build_artifacts(settings)
    recommender = load_recommender(artifacts_dir)

    results = recommend_movies(recommender, [101], count=5, mode="similar")

    assert manifest["movie_count"] == 3
    assert results["id"].tolist() == [102]
    assert results.iloc[0]["similarity"] > 0
    assert 101 not in results["id"].tolist()
