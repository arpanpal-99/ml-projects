from __future__ import annotations

from src.data.loader import load_and_merge_data
from src.data.preprocessing import preprocess_movies
from src.features.feature_engineering import build_feature_matrix
from src.recommender.artifacts import load_artifacts, save_artifacts
from src.recommender.engine import MovieRecommender


def test_source_to_ranked_recommendation_pipeline(tmp_path, raw_frames):
    raw_dir = tmp_path / "raw"
    artifacts_dir = tmp_path / "artifacts"
    raw_dir.mkdir()
    movies, credits = raw_frames
    movies.to_csv(raw_dir / "tmdb_5000_movies.csv", index=False)
    credits.to_csv(raw_dir / "tmdb_5000_credits.csv", index=False)

    merged, load_summary = load_and_merge_data(raw_dir)
    prepared, preprocessing_summary = preprocess_movies(merged, max_parse_failure_rate=0.1)
    vectorizer, matrix = build_feature_matrix(prepared["tags"].tolist())
    save_artifacts(
        prepared,
        vectorizer,
        matrix,
        artifacts_dir,
        {**load_summary, **preprocessing_summary},
        max_features=5000,
    )
    loaded_movies, loaded_vectorizer, loaded_matrix = load_artifacts(artifacts_dir)
    recommender = MovieRecommender(loaded_movies, loaded_vectorizer, loaded_matrix)

    results = recommender.recommend(101, k=5)

    assert results["id"].tolist() == [102]
    assert results.iloc[0]["similarity"] > 0
    assert 101 not in results["id"].tolist()
