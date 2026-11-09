"""Small inspection helpers for example recommendation outputs."""

from __future__ import annotations

import pandas as pd

from src.recommender.engine import MovieRecommender


def example_recommendations(
    recommender: MovieRecommender, movie_ids: list[int], k: int = 5
) -> pd.DataFrame:
    """Return inspectable examples; this is not a relevance or accuracy metric."""
    records: list[dict[str, object]] = []
    for movie_id in movie_ids:
        seed = recommender.get_movie(movie_id)
        if seed is None:
            continue
        recommendations = recommender.recommend(movie_id, k).to_dict("records")
        for rank, recommendation in enumerate(recommendations, 1):
            records.append(
                {
                    "seed_id": movie_id,
                    "seed_title": seed["title"],
                    "rank": rank,
                    "movie_id": recommendation["id"],
                    "title": recommendation["title"],
                    "similarity": recommendation["similarity"],
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "seed_id",
            "seed_title",
            "rank",
            "movie_id",
            "title",
            "similarity",
        ],
    )
