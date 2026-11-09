from __future__ import annotations

import pandas as pd
import pytest

from src.features.feature_engineering import build_feature_matrix
from src.recommender.engine import MovieRecommender


@pytest.fixture
def recommender():
    movies = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Space One",
                "release_date": "2020-01-01",
                "tags": "space galaxy journey",
            },
            {
                "id": 2,
                "title": "Space Two",
                "release_date": "2019-01-01",
                "tags": "space galaxy travel",
            },
            {
                "id": 3,
                "title": "Garden",
                "release_date": "2021-01-01",
                "tags": "garden romance flowers",
            },
            {"id": 4, "title": "No words", "release_date": "2018-01-01", "tags": "the and or"},
        ]
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    return MovieRecommender(movies, vectorizer, matrix)


@pytest.fixture
def discovery_recommender():
    movies = pd.DataFrame(
        [
            {
                "id": 10,
                "title": "Orbit",
                "release_date": "2020-01-01",
                "tags": "space pilot mission galaxy",
                "genres_list": ["science_fiction", "adventure"],
                "keywords_list": ["space", "mission"],
                "cast_names": ["alex_star"],
                "director_names": ["sky_director"],
                "vote_average": 7.4,
                "vote_count": 1200,
                "popularity": 70,
            },
            {
                "id": 11,
                "title": "Orbit Again",
                "release_date": "2021-01-01",
                "tags": "space pilot mission galaxy sequel",
                "genres_list": ["science_fiction", "adventure"],
                "keywords_list": ["space", "mission"],
                "cast_names": ["alex_star"],
                "director_names": ["sky_director"],
                "vote_average": 8.1,
                "vote_count": 450,
                "popularity": 82,
            },
            {
                "id": 12,
                "title": "One Perfect Vote",
                "release_date": "2022-01-01",
                "tags": "space astronaut mission",
                "genres_list": ["science_fiction"],
                "keywords_list": ["space"],
                "cast_names": ["different_star"],
                "director_names": ["new_director"],
                "vote_average": 10.0,
                "vote_count": 1,
                "popularity": 1,
            },
            {
                "id": 13,
                "title": "Quiet Stars",
                "release_date": "2019-01-01",
                "tags": "space galaxy astronaut mission",
                "genres_list": ["science_fiction"],
                "keywords_list": ["space", "astronaut"],
                "cast_names": ["casey_star"],
                "director_names": ["indie_director"],
                "vote_average": 8.8,
                "vote_count": 180,
                "popularity": 10,
            },
            {
                "id": 14,
                "title": "City Lights",
                "release_date": "2018-01-01",
                "tags": "city romance musician",
                "genres_list": ["romance"],
                "keywords_list": ["city"],
                "cast_names": ["romance_star"],
                "director_names": ["heart_director"],
                "vote_average": 7.8,
                "vote_count": 800,
                "popularity": 96,
            },
        ]
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    return MovieRecommender(movies, vectorizer, matrix)


def test_recommendations_exclude_seed_and_are_ranked(recommender):
    results = recommender.recommend(1, k=5)

    assert results["id"].tolist() == [2]
    assert results["similarity"].is_monotonic_decreasing
    assert 1 not in results["id"].tolist()


def test_empty_seed_vector_returns_no_arbitrary_matches(recommender):
    assert recommender.recommend(4).empty


def test_search_disambiguates_duplicate_titles_by_year_and_id(recommender):
    movies = pd.concat(
        [
            recommender.movies,
            pd.DataFrame(
                [
                    {
                        "id": 5,
                        "title": "Space One",
                        "release_date": "2022-05-01",
                        "tags": "space galaxy trip",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    updated = MovieRecommender(movies, vectorizer, matrix)

    found = updated.find_movies("space one")

    assert found["id"].tolist() == [5, 1]


def test_search_suggests_prefix_matches_then_best_rated_contains(recommender):
    movies = pd.DataFrame(
        [
            {
                "id": 10,
                "title": "Space: The Journey",
                "release_date": "2021-01-01",
                "tags": "space journey",
                "vote_average": 6.0,
                "vote_count": 500,
            },
            {
                "id": 11,
                "title": "The Space Adventure",
                "release_date": "2022-01-01",
                "tags": "space adventure",
                "vote_average": 9.0,
                "vote_count": 120,
            },
            {
                "id": 12,
                "title": "Space Legends",
                "release_date": "2020-01-01",
                "tags": "space legends",
                "vote_average": 8.5,
                "vote_count": 400,
            },
            {
                "id": 13,
                "title": "The Space Story",
                "release_date": "2023-01-01",
                "tags": "space story",
                "vote_average": 9.0,
                "vote_count": 300,
            },
        ]
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    updated = MovieRecommender(movies, vectorizer, matrix)

    suggestions = updated.find_movies("space")

    assert suggestions["id"].tolist() == [12, 10, 13, 11]


def test_invalid_arguments_are_explicit(recommender):
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend(1, k=0)
    with pytest.raises(KeyError, match="not found"):
        recommender.recommend(999)


def test_top_movies_shrink_a_single_vote_outlier_and_filter_by_genre(discovery_recommender):
    top = discovery_recommender.top_movies(k=5)
    science_fiction = discovery_recommender.top_movies(k=5, genre="science fiction")

    assert top.iloc[0]["id"] != 12
    assert 12 not in discovery_recommender.top_movies(k=5, min_votes=2)["id"].tolist()
    assert science_fiction["id"].tolist()
    assert all(
        "science_fiction" in movie_genres
        for movie_genres in science_fiction["genres_list"].tolist()
    )


def test_popular_and_hybrid_recommendations_are_ranked_and_explainable(discovery_recommender):
    popular = discovery_recommender.popular_movies(k=2)
    recommendations = discovery_recommender.recommend_for_movies([10], k=3, mode="balanced")
    reasons = discovery_recommender.match_reasons([10], 11)

    assert popular.iloc[0]["id"] == 14
    assert 10 not in recommendations["id"].tolist()
    assert recommendations["recommendation_mode"].eq("balanced").all()
    assert recommendations["recommendation_score"].is_monotonic_decreasing
    assert any(reason.startswith("Shared genre:") for reason in reasons)
    assert any(reason.startswith("Same director:") for reason in reasons)


def test_profile_recommendations_exclude_all_liked_titles_and_validate_options(discovery_recommender):
    recommendations = discovery_recommender.recommend_for_movies(
        [10, 11],
        k=5,
        genre="science_fiction",
        mode="hidden_gems",
    )

    assert not {10, 11}.intersection(recommendations["id"].tolist())
    assert recommendations["recommendation_mode"].eq("hidden_gems").all()
    with pytest.raises(ValueError, match="mode"):
        discovery_recommender.recommend_for_movies([10], mode="unknown")
    with pytest.raises(ValueError, match="at least one"):
        discovery_recommender.recommend_for_movies([])
