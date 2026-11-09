"""Shared pytest fixtures providing small, offline movie and credits datasets."""

from __future__ import annotations

import json

import pandas as pd
import pytest


@pytest.fixture
def raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return three synthetic movie rows and their corresponding credits."""
    movies = pd.DataFrame(
        [
            {
                "id": 101,
                "title": "Star Journey",
                "genres": json.dumps([{"id": 1, "name": "Science Fiction"}]),
                "keywords": json.dumps([{"id": 2, "name": "space travel"}]),
                "overview": "A crew explores a distant galaxy.",
                "release_date": "2020-01-01",
                "poster_path": None,
                "popularity": 80.0,
            },
            {
                "id": 102,
                "title": "Galaxy Crew",
                "genres": json.dumps([{"id": 1, "name": "Science Fiction"}]),
                "keywords": json.dumps([{"id": 2, "name": "space travel"}]),
                "overview": "A crew journeys across a galaxy.",
                "release_date": "2019-01-01",
                "poster_path": "/galaxy.jpg",
                "popularity": 60.0,
            },
            {
                "id": 103,
                "title": "Garden Days",
                "genres": json.dumps([{"id": 3, "name": "Romance"}]),
                "keywords": json.dumps([{"id": 4, "name": "garden"}]),
                "overview": "Two people meet in a quiet garden.",
                "release_date": "2018-01-01",
                "poster_path": None,
                "popularity": 40.0,
            },
        ]
    )
    credits = pd.DataFrame(
        [
            {
                "movie_id": 101,
                "cast": json.dumps(
                    [
                        {"id": 10, "name": "Alex Actor"},
                        {"id": 11, "name": "Casey Star"},
                    ]
                ),
                "crew": json.dumps([{"id": 12, "name": "Sam Director", "job": "Director"}]),
            },
            {
                "movie_id": 102,
                "cast": json.dumps([{"id": 10, "name": "Alex Actor"}]),
                "crew": json.dumps([{"id": 12, "name": "Sam Director", "job": "Director"}]),
            },
            {
                "movie_id": 103,
                "cast": "not valid json",
                "crew": "[]",
            },
        ]
    )
    return movies, credits
