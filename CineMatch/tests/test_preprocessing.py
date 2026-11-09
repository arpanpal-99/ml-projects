from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import normalize_name, parse_json_list, preprocess_movies


def test_parses_metadata_and_builds_deterministic_tags(raw_frames):
    movies, credits = raw_frames
    merged = movies.merge(credits, left_on="id", right_on="movie_id")

    processed, summary = preprocess_movies(merged, max_parse_failure_rate=0.2)

    assert processed.loc[0, "tags"] == (
        "science_fiction space_travel alex_actor casey_star sam_director "
        "a crew explores a distant galaxy."
    )
    assert processed.loc[0, "director_names"] == ["sam_director"]
    assert summary["parse_failures_by_column"]["cast"] == 1
    assert summary["empty_tag_rows"] == 0


def test_parser_never_evaluates_malformed_values():
    assert parse_json_list("not-json") == ([], True)
    assert parse_json_list("null") == ([], True)
    assert normalize_name("  Alex   Actor ") == "alex_actor"


def test_parse_failure_threshold_is_enforced(raw_frames):
    movies, credits = raw_frames
    merged = movies.merge(credits, left_on="id", right_on="movie_id")

    with pytest.raises(ValueError, match="exceeds the configured limit"):
        preprocess_movies(merged, max_parse_failure_rate=0.01)


def test_missing_metadata_is_not_converted_to_nan_text():
    frame = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "No details",
                "genres": None,
                "keywords": None,
                "cast": None,
                "crew": None,
                "overview": None,
            }
        ]
    )

    processed, summary = preprocess_movies(frame)

    assert processed.loc[0, "tags"] == ""
    assert summary["empty_tag_rows"] == 1
    assert "nan" not in processed.loc[0, "tags"]
