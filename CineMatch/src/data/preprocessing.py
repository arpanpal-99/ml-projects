from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
STRUCTURED_COLUMNS = ("genres", "keywords", "cast", "crew")
WHITESPACE = re.compile(r"\s+")


def parse_json_list(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if value is None:
        return [], False
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return [], False
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [], True
    else:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return [], False
        return [], True

    if not isinstance(parsed, list):
        return [], True
    return ([item for item in parsed if isinstance(item, Mapping)], False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    missing = pd.isna(value)
    if not hasattr(missing, "__len__") and bool(missing):
        return ""
    return WHITESPACE.sub(" ", str(value).strip().lower())


def normalize_name(value: Any) -> str:
    return "_".join(normalize_text(value).split())


def _names(items: list[dict[str, Any]], limit: int | None = None) -> list[str]:
    selected = items if limit is None else items[:limit]
    return [
        normalize_name(item.get("name")) for item in selected if normalize_name(item.get("name"))
    ]


def preprocess_movies(
    merged: pd.DataFrame,
    principal_cast_size: int = 5,
    max_parse_failure_rate: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if principal_cast_size < 1:
        raise ValueError("principal_cast_size must be positive.")
    if not 0 <= max_parse_failure_rate <= 1:
        raise ValueError("max_parse_failure_rate must be between 0 and 1.")

    movies = merged.copy()
    failures = {column: 0 for column in STRUCTURED_COLUMNS}
    parsed_columns: dict[str, list[list[dict[str, Any]]]] = {}
    for column in STRUCTURED_COLUMNS:
        values: list[list[dict[str, Any]]] = []
        if column not in movies:
            values = [[] for _ in range(len(movies))]
        else:
            for value in movies[column].tolist():
                parsed, failed = parse_json_list(value)
                values.append(parsed)
                failures[column] += int(failed)
        parsed_columns[column] = values

    total_fields = max(1, len(movies) * len(STRUCTURED_COLUMNS))
    failure_count = sum(failures.values())
    failure_rate = failure_count / total_fields
    if failure_rate > max_parse_failure_rate:
        raise ValueError(
            f"Malformed structured metadata rate {failure_rate:.1%} exceeds the "
            f"configured limit {max_parse_failure_rate:.1%}; failures by column: "
            f"{failures}. Inspect the source CSV or adjust "
            "CINEMATCH_MAX_PARSE_FAILURE_RATE deliberately."
        )
    for column, count in failures.items():
        if count:
            LOGGER.warning("Treated %d malformed %s values as empty lists.", count, column)

    genres: list[list[str]] = []
    keywords: list[list[str]] = []
    cast: list[list[str]] = []
    directors: list[list[str]] = []
    tags: list[str] = []
    for index in range(len(movies)):
        genre_names = _names(parsed_columns["genres"][index])
        keyword_names = _names(parsed_columns["keywords"][index])
        cast_names = _names(parsed_columns["cast"][index], principal_cast_size)
        director_names = [
            normalize_name(item.get("name"))
            for item in parsed_columns["crew"][index]
            if item.get("job") == "Director" and normalize_name(item.get("name"))
        ]
        overview = normalize_text(movies.iloc[index].get("overview"))

        genres.append(genre_names)
        keywords.append(keyword_names)
        cast.append(cast_names)
        directors.append(director_names)
        tags.append(
            " ".join(
                part
                for part in (
                    " ".join(genre_names),
                    " ".join(keyword_names),
                    " ".join(cast_names),
                    " ".join(director_names),
                    overview,
                )
                if part
            )
        )

    movies["genres_list"] = genres
    movies["keywords_list"] = keywords
    movies["cast_names"] = cast
    movies["director_names"] = directors
    movies["tags"] = tags
    summary = {
        "parse_failures_by_column": failures,
        "parse_failure_rate": failure_rate,
        "empty_tag_rows": int(movies["tags"].str.strip().eq("").sum()),
        "missing_value_rates": {
            column: float(movies[column].isna().mean())
            for column in ("overview", "poster_path", "release_date")
            if column in movies
        },
    }
    return movies, summary
