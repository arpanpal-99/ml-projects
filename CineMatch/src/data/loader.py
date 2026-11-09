from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CREDITS_FILENAME, MOVIES_FILENAME

LOGGER = logging.getLogger(__name__)

MOVIE_COLUMNS = {"id", "title", "genres", "keywords", "overview"}
CREDIT_COLUMNS = {"movie_id", "cast", "crew"}


class DataValidationError(ValueError):
    """Raised when source data does not satisfy the documented data contract."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required dataset not found: {path}. Download the TMDB 5000 Movies "
            f"and Credits CSV files and place them in data/raw/; see README.md."
        )
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not read CSV file {path}: {exc}") from exc


def _validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"{label} CSV is missing required columns: {', '.join(missing)}.")


def _canonicalize_ids(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    result = frame.copy()
    numeric_ids = pd.to_numeric(result[column], errors="coerce")
    invalid = numeric_ids.isna() | (numeric_ids % 1 != 0)
    if invalid.any():
        examples = result.loc[invalid, column].head(5).tolist()
        raise DataValidationError(
            f"{label} contains {int(invalid.sum())} invalid movie IDs "
            f"(examples: {examples}). Correct the source CSV; IDs cannot be guessed."
        )
    result[column] = numeric_ids.astype("int64")
    duplicates = result[column].duplicated(keep=False)
    if duplicates.any():
        examples = result.loc[duplicates, column].drop_duplicates().head(5).tolist()
        raise DataValidationError(
            f"{label} contains duplicate movie IDs (examples: {examples}). "
            "Resolve source duplicates explicitly before building artifacts."
        )
    return result


def load_and_merge_data(
    raw_data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    movies = _read_csv(raw_data_dir / MOVIES_FILENAME)
    credits = _read_csv(raw_data_dir / CREDITS_FILENAME)
    _validate_columns(movies, MOVIE_COLUMNS, "Movies")
    _validate_columns(credits, CREDIT_COLUMNS, "Credits")

    movies = _canonicalize_ids(movies, "id", "Movies")
    credits = _canonicalize_ids(credits, "movie_id", "Credits")

    credits_small = credits[["movie_id", "cast", "crew"]].copy()
    merged = movies.merge(
        credits_small,
        how="left",
        left_on="id",
        right_on="movie_id",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    unmatched_movie_ids = [
        int(movie_id) for movie_id in merged.loc[merged["_merge"] == "left_only", "id"].tolist()
    ]
    unmatched_credit_ids = [
        int(movie_id)
        for movie_id in credits.loc[~credits["movie_id"].isin(movies["id"]), "movie_id"].tolist()
    ]
    unmatched_movie_count = len(unmatched_movie_ids)
    unmatched_credit_count = len(unmatched_credit_ids)
    merged = merged.drop(columns=["movie_id", "_merge"])
    merged = merged.loc[merged["title"].notna()].copy()
    merged["title"] = merged["title"].astype(str).str.strip()
    blank_title = merged["title"].eq("")
    excluded_title_count = int(blank_title.sum())
    merged = merged.loc[~blank_title].reset_index(drop=True)

    if merged.empty:
        raise DataValidationError(
            "No movies with both a valid ID and a non-empty title remain after validation."
        )

    diagnostics = {
        "movies_input_rows": len(movies),
        "credits_input_rows": len(credits),
        "merged_rows_before_title_filter": len(movies),
        "duplicate_movie_ids": 0,
        "duplicate_credit_movie_ids": 0,
        "movies_without_matching_credits": unmatched_movie_count,
        "unmatched_movie_ids": unmatched_movie_ids,
        "credits_without_matching_movie": unmatched_credit_count,
        "unmatched_credit_movie_ids": unmatched_credit_ids,
        "excluded_missing_or_blank_titles": excluded_title_count,
        "merged_rows": len(merged),
    }
    LOGGER.info(
        "Loaded %d movies; %d movie rows lack credits and %d credit rows lack a movie.",
        len(merged),
        unmatched_movie_count,
        unmatched_credit_count,
    )
    return merged, diagnostics
