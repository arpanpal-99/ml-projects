"""A small, teachable content-based movie recommender built with the TMDB 5000 data."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_DIR = Path(__file__).resolve().parent
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"
TMDB_ENV_FILE = PROJECT_DIR.parent / "CineMatch" / ".env"
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


def _find_dataset() -> tuple[Path, Path]:
    """Find both dataset CSVs in this project."""
    data_dirs = (PROJECT_DIR / "data" / "raw", PROJECT_DIR / "data")
    for data_dir in data_dirs:
        movies_path = data_dir / MOVIES_FILE
        credits_path = data_dir / CREDITS_FILE
        if movies_path.is_file() and credits_path.is_file():
            return movies_path, credits_path
    expected = PROJECT_DIR / "data" / "raw"
    raise FileNotFoundError(
        f"Could not find both dataset CSVs. Put {MOVIES_FILE} and {CREDITS_FILE} "
        f"in {expected}."
    )


def _get_tmdb_api_key() -> str | None:
    """Read the TMDB API key from the environment or neighboring CineMatch .env."""
    load_dotenv(TMDB_ENV_FILE, override=False)
    return os.getenv("TMDB_API_KEY")


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_tmdb_details(movie_id: int, _api_key: str) -> dict[str, Any]:
    """Fetch TMDB metadata for one movie without exposing credentials in errors."""
    try:
        response = requests.get(
            f"{TMDB_API_BASE}/movie/{movie_id}",
            params={"api_key": _api_key},
            timeout=(3, 10),
        )
        response.raise_for_status()
        details = response.json()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        message = f"TMDB details request failed (HTTP {status})." if status else (
            "TMDB details request failed due to a network error."
        )
        raise RuntimeError(message) from exc
    except ValueError as exc:
        raise RuntimeError("TMDB returned an invalid details response.") from exc

    if not isinstance(details, dict) or details.get("id") != movie_id:
        raise RuntimeError("TMDB returned details for an unexpected movie.")
    return details


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_poster(poster_url: str) -> bytes:
    """Download a poster image, validating the response before displaying it."""
    try:
        response = requests.get(poster_url, timeout=(3, 10))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("The movie poster could not be downloaded.") from exc
    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise RuntimeError("TMDB returned an invalid poster image.")
    return response.content


def _parse_metadata(value: Any) -> list[dict[str, Any]]:
    """Turn a CSV JSON cell into dictionaries; malformed or missing cells become empty."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # literal_eval accepts Python-style list strings without executing arbitrary code.
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def _names(value: Any, *, limit: int | None = None) -> list[str]:
    """Extract normalized names from a genres, keywords, or cast JSON cell."""
    names = [
        "_".join(str(item.get("name", "")).lower().split())
        for item in _parse_metadata(value)
    ]
    names = [name for name in names if name]
    return names if limit is None else names[:limit]


def _original_names(value: Any, *, limit: int | None = None) -> list[str]:
    """Extract names as written in the source metadata for display."""
    names = [
        str(item["name"]).strip()
        for item in _parse_metadata(value)
        if item.get("name")
    ]
    return names if limit is None else names[:limit]


def _directors(value: Any) -> list[str]:
    """Extract director names from the credits crew JSON cell."""
    return [
        "_".join(str(person.get("name", "")).lower().split())
        for person in _parse_metadata(value)
        if person.get("job") == "Director" and person.get("name")
    ]


def _original_directors(value: Any) -> list[str]:
    """Extract director names without changing their capitalization or spacing."""
    return [
        str(person["name"]).strip()
        for person in _parse_metadata(value)
        if person.get("job") == "Director" and person.get("name")
    ]


@st.cache_resource(show_spinner="Preparing movie features...")
def load_movies_and_features() -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """Join the two CSVs, engineer text tags, and create sparse word-count vectors."""
    movies_path, credits_path = _find_dataset()
    movies = pd.read_csv(movies_path, low_memory=False)
    credits = pd.read_csv(credits_path, low_memory=False)

    required_movie_columns = {"id", "title", "genres", "keywords", "overview"}
    required_credit_columns = {"movie_id", "cast", "crew"}
    if not required_movie_columns.issubset(movies.columns):
        missing = sorted(required_movie_columns - set(movies.columns))
        raise ValueError(f"Movies CSV is missing required columns: {', '.join(missing)}.")
    if not required_credit_columns.issubset(credits.columns):
        missing = sorted(required_credit_columns - set(credits.columns))
        raise ValueError(f"Credits CSV is missing required columns: {', '.join(missing)}.")

    # Keep only the credit columns needed for features, avoiding duplicate title fields.
    catalogue = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        how="left",
        left_on="id",
        right_on="movie_id",
        validate="one_to_one",
        sort=False,
    )

    # A movie is one text document. Genres, keywords, leading cast, director(s),
    # and overview are deliberately combined so the feature engineering is visible.
    catalogue["genres_list"] = catalogue["genres"].map(_names)
    catalogue["keywords_list"] = catalogue["keywords"].map(_names)
    catalogue["cast_names"] = catalogue["cast"].map(lambda value: _names(value, limit=3))
    catalogue["director_names"] = catalogue["crew"].map(_directors)
    catalogue["tags"] = catalogue.apply(
        lambda movie: " ".join(
            [
                *movie["genres_list"],
                *movie["keywords_list"],
                *movie["cast_names"],
                *movie["director_names"],
                str(movie["overview"]).lower()
                if pd.notna(movie["overview"])
                else "",
            ]
        ),
        axis=1,
    )

    # CountVectorizer learns a vocabulary and turns every movie's tags into a
    # sparse vector of word counts. Common English filler words are omitted.
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    try:
        feature_matrix = vectorizer.fit_transform(catalogue["tags"]).tocsr()
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            raise ValueError("No useful text was found in the movie metadata.") from exc
        raise

    catalogue["year"] = pd.to_datetime(
        catalogue["release_date"], errors="coerce"
    ).dt.year
    return catalogue, feature_matrix


def recommend(
    movie_index: int,
    movies: pd.DataFrame,
    feature_matrix: sparse.csr_matrix,
    count: int = 5,
) -> pd.DataFrame:
    """Rank movies by cosine similarity, excluding the selected movie itself."""
    # Only one row of similarity scores is needed. This avoids building the much
    # larger all-movies-by-all-movies similarity matrix.
    scores = cosine_similarity(feature_matrix[movie_index], feature_matrix).ravel()
    candidate_mask = (np.arange(len(movies)) != movie_index) & (scores > 0)
    candidate_indices = np.flatnonzero(candidate_mask)
    if not len(candidate_indices):
        return pd.DataFrame()

    titles = movies["title"].astype(str).str.casefold().to_numpy()
    movie_ids = movies["id"].to_numpy()
    order = np.lexsort(
        (movie_ids[candidate_indices], titles[candidate_indices], -scores[candidate_indices])
    )
    selected = candidate_indices[order[:count]]

    results = movies.iloc[selected][
        ["id", "title", "year", "genres_list", "vote_average"]
    ].copy()
    results["similarity"] = scores[selected]
    results["genres"] = results["genres_list"].map(
        lambda genres: ", ".join(genre.replace("_", " ").title() for genre in genres)
    )
    return results.drop(columns="genres_list").reset_index(drop=True)


def main() -> None:
    """Render the minimal movie picker and recommendation results."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 12% 8%, rgba(255, 91, 156, .34), transparent 38%),
                radial-gradient(ellipse at 88% 18%, rgba(85, 115, 255, .34), transparent 40%),
                radial-gradient(ellipse at 55% 92%, rgba(24, 196, 180, .25), transparent 42%),
                linear-gradient(135deg, #101225 0%, #171a36 48%, #111629 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: rgba(16, 18, 37, .35);
        }
        .stButton button[kind="tertiary"] {
            color: #9bd1ff;
            justify-content: flex-start;
            text-decoration: underline;
            text-underline-offset: 3px;
        }
        .stButton button[kind="tertiary"]:hover {
            color: #ffffff;
            background: rgba(155, 209, 255, .12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🎬 Movie Recommendation System")
    st.write(
        "Choose a movie to find others with similar genres, keywords, cast, "
        "directors, and plot descriptions."
    )

    try:
        movies, feature_matrix = load_movies_and_features()
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    labels = movies.apply(
        lambda movie: (
            f"{movie['title']} ({int(movie['year'])})"
            if pd.notna(movie["year"])
            else str(movie["title"])
        ),
        axis=1,
    )
    selected_index = st.selectbox(
        "Select a movie",
        options=range(len(movies)),
        format_func=lambda index: labels.iloc[index],
    )
    result_count = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

    if st.button("Recommend movies", type="primary"):
        results = recommend(selected_index, movies, feature_matrix, result_count)
        st.session_state["recommendation_context"] = (selected_index, result_count)
        st.session_state["recommendation_results"] = results
        st.session_state.pop("selected_movie_details_id", None)

    if st.session_state.get("recommendation_context") == (
        selected_index,
        result_count,
    ):
        results = st.session_state["recommendation_results"]
        if results.empty:
            st.info("No movies with shared text features were found.")
            return

        st.divider()
        st.subheader(f"Movies similar to {labels.iloc[selected_index]}")
        display = results.rename(
            columns={
                "title": "Movie",
                "year": "Year",
                "vote_average": "Rating / 10",
                "genres": "Genres",
            }
        )
        display = display.drop(columns="id")
        headers = ["Movie", "Year", "Rating / 10", "Genres"]
        widths = [3, 0.8, 1, 2.4]
        for column, heading in zip(st.columns(widths), headers):
            column.markdown(f"**{heading}**")
        for row_index, row in display.iterrows():
            columns = st.columns(widths)
            movie_id = int(results.iloc[row_index]["id"])
            if columns[0].button(
                f"{row['Movie']}  ›",
                key=f"movie-details-{selected_index}-{result_count}-{movie_id}",
                type="tertiary",
                help="Show movie details",
            ):
                st.session_state["selected_movie_details_id"] = movie_id
            columns[1].write(str(int(row["Year"])) if pd.notna(row["Year"]) else "N/A")
            columns[2].write(
                f"{row['Rating / 10']:.1f}"
                if pd.notna(row["Rating / 10"])
                else "N/A"
            )
            columns[3].write(row["Genres"] or "N/A")
            if row_index < len(display) - 1:
                st.markdown(
                    "<hr style='margin: 0.35rem 0; border-color: "
                    "rgba(255, 255, 255, 0.18);'>",
                    unsafe_allow_html=True,
                )
        st.caption(
            "Similarity is based on shared text features; it is not a prediction "
            "of how much you will enjoy a movie."
        )

        details_id = st.session_state.get("selected_movie_details_id")
        result_ids = set(results["id"].astype(int))
        if details_id not in result_ids:
            details_id = None
        if details_id is not None:
            movie = movies.loc[movies["id"] == details_id].iloc[0]
            st.divider()
            st.subheader(str(movie["title"]))

            api_key = _get_tmdb_api_key()
            tmdb_details: dict[str, Any] = {}
            if not api_key:
                st.warning(
                    f"TMDB_API_KEY was not found in the environment or {TMDB_ENV_FILE}."
                )
            else:
                try:
                    tmdb_details = _fetch_tmdb_details(int(details_id), api_key)
                except RuntimeError as exc:
                    st.warning(str(exc))

            poster_path = tmdb_details.get("poster_path") or movie.get("poster_path")
            poster_url = (
                f"{TMDB_IMAGE_BASE}/{str(poster_path).lstrip('/')}"
                if isinstance(poster_path, str) and poster_path.strip()
                else None
            )
            detail_columns = st.columns([1, 2.5])
            with detail_columns[0]:
                if poster_url:
                    try:
                        st.image(_fetch_poster(poster_url), width=260)
                    except RuntimeError as exc:
                        st.warning(str(exc))
                else:
                    st.info("No poster is available for this movie.")

            with detail_columns[1]:
                overview = tmdb_details.get("overview") or movie.get("overview")
                st.write(
                    str(overview)
                    if isinstance(overview, str) and overview.strip()
                    else "No overview is available for this movie."
                )

                release_date = (
                    tmdb_details.get("release_date") or movie.get("release_date")
                )
                release_year = (
                    str(release_date)[:4]
                    if isinstance(release_date, str) and release_date
                    else "N/A"
                )
                runtime = tmdb_details.get("runtime") or movie.get("runtime")
                rating = tmdb_details.get("vote_average", movie.get("vote_average"))
                vote_count = tmdb_details.get("vote_count", movie.get("vote_count"))
                metrics = st.columns(4)
                metrics[0].metric("Release year", release_year)
                metrics[1].metric(
                    "Runtime",
                    f"{runtime} min" if pd.notna(runtime) and runtime else "N/A",
                )
                metrics[2].metric(
                    "Rating",
                    f"{rating:.1f} / 10" if pd.notna(rating) else "N/A",
                )
                metrics[3].metric(
                    "Votes",
                    f"{int(vote_count):,}" if pd.notna(vote_count) else "N/A",
                )

                api_genres = tmdb_details.get("genres")
                genres = (
                    [str(genre["name"]) for genre in api_genres if genre.get("name")]
                    if isinstance(api_genres, list)
                    else [genre.replace("_", " ").title() for genre in movie["genres_list"]]
                )
                directors = _original_directors(movie.get("crew"))
                cast = _original_names(movie.get("cast"), limit=3)
                st.markdown(
                    f"**Genres:** {', '.join(genres) or 'N/A'}  \n"
                    f"**Director(s):** {', '.join(directors) or 'N/A'}  \n"
                    f"**Cast:** {', '.join(cast) or 'N/A'}"
                )


if __name__ == "__main__":
    main()
