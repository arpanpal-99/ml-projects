"""CineMatch: a polished, local-first movie discovery experience."""

from __future__ import annotations

import html
import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from ml_pipeline import load_recommender as load_movie_recommender
from ml_pipeline import recommend_movies
from src.recommender.artifacts import ArtifactError
from src.recommender.engine import MovieRecommender
from src.services.tmdb_client import TMDBClient
from src.ui_components import (
    apply_product_styles,
    display_name,
    metadata_list,
    movie_year,
    numeric_value,
    poster_url,
    render_movie_card,
    render_movie_details,
)
from src.utils.logging_config import configure_logging

load_dotenv()
configure_logging()

st.set_page_config(
    page_title="CineMatch — movie recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_recommender() -> MovieRecommender:
    """Cache the model between Streamlit reruns; model loading lives in ml_pipeline."""
    return load_movie_recommender()


@st.cache_resource
def load_tmdb_client() -> TMDBClient:
    """Create the cached optional TMDB client from environment or Streamlit secrets."""
    api_key = os.getenv("TMDB_API_KEY")
    secrets_files = (
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if not api_key and any(path.is_file() for path in secrets_files):
        api_key = st.secrets.get("TMDB_API_KEY")
    return TMDBClient(api_key=api_key)


# Streamlit reruns this module after interactions, so transient choices are stored
# in session_state rather than module globals or a user database.
_STATE_DEFAULTS: dict[str, object] = {
    "active_movie_id": None,
    "show_next_watch": False,
    "details_movie_id": None,
    "details_return_target": "recommendations",
    "saved_movie_ids": [],
    "liked_movie_ids": [],
    "recent_movie_ids": [],
    "recent_history_started": False,
    "discovery_seed": None,
    "surprise_counter": 0,
    "scroll_target": None,
    "scroll_request_id": 0,
    "notice": None,
    "loading_action": None,
    "details_open_request": False,
    "picker_value_set_by_activation": None,
}

_LOADING_PRESENTATIONS: dict[str, tuple[str, str]] = {
    "picker-details": ("Getting the full picture...", "selection"),
    "card-details": ("Unfolding the plot...", "details"),
    "genre-selection": ("Shuffling the genre deck...", "genre"),
    "similar-movies": ("Following the movie trail...", "recommendations"),
    "surprise-me": ("Summoning a little movie magic...", "selection"),
    "style-balanced": ("Striking the perfect balance...", "recommendations"),
    "style-closest": ("Tracking down cinematic twins...", "recommendations"),
    "style-hidden-gems": ("Digging up cinematic treasures...", "recommendations"),
}


def _init_session_state() -> None:
    """Initialize each session key once, copying mutable defaults per session."""
    for key, value in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def _record(movie: pd.Series | dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert a catalogue row into a plain dictionary for UI rendering."""
    if movie is None:
        return None
    return movie.to_dict() if isinstance(movie, pd.Series) else dict(movie)


def _movie(recommender: MovieRecommender, movie_id: int) -> dict[str, Any] | None:
    """Look up one catalogue movie and return its UI-friendly record."""
    return _record(recommender.get_movie(int(movie_id)))


def _movie_label(movie_id: int, recommender: MovieRecommender) -> str:
    """Format a search option with title, year, and rating for disambiguation."""
    movie = _movie(recommender, movie_id)
    if movie is None:
        return f"Unknown movie ({movie_id})"
    title = str(movie.get("title", "Untitled movie"))
    rating = numeric_value(movie, "vote_average")
    rating_text = f"★ {rating:.1f}/10" if rating is not None else "Rating unavailable"
    return f"{title} · {movie_year(movie)} · {rating_text}"


def _valid_ids(recommender: MovieRecommender, state_key: str) -> list[int]:
    """Drop invalid, duplicate, or no-longer-catalogued IDs from session state."""
    available = {int(movie_id) for movie_id in recommender.movies["id"].tolist()}
    valid: list[int] = []
    for raw_id in st.session_state.get(state_key, []):
        try:
            movie_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if movie_id in available and movie_id not in valid:
            valid.append(movie_id)
    if valid != st.session_state.get(state_key, []):
        st.session_state[state_key] = valid
    return valid


def _remember_recent(movie_id: int) -> None:
    """Move a selected movie to the front of the short session history."""
    current = [int(item) for item in st.session_state.get("recent_movie_ids", [])]
    st.session_state["recent_movie_ids"] = [movie_id] + [
        item for item in current if item != movie_id
    ][:5]
    st.session_state["recent_history_started"] = True


def _set_notice(message: str) -> None:
    """Queue a one-time notice for display on the next Streamlit rerun."""
    st.session_state["notice"] = message


def _activate_movie(movie_id: int) -> None:
    """Atomically make a movie the active recommendation seed."""
    movie_id = int(movie_id)
    st.session_state["loading_action"] = "similar-movies"
    st.session_state["active_movie_id"] = movie_id
    st.session_state["show_next_watch"] = True
    if st.session_state.get("movie-picker") != movie_id:
        st.session_state["picker_value_set_by_activation"] = movie_id
    st.session_state["movie-picker"] = movie_id
    st.session_state["details_movie_id"] = None
    st.session_state["scroll_target"] = "recommendations"
    _remember_recent(movie_id)
    _set_notice("Your recommendations are ready.")


def _activate_selected_movie() -> None:
    """Use the current search selection as the recommendation seed."""
    selected_id = st.session_state.get("movie-picker")
    if selected_id is not None:
        _activate_movie(int(selected_id))


def _open_selected_movie_details() -> None:
    """Open the selected title's profile unless the picker changed programmatically."""
    selected_id = st.session_state.get("movie-picker")
    activated_picker_value = st.session_state.pop("picker_value_set_by_activation", None)
    if selected_id is not None and selected_id == activated_picker_value:
        return
    st.session_state["show_next_watch"] = False
    st.session_state["active_movie_id"] = None
    if selected_id is None:
        st.session_state["loading_action"] = None
        st.session_state["details_movie_id"] = None
        st.session_state["details_open_request"] = False
        return
    _remember_recent(int(selected_id))
    st.session_state["loading_action"] = "picker-details"
    st.session_state["details_movie_id"] = int(selected_id)
    st.session_state["details_open_request"] = True


def _open_movie_details(movie_id: int, return_target: str) -> None:
    """Queue a card-triggered details dialog and remember its originating section."""
    st.session_state["loading_action"] = "card-details"
    st.session_state["details_movie_id"] = int(movie_id)
    st.session_state["details_return_target"] = return_target
    st.session_state["details_open_request"] = True


def _toggle_collection(collection_key: str, movie_id: int, add_message: str, remove_message: str) -> None:
    """Add or remove a movie ID in a named session-only collection."""
    movie_id = int(movie_id)
    ids = [int(item) for item in st.session_state.get(collection_key, [])]
    if movie_id in ids:
        st.session_state[collection_key] = [item for item in ids if item != movie_id]
        _set_notice(remove_message)
    else:
        st.session_state[collection_key] = [movie_id, *ids]
        _set_notice(add_message)


def _toggle_saved(movie_id: int) -> None:
    """Toggle a movie in the user's session-only saved list."""
    _toggle_collection(
        "saved_movie_ids",
        movie_id,
        "Saved to My List.",
        "Removed from My List.",
    )


def _is_saved(movie_id: int) -> bool:
    """Check whether a movie is currently in the saved list."""
    return int(movie_id) in {
        int(saved_id) for saved_id in st.session_state.get("saved_movie_ids", [])
    }


def _toggle_liked(movie_id: int) -> None:
    """Toggle a movie in the session-only taste profile."""
    _toggle_collection(
        "liked_movie_ids",
        movie_id,
        "Added to your taste profile.",
        "Removed from your taste profile.",
    )


def _advance_shelf() -> None:
    """Refresh the discovery seed so both home-page shelves rotate."""
    _refresh_discovery_seed()
    _set_notice("Here are a few more great choices.")


def _surprise_me(candidate_ids: Iterable[int]) -> None:
    """Choose the next popular candidate and open its details as a recommendation."""
    ids = [int(movie_id) for movie_id in candidate_ids]
    if not ids:
        _set_notice("There are no titles available for a surprise pick yet.")
        return
    counter = int(st.session_state.get("surprise_counter", 0))
    movie_id = ids[counter % len(ids)]
    st.session_state["surprise_counter"] = counter + 1
    _activate_movie(movie_id)
    st.session_state["loading_action"] = "surprise-me"
    st.session_state["details_movie_id"] = movie_id
    st.session_state["details_open_request"] = True
    _set_notice("A surprise pick is on deck — see why it matches below.")


def _clear_collection(collection_key: str, message: str) -> None:
    """Clear a session collection and queue confirmation feedback."""
    st.session_state[collection_key] = []
    _set_notice(message)


def _scroll_to(target_id: str) -> None:
    """Scroll only after the target exists and put keyboard focus on it."""
    request_id = int(st.session_state.get("scroll_request_id", 0)) + 1
    st.session_state["scroll_request_id"] = request_id
    target_literal = json.dumps(target_id)
    components.html(
        f"""
        <script>
        (() => {{
          const root = window.parent;
          const targetId = {target_literal};
          const requestId = {request_id};
          root.__cinematchScrollRequest = requestId;
          let attempts = 0;
          const reveal = () => {{
            if (root.__cinematchScrollRequest !== requestId) return;
            const target = root.document.getElementById(targetId);
            if (!target && attempts++ < 45) {{
              root.setTimeout(reveal, 55);
              return;
            }}
            if (!target) return;
            target.setAttribute("tabindex", "-1");
            const reducedMotion = root.matchMedia("(prefers-reduced-motion: reduce)").matches;
            target.scrollIntoView({{ behavior: reducedMotion ? "auto" : "smooth", block: "start" }});
            root.setTimeout(() => target.focus({{ preventScroll: true }}), reducedMotion ? 0 : 420);
          }};
          root.requestAnimationFrame(reveal);
        }})();
        </script>
        """,
        height=0,
    )


def _section_intro(anchor: str, kicker: str, title: str, description: str) -> None:
    """Render a consistent accessible heading block for a page section."""
    st.markdown(
        "<div "
        f"id='{html.escape(anchor, quote=True)}' class='cm-section-intro cm-anchor' tabindex='-1'>"
        "<div>"
        f"<div class='cm-label'>{html.escape(kicker)}</div>"
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(description)}</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def _animate_overflowing_card_titles() -> None:
    """Scroll only movie titles that exceed their card's available width."""
    components.html(
        """
        <script>
        (() => {
          const root = window.parent;
          const update = () => {
            root.document.querySelectorAll(".cm-card-title").forEach((title) => {
              root.__cinematchTitleResizeObserver.observe(title);
              const distance = Math.ceil(title.scrollWidth - title.clientWidth);
              const overflowing = distance > 2;
              title.classList.toggle("cm-card-title--overflow", overflowing);
              if (overflowing) {
                title.style.setProperty("--cm-title-distance", `-${distance}px`);
                title.style.setProperty(
                  "--cm-title-duration",
                  `${Math.max(7, distance / 24 + 3)}s`
                );
              }
            });
          };
          if (root.__cinematchTitleMutationObserver) {
            root.__cinematchTitleMutationObserver.disconnect();
          }
          if (root.__cinematchTitleResizeObserver) {
            root.__cinematchTitleResizeObserver.disconnect();
          }
          root.__cinematchTitleResizeObserver = new ResizeObserver(update);
          root.__cinematchTitleMutationObserver = new MutationObserver(update);
          root.__cinematchTitleMutationObserver.observe(root.document.body, {
            childList: true,
            characterData: true,
            subtree: true
          });
          root.document.fonts.ready.then(() => {
            update();
            root.document.querySelectorAll(".cm-card-title").forEach((title) => {
              root.__cinematchTitleResizeObserver.observe(title);
            });
          });
          root.requestAnimationFrame(update);
        })();
        </script>
        """,
        height=0,
    )


def _refresh_discovery_seed() -> int:
    """Generate and store a new seed for reproducible random discovery shelves."""
    seed = random.SystemRandom().randrange(0, 2**32)
    st.session_state["discovery_seed"] = seed
    return seed


def _discovery_seed() -> int:
    """Return the current shelf seed, creating one if session state lacks it."""
    seed = st.session_state.get("discovery_seed")
    if not isinstance(seed, int):
        return _refresh_discovery_seed()
    return seed


def _random_rated_movies(
    recommender: MovieRecommender,
    tmdb: TMDBClient,
    *,
    minimum_rating: float,
    seed: int,
    count: int = 8,
    genres: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sample above-threshold, optionally genre-filtered movies with poster art."""
    movies = recommender.movies.copy()
    ratings = pd.to_numeric(
        movies.get("vote_average", pd.Series(index=movies.index, dtype=float)),
        errors="coerce",
    )
    candidates = movies.loc[ratings >= minimum_rating].copy()
    selected_genres = {
        str(genre).casefold().replace(" ", "_")
        for genre in genres or ()
    }
    if selected_genres and "genres_list" in candidates:
        candidates = candidates.loc[
            candidates["genres_list"].map(
                lambda values: bool(
                    selected_genres.intersection(
                        genre.casefold().replace(" ", "_")
                        for genre in metadata_list(values)
                    )
                )
            )
        ]
    if candidates.empty:
        return candidates.reset_index(drop=True)
    shuffled = candidates.sample(frac=1, random_state=seed)
    selected: list[dict[str, Any]] = []
    for movie in shuffled.to_dict("records"):
        if poster_url(movie, tmdb, include_remote_fallback=True) is None:
            continue
        selected.append(movie)
        if len(selected) == count:
            break
    return pd.DataFrame(selected, columns=candidates.columns).reset_index(drop=True)


def _random_top_picks(
    recommender: MovieRecommender,
    tmdb: TMDBClient,
    *,
    seed: int,
    genre: str | None = None,
    count: int = 8,
) -> tuple[pd.DataFrame, list[str]]:
    """Select a deterministic genre mix and sample its highly rated movies."""
    available_genres = recommender.genres()
    rng = random.Random(seed)
    if genre is not None:
        selected_genres = [genre]
    elif len(available_genres) == 1:
        selected_genres = available_genres
    elif available_genres:
        number_of_genres = rng.randint(2, min(3, len(available_genres)))
        selected_genres = rng.sample(available_genres, number_of_genres)
    else:
        selected_genres = []
    movies = _random_rated_movies(
        recommender,
        tmdb,
        minimum_rating=7.5,
        seed=seed,
        count=count,
        genres=selected_genres,
    )
    return movies, selected_genres


def _render_cards(
    movies: pd.DataFrame,
    recommender: MovieRecommender,
    tmdb: TMDBClient,
    *,
    rail_key: str,
    origin: str,
    seed_ids: Iterable[int] = (),
    show_match_scores: bool = False,
    primary_label: str = "Find similar",
    featured_detail_keys: bool = False,
) -> None:
    """Render movie cards with match explanations and session action state."""
    records = movies.to_dict("records")
    saved_ids = set(st.session_state.get("saved_movie_ids", []))
    liked_ids = set(st.session_state.get("liked_movie_ids", []))
    seeds = [int(movie_id) for movie_id in seed_ids]

    # The desktop shelf uses four compact cards per row; the mobile breakpoint
    # stacks columns before their controls become cramped.
    for first in range(0, len(records), 4):
        row = records[first : first + 4]
        columns = st.columns(len(row), gap="medium")
        for column, movie in zip(columns, row, strict=True):
            movie_id = int(movie["id"])
            reasons: list[str] = []
            if seeds:
                try:
                    reasons = recommender.match_reasons(seeds, movie_id)
                except (KeyError, ValueError):
                    reasons = ["Similar story, genre, and talent signals"]
            with column:
                render_movie_card(
                    movie,
                    tmdb,
                    card_key=f"{rail_key}-{movie_id}",
                    primary_key=(f"similar-{movie_id}" if rail_key == "recommendations" else None),
                    details_key=(f"featured-explore-{movie_id}" if featured_detail_keys else None),
                    on_seed=_activate_movie,
                    on_details=_open_movie_details,
                    on_save=_toggle_saved,
                    on_like=_toggle_liked,
                    saved=movie_id in saved_ids,
                    liked=movie_id in liked_ids,
                    origin=origin,
                    match_reasons=reasons,
                    primary_label=primary_label,
                    show_match_score=show_match_scores,
                )


def _render_nav(saved_count: int) -> None:
    """Render the top navigation links and current saved-list count."""
    st.markdown(
        f"""
        <nav class="cm-nav" aria-label="CineMatch navigation">
          <a class="cm-wordmark" href="#page-top">CINE<span>MATCH</span></a>
          <div class="cm-nav-links">
            <a href="#search-panel">Search</a>
            <a href="#top-picks">Top picks</a>
            <a href="#recommendations">For you</a>
            <a href="#my-list">My list ({saved_count})</a>
          </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _render_hero(catalogue_size: int) -> None:
    """Render the landing-page introduction and catalogue summary."""
    st.markdown(
        f"""
        <div id="page-top" class="cm-hero">
          <div class="cm-kicker">A better movie night starts here</div>
          <h1>Find the story<br>you’ll <em>love next.</em></h1>
          <p class="cm-hero-copy">CineMatch turns the films you already enjoy into a
          thoughtful next-watch shortlist. Matched by story, genre, cast, and the people behind the camera.</p>
          <div class="cm-hero-actions">
            <a href="#search-panel">Find a movie</a>
            <a href="#top-picks">Browse top picks</a>
          </div>
          <div class="cm-hero-trust">
            <span>{catalogue_size:,} films in this catalogue</span>
            <span>Content-based matching, not hype</span>
            <span>Build a taste profile as you browse</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_notice() -> None:
    """Display and consume any queued session notice."""
    notice = st.session_state.pop("notice", None)
    if notice:
        # Decorative glyphs are not valid Streamlit toast icons. A text-only
        # toast is compatible with all supported Streamlit versions.
        st.toast(str(notice))


def _render_search(recommender: MovieRecommender, surprise_ids: list[int]) -> int | None:
    """Render movie search, recommendation activation, and surprise controls."""
    st.markdown('<div id="search-panel" class="cm-anchor" tabindex="-1"></div>', unsafe_allow_html=True)
    with st.container(border=True, key="movie-search-panel"):
        st.markdown('<div class="cm-label">Start with a film you know</div>', unsafe_allow_html=True)
        st.markdown("<h2>Find movies like a film you love</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>Type a title, choose the right release, then let CineMatch surface its closest companions.</p>",
            unsafe_allow_html=True,
        )
        search_column, action_column = st.columns((2.4, 1), gap="medium")
        choice_ids = [int(movie_id) for movie_id in recommender.movies["id"].tolist()]
        with search_column:
            selected_id = st.selectbox(
                "Search the catalogue",
                options=choice_ids,
                index=None,
                placeholder="Start typing a movie title…",
                key="movie-picker",
                on_change=_open_selected_movie_details,
                format_func=lambda movie_id: _movie_label(movie_id, recommender),
                help=(
                    "Search by title. CineMatch shows the release year and rating "
                    "to help distinguish remakes."
                ),
            )
        with action_column:
            st.markdown("<div style='height: 1.78rem'></div>", unsafe_allow_html=True)
            st.button(
                "Find similar movies",
                key="find-selected-movie",
                type="primary",
                use_container_width=True,
                disabled=selected_id is None,
                on_click=_activate_selected_movie,
            )
            st.button(
                "Surprise me",
                key="surprise-me",
                use_container_width=True,
                on_click=_surprise_me,
                args=(surprise_ids,),
            )

        if selected_id is not None:
            selected_movie = _movie(recommender, int(selected_id))
            if selected_movie:
                active_id = st.session_state.get("active_movie_id")
                state_text = (
                    "This is already your active match."
                    if active_id == int(selected_id)
                    else "Ready when you are — select Find similar movies to make it your match."
                )
                genres = ", ".join(
                    display_name(item) for item in metadata_list(selected_movie.get("genres_list"))[:3]
                )
                st.markdown(
                    "<div class='cm-now-playing'>"
                    f"Selected: <strong>{html.escape(str(selected_movie.get('title', 'Untitled movie')))}</strong>"
                    f" · {html.escape(movie_year(selected_movie))}"
                    f"{' · ' + html.escape(genres) if genres else ''}"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(state_text)
        else:
            st.caption("Tip: a film you genuinely liked gives the clearest first set of matches.")
    return int(selected_id) if selected_id is not None else None


def _render_recently_explored(recommender: MovieRecommender) -> None:
    """Render recent recommendation seeds that are still present in the catalog."""
    recent_slot = st.empty()
    recent_ids = _valid_ids(recommender, "recent_movie_ids")
    if recent_ids:
        st.session_state["recent_history_started"] = True
    if not st.session_state.get("recent_history_started", False):
        return
    with recent_slot.container(key="recently-explored"):
        st.markdown("<div class='cm-label'>Continue exploring</div>", unsafe_allow_html=True)
        if not recent_ids:
            st.caption("Your recent picks are not available in the current catalogue.")
            return
        st.caption("Jump back into a recent recommendation trail.")
        recent_movies = [
            (movie_id, movie)
            for movie_id in recent_ids[:4]
            if (movie := _movie(recommender, movie_id)) is not None
        ]
        if not recent_movies:
            return
        labels = [
            str(movie.get("title", "Untitled movie"))
            for _, movie in recent_movies
        ]
        column_widths = [min(36, max(8, len(title))) for title in labels]
        columns = st.columns(column_widths, gap="small")
        for column, (movie_id, _movie_data), title in zip(
            columns, recent_movies, labels, strict=True
        ):
            with column:
                st.button(
                    title,
                    key=f"recent-{movie_id}",
                    help=title,
                    on_click=_activate_movie,
                    args=(movie_id,),
                )


@st.dialog("Movie details", width="large")
def _render_details_dialog(movie_id: int) -> None:
    """Render the selected movie's profile dialog and its seed/save actions."""
    if st.session_state.get("details_movie_id") != movie_id:
        st.rerun()
    recommender = load_recommender()
    tmdb = load_tmdb_client()
    movie = _movie(recommender, movie_id)
    if movie is None:
        st.session_state["details_movie_id"] = None
        st.rerun()
        return
    render_movie_details(
        movie,
        tmdb,
        on_seed=_activate_movie,
        on_save=_toggle_saved,
        is_saved=_is_saved,
    )


def _selected_genre(recommender: MovieRecommender) -> str | None:
    """Translate the genre selectbox label into the normalized genre key."""
    genre_keys = recommender.genres()
    selected_label = st.session_state.get("discovery-genre", "All genres")
    genre_labels = ["All genres", *[display_name(genre) for genre in genre_keys]]
    if selected_label not in genre_labels or selected_label == "All genres":
        return None
    return genre_keys[genre_labels.index(selected_label) - 1]


def _begin_genre_selection() -> None:
    """Refresh discovery results and show genre-specific loading feedback."""
    _refresh_discovery_seed()
    st.session_state["loading_action"] = "genre-selection"


def _begin_recommendation_style_change() -> None:
    """Choose contextual loading feedback for the selected ranking mode."""
    style_actions = {
        "Balanced": "style-balanced",
        "Closest match": "style-closest",
        "Hidden gems": "style-hidden-gems",
    }
    selected_style = st.session_state.get("recommendation-style")
    action = style_actions.get(selected_style)
    if action is not None:
        st.session_state["loading_action"] = action


def _render_discovery(recommender: MovieRecommender, tmdb: TMDBClient) -> str | None:
    """Render genre-filtered top picks and the all-catalogue popular shelf."""
    st.divider()
    _section_intro(
        "top-picks",
        "Discover with confidence",
        "Top picks for tonight",
        "A fresh mix from two or three genres, featuring films rated 7.5 or higher.",
    )
    genre_keys = recommender.genres()
    genre_labels = ["All genres", *[display_name(genre) for genre in genre_keys]]
    filters, refresh = st.columns((2.2, 1), gap="medium")
    with filters:
        st.selectbox(
            "Browse a genre",
            genre_labels,
            key="discovery-genre",
            on_change=_begin_genre_selection,
            help="Choose one genre, or leave all genres selected for a rotating two- or three-genre mix. Popular picks always span every genre.",
        )
    with refresh:
        st.markdown("<div style='height: 1.78rem'></div>", unsafe_allow_html=True)
        st.button(
            "Show more picks",
            key="next-discovery-page",
            use_container_width=True,
            on_click=_advance_shelf,
        )

    genre = _selected_genre(recommender)
    seed = _discovery_seed()
    top_movies, selected_genres = _random_top_picks(
        recommender,
        tmdb,
        seed=seed,
        genre=genre,
    )
    if top_movies.empty:
        st.info("No films rated 7.5 or higher were found for that genre mix. Try another genre.")
    else:
        st.caption(
            "Genre mix: " + " · ".join(display_name(item) for item in selected_genres)
        )
        _render_cards(
            top_movies,
            recommender,
            tmdb,
            rail_key="top-picks",
            origin="top-picks",
            primary_label="Find similar movies",
            featured_detail_keys=True,
        )

    st.divider()
    _section_intro(
        "popular-now",
        "Crowd signal",
        "Popular across the catalogue",
        "A random selection of highly rated films, drawn from every genre.",
    )
    popular_movies = _random_rated_movies(
        recommender,
        tmdb,
        minimum_rating=8.3,
        seed=(seed + 1) % (2**32),
    )
    if popular_movies.empty:
        st.info("No films rated 8.3 or higher are available in the catalogue.")
    else:
        _render_cards(
            popular_movies,
            recommender,
            tmdb,
            rail_key="popular",
            origin="popular-now",
            primary_label="Find similar movies",
        )
    return genre


def _render_profile_recommendations(
    recommender: MovieRecommender,
    tmdb: TMDBClient,
    genre: str | None,
) -> None:
    """Build and display recommendations from the user's liked titles."""
    liked_ids = _valid_ids(recommender, "liked_movie_ids")
    st.divider()
    _section_intro(
        "taste-profile",
        "Your taste profile",
        "More like what you like",
        "Like a few films and CineMatch blends their shared signals into a personal discovery rail for this browser session.",
    )
    if not liked_ids:
        st.markdown(
            "<div class='cm-empty'>Like titles from any card to build a small, private taste profile. "
            "CineMatch will use those choices to create a blended recommendation rail here.</div>",
            unsafe_allow_html=True,
        )
        return
    liked_titles = [
        str(movie.get("title", "Untitled movie"))
        for movie_id in liked_ids[:4]
        if (movie := _movie(recommender, movie_id)) is not None
    ]
    st.caption("Building from: " + " · ".join(liked_titles))
    try:
        recommendations = recommend_movies(
            recommender,
            liked_ids,
            count=8,
            genre=genre,
            mode="balanced",
        )
    except (KeyError, ValueError):
        recommendations = pd.DataFrame()
    if recommendations.empty:
        st.info("Those titles do not share enough catalogue text signals yet. Add another liked film or remove the genre filter.")
    else:
        _render_cards(
            recommendations,
            recommender,
            tmdb,
            rail_key="profile",
            origin="taste-profile",
            seed_ids=liked_ids,
            show_match_scores=True,
        )
    st.button(
        "Clear taste profile",
        key="clear-taste-profile",
        on_click=_clear_collection,
        args=("liked_movie_ids", "Your taste profile has been cleared."),
    )


def _render_active_recommendations(
    recommender: MovieRecommender,
    tmdb: TMDBClient,
    genre: str | None,
) -> None:
    """Render the current seed's recommendation rail and ranking controls."""
    if not st.session_state.get("show_next_watch", False):
        return
    st.divider()
    active_id = st.session_state.get("active_movie_id")
    if active_id is not None and _movie(recommender, int(active_id)) is None:
        st.session_state["active_movie_id"] = None
        active_id = None
    if active_id is None:
        _section_intro(
            "recommendations",
            "Your next watch",
            "Recommendations will appear here",
            "Choose a title above, or use Surprise me, and we’ll turn its themes and talent into a focused next-watch list.",
        )
        st.markdown(
            "<div class='cm-empty'>Your discovery rail is waiting. Search for a film you know to start a recommendation trail.</div>",
            unsafe_allow_html=True,
        )
        return

    seed = _movie(recommender, int(active_id))
    if seed is None:
        return
    _section_intro(
        "recommendations",
        "Your next watch",
        f"Because you chose {str(seed.get('title', 'this movie'))}",
        "Choose how tightly CineMatch should follow the original film. Match signals are based on the catalogue’s story, genre, cast, director, and keyword metadata.",
    )
    st.markdown(
        "<div class='cm-now-playing'>Matching from <strong>"
        f"{html.escape(str(seed.get('title', 'Untitled movie')))}</strong> · {html.escape(movie_year(seed))}</div>",
        unsafe_allow_html=True,
    )
    mode_label = st.radio(
        "Recommendation style",
        ("Balanced", "Closest match", "Hidden gems"),
        horizontal=True,
        key="recommendation-style",
        on_change=_begin_recommendation_style_change,
        help=(
            "Balanced adds small quality and popularity signals; Closest match is pure content similarity; "
            "Hidden gems favours well-rated, less broadly popular matches."
        ),
    )
    mode = {
        "Balanced": "balanced",
        "Closest match": "similar",
        "Hidden gems": "hidden_gems",
    }[mode_label]
    recommendations = recommend_movies(
        recommender,
        [int(active_id)],
        count=8,
        genre=genre,
        mode=mode,
    )
    if recommendations.empty:
        message = "No matching titles"
        if genre:
            message += f" in {display_name(genre)}"
        st.info(message + ". Try another recommendation style or clear the genre filter.")
        return
    _render_cards(
        recommendations,
        recommender,
        tmdb,
        rail_key="recommendations",
        origin="recommendations",
        seed_ids=[int(active_id)],
        show_match_scores=True,
        primary_label="Find similar movies",
    )
    st.caption(
        "A match signal describes shared catalogue features. It is not a prediction of how much you will enjoy a film."
    )


def _render_my_list(recommender: MovieRecommender, tmdb: TMDBClient) -> None:
    """Render saved titles and actions for starting new recommendation trails."""
    saved_ids = _valid_ids(recommender, "saved_movie_ids")
    st.divider()
    _section_intro(
        "my-list",
        "Your shortlist",
        "My List",
        "Save options as you browse, then return here to compare them or spin up a fresh recommendation trail.",
    )
    if not saved_ids:
        st.markdown(
            "<div class='cm-empty'>Nothing saved yet. Use Save on any card to keep a possible movie night in one place.</div>",
            unsafe_allow_html=True,
        )
        return
    saved_movies = [
        movie for movie_id in saved_ids if (movie := _movie(recommender, movie_id)) is not None
    ]
    _render_cards(
        pd.DataFrame(saved_movies),
        recommender,
        tmdb,
        rail_key="saved",
        origin="my-list",
        primary_label="Find similar movies",
    )
    st.button(
        "Clear My List",
        key="clear-saved-movies",
        on_click=_clear_collection,
        args=("saved_movie_ids", "My List has been cleared."),
    )


def _render_footer() -> None:
    """Render the TMDB attribution and project disclaimer."""
    st.markdown(
        """
        <div class="cm-footer">
          CineMatch uses a local content-based recommendation model built from the TMDB 5000 dataset distribution.
          Poster paths and optional movie details are associated with <a href="https://www.themoviedb.org/" target="_blank" rel="noreferrer">TMDB</a>.
          This product uses the TMDB API but is not endorsed or certified by TMDB.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_application() -> None:
    """Load project services and render the complete page in interaction order."""
    _init_session_state()
    try:
        recommender = load_recommender()
    except (ArtifactError, OSError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    if recommender.movies.empty:
        st.error("The movie catalogue is empty. Rebuild the CineMatch artifacts before launching the app.")
        return
    tmdb = load_tmdb_client()

    # Validate session-held movie IDs against the current catalogue before using
    # them; a rebuilt catalogue may no longer contain an earlier selection.
    _valid_ids(recommender, "saved_movie_ids")
    _valid_ids(recommender, "liked_movie_ids")
    _valid_ids(recommender, "recent_movie_ids")

    active_id = st.session_state.get("active_movie_id")
    if active_id is not None and _movie(recommender, int(active_id)) is None:
        st.session_state["active_movie_id"] = None

    surprise_ids = recommender.popular_movies(k=24)["id"].astype(int).tolist()
    _render_nav(len(st.session_state.get("saved_movie_ids", [])))
    _render_hero(len(recommender.movies))
    _render_notice()
    _render_search(recommender, surprise_ids)
    should_open_details = st.session_state.pop("details_open_request", False)
    details_id = st.session_state.get("details_movie_id")
    if should_open_details and details_id is not None:
        _render_details_dialog(int(details_id))
    else:
        st.session_state["details_movie_id"] = None
    recommendation_slot = st.empty()
    with recommendation_slot.container(key="active-recommendations"):
        _render_active_recommendations(recommender, tmdb, _selected_genre(recommender))
    _render_recently_explored(recommender)
    _render_discovery(recommender, tmdb)
    _render_profile_recommendations(recommender, tmdb, _selected_genre(recommender))
    _render_my_list(recommender, tmdb)
    _render_footer()
    _animate_overflowing_card_titles()

    target = st.session_state.pop("scroll_target", None)
    if target:
        _scroll_to(str(target))


def _loading_presentation(action: str | None) -> tuple[str, str] | None:
    """Map a session action name to its spinner message and visual category."""
    if action is None:
        return None
    return _LOADING_PRESENTATIONS.get(action)


def _style_loading_indicator(loading: tuple[str, str] | None) -> None:
    """Keep the spinner render tree stable while showing it only for actions."""
    accents = {
        "selection": ("#ff6b86", "#b6a1ff"),
        "details": ("#8ce1d0", "#8faeff"),
        "genre": ("#ffd166", "#ff8fc7"),
        "recommendations": ("#ff6b86", "#8ce1d0"),
    }
    message, variant = loading or ("", "selection")
    primary, secondary = accents.get(variant, ("#ff6b86", "#b6a1ff"))
    message_literal = json.dumps(message)
    active = "true" if loading is not None else "false"
    st.markdown(
        f"""
        <style>
        body:has(#cm-loader-state[data-active="false"]) [data-testid="stSpinner"] {{
          display: none !important;
        }}
        [data-testid="stSpinner"]::before {{
          border-top-color: {primary};
          border-right-color: {secondary};
          box-shadow: 0 0 0 12px rgba(18,16,34,.50), 0 18px 50px rgba(0,0,0,.38),
            0 0 28px {secondary}55;
        }}
        [data-testid="stSpinner"]::after {{
          content: {message_literal};
        }}
        </style>
        <div id="cm-loader-state" data-active="{active}"></div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Apply shared styles and render the app inside the contextual spinner."""
    apply_product_styles()
    loading_action = st.session_state.pop("loading_action", None)
    loading = _loading_presentation(loading_action)
    _style_loading_indicator(loading)
    with st.spinner(loading[0] if loading is not None else ""):
        _render_application()


if __name__ == "__main__":
    main()
