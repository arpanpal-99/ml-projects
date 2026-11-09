from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.features.feature_engineering import build_feature_matrix
from src.recommender.artifacts import save_artifacts
from src.recommender.engine import MovieRecommender
from src.services.tmdb_client import TMDBClient


def _catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "title": "Space One",
                "release_date": "2020-01-01",
                "tags": "space galaxy journey pilot",
                "genres_list": ["science_fiction", "adventure"],
                "keywords_list": ["space", "journey"],
                "overview": "A trip through space.",
                "poster_path": "/space-one.jpg",
                "popularity": 80,
                "vote_average": 7.5,
                "vote_count": 120,
                "original_language": "en",
                "director_names": ["alex_director"],
                "cast_names": ["alex_actor"],
            },
            {
                "id": 2,
                "title": "Space Two",
                "release_date": "2019-01-01",
                "tags": "space galaxy travel pilot",
                "genres_list": ["science_fiction"],
                "keywords_list": ["space", "travel"],
                "overview": "A journey through a galaxy.",
                "poster_path": "/space-two.jpg",
                "popularity": 60,
                "vote_average": 7.0,
                "vote_count": 90,
                "original_language": "en",
                "director_names": ["sam_director"],
                "cast_names": ["casey_star"],
            },
            {
                "id": 3,
                "title": "Garden Story",
                "release_date": "2021-01-01",
                "tags": "garden romance flowers",
                "genres_list": ["romance"],
                "keywords_list": ["garden"],
                "overview": "A quiet romance in a city garden.",
                "poster_path": "/garden-story.jpg",
                "popularity": 35,
                "vote_average": 8.4,
                "vote_count": 140,
                "original_language": "en",
                "director_names": ["casey_director"],
                "cast_names": ["garden_actor"],
            },
        ]
    )


def _app(monkeypatch, tmp_path) -> AppTest:
    movies = _catalogue()
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    save_artifacts(movies, vectorizer, matrix, tmp_path, {}, 5000)
    monkeypatch.setenv("CINEMATCH_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "true")
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run()
    app.session_state["discovery_seed"] = 5
    return app.run()


def _button(app: AppTest, key: str):
    return next(button for button in app.button if button.key == key)


def _contains_markdown(app: AppTest, text: str) -> bool:
    return any(text in item.value for item in app.markdown)


def test_random_discovery_shelves_respect_rating_genre_seed_and_posters(monkeypatch):
    from app import _random_rated_movies, _random_top_picks

    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    genres = ["action", "comedy", "drama", "fantasy", "romance"]
    movies = pd.DataFrame(
        [
            {
                "id": movie_id,
                "title": f"Film {movie_id}",
                "tags": f"film story {movie_id}",
                "genres_list": [genres[movie_id % len(genres)]],
                "vote_average": 7.4 + (movie_id % 10) * 0.12,
                "poster_path": None if movie_id % 3 == 0 else f"/poster-{movie_id}.jpg",
            }
            for movie_id in range(40)
        ]
    )
    vectorizer, matrix = build_feature_matrix(movies["tags"].tolist())
    recommender = MovieRecommender(movies, vectorizer, matrix)

    tmdb = TMDBClient(api_key=None)
    top_picks, selected_genres = _random_top_picks(recommender, tmdb, seed=7)
    next_top_picks, next_genres = _random_top_picks(recommender, tmdb, seed=8)
    popular_picks = _random_rated_movies(
        recommender,
        tmdb,
        minimum_rating=8.3,
        seed=9,
    )

    assert 2 <= len(selected_genres) <= 3
    assert 2 <= len(next_genres) <= 3
    assert top_picks["vote_average"].ge(7.5).all()
    allowed_genres = set(selected_genres)
    assert all(
        allowed_genres.intersection(movie_genres)
        for movie_genres in top_picks["genres_list"]
    )
    assert top_picks["poster_path"].notna().all()
    assert top_picks["poster_path"].str.strip().ne("").all()
    assert set(top_picks["id"]) != set(next_top_picks["id"])
    assert 0 < len(popular_picks) <= 8
    assert popular_picks["vote_average"].ge(8.3).all()
    assert popular_picks["poster_path"].notna().all()
    assert popular_picks["poster_path"].str.strip().ne("").all()
    assert len({genre for values in popular_picks["genres_list"] for genre in values}) > 1


def test_product_home_renders_discovery_rails_and_surprise_flow(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    from app import _loading_presentation, _movie_label, load_recommender

    assert not app.exception
    assert not app.error
    assert app.selectbox[0].label == "Search the catalogue"
    assert _movie_label(1, load_recommender()) == "Space One · 2020 · ★ 7.5/10"
    assert _loading_presentation("picker-details") == ("Getting the full picture...", "selection")
    assert _loading_presentation("card-details") == ("Unfolding the plot...", "details")
    assert _loading_presentation("genre-selection") == ("Shuffling the genre deck...", "genre")
    assert _loading_presentation("similar-movies") == (
        "Following the movie trail...",
        "recommendations",
    )
    assert _loading_presentation("surprise-me") == (
        "Summoning a little movie magic...",
        "selection",
    )
    assert _loading_presentation("style-balanced") == (
        "Striking the perfect balance...",
        "recommendations",
    )
    assert _loading_presentation("style-closest") == (
        "Tracking down cinematic twins...",
        "recommendations",
    )
    assert _loading_presentation("style-hidden-gems") == (
        "Digging up cinematic treasures...",
        "recommendations",
    )
    assert _button(app, "find-selected-movie").label == "Find similar movies"
    assert _contains_markdown(app, "Find movies like a film you love")
    assert sum("Find movies like a film you love" in item.value for item in app.markdown) == 1
    assert _contains_markdown(app, "Top picks for tonight")
    assert _contains_markdown(app, "Popular across the catalogue")
    assert not _contains_markdown(app, "id='recommendations'")
    assert not _contains_markdown(app, "A trip through space.")
    assert _contains_markdown(
        app,
        'class="cm-card-title" title="Space One"><span>Space One</span>',
    )
    assert not any(button.label.lower() == "match this film" for button in app.button)
    assert not any(button.label == "Clear selection" for button in app.button)
    assert any(
        button.key == "top-picks-1-similar" and button.label == "Find similar movies"
        for button in app.button
    )
    assert any(button.key and button.key.startswith("featured-explore-") for button in app.button)
    assert not _contains_markdown(app, "cinematch-loader")

    _button(app, "surprise-me").click().run()

    assert app.session_state["active_movie_id"] in {1, 2, 3}
    assert app.session_state["details_movie_id"] == app.session_state["active_movie_id"]
    assert _contains_markdown(app, "Movie profile")
    assert app.session_state["scroll_request_id"] == 1
    assert _contains_markdown(app, "Because you chose")
    assert any(button.key and button.key.startswith("similar-") for button in app.button)
    assert any(
        button.key and button.key.startswith("similar-") and button.label == "Find similar movies"
        for button in app.button
    )
    sections = [item.value for item in app.markdown]
    recommendation_index = next(
        index for index, value in enumerate(sections) if "id='recommendations'" in value
    )
    discovery_index = next(
        index for index, value in enumerate(sections) if "id='top-picks'" in value
    )
    assert recommendation_index < discovery_index


def test_seed_details_watchlist_and_taste_state_are_synchronised(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    app.selectbox[0].select(1).run()
    assert app.session_state["details_movie_id"] == 1
    assert app.session_state["show_next_watch"] is False
    assert not _contains_markdown(app, "id='recommendations'")
    assert app.session_state["recent_movie_ids"] == [1]
    assert app.session_state["recent_history_started"] is True
    assert _contains_markdown(app, "Continue exploring")
    assert _button(app, "recent-1").label == "Space One"
    assert _contains_markdown(app, "A trip through space.")
    assert _contains_markdown(app, "Space One")
    assert _contains_markdown(app, "/space-one.jpg")
    assert _contains_markdown(app, 'class="cm-detail-stats"')
    assert _contains_markdown(app, "Audience votes")
    assert _contains_markdown(app, "120")
    assert _contains_markdown(app, "Popularity")
    assert _contains_markdown(app, "80")
    assert not _contains_markdown(app, 'class="cm-detail-stat__label">Rating')
    assert _button(app, "details-save").label == "Save to list"
    assert _button(app, "details-find-similar").label == "Find similar movies"

    app.selectbox[0].select(2).run()
    assert app.session_state["details_movie_id"] == 2
    assert _contains_markdown(app, "Space Two")
    assert _contains_markdown(app, "/space-two.jpg")

    app.selectbox[0].select(1).run()
    _button(app, "find-selected-movie").click().run()

    assert app.session_state["active_movie_id"] == 1
    assert app.session_state["show_next_watch"] is True
    assert app.session_state["movie-picker"] == 1
    assert _contains_markdown(app, "Because you chose Space One")
    assert _contains_markdown(app, "Continue exploring")
    assert _button(app, "recent-1").label == "Space One"
    assert _button(app, "recent-1").help == "Space One"
    recommendation_scroll_request = app.session_state["scroll_request_id"]

    app.selectbox[0].select(None).run()
    assert app.session_state["show_next_watch"] is False
    assert not _contains_markdown(app, "id='recommendations'")
    assert sum("Find movies like a film you love" in item.value for item in app.markdown) == 1
    assert _contains_markdown(app, "Continue exploring")
    assert _button(app, "recent-1").label == "Space One"

    _button(app, "featured-explore-1").click().run()

    assert app.session_state["details_movie_id"] == 1
    assert _contains_markdown(app, "Movie profile")
    assert _contains_markdown(app, "Space One")
    assert app.session_state["scroll_request_id"] == recommendation_scroll_request

    _button(app, "details-save").click().run()
    assert app.session_state["saved_movie_ids"] == [1]

    _button(app, "featured-explore-1").click().run()
    assert _button(app, "details-save").label == "Remove from list"
    _button(app, "details-save").click().run()
    assert app.session_state["saved_movie_ids"] == []

    _button(app, "next-discovery-page").click().run()
    assert app.session_state["details_movie_id"] is None
    assert not _contains_markdown(app, "Movie profile")

    _button(app, "top-picks-1-save").click().run()
    assert app.session_state["saved_movie_ids"] == [1]
    assert _contains_markdown(app, "My List")

    _button(app, "top-picks-1-like").click().run()
    assert app.session_state["liked_movie_ids"] == [1]
    assert _contains_markdown(app, "More like what you like")


def test_continue_exploring_remains_visible_after_recent_ids_become_unavailable(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    app.session_state["recent_movie_ids"] = [999999]
    app.session_state["recent_history_started"] = True

    app.run()

    assert _contains_markdown(app, "Continue exploring")
    assert any("Your recent picks are not available" in item.value for item in app.caption)


def test_continue_exploring_remains_after_find_similar_in_details_dialog(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    app.selectbox[0].select(2).run()
    _button(app, "find-selected-movie").click().run()
    _button(app, "featured-explore-1").click().run()
    _button(app, "details-find-similar").click().run()

    assert app.session_state["active_movie_id"] == 1
    assert app.session_state["recent_movie_ids"][0] == 1
    assert app.session_state["recent_history_started"] is True
    assert app.session_state["show_next_watch"] is True
    assert app.session_state["movie-picker"] == 1
    assert app.session_state["details_movie_id"] is None
    assert _contains_markdown(app, "Continue exploring")
    assert _button(app, "recent-1").label == "Space One"
    assert not _contains_markdown(app, "Movie profile")
