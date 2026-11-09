"""Offline tests for TMDB URL construction, caching, and graceful API failures."""

from __future__ import annotations

from unittest.mock import Mock

import requests

from src.services.tmdb_client import TMDBClient


def test_local_poster_url_does_not_need_network():
    assert TMDBClient.poster_url("/poster.jpg") == ("https://image.tmdb.org/t/p/w500/poster.jpg")
    assert TMDBClient.poster_url(None) is None


def test_no_key_skips_external_request(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    session = Mock(spec=requests.Session)
    client = TMDBClient(api_key=None, session=session)

    assert client.get_movie_details(1) is None
    session.get.assert_not_called()


def test_details_are_cached_and_ids_are_checked():
    session = Mock(spec=requests.Session)
    response = Mock()
    response.json.return_value = {"id": 1, "title": "Example"}
    session.get.return_value = response
    client = TMDBClient(api_key="test-key", session=session)

    assert client.get_movie_details(1)["title"] == "Example"
    assert client.get_movie_details(1)["title"] == "Example"
    session.get.assert_called_once()


def test_network_failure_degrades_gracefully():
    session = Mock(spec=requests.Session)
    session.get.side_effect = requests.Timeout("timeout")
    client = TMDBClient(api_key="test-key", session=session)

    assert client.get_movie_details(1) is None
    assert client.get_movie_details(1) is None
    session.get.assert_called_once()


def test_network_failure_log_does_not_include_credentials(caplog):
    session = Mock(spec=requests.Session)
    session.get.side_effect = requests.RequestException(
        "request failed: https://example.test/?api_key=private-value"
    )
    client = TMDBClient(api_key="private-value", session=session)

    client.get_movie_details(1)

    assert "private-value" not in caplog.text
