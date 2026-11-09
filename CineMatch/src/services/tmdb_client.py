from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
FAILED_LOOKUP_CACHE_TTL_SECONDS = 30


class TMDBClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: tuple[float, float] = (1.0, 3.0),
        cache_ttl_seconds: int = 3600,
        cache_size: int = 256,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_size = cache_size
        self.session = session or requests.Session()
        self._cache: OrderedDict[int, tuple[float, dict[str, Any] | None]] = OrderedDict()
        self._lock = threading.Lock()

    def _cache_details(self, movie_id: int, details: dict[str, Any] | None, ttl: int) -> None:
        with self._lock:
            cached_details = details.copy() if details else None
            self._cache[movie_id] = (time.monotonic() + ttl, cached_details)
            self._cache.move_to_end(movie_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    @staticmethod
    def poster_url(poster_path: Any, size: str = "w500") -> str | None:
        if not isinstance(poster_path, str) or not poster_path.strip():
            return None
        if size not in {"w92", "w154", "w185", "w342", "w500", "w780", "original"}:
            raise ValueError("Unsupported TMDB image size.")
        return f"{TMDB_IMAGE_BASE}/{size}/{poster_path.lstrip('/')}"

    def get_movie_details(self, movie_id: int) -> dict[str, Any] | None:
        if not self.api_key:
            return None
        movie_id = int(movie_id)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(movie_id)
            if cached and cached[0] > now:
                self._cache.move_to_end(movie_id)
                return cached[1].copy() if cached[1] else None
            if cached:
                del self._cache[movie_id]

        try:
            response = self.session.get(
                f"{TMDB_API_BASE}/movie/{movie_id}",
                params={"api_key": self.api_key},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning(
                "TMDB details lookup failed for movie ID %s (%s).",
                movie_id,
                type(exc).__name__,
            )
            self._cache_details(movie_id, None, FAILED_LOOKUP_CACHE_TTL_SECONDS)
            return None

        if not isinstance(payload, dict) or payload.get("id") != movie_id:
            LOGGER.warning("TMDB returned an invalid or mismatched movie ID for %s.", movie_id)
            self._cache_details(movie_id, None, FAILED_LOOKUP_CACHE_TTL_SECONDS)
            return None

        self._cache_details(movie_id, payload, self.cache_ttl_seconds)
        return payload
