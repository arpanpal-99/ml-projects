from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, vectorizer: object, matrix: sparse.spmatrix) -> None:
        if len(movies) != matrix.shape[0]:
            raise ValueError("Movie table and feature matrix row counts must match.")
        if movies["id"].isna().any() or movies["id"].duplicated().any():
            raise ValueError("Movie IDs must be non-null and unique.")
        self.movies = movies.reset_index(drop=True)
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        self._id_to_row = {int(movie_id): row for row, movie_id in enumerate(self.movies["id"])}
        self._nonzero_rows = np.asarray(self.matrix.getnnz(axis=1)).ravel() > 0

    @staticmethod
    def _validate_limit(limit: int, name: str = "k") -> int:
        if isinstance(limit, bool) or not isinstance(limit, int | np.integer) or limit < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(limit)

    @staticmethod
    def _metadata_values(value: object) -> list[str]:
        """Return a clean list for list-like metadata stored in artifacts."""
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if isinstance(item, str) and item.strip()]
        return []

    @staticmethod
    def _display_name(value: str) -> str:
        return value.replace("_", " ").strip().title()

    def _numeric_values(self, column: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(
            self.movies.get(column, pd.Series(index=self.movies.index, dtype=float)),
            errors="coerce",
        ).fillna(default)

    def _quality_scores(self) -> tuple[pd.Series, pd.Series]:
        """Return Bayesian quality and popularity scores on a stable 0..1 scale.

        A raw average rating over-rewards little-known titles.  The Bayesian score
        shrinks those ratings toward the catalogue average according to vote count,
        giving the discovery rails a useful, repeatable definition of "top".
        """
        ratings = self._numeric_values("vote_average")
        votes = self._numeric_values("vote_count")
        rated = ratings[ratings > 0]
        catalogue_average = float(rated.mean()) if not rated.empty else 0.0
        minimum_votes = float(votes.quantile(0.60)) if not votes.empty else 0.0
        if minimum_votes <= 0:
            minimum_votes = 1.0
        weighted = (
            (votes / (votes + minimum_votes)) * ratings
            + (minimum_votes / (votes + minimum_votes)) * catalogue_average
        )
        quality = weighted.rank(method="average", pct=True).fillna(0.0)
        popularity = self._numeric_values("popularity").rank(method="average", pct=True).fillna(0.0)
        return quality, popularity

    def _genre_mask(self, genre: str | None) -> np.ndarray:
        if genre is None or not str(genre).strip():
            return np.ones(len(self.movies), dtype=bool)
        expected = str(genre).strip().casefold().replace(" ", "_")
        return self.movies.get(
            "genres_list", pd.Series([[] for _ in range(len(self.movies))], index=self.movies.index)
        ).map(
            lambda values: expected
            in {item.casefold().replace(" ", "_") for item in self._metadata_values(values)}
        ).to_numpy(dtype=bool)

    def get_movie(self, movie_id: int) -> pd.Series | None:
        row = self._id_to_row.get(int(movie_id))
        return None if row is None else self.movies.iloc[row].copy()

    def genres(self, limit: int | None = None) -> list[str]:
        """Return available genres ordered by catalogue coverage, then name."""
        counts: dict[str, int] = {}
        for values in self.movies.get("genres_list", pd.Series(dtype=object)):
            for genre in self._metadata_values(values):
                normalized = genre.casefold().replace(" ", "_")
                counts[normalized] = counts.get(normalized, 0) + 1
        ordered = sorted(counts, key=lambda item: (-counts[item], self._display_name(item)))
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int | np.integer) or limit < 1:
                raise ValueError("limit must be a positive integer when supplied.")
            ordered = ordered[: int(limit)]
        return ordered

    def top_movies(
        self,
        k: int = 10,
        *,
        genre: str | None = None,
        min_votes: int | None = None,
    ) -> pd.DataFrame:
        """Rank catalogue favourites using a Bayesian rating instead of randomness."""
        k = self._validate_limit(k)
        if min_votes is not None and (
            isinstance(min_votes, bool) or not isinstance(min_votes, int | np.integer) or min_votes < 0
        ):
            raise ValueError("min_votes must be a non-negative integer when supplied.")

        quality, popularity = self._quality_scores()
        ratings = self._numeric_values("vote_average")
        votes = self._numeric_values("vote_count")
        candidates = self.movies.loc[self._genre_mask(genre)].copy()
        candidates["top_score"] = quality.loc[candidates.index]
        candidates["popularity_score"] = popularity.loc[candidates.index]
        candidates["_rating_sort"] = ratings.loc[candidates.index]
        candidates["_vote_sort"] = votes.loc[candidates.index]
        if min_votes is not None:
            candidates = candidates.loc[candidates["_vote_sort"] >= int(min_votes)]
        if candidates.empty:
            return candidates.drop(columns=["_rating_sort", "_vote_sort"], errors="ignore")
        candidates["_title_sort"] = candidates["title"].astype(str).str.casefold()
        return (
            candidates.sort_values(
                ["top_score", "_rating_sort", "_vote_sort", "popularity_score", "_title_sort", "id"],
                ascending=[False, False, False, False, True, True],
                kind="mergesort",
            )
            .drop(columns=["_rating_sort", "_vote_sort", "_title_sort"])
            .head(k)
            .reset_index(drop=True)
        )

    def popular_movies(self, k: int = 10, *, genre: str | None = None) -> pd.DataFrame:
        """Return broadly popular titles, with quality breaking popularity ties."""
        k = self._validate_limit(k)
        quality, popularity = self._quality_scores()
        ratings = self._numeric_values("vote_average")
        votes = self._numeric_values("vote_count")
        candidates = self.movies.loc[self._genre_mask(genre)].copy()
        candidates["popularity_score"] = popularity.loc[candidates.index]
        candidates["top_score"] = quality.loc[candidates.index]
        candidates["_rating_sort"] = ratings.loc[candidates.index]
        candidates["_vote_sort"] = votes.loc[candidates.index]
        candidates["_title_sort"] = candidates["title"].astype(str).str.casefold()
        return (
            candidates.sort_values(
                ["popularity_score", "top_score", "_vote_sort", "_rating_sort", "_title_sort", "id"],
                ascending=[False, False, False, False, True, True],
                kind="mergesort",
            )
            .drop(columns=["_rating_sort", "_vote_sort", "_title_sort"])
            .head(k)
            .reset_index(drop=True)
        )

    def find_movies(self, query: str, limit: int = 20) -> pd.DataFrame:
        limit = self._validate_limit(limit, "limit")
        normalized_query = str(query).strip().casefold()
        candidates = self.movies.copy()
        titles = candidates["title"].astype(str).str.casefold()
        if normalized_query:
            matches = titles.str.contains(normalized_query, regex=False, na=False)
            candidates = candidates.loc[matches].copy()
        if candidates.empty:
            return candidates.head(0).copy()

        titles = candidates["title"].astype(str).str.casefold()
        candidates["_prefix_sort"] = (
            0 if not normalized_query else (~titles.str.startswith(normalized_query)).astype(int)
        )
        candidates["_rating_sort"] = pd.to_numeric(
            candidates.get("vote_average", pd.Series(index=candidates.index, dtype=float)),
            errors="coerce",
        ).fillna(-1)
        candidates["_votes_sort"] = pd.to_numeric(
            candidates.get("vote_count", pd.Series(index=candidates.index, dtype=float)),
            errors="coerce",
        ).fillna(0)
        candidates["_year_sort"] = pd.to_numeric(
            candidates.get("release_date", pd.Series(index=candidates.index, dtype=object))
            .astype(str)
            .str.slice(0, 4),
            errors="coerce",
        )
        candidates["_title_sort"] = titles
        result = candidates.sort_values(
            [
                "_prefix_sort",
                "_rating_sort",
                "_votes_sort",
                "_title_sort",
                "_year_sort",
                "id",
            ],
            ascending=[True, False, False, True, False, True],
            kind="mergesort",
            na_position="last",
        )
        return (
            result.drop(
                columns=[
                    "_prefix_sort",
                    "_rating_sort",
                    "_votes_sort",
                    "_year_sort",
                    "_title_sort",
                ]
            )
            .head(limit)
            .reset_index(drop=True)
        )

    def recommend(self, movie_id: int, k: int = 5) -> pd.DataFrame:
        k = self._validate_limit(k)
        movie_id = int(movie_id)
        row = self._id_to_row.get(movie_id)
        if row is None:
            raise KeyError(f"Movie ID {movie_id} was not found in the movie catalog.")
        columns = list(self.movies.columns) + ["similarity"]
        if not self._nonzero_rows[row]:
            return pd.DataFrame(columns=columns)

        scores = cosine_similarity(self.matrix[row], self.matrix).ravel()
        valid = self._nonzero_rows & (self.movies["id"].to_numpy() != movie_id) & (scores > 0)
        candidate_rows = np.flatnonzero(valid)
        if not len(candidate_rows):
            return pd.DataFrame(columns=columns)

        titles = self.movies["title"].astype(str).str.casefold().to_numpy()
        ids = self.movies["id"].to_numpy()
        order = np.lexsort(
            (
                ids[candidate_rows],
                titles[candidate_rows],
                -scores[candidate_rows],
            )
        )
        selected = candidate_rows[order[:k]]
        result = self.movies.iloc[selected].copy()
        result["similarity"] = scores[selected]
        return result.reset_index(drop=True)

    def recommend_for_movies(
        self,
        movie_ids: Iterable[int],
        k: int = 8,
        *,
        genre: str | None = None,
        mode: str = "balanced",
    ) -> pd.DataFrame:
        """Create explainable discovery candidates from one or more liked titles.

        ``similar`` preserves the pure content model, while ``balanced`` adds a
        modest quality/popularity signal so weakly rated titles do not dominate the
        first screen. ``hidden_gems`` gives quality more weight and rewards titles
        that are less broadly popular.  The original :meth:`recommend` remains a
        pure cosine-similarity API for callers that need that behaviour.
        """
        k = self._validate_limit(k)
        normalized_mode = str(mode).strip().casefold()
        if normalized_mode not in {"similar", "balanced", "hidden_gems"}:
            raise ValueError("mode must be 'similar', 'balanced', or 'hidden_gems'.")

        ids = list(dict.fromkeys(int(movie_id) for movie_id in movie_ids))
        if not ids:
            raise ValueError("movie_ids must contain at least one movie ID.")
        missing = [movie_id for movie_id in ids if movie_id not in self._id_to_row]
        if missing:
            raise KeyError(f"Movie ID {missing[0]} was not found in the movie catalog.")
        rows = np.asarray([self._id_to_row[movie_id] for movie_id in ids], dtype=int)
        usable_rows = rows[self._nonzero_rows[rows]]
        columns = list(self.movies.columns) + [
            "similarity",
            "recommendation_score",
            "recommendation_mode",
        ]
        if not len(usable_rows):
            return pd.DataFrame(columns=columns)

        profile = sparse.csr_matrix(self.matrix[usable_rows].mean(axis=0))
        similarities = cosine_similarity(profile, self.matrix).ravel()
        candidate_mask = self._nonzero_rows & ~self.movies["id"].isin(ids).to_numpy()
        candidate_mask &= similarities > 0
        candidate_mask &= self._genre_mask(genre)
        candidate_rows = np.flatnonzero(candidate_mask)
        if not len(candidate_rows):
            return pd.DataFrame(columns=columns)

        quality, popularity = self._quality_scores()
        if normalized_mode == "similar":
            scores = similarities
        elif normalized_mode == "balanced":
            scores = 0.72 * similarities + 0.20 * quality.to_numpy() + 0.08 * popularity.to_numpy()
        else:
            scores = 0.55 * similarities + 0.35 * quality.to_numpy() + 0.10 * (1 - popularity.to_numpy())

        titles = self.movies["title"].astype(str).str.casefold().to_numpy()
        ids_array = self.movies["id"].to_numpy()
        order = np.lexsort(
            (
                ids_array[candidate_rows],
                titles[candidate_rows],
                -similarities[candidate_rows],
                -scores[candidate_rows],
            )
        )
        selected = candidate_rows[order[:k]]
        result = self.movies.iloc[selected].copy()
        result["similarity"] = similarities[selected]
        result["recommendation_score"] = scores[selected]
        result["recommendation_mode"] = normalized_mode
        return result.reset_index(drop=True)

    def match_reasons(
        self,
        seed_ids: Iterable[int],
        candidate_id: int,
        limit: int = 3,
    ) -> list[str]:
        """Describe the strongest human-readable metadata signals shared by a match."""
        limit = self._validate_limit(limit, "limit")
        candidate = self.get_movie(candidate_id)
        if candidate is None:
            raise KeyError(f"Movie ID {candidate_id} was not found in the movie catalog.")
        seed_movies = [self.get_movie(movie_id) for movie_id in dict.fromkeys(seed_ids)]
        seeds = [movie for movie in seed_movies if movie is not None]
        if not seeds:
            raise ValueError("seed_ids must contain at least one movie ID.")

        reasons: list[str] = []
        labels = (
            ("genres_list", "Shared genre"),
            ("director_names", "Same director"),
            ("cast_names", "Shared cast"),
            ("keywords_list", "Shared theme"),
        )
        for field, label in labels:
            candidate_values = set(self._metadata_values(candidate.get(field)))
            profile_values = set().union(
                *(set(self._metadata_values(seed.get(field))) for seed in seeds)
            )
            shared = candidate_values & profile_values
            if not shared:
                continue
            names = ", ".join(self._display_name(value) for value in sorted(shared)[:2])
            reasons.append(f"{label}: {names}")
            if len(reasons) >= limit:
                return reasons
        return reasons or ["Similar story, genre, and talent signals"]
