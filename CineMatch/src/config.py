"""Project paths and validated environment-based settings for CineMatch."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Resolved paths and tunable limits shared by the build and app workflows."""

    project_root: Path = PROJECT_ROOT
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    max_features: int = 5000
    max_parse_failure_rate: float = 0.05
    principal_cast_size: int = 5

    @classmethod
    def from_environment(cls) -> Settings:
        """Create settings from CINEMATCH_* variables, validating numeric limits."""
        root = Path(os.getenv("CINEMATCH_ROOT", str(PROJECT_ROOT))).expanduser().resolve()
        max_features = int(os.getenv("CINEMATCH_MAX_FEATURES", "5000"))
        parse_failure_rate = float(os.getenv("CINEMATCH_MAX_PARSE_FAILURE_RATE", "0.05"))
        cast_size = int(os.getenv("CINEMATCH_PRINCIPAL_CAST_SIZE", "5"))

        if max_features < 1:
            raise ValueError("CINEMATCH_MAX_FEATURES must be a positive integer.")
        if not 0 <= parse_failure_rate <= 1:
            raise ValueError("CINEMATCH_MAX_PARSE_FAILURE_RATE must be between 0 and 1.")
        if cast_size < 1:
            raise ValueError("CINEMATCH_PRINCIPAL_CAST_SIZE must be a positive integer.")

        return cls(
            project_root=root,
            raw_data_dir=Path(
                os.getenv("CINEMATCH_RAW_DATA_DIR", str(root / "data" / "raw"))
            ).expanduser(),
            processed_data_dir=Path(
                os.getenv(
                    "CINEMATCH_PROCESSED_DATA_DIR",
                    str(root / "data" / "processed"),
                )
            ).expanduser(),
            artifacts_dir=Path(
                os.getenv("CINEMATCH_ARTIFACTS_DIR", str(root / "artifacts"))
            ).expanduser(),
            max_features=max_features,
            max_parse_failure_rate=parse_failure_rate,
            principal_cast_size=cast_size,
        )


MOVIES_FILENAME = "tmdb_5000_movies.csv"
CREDITS_FILENAME = "tmdb_5000_credits.csv"
