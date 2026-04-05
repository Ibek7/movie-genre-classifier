"""Preprocessing sub-package for the movie-genre-classifier."""

from .cleaner import (
    load_data,
    drop_duplicates,
    drop_missing,
    normalize_text,
    clean_and_save,
)

__all__ = [
    "load_data",
    "drop_duplicates",
    "drop_missing",
    "normalize_text",
    "clean_and_save",
]
