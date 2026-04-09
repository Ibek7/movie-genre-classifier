"""Unit tests for src.constants — verifies invariants on all exported constants."""

import pytest

from src.constants import (
    TOP_GENRES,
    RARE_GENRE_LABEL,
    MIN_GENRE_SAMPLES,
    DEFAULT_VECTORIZER_FILENAME,
    DEFAULT_MODEL_FILENAMES,
    DEFAULT_MAX_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)


# ---------------------------------------------------------------------------
# TOP_GENRES
# ---------------------------------------------------------------------------

def test_top_genres_is_list():
    assert isinstance(TOP_GENRES, list)


def test_top_genres_non_empty():
    assert len(TOP_GENRES) > 0


def test_top_genres_all_strings():
    assert all(isinstance(g, str) for g in TOP_GENRES)


def test_top_genres_contains_rare_label():
    assert RARE_GENRE_LABEL in TOP_GENRES


def test_top_genres_no_duplicates():
    assert len(TOP_GENRES) == len(set(TOP_GENRES))


def test_top_genres_all_lowercase():
    assert all(g == g.lower() for g in TOP_GENRES)


# ---------------------------------------------------------------------------
# RARE_GENRE_LABEL
# ---------------------------------------------------------------------------

def test_rare_genre_label_is_string():
    assert isinstance(RARE_GENRE_LABEL, str)


def test_rare_genre_label_non_empty():
    assert RARE_GENRE_LABEL.strip() != ""


# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------

def test_min_genre_samples_positive():
    assert MIN_GENRE_SAMPLES > 0


def test_default_max_features_positive():
    assert DEFAULT_MAX_FEATURES > 0


def test_default_test_size_in_unit_interval():
    assert 0.0 < DEFAULT_TEST_SIZE < 1.0


def test_default_random_state_non_negative():
    assert DEFAULT_RANDOM_STATE >= 0


# ---------------------------------------------------------------------------
# File-name constants
# ---------------------------------------------------------------------------

def test_default_vectorizer_filename_has_joblib_extension():
    assert DEFAULT_VECTORIZER_FILENAME.endswith(".joblib")


def test_default_model_filenames_has_nb_and_lr_keys():
    assert "nb" in DEFAULT_MODEL_FILENAMES
    assert "lr" in DEFAULT_MODEL_FILENAMES


def test_default_model_filenames_are_joblib():
    for name in DEFAULT_MODEL_FILENAMES.values():
        assert name.endswith(".joblib"), f"{name!r} should end with .joblib"
