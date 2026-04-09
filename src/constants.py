"""Project-wide constants for the movie-genre-classifier.

Centralises fixed values such as the canonical genre label list so that
training, inference, and evaluation code all reference a single source of
truth rather than repeating magic strings.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Genre labels
# ---------------------------------------------------------------------------

#: The 15 primary genres used after consolidation (ordered alphabetically).
#: Plots whose primary genre falls outside this set are mapped to "other".
TOP_GENRES: list[str] = [
    "action",
    "adventure",
    "animation",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "musical",
    "mystery",
    "other",
    "romance",
    "science fiction",
    "thriller",
    "western",
]

#: Alias used for consolidating rare genres.
RARE_GENRE_LABEL: str = "other"

#: Minimum number of training samples required for a genre to be kept as its
#: own class (rather than collapsed into :data:`RARE_GENRE_LABEL`).
MIN_GENRE_SAMPLES: int = 100

# ---------------------------------------------------------------------------
# Default model artefact names
# ---------------------------------------------------------------------------

#: Default filename for the fitted TF-IDF vectorizer.
DEFAULT_VECTORIZER_FILENAME: str = "tfidf_vectorizer.joblib"

#: Default filenames for the two baseline classifiers.
DEFAULT_MODEL_FILENAMES: dict[str, str] = {
    "nb": "nb_model.joblib",
    "lr": "lr_model.joblib",
}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

#: Default maximum vocabulary size for :class:`sklearn.feature_extraction.text.TfidfVectorizer`.
DEFAULT_MAX_FEATURES: int = 5_000

#: Default random seed for reproducible train/test splits.
DEFAULT_RANDOM_STATE: int = 42

#: Default fraction of data reserved for evaluation.
DEFAULT_TEST_SIZE: float = 0.2
