"""Shared pytest fixtures available to all test modules.

Fixtures defined here are automatically discovered by pytest and can be
used in any test file without an explicit import.
"""

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Reusable toy corpus
# ---------------------------------------------------------------------------

SAMPLE_PLOTS = [
    "A brave hero fights dragons and saves the kingdom",
    "Two people fall in love during a summer vacation",
    "A detective investigates a mysterious murder case",
    "Soldiers battle through enemy lines in World War Two",
    "A young wizard discovers magical powers at school",
    "Two lovers reunite after years apart in Paris",
]

SAMPLE_LABELS = ["Action", "Romance", "Mystery", "War", "Fantasy", "Romance"]


@pytest.fixture(scope="session")
def sample_plots() -> list[str]:
    """Return a small list of raw plot strings."""
    return list(SAMPLE_PLOTS)


@pytest.fixture(scope="session")
def sample_labels() -> list[str]:
    """Return genre labels aligned with :func:`sample_plots`."""
    return list(SAMPLE_LABELS)


@pytest.fixture(scope="session")
def sample_series(sample_plots) -> pd.Series:
    """Return plot strings as a pandas Series."""
    return pd.Series(sample_plots)


# ---------------------------------------------------------------------------
# Pre-trained artefacts (session-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained_vectorizer(sample_series) -> TfidfVectorizer:
    """Return a TF-IDF vectorizer fitted on the sample corpus."""
    vec = TfidfVectorizer(max_features=50, stop_words="english")
    vec.fit(sample_series)
    return vec


@pytest.fixture(scope="session")
def trained_lr(trained_vectorizer, sample_series, sample_labels) -> LogisticRegression:
    """Return a LogisticRegression model trained on the sample corpus."""
    X = trained_vectorizer.transform(sample_series)
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X, sample_labels)
    return clf


# ---------------------------------------------------------------------------
# Serialised artefact paths (function-scoped so each test gets a fresh dir)
# ---------------------------------------------------------------------------

@pytest.fixture()
def artefact_paths(tmp_path, trained_vectorizer, trained_lr):
    """Persist vectorizer and LR model to tmp_path; return path dict."""
    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "lr.joblib"
    joblib.dump(trained_vectorizer, vec_path)
    joblib.dump(trained_lr, model_path)
    return {"vec": vec_path, "model": model_path}


# ---------------------------------------------------------------------------
# Minimal labelled CSV fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_csv(tmp_path) -> "pathlib.Path":
    """Write a tiny labelled CSV and return its path."""
    import pathlib  # local import to keep module-level imports clean
    path: pathlib.Path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "Title": [f"Movie {i}" for i in range(len(SAMPLE_PLOTS))],
            "Plot": SAMPLE_PLOTS,
            "Genre": [f"{lbl}|Drama" for lbl in SAMPLE_LABELS],
        }
    ).to_csv(path, index=False)
    return path
