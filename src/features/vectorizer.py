"""TF-IDF feature-engineering utilities.

Wraps scikit-learn's :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
with project-specific defaults tuned during notebook experiments.

Public API
----------
fit_vectorizer : Fit a new vectorizer on a corpus of plot strings.
transform_plots : Transform plot strings using a fitted vectorizer.
"""

from pathlib import Path
from typing import Tuple, Optional, List, Union

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["fit_vectorizer", "transform_plots", "save_vectorizer", "load_vectorizer"]

def fit_vectorizer(
    plots: pd.Series,
    max_features: int = 5000,  # Optimized from notebook testing
    ngram_range: Tuple[int, int] = (1, 1),
    max_df: float = 0.95,
    min_df: int = 15,
    stop_words: Optional[Union[str, List[str]]] = 'english'
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the given plot texts.
    Returns the fitted vectorizer.
    """
    adjusted_min_df = min(min_df, max(1, len(plots) - 1))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=adjusted_min_df,
        stop_words=stop_words,
        token_pattern=r"(?u)\b\w+\b"
    )
    vectorizer.fit(plots)
    return vectorizer

def transform_plots(
    vectorizer: TfidfVectorizer,
    plots: pd.Series
):
    """
    Transform plot texts into TF-IDF feature matrix.
    """
    return vectorizer.transform(plots)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> Path:
    """Persist a fitted TF-IDF vectorizer to disk using joblib.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    vectorizer:
        A fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
    path:
        Destination file path (conventionally ``*.joblib``).

    Returns
    -------
    pathlib.Path
        The resolved path where the file was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, p)
    return p


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Load a previously saved TF-IDF vectorizer from disk.

    Parameters
    ----------
    path:
        Path to a joblib file produced by :func:`save_vectorizer`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.

    Returns
    -------
    TfidfVectorizer
        The deserialized, ready-to-use vectorizer.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {p}")
    return joblib.load(p)