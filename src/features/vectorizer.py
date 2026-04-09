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

__all__ = ["fit_vectorizer", "transform_plots", "save_vectorizer", "load_vectorizer", "get_vocabulary_size"]

def fit_vectorizer(
    plots: pd.Series,
    max_features: int = 5000,  # Optimized from notebook testing
    ngram_range: Tuple[int, int] = (1, 1),
    max_df: float = 0.95,
    min_df: int = 15,
    stop_words: Optional[Union[str, List[str]]] = 'english'
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on the given plot texts.

    Parameters
    ----------
    plots:
        Series of raw or normalised plot strings.
    max_features:
        Maximum size of the vocabulary.  Only the top *max_features* terms
        ordered by corpus-wide TF-IDF are kept.
    ngram_range:
        Lower and upper boundary of the n-gram range, e.g. ``(1, 1)`` for
        unigrams only or ``(1, 2)`` to include bigrams.
    max_df:
        Ignore terms with a document frequency higher than this threshold.
        Float values are fractions; int values are absolute counts.
    min_df:
        Ignore terms with a document frequency lower than this threshold.
        Automatically capped to ``len(plots) - 1`` for small corpora.
    stop_words:
        Stop-word list to apply.  Pass ``'english'`` (default) or a custom
        list, or ``None`` to disable stop-word removal.

    Returns
    -------
    TfidfVectorizer
        A fitted scikit-learn :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
        ready for use with :func:`transform_plots`.
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
    """Transform plot texts into a sparse TF-IDF feature matrix.

    Parameters
    ----------
    vectorizer:
        A fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
        returned by :func:`fit_vectorizer`.
    plots:
        Series of raw or normalised plot strings to transform.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape ``(n_plots, n_features)``.
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


def get_vocabulary_size(vectorizer: TfidfVectorizer) -> int:
    """Return the number of features (vocabulary size) of a fitted vectorizer.

    This is a convenience wrapper around ``len(vectorizer.vocabulary_)`` that
    also provides a clear error message when called on an unfitted instance.

    Parameters
    ----------
    vectorizer:
        A fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.

    Returns
    -------
    int
        The number of unique tokens in the fitted vocabulary.

    Raises
    ------
    ValueError
        If *vectorizer* has not been fitted yet (no ``vocabulary_`` attribute).

    Examples
    --------
    >>> import pandas as pd
    >>> vec = fit_vectorizer(pd.Series(["action drama", "comedy horror"]), min_df=1)
    >>> get_vocabulary_size(vec) > 0
    True
    """
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "The vectorizer has not been fitted yet.  "
            "Call fit_vectorizer() before get_vocabulary_size()."
        )
    return len(vectorizer.vocabulary_)
