"""TF-IDF feature-engineering utilities.

Wraps scikit-learn's :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
with project-specific defaults tuned during notebook experiments.

Public API
----------
fit_vectorizer : Fit a new vectorizer on a corpus of plot strings.
transform_plots : Transform plot strings using a fitted vectorizer.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import Tuple, Optional, List, Union

__all__ = ["fit_vectorizer", "transform_plots"]

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