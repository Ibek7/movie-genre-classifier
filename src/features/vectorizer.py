from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from typing import Tuple, Optional, List, Union

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
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
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