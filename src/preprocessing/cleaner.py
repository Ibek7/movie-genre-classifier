"""Text-cleaning utilities for raw movie plot data.

Provides functions to load, deduplicate, prune, normalize, and persist
the cleaned dataset.  The main entry-point for a full run is
:func:`clean_and_save`.
"""

import pandas as pd
import re
from pathlib import Path


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.
    """
    return pd.read_csv(input_path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicates based on available columns (Title and/or Plot).
    """
    # Determine which columns exist for deduplication
    subset_cols = [col for col in ("Title", "Plot") if col in df.columns]
    if not subset_cols:
        return df
    return df.drop_duplicates(subset=subset_cols)


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing Plot or Genre.
    """
    return df.dropna(subset=["Plot", "Genre"])


def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing, removing HTML tags, non-alphanumeric chars, and collapsing whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)            # strip HTML tags
    text = re.sub(r"[^a-z0-9\s\|]", " ", text)    # keep letters/numbers/pipes
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*.

    Parameters
    ----------
    text:
        Raw or pre-normalised plot string.

    Returns
    -------
    int
        Token count; 0 for empty or non-string input.

    Examples
    --------
    >>> word_count("A hero saves the world")
    5
    """
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def average_word_length(text: str) -> float:
    """Return the average token length for whitespace-separated words.

    Parameters
    ----------
    text:
        Raw or pre-normalised plot string.

    Returns
    -------
    float
        Mean token length. Returns ``0.0`` for empty/non-string input.
    """
    if not text or not isinstance(text, str):
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def sentence_count(text: str) -> int:
    """Return an approximate sentence count by splitting on ``.``, ``!``, ``?``.

    Parameters
    ----------
    text:
        Raw plot string (not pre-normalised, since normalisation strips punctuation).

    Returns
    -------
    int
        Sentence count; at least 1 for any non-empty string.

    Examples
    --------
    >>> sentence_count("He ran. She laughed! Why?")
    3
    """
    if not text or not isinstance(text, str):
        return 0
    parts = re.split(r"[.!?]+", text.strip())
    return max(1, sum(1 for p in parts if p.strip()))


def truncate_plot(text: str, max_words: int = 200) -> str:
    """Truncate *text* to at most *max_words* whitespace-separated tokens.

    Long Wikipedia plot summaries can exceed 1 000 words.  Truncating to a
    fixed budget keeps memory use predictable and prevents outlier documents
    from dominating TF-IDF statistics.

    Parameters
    ----------
    text:
        Raw or pre-normalised plot string.
    max_words:
        Maximum number of tokens to keep (default 200).  Must be a positive
        integer; values ≤ 0 raise :class:`ValueError`.

    Returns
    -------
    str
        The (possibly truncated) text.  If *text* has fewer than *max_words*
        tokens it is returned unchanged.  Non-string or empty input returns
        an empty string.

    Examples
    --------
    >>> truncate_plot("one two three four five", max_words=3)
    'one two three'
    >>> truncate_plot("short", max_words=100)
    'short'
    """
    if max_words <= 0:
        raise ValueError(f"max_words must be a positive integer, got {max_words}")
    if not text or not isinstance(text, str):
        return ""
    tokens = text.split()
    return " ".join(tokens[:max_words])


def filter_short_plots(df: pd.DataFrame, min_words: int = 20, plot_col: str = "Plot") -> pd.DataFrame:
    """Remove rows whose plot contains fewer than *min_words* tokens.

    Very short plots (stub articles, redirect pages, etc.) add noise and
    rarely carry enough signal for reliable genre classification.

    Parameters
    ----------
    df:
        DataFrame that must contain a column named *plot_col*.
    min_words:
        Minimum number of whitespace-separated tokens required to keep a row.
        Default is 20.  Must be a positive integer.
    plot_col:
        Name of the column containing plot text (default ``"Plot"``).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with short-plot rows removed.  The original index
        is preserved (not reset).

    Raises
    ------
    ValueError
        If *plot_col* is not present in *df* or *min_words* ≤ 0.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Plot": ["short", "a much longer plot summary with many words here"]})
    >>> filter_short_plots(df, min_words=5)
       Plot
    1  a much longer plot summary with many words here
    """
    if min_words <= 0:
        raise ValueError(f"min_words must be a positive integer, got {min_words}")
    if plot_col not in df.columns:
        raise ValueError(f"Column '{plot_col}' not found in DataFrame")
    mask = df[plot_col].fillna("").apply(word_count) >= min_words
    return df[mask]


def clean_and_save(input_path: str, output_path: str) -> None:
    """
    Run full cleaning pipeline and save processed CSV.

    Steps:
    1. Load raw data.
    2. Drop duplicates.
    3. Drop missing values.
    4. Normalize the 'Plot' text.
    5. Save cleaned DataFrame to output_path.
    """
    df = load_data(input_path)
    df = drop_duplicates(df)
    df = drop_missing(df)
    df["Plot"] = df["Plot"].apply(normalize_text)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
