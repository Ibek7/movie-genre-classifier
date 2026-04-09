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
