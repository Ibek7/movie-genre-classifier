"""Utility helpers shared across the movie-genre-classifier project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import classification_report


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any missing parents) if it does not already exist.

    Returns the resolved :class:`~pathlib.Path` so callers can chain calls.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load and return the contents of a JSON file as a Python dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Serialise *data* to JSON and write it to *path*.

    Parent directories are created automatically if they do not exist.
    """
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)


def format_accuracy(value: float, decimals: int = 2) -> str:
    """Return *value* formatted as a percentage string.

    Examples
    --------
    >>> format_accuracy(0.724)
    '72.40%'
    """
    return f"{value * 100:.{decimals}f}%"


def get_top_genres(
    genre_series: pd.Series,
    top_n: int = 15,
    sep: str = "|",
    primary_only: bool = True,
) -> List[str]:
    """Return the *top_n* most frequent genres from a pipe-separated Series.

    Parameters
    ----------
    genre_series:
        A pandas Series of genre strings such as ``"Action|Drama"``.
    top_n:
        Maximum number of genres to return, ordered by descending frequency.
    sep:
        Delimiter used to split multi-label genre strings.
    primary_only:
        When *True* (default), only the first label in each value is counted.
        When *False*, all labels are counted individually.

    Returns
    -------
    list[str]
        Genre names sorted from most to least frequent, length ≤ *top_n*.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series(["Action|Drama", "Drama|Comedy", "Action"])
    >>> get_top_genres(s, top_n=2)
    ['Action', 'Drama']
    """
    if primary_only:
        labels = genre_series.dropna().str.split(sep).str[0]
    else:
        labels = genre_series.dropna().str.split(sep).explode()

    return labels.value_counts().head(top_n).index.tolist()


def compute_classification_report(
    y_true: List[str],
    y_pred: List[str],
    output_dict: bool = True,
) -> Dict[str, Any]:
    """Thin wrapper around :func:`sklearn.metrics.classification_report`.

    Parameters
    ----------
    y_true:
        Ground-truth genre labels.
    y_pred:
        Predicted genre labels.
    output_dict:
        When *True* (default) return the report as a nested dict suitable for
        JSON serialisation.  When *False* return the human-readable string.

    Returns
    -------
    dict | str
        Per-class precision, recall, F1, support plus macro/weighted averages.

    Examples
    --------
    >>> report = compute_classification_report(["Action", "Drama"], ["Action", "Action"])
    >>> "Drama" in report
    True
    """
    return classification_report(y_true, y_pred, output_dict=output_dict, zero_division=0)
