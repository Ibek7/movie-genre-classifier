"""Utility helpers shared across the movie-genre-classifier project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


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
