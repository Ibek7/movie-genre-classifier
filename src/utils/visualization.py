"""Visualization helpers for exploring and reporting model results.

All functions return the :class:`matplotlib.figure.Figure` so callers can
either display it interactively or save it without side-effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_genre_distribution(
    labels: pd.Series,
    title: str = "Genre Distribution",
    top_n: int = 15,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart of the *top_n* most common genres in *labels*.

    Parameters
    ----------
    labels:
        Series (or array-like) of genre strings.
    title:
        Chart title.
    top_n:
        Number of genres to display; the rest are omitted.
    save_path:
        When provided the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    counts = pd.Series(labels).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_model_comparison(
    metrics: dict[str, float],
    title: str = "Model Accuracy Comparison",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing accuracy scores across models.

    Parameters
    ----------
    metrics:
        Mapping of model name → accuracy (0–1).
    title:
        Chart title.
    save_path:
        When provided the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    names = list(metrics.keys())
    values = [metrics[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, values, color=["steelblue", "coral"][: len(names)], edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
