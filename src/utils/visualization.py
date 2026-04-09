"""Visualization helpers for exploring and reporting model results.

All functions return the :class:`matplotlib.figure.Figure` so callers can
either display it interactively or save it without side-effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


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


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Heatmap of the confusion matrix for *y_true* vs *y_pred*.

    Parameters
    ----------
    y_true:
        Ground-truth genre labels.
    y_pred:
        Predicted genre labels from the model.
    labels:
        Ordered list of class names.  When *None* the sorted unique values of
        *y_true* are used.
    title:
        Figure title.
    save_path:
        When provided the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if labels is None:
        labels = sorted(set(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
