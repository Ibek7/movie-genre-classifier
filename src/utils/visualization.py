"""Visualization helpers for exploring and reporting model results.

All functions return the :class:`matplotlib.figure.Figure` so callers can
either display it interactively or save it without side-effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def normalize_confusion_matrix_values(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion matrix so each row sums to 1 when possible.

    Rows with sum 0 are left as all zeros.
    """
    matrix = np.asarray(cm, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


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


def plot_precision_recall_per_class(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    title: str = "Precision & Recall per Class",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Grouped bar chart showing precision and recall for every genre class.

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

    precision, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 5))
    ax.bar(x - width / 2, precision, width, label="Precision", color="steelblue", edgecolor="white")
    ax.bar(x + width / 2, recall, width, label="Recall", color="coral", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_learning_curve(
    train_sizes: List[int],
    train_scores: List[float],
    val_scores: List[float],
    title: str = "Learning Curve",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Line chart comparing training and validation accuracy across dataset sizes.

    Useful for diagnosing bias/variance trade-offs: if train accuracy is high
    but val accuracy is low the model is overfitting; if both are low the model
    is underfitting.

    Parameters
    ----------
    train_sizes:
        List of training-set sizes (x-axis).
    train_scores:
        Mean training accuracy at each size.
    val_scores:
        Mean validation accuracy at each size.
    title:
        Figure title.
    save_path:
        When provided the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores, "o-", color="steelblue", label="Train accuracy")
    ax.plot(train_sizes, val_scores, "s--", color="coral", label="Validation accuracy")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_f1_heatmap(
    report: Dict[str, Dict[str, float]],
    title: str = "F1 Score per Class",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Single-column heatmap of per-class F1 scores from a classification report.

    Parameters
    ----------
    report:
        Dict produced by :func:`src.utils.helpers.compute_classification_report`
        (``output_dict=True``).  Keys are class names; values are dicts
        containing at least ``"f1-score"``.
    title:
        Figure title.
    save_path:
        When provided the figure is saved to this path before being returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    skip = {"accuracy", "macro avg", "weighted avg"}
    classes = [c for c in report if c not in skip]
    f1_scores = np.array([report[c]["f1-score"] for c in classes]).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(3, max(4, len(classes) // 2)))
    im = ax.imshow(f1_scores, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks([])
    ax.set_title(title)
    for i, score in enumerate(f1_scores.flatten()):
        ax.text(0, i, f"{score:.2f}", ha="center", va="center",
                color="black", fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
