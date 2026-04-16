"""Tests for src.utils.visualization helpers."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.utils.visualization import normalize_confusion_matrix_values, plot_top_n_genres


def test_normalize_confusion_matrix_values_row_sums_to_one_when_nonzero_rows():
    cm = np.array([[8, 2], [1, 9]])
    norm = normalize_confusion_matrix_values(cm)
    assert np.allclose(norm.sum(axis=1), np.array([1.0, 1.0]))


def test_normalize_confusion_matrix_values_handles_zero_rows():
    cm = np.array([[0, 0], [3, 1]])
    norm = normalize_confusion_matrix_values(cm)
    assert np.allclose(norm[0], np.array([0.0, 0.0]))
    assert np.allclose(norm[1], np.array([0.75, 0.25]))


def test_normalize_confusion_matrix_values_returns_float_matrix():
    cm = np.array([[1, 1], [2, 0]], dtype=int)
    norm = normalize_confusion_matrix_values(cm)
    assert np.issubdtype(norm.dtype, np.floating)


# ---------------------------------------------------------------------------
# plot_top_n_genres
# ---------------------------------------------------------------------------

def test_plot_top_n_genres_returns_figure():
    import matplotlib.pyplot as plt
    labels = pd.Series(["Drama", "Action", "Drama", "Comedy", "Action", "Drama"])
    fig = plot_top_n_genres(labels, top_n=3)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_plot_top_n_genres_respects_top_n():
    import matplotlib.pyplot as plt
    labels = pd.Series(["Drama", "Action", "Comedy", "Horror", "Thriller"] * 4)
    fig = plot_top_n_genres(labels, top_n=3)
    ax = fig.axes[0]
    assert len(ax.patches) <= 3
    plt.close("all")
