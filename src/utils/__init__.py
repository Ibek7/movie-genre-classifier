"""Utils sub-package for the movie-genre-classifier.

Public API
----------
helpers
  ensure_dir                   — create a directory (and parents) if absent.
  load_json / save_json        — read / write JSON with automatic dir creation.
  format_accuracy              — format a float as a percentage string.
  get_top_genres               — top-N most frequent genres from a Series.
  compute_classification_report — sklearn classification_report as a dict.
  format_duration              — human-readable seconds string (e.g. '2m 34s').
  elapsed_time                 — formatted wall-clock elapsed time since a start.
  safe_divide                  — zero-safe division helper.
  clamp_probability            — clamp numeric values to [0, 1].

visualization
  plot_genre_distribution      — bar chart of genre frequencies.
  plot_model_comparison        — horizontal bar chart of model accuracies.
  plot_confusion_matrix        — annotated confusion-matrix heatmap.
  plot_precision_recall_per_class — grouped precision/recall bar chart.
  plot_learning_curve          — train vs. validation accuracy over dataset size.
  plot_f1_heatmap              — per-class F1 single-column heatmap.
"""

from .helpers import (
    ensure_dir,
    load_json,
    save_json,
    format_accuracy,
    get_top_genres,
    compute_classification_report,
    format_duration,
    elapsed_time,
    safe_divide,
    clamp_probability,
)
from .visualization import (
    plot_genre_distribution,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_precision_recall_per_class,
    plot_learning_curve,
    plot_f1_heatmap,
)

__all__ = [
    # helpers
    "ensure_dir",
    "load_json",
    "save_json",
    "format_accuracy",
    "get_top_genres",
    "compute_classification_report",
    "format_duration",
    "elapsed_time",
    "safe_divide",
    "clamp_probability",
    # visualization
    "plot_genre_distribution",
    "plot_model_comparison",
    "plot_confusion_matrix",
    "plot_precision_recall_per_class",
    "plot_learning_curve",
    "plot_f1_heatmap",
]
