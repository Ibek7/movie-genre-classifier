"""Models sub-package for the movie-genre-classifier.

Public API
----------
Training
  train_and_save_models  — fit NB + LR classifiers and persist artefacts.

Inference
  predict                 — batch genre prediction from raw plot strings.
  predict_genre           — single-plot convenience wrapper.
  predict_proba           — per-class probability scores.
  predict_with_confidence — top genre + confidence score per plot.
  predict_top_k           — ranked top-k genres with probabilities.
  predict_from_csv        — CSV in → CSV out batch inference.
  batch_predict_from_dir  — run inference on a whole directory of CSVs.
"""

from .train import train_and_save_models
from .predict import (
    predict,
    predict_genre,
    predict_proba,
    predict_with_confidence,
    predict_top_k,
    predict_from_csv,
    batch_predict_from_dir,
)

__all__ = [
    "train_and_save_models",
    "predict",
    "predict_genre",
    "predict_proba",
    "predict_with_confidence",
    "predict_top_k",
    "predict_from_csv",
    "batch_predict_from_dir",
]
