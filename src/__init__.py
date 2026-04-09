"""movie-genre-classifier — top-level package.

Classifies movie genres from Wikipedia plot summaries using TF-IDF features and
scikit-learn models (Multinomial Naive Bayes, Logistic Regression).

Sub-packages
------------
preprocessing : Data cleaning and tokenization utilities.
features      : TF-IDF vectorization.
models        : Training and inference pipelines.
utils         : Shared helpers (I/O, formatting, visualization).
constants     : Centralised genre labels and hyper-parameter defaults.

Quick-start
-----------
>>> from src.models.predict import predict_with_confidence
>>> results = predict_with_confidence(["A hero saves the world"], vec_path, model_path)
>>> results[0]
{'genre': 'Action', 'confidence': 0.87}
"""

__version__ = "0.3.0"
__author__ = "Bekam Guta"
__email__ = "bekamdawit551@gmail.com"
__description__ = (
    "Movie genre classifier using TF-IDF features and scikit-learn models, "
    "trained on Wikipedia plot summaries."
)
__license__ = "MIT"
__url__ = "https://github.com/Ibek7/movie-genre-classifier"

__all__ = [
    "preprocessing",
    "features",
    "models",
    "utils",
    "constants",
]
