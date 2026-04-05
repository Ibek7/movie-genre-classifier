"""Hyperparameter tuning utilities using scikit-learn GridSearchCV.

Run a grid search over TF-IDF + classifier pipelines and return the best
estimator along with a results summary.  Intended to be called from a
notebook or as a standalone script for experimentation.

Usage example
-------------
>>> from src.models.tune_hyperparameters import tune_logistic_regression
>>> best, results = tune_logistic_regression(X_train, y_train)
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

__all__ = ["tune_logistic_regression", "tune_naive_bayes"]


def tune_logistic_regression(
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple:
    """Grid-search over TF-IDF + Logistic Regression hyperparameters.

    Parameters
    ----------
    X_train:
        Raw (or cleaned) plot strings for training.
    y_train:
        Corresponding genre labels.
    cv:
        Number of cross-validation folds.
    n_jobs:
        Parallel jobs for GridSearchCV (``-1`` uses all cores).
    verbose:
        Verbosity level forwarded to :class:`~sklearn.model_selection.GridSearchCV`.

    Returns
    -------
    best_estimator : sklearn.pipeline.Pipeline
        The fitted pipeline with the best found parameters.
    results_df : pandas.DataFrame
        Full CV results table sorted by mean test score (descending).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)

    results_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Best LR params : {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    return search.best_estimator_, results_df


def tune_naive_bayes(
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 3,
    n_jobs: int = -1,
    verbose: int = 1,
) -> tuple:
    """Grid-search over TF-IDF + Multinomial Naive Bayes hyperparameters.

    Parameters
    ----------
    X_train:
        Raw (or cleaned) plot strings for training.
    y_train:
        Corresponding genre labels.
    cv:
        Number of cross-validation folds.
    n_jobs:
        Parallel jobs for GridSearchCV (``-1`` uses all cores).
    verbose:
        Verbosity level forwarded to :class:`~sklearn.model_selection.GridSearchCV`.

    Returns
    -------
    best_estimator : sklearn.pipeline.Pipeline
        The fitted pipeline with the best found parameters.
    results_df : pandas.DataFrame
        Full CV results table sorted by mean test score (descending).
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__alpha": [0.01, 0.1, 1.0],
    }

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)

    results_df = (
        pd.DataFrame(search.cv_results_)
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Best NB params : {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.4f}")

    return search.best_estimator_, results_df
