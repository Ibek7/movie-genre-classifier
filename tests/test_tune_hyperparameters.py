"""Tests for src.models.tune_hyperparameters."""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.models.tune_hyperparameters import tune_logistic_regression, tune_naive_bayes

# ---------------------------------------------------------------------------
# Minimal corpus that gives GridSearchCV something to work with (cv=2)
# ---------------------------------------------------------------------------

PLOTS = pd.Series([
    "action hero fights dragons saves kingdom",
    "romance love story summer vacation paris",
    "detective mystery murder crime investigation",
    "science fiction spaceship galaxy exploration",
    "action battle war soldiers brave hero",
    "comedy funny humor jokes laughter friends",
])

LABELS = ["Action", "Romance", "Mystery", "SciFi", "Action", "Comedy"]


class TestTuneLogisticRegression:
    def test_returns_pipeline(self):
        best, _ = tune_logistic_regression(PLOTS, LABELS, cv=2, verbose=0)
        assert isinstance(best, Pipeline)

    def test_returns_dataframe_with_mean_test_score(self):
        _, results = tune_logistic_regression(PLOTS, LABELS, cv=2, verbose=0)
        assert "mean_test_score" in results.columns

    def test_results_sorted_descending(self):
        _, results = tune_logistic_regression(PLOTS, LABELS, cv=2, verbose=0)
        scores = results["mean_test_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_estimator_can_predict(self):
        best, _ = tune_logistic_regression(PLOTS, LABELS, cv=2, verbose=0)
        preds = best.predict(PLOTS)
        assert len(preds) == len(PLOTS)
        assert all(isinstance(p, str) for p in preds)


class TestTuneNaiveBayes:
    def test_returns_pipeline(self):
        best, _ = tune_naive_bayes(PLOTS, LABELS, cv=2, verbose=0)
        assert isinstance(best, Pipeline)

    def test_returns_dataframe_with_mean_test_score(self):
        _, results = tune_naive_bayes(PLOTS, LABELS, cv=2, verbose=0)
        assert "mean_test_score" in results.columns

    def test_results_sorted_descending(self):
        _, results = tune_naive_bayes(PLOTS, LABELS, cv=2, verbose=0)
        scores = results["mean_test_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_estimator_can_predict(self):
        best, _ = tune_naive_bayes(PLOTS, LABELS, cv=2, verbose=0)
        preds = best.predict(PLOTS)
        assert len(preds) == len(PLOTS)
