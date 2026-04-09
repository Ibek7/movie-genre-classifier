import pandas as pd
import pytest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.predict import (
    predict,
    predict_from_csv,
    predict_proba,
    predict_with_confidence,
    predict_top_k,
)


def test_predict_rejects_empty_plot_list():
    with pytest.raises(ValueError, match="at least one plot summary"):
        predict([], "missing_vec.joblib", "missing_model.joblib")


def test_predict_rejects_all_empty_normalized_plots(tmp_path):
    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "model.joblib"
    vec_path.write_text("placeholder")
    model_path.write_text("placeholder")

    with pytest.raises(ValueError, match="at least one non-empty summary"):
        predict(["   ", "\n\t"], vec_path, model_path)


def test_predict_rejects_missing_vectorizer(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("placeholder")

    with pytest.raises(FileNotFoundError, match="Vectorizer not found"):
        predict(["A plot"], tmp_path / "vec.joblib", model_path)


def test_predict_rejects_missing_model(tmp_path):
    vec_path = tmp_path / "vec.joblib"
    vec_path.write_text("placeholder")

    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict(["A plot"], vec_path, tmp_path / "model.joblib")


def test_predict_from_csv_rejects_empty_input(tmp_path):
    input_csv = tmp_path / "empty.csv"
    output_csv = tmp_path / "out.csv"
    pd.DataFrame(columns=["Plot"]).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="Input CSV is empty"):
        predict_from_csv(
            input_csv=input_csv,
            output_csv=output_csv,
            vec_path=tmp_path / "vec.joblib",
            model_path=tmp_path / "model.joblib",
        )


def test_predict_from_csv_rejects_missing_input_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input CSV not found"):
        predict_from_csv(
            input_csv=tmp_path / "missing.csv",
            output_csv=tmp_path / "out.csv",
            vec_path=tmp_path / "vec.joblib",
            model_path=tmp_path / "model.joblib",
        )


def test_predict_from_csv_rejects_missing_plot_column(tmp_path):
    input_csv = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="must contain a 'Plot' column"):
        predict_from_csv(
            input_csv=input_csv,
            output_csv=tmp_path / "out.csv",
            vec_path=tmp_path / "vec.joblib",
            model_path=tmp_path / "model.joblib",
        )


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------

def test_predict_proba_rejects_empty_plot_list():
    with pytest.raises(ValueError, match="at least one plot summary"):
        predict_proba([], "missing_vec.joblib", "missing_model.joblib")


def test_predict_proba_raises_on_missing_vectorizer(tmp_path):
    model_path = tmp_path / "model.joblib"
    model_path.write_text("placeholder")
    with pytest.raises(FileNotFoundError, match="Vectorizer not found"):
        predict_proba(["A plot"], tmp_path / "vec.joblib", model_path)


def test_predict_proba_raises_when_model_has_no_predict_proba(tmp_path):
    """Models that don't support predict_proba (e.g. LinearSVC) should raise AttributeError."""
    from sklearn.svm import LinearSVC

    corpus = pd.Series(["action hero saves world", "romance love story"])
    labels = ["Action", "Romance"]
    vec = TfidfVectorizer(max_features=20)
    X = vec.fit_transform(corpus)
    clf = LinearSVC()
    clf.fit(X, labels)

    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "model.joblib"
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)

    with pytest.raises(AttributeError, match="does not support predict_proba"):
        predict_proba(["action hero"], vec_path, model_path)


def test_predict_proba_returns_class_scores(tmp_path):
    """predict_proba should return one dict per plot with probabilities summing to ~1."""
    corpus = pd.Series([
        "action hero saves world",
        "romance love story",
        "action fight battle",
        "love kiss romance",
    ])
    labels = ["Action", "Romance", "Action", "Romance"]
    vec = TfidfVectorizer(max_features=50)
    X = vec.fit_transform(corpus)
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X, labels)

    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "model.joblib"
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)

    results = predict_proba(["action hero fights"], vec_path, model_path)
    assert len(results) == 1
    row = results[0]
    assert set(row.keys()) == {"Action", "Romance"}
    assert abs(sum(row.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Shared fixture — tiny trained vectorizer + LR model
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_model(tmp_path):
    """Return (vec_path, model_path) for a two-class micro model."""
    corpus = pd.Series([
        "action hero saves world",
        "romance love story kiss",
        "action battle fight war",
        "romance couple wedding date",
        "action explosion stunts",
        "romance heartbreak feelings",
    ])
    labels = ["Action", "Romance", "Action", "Romance", "Action", "Romance"]

    vec = TfidfVectorizer(max_features=50)
    X = vec.fit_transform(corpus)
    clf = LogisticRegression(max_iter=300, random_state=42)
    clf.fit(X, labels)

    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "model.joblib"
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)
    return vec_path, model_path


# ---------------------------------------------------------------------------
# predict_with_confidence tests
# ---------------------------------------------------------------------------

def test_predict_with_confidence_returns_genre_and_confidence_keys(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_with_confidence(["action hero fights"], vec_path, model_path)
    assert len(results) == 1
    assert set(results[0].keys()) == {"genre", "confidence"}


def test_predict_with_confidence_confidence_in_unit_interval(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_with_confidence(
        ["romantic love story", "action battle war"], vec_path, model_path
    )
    for res in results:
        assert 0.0 <= res["confidence"] <= 1.0


def test_predict_with_confidence_genre_is_known_label(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_with_confidence(["epic battle scene"], vec_path, model_path)
    assert results[0]["genre"] in {"Action", "Romance"}


def test_predict_with_confidence_returns_one_result_per_plot(tiny_model):
    vec_path, model_path = tiny_model
    plots = ["plot one", "plot two", "plot three"]
    results = predict_with_confidence(plots, vec_path, model_path)
    assert len(results) == len(plots)


def test_predict_with_confidence_propagates_attribute_error_on_no_proba(tmp_path):
    from sklearn.svm import LinearSVC

    corpus = pd.Series(["action hero", "romance love"])
    labels = ["Action", "Romance"]
    vec = TfidfVectorizer(max_features=10)
    X = vec.fit_transform(corpus)
    clf = LinearSVC()
    clf.fit(X, labels)

    vec_path = tmp_path / "vec.joblib"
    model_path = tmp_path / "model.joblib"
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)

    with pytest.raises(AttributeError, match="does not support predict_proba"):
        predict_with_confidence(["action hero"], vec_path, model_path)


# ---------------------------------------------------------------------------
# predict_top_k tests
# ---------------------------------------------------------------------------

def test_predict_top_k_zero_raises_value_error(tiny_model):
    vec_path, model_path = tiny_model
    with pytest.raises(ValueError, match="k must be a positive integer"):
        predict_top_k(["action hero"], vec_path, model_path, k=0)


def test_predict_top_k_negative_raises_value_error(tiny_model):
    vec_path, model_path = tiny_model
    with pytest.raises(ValueError):
        predict_top_k(["action hero"], vec_path, model_path, k=-1)


def test_predict_top_k_returns_nested_list(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_top_k(["action hero saves world"], vec_path, model_path, k=2)
    assert isinstance(results, list)
    assert isinstance(results[0], list)


def test_predict_top_k_k1_returns_single_candidate_per_plot(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_top_k(["romantic love story"], vec_path, model_path, k=1)
    assert len(results[0]) == 1
    assert set(results[0][0].keys()) == {"genre", "confidence"}


def test_predict_top_k_sorted_descending(tiny_model):
    vec_path, model_path = tiny_model
    results = predict_top_k(["action battle fight"], vec_path, model_path, k=2)
    confidences = [item["confidence"] for item in results[0]]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_top_k_clamps_to_n_classes(tiny_model):
    """Requesting more candidates than classes should return at most n_classes."""
    vec_path, model_path = tiny_model
    results = predict_top_k(["any plot text"], vec_path, model_path, k=100)
    # Model has 2 classes; result should be clamped to 2
    assert len(results[0]) <= 2

