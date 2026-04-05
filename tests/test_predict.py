import pandas as pd
import pytest
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.predict import predict, predict_from_csv, predict_proba


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
