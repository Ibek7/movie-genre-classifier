"""End-to-end integration tests covering the full pipeline.

These tests exercise the complete flow:
  raw CSV → clean → train → save artefacts → predict → predict_proba
"""

import pandas as pd
import pytest

from src.preprocessing.cleaner import clean_and_save
from src.models.train import train_and_save_models
from src.models.predict import predict, predict_genre, predict_proba


@pytest.fixture()
def toy_dataset(tmp_path):
    """Write a small labelled CSV and return its path."""
    df = pd.DataFrame(
        {
            "Title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E", "Movie F"],
            "Plot": [
                "A brave hero fights dragons and saves the kingdom",
                "Two people fall in love during a summer vacation",
                "A detective investigates a mysterious murder case",
                "Soldiers battle through enemy lines in World War Two",
                "A young wizard discovers magical powers at school",
                "Two lovers reunite after years apart in Paris",
            ],
            "Genre": [
                "Action|Adventure",
                "Romance|Drama",
                "Mystery|Thriller",
                "War|Drama",
                "Fantasy|Adventure",
                "Romance|Drama",
            ],
        }
    )
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def trained_artefacts(tmp_path, toy_dataset):
    """Clean, train, and return paths to all saved artefacts."""
    cleaned = tmp_path / "cleaned.csv"
    clean_and_save(str(toy_dataset), str(cleaned))

    vec_path = tmp_path / "vec.joblib"
    nb_path = tmp_path / "nb.joblib"
    lr_path = tmp_path / "lr.joblib"

    train_and_save_models(
        str(cleaned),
        str(vec_path),
        {"nb": str(nb_path), "lr": str(lr_path)},
        test_size=0.34,
        random_state=0,
        min_genre_samples=1,
    )
    return {"vec": vec_path, "nb": nb_path, "lr": lr_path}


class TestCleaningStage:
    def test_cleaned_csv_is_created(self, tmp_path, toy_dataset):
        out = tmp_path / "cleaned.csv"
        clean_and_save(str(toy_dataset), str(out))
        assert out.exists()

    def test_cleaned_csv_has_no_nulls(self, tmp_path, toy_dataset):
        out = tmp_path / "cleaned.csv"
        clean_and_save(str(toy_dataset), str(out))
        df = pd.read_csv(out)
        assert df["Plot"].isnull().sum() == 0
        assert df["Genre"].isnull().sum() == 0


class TestTrainingStage:
    def test_all_artefact_files_are_created(self, trained_artefacts):
        for key, path in trained_artefacts.items():
            assert path.exists(), f"Missing artefact: {key}"

    def test_performance_summary_json_is_written(self, tmp_path, toy_dataset):
        cleaned = tmp_path / "cleaned.csv"
        clean_and_save(str(toy_dataset), str(cleaned))
        vec_path = tmp_path / "vec.joblib"
        _, _, summary = train_and_save_models(
            str(cleaned),
            str(vec_path),
            {"nb": str(tmp_path / "nb.joblib"), "lr": str(tmp_path / "lr.joblib")},
            test_size=0.34,
            random_state=0,
            min_genre_samples=1,
        )
        assert "model_performance" in summary
        assert "naive_bayes" in summary["model_performance"]
        assert "logistic_regression" in summary["model_performance"]


class TestInferenceStage:
    def test_predict_returns_one_label_per_plot(self, trained_artefacts):
        plots = ["A hero fights a dragon", "A love story in Paris"]
        preds = predict(plots, trained_artefacts["vec"], trained_artefacts["lr"])
        assert len(preds) == 2
        assert all(isinstance(p, str) for p in preds)

    def test_predict_genre_returns_string(self, trained_artefacts):
        result = predict_genre(
            "A detective solves a mystery",
            trained_artefacts["vec"],
            trained_artefacts["lr"],
        )
        assert isinstance(result, str)

    def test_predict_proba_sums_to_one(self, trained_artefacts):
        results = predict_proba(
            ["A brave soldier fights in war"],
            trained_artefacts["vec"],
            trained_artefacts["lr"],
        )
        assert len(results) == 1
        assert abs(sum(results[0].values()) - 1.0) < 1e-5
