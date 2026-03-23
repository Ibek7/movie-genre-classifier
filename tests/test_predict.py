import pandas as pd
import pytest

from src.models.predict import predict, predict_from_csv


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