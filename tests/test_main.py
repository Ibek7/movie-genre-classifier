from src.main import build_parser
import pytest


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.data_path == "data/processed/cleaned_plots.csv"
    assert args.vectorizer_path == "models/production_vectorizer.joblib"
    assert args.nb_model_path == "models/nb.joblib"
    assert args.lr_model_path == "models/lr.joblib"
    assert args.test_size == 0.2
    assert args.random_state == 42
    assert args.max_features == 5000
    assert args.min_genre_samples == 100


def test_build_parser_custom_values():
    parser = build_parser()
    args = parser.parse_args([
        "--data-path", "custom.csv",
        "--max-features", "2500",
        "--min-genre-samples", "10",
    ])

    assert args.data_path == "custom.csv"
    assert args.max_features == 2500
    assert args.min_genre_samples == 10


def test_build_parser_rejects_non_positive_max_features():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--max-features", "0"])


def test_build_parser_rejects_test_size_out_of_range():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--test-size", "1.2"])