import argparse

from src.models.train import train_and_save_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train movie genre classification models")
    parser.add_argument("--data-path", default="data/processed/cleaned_plots.csv")
    parser.add_argument("--vectorizer-path", default="models/production_vectorizer.joblib")
    parser.add_argument("--nb-model-path", default="models/nb.joblib")
    parser.add_argument("--lr-model-path", default="models/lr.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-genre-samples", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_and_save_models(
        data_path=args.data_path,
        vec_path=args.vectorizer_path,
        model_paths={"nb": args.nb_model_path, "lr": args.lr_model_path},
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        min_genre_samples=args.min_genre_samples,
    )


if __name__ == "__main__":
    main()
